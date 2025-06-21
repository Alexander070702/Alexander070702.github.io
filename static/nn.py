#!/usr/bin/env python3
"""
pretrain_and_export.py

Train a GFlowNetAgent on Tetris (10×20), save a PyTorch checkpoint, and export a correctly-batched ONNX model for front-end inference.
Usage:
    python3 nn.py --episodes 10000 --checkpoint-path gfn.pt --onnx-path gfn.onnx
"""

import argparse
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ------------------------------------------------------------------------------
# --- Environment (standard Tetris, fixed 10×20, 7-piece sequence) ------------
# ------------------------------------------------------------------------------
BOARD_WIDTH, BOARD_HEIGHT = 10, 20
PIECE_ORDER = ['I','O','T','S','Z','J','L']
MAX_STEPS = 2000

TETROMINOES = {
    'I': np.array([[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]]),
    'O': np.array([[1,1],[1,1]]),
    'T': np.array([[0,1,0],[1,1,1],[0,0,0]]),
    'S': np.array([[0,1,1],[1,1,0],[0,0,0]]),
    'Z': np.array([[1,1,0],[0,1,1],[0,0,0]]),
    'J': np.array([[1,0,0],[1,1,1],[0,0,0]]),
    'L': np.array([[0,0,1],[1,1,1],[0,0,0]])
}

class TetrisEnv:
    def __init__(self): self.reset()
    def reset(self):
        self.board      = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.int32)
        self.next_idx   = 0
        self.step_count = 0
        self.score      = 0
        self.current    = self._spawn()
        return self._get_state()
    def _spawn(self):
        t = PIECE_ORDER[self.next_idx]
        self.next_idx = (self.next_idx + 1) % len(PIECE_ORDER)
        return {'type': t, 'shape': TETROMINOES[t].copy(), 'x':0,'y':0}
    def _get_state(self):
        return {'board': self.board.copy(),
                'piece': {'type': self.current['type'],
                          'shape': self.current['shape'].tolist(),
                          'x':0,'y':0}}
    def _collides(self, shape, x, y):
        h,w = shape.shape
        for i in range(h):
            for j in range(w):
                if shape[i,j]:
                    xi, yi = x+j, y+i
                    if xi<0 or xi>=BOARD_WIDTH or yi<0 or yi>=BOARD_HEIGHT:
                        return True
                    if self.board[yi,xi]: return True
        return False
    def _clear(self):
        new, cleared = [], 0
        for row in self.board:
            if row.all(): cleared+=1
            else: new.append(row)
        for _ in range(cleared): new.insert(0, np.zeros(BOARD_WIDTH, dtype=np.int32))
        self.board = np.stack(new, axis=0)
        return cleared
    def step(self, action):
        r,x = action
        shape = np.rot90(self.current['shape'], -r) if r>0 else self.current['shape']
        y = 0
        while not self._collides(shape, x, y+1): y+=1
        h,w = shape.shape
        for i in range(h):
            for j in range(w):
                if shape[i,j]:
                    yi, xi = y+i, x+j
                    if 0<=yi<BOARD_HEIGHT and 0<=xi<BOARD_WIDTH:
                        self.board[yi,xi] = 1
        lines = self._clear()
        self.score += lines
        self.current = self._spawn()
        self.step_count += 1
        done   = (self.step_count >= MAX_STEPS)
        reward = float(lines + 1e-8)
        return self._get_state(), reward, done

# ------------------------------------------------------------------------------
# --- Model Definition (Residual + Self-Attention) ---------------------------
# ------------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)
    def forward(self, x):
        h = torch.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return torch.relu(h + x)

class AttentionPool(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.qkv   = nn.Linear(dim, dim*3)
        self.proj  = nn.Linear(dim, dim)
        self.heads = heads
    def forward(self, x):
        B,N,D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, D//self.heads)
        q,k,v = qkv.unbind(2)
        att = (q @ k.transpose(-1,-2)) * (1/(D//self.heads)**0.5)
        w   = torch.softmax(att, dim=-1)
        h   = (w @ v).transpose(1,2).reshape(B, N, D)
        return self.proj(h)

class GFlowNetAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(1, 16, 3, padding=1)
        self.res1    = ResidualBlock(16)
        self.res2    = ResidualBlock(16)
        self.pool    = nn.AvgPool2d(kernel_size=(4,2), stride=(4,2))
        self.attn    = AttentionPool(5*5*16, heads=4)
        self.fc1     = nn.Linear(5*5*16, 128)
        self.fc2     = nn.Linear(128 + len(PIECE_ORDER), 4*BOARD_WIDTH)
        self.logZ    = nn.Parameter(torch.zeros(()))

    def forward(self, board, piece_onehot):
        if board.dim() == 2:
            x = board.unsqueeze(0).unsqueeze(0).float()
            B = 1
        else:
            x = board.unsqueeze(1).float()
            B = board.shape[0]
        x = torch.relu(self.conv_in(x))
        x = self.res1(x); x = self.res2(x)
        x = self.pool(x).flatten(1)
        x = self.attn(x.unsqueeze(1)).squeeze(1)
        x = torch.relu(self.fc1(x))
        if piece_onehot.dim() == 1:
            p = piece_onehot.unsqueeze(0)
        else:
            p = piece_onehot
        x = torch.cat([x, p], dim=1)
        logF = self.fc2(x)
        return logF.view(B, 4, BOARD_WIDTH)

# ------------------------------------------------------------------------------
# --- Loss & Utility -----------------------------------------------------------
# ------------------------------------------------------------------------------
def trajectory_balance_loss(agent, traj, final_reward):
    logR     = torch.log(torch.tensor(final_reward + 1e-8))
    sum_logf = torch.zeros((), device=agent.logZ.device)
    for b,p,idx in traj:
        logF = agent(b, p).flatten()
        sum_logf += logF[idx]
    return (sum_logf + logR - agent.logZ).pow(2)

def one_hot_piece(ptype):
    v = torch.zeros(len(PIECE_ORDER))
    v[PIECE_ORDER.index(ptype)] = 1.0
    return v

# ------------------------------------------------------------------------------
# --- Main: Training & Export -------------------------------------------------
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes',       type=int,   default=5000)
    parser.add_argument('--lr',             type=float, default=1e-3)
    parser.add_argument('--topk',           type=int,   default=3)
    parser.add_argument('--log-interval',   type=int,   default=100)
    parser.add_argument('--checkpoint-path',type=str,   default='gfn.pt')
    parser.add_argument('--onnx-path',      type=str,   default='gfn.onnx')
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env    = TetrisEnv()
    agent  = GFlowNetAgent().to(device)
    opt    = optim.Adam(agent.parameters(), lr=args.lr)
    best   = deque(maxlen=args.topk)

    # prepare lists for logging
    ep_rewards = []
    ep_losses  = []

    for ep in range(1, args.episodes+1):
        state, traj, done = env.reset(), [], False
        while not done:
            board = torch.tensor(state['board'], device=device)
            piece = one_hot_piece(state['piece']['type']).to(device)
            logF  = agent(board, piece).detach()
            probs = torch.softmax(logF.flatten(), dim=0)
            idx   = torch.multinomial(probs, 1).item()
            r_act, x_act = divmod(idx, BOARD_WIDTH)
            traj.append((board, piece, idx))
            state, reward, done = env.step((r_act, x_act))
        score = env.score

        loss = trajectory_balance_loss(agent, traj, score + 1e-8)
        opt.zero_grad(); loss.backward(); opt.step()

        best.append(score)
        ep_rewards.append(score)
        ep_losses.append(loss.item())

        if ep % args.log_interval == 0 or ep == 1 or ep == args.episodes:
            avg_reward = np.mean(ep_rewards[-args.log_interval:])
            avg_loss   = np.mean(ep_losses[-args.log_interval:])
            print(f"Ep {ep}/{args.episodes} | "
                  f"Score={score:.1f} | "
                  f"AvgScore(last{args.log_interval})={avg_reward:.1f} | "
                  f"Loss={loss.item():.4f} | "
                  f"AvgLoss(last{args.log_interval})={avg_loss:.4f}")

    torch.save(agent.state_dict(), args.checkpoint_path)
    print(f"✅ Saved checkpoint: {args.checkpoint_path}")

    # export ONNX with batch dim
    dummy_b = torch.zeros(1, BOARD_HEIGHT, BOARD_WIDTH, device=device)
    dummy_p = torch.zeros(1, len(PIECE_ORDER),   device=device)
    torch.onnx.export(
        agent,
        (dummy_b, dummy_p),
        args.onnx_path,
        export_params=True,
        opset_version=11,
        input_names=['board','piece_onehot'],
        output_names=['logF'],
        dynamic_axes={'board':{0:'batch'}, 'piece_onehot':{0:'batch'}, 'logF':{0:'batch'}}
    )
    print(f"✅ Exported ONNX model: {args.onnx_path}")

if __name__ == '__main__':
    main()
