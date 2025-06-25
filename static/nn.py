#!/usr/bin/env python3
"""
train_and_export_gfn_tetris_path_b.py

GFlowNet agent for Tetris using Path B:
 - A "trajectory" is a single piece placement.
 - The GFN reward R(x) is a heuristic score of the *resulting board state*.
 - The agent learns a policy to sample board states proportional to their heuristic score.

Key Changes:
 - Added score_board_state(board) to evaluate static board quality.
 - Training loop uses this board score for the Trajectory Balance loss.
 - Fixed NameError by moving FlowNet class definition.
 - Kept imitation learning and stabilization techniques.
 - FIXED: Replaced AdaptiveAvgPool2d with AvgPool2d for ONNX compatibility.

Usage:
    python3 train_and_export_gfn_tetris_path_b.py \
        --episodes 50000 --batch-size 16 \
        --imitation-epochs 2000 --eps-decay-episodes 500 \
        --lr 1e-4 --checkpoint-path gfn_path_b.pt \
        --onnx-path gfn_path_b.onnx
"""
import argparse
import random
import math
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ------------------------------------------------------------------------------
# --- Environment --------------------------------------------------------------
# ------------------------------------------------------------------------------
BOARD_WIDTH, BOARD_HEIGHT = 6, 10
N_ROTATIONS = {'I': 2, 'O': 1, 'T': 4, 'S': 2, 'Z': 2, 'J': 4, 'L': 4}
TETROMINOES = {
    'I': np.array([[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]], dtype=np.int32),
    'O': np.array([[1,1],[1,1]], dtype=np.int32),
    'T': np.array([[0,1,0],[1,1,1],[0,0,0]], dtype=np.int32),
    'S': np.array([[0,1,1],[1,1,0],[0,0,0]], dtype=np.int32),
    'Z': np.array([[1,1,0],[0,1,1],[0,0,0]], dtype=np.int32),
    'J': np.array([[1,0,0],[1,1,1],[0,0,0]], dtype=np.int32),
    'L': np.array([[0,0,1],[1,1,1],[0,0,0]], dtype=np.int32)
}

class TetrisEnv:
    def __init__(self):
        self.board = None
        self.bag = []
        self.reset()

    def reset(self):
        self.board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.int32)
        self.step_count = 0
        self.score = 0
        self.game_over = False
        self._refill_bag()
        self.current = self._spawn()
        if self._collides(self.current['shape'], self.current['x'], self.current['y']):
            self.game_over = True
        return self._get_state()

    def _refill_bag(self):
        self.bag = list(TETROMINOES.keys())
        random.shuffle(self.bag)

    def _spawn(self):
        if len(self.bag) == 0:
            self._refill_bag()
        ptype = self.bag.pop()
        shape = TETROMINOES[ptype].copy()
        x = (BOARD_WIDTH - shape.shape[1]) // 2
        y = 0
        return {'type': ptype, 'shape': shape, 'x': x, 'y': y}

    def _get_state(self):
        return {
            'board': self.board.copy(),
            'piece': {
                'type': self.current['type'],
                'shape': self.current['shape'].tolist(),
                'x': self.current['x'],
                'y': self.current['y']
            }
        }

    def _collides(self, shape, x, y):
        h, w = shape.shape
        for i in range(h):
            for j in range(w):
                if shape[i, j]:
                    xi, yi = x + j, y + i
                    if xi < 0 or xi >= BOARD_WIDTH or yi < 0 or yi >= BOARD_HEIGHT:
                        return True
                    if self.board[yi, xi]:
                        return True
        return False

    def _clear_lines(self):
        new_rows = []
        cleared = 0
        for row_idx in range(BOARD_HEIGHT):
            if np.all(self.board[row_idx]):
                cleared += 1
            else:
                new_rows.append(self.board[row_idx])
        
        for _ in range(cleared):
            new_rows.insert(0, np.zeros(BOARD_WIDTH, dtype=np.int32))
        
        if cleared > 0:
            self.board = np.stack(new_rows, axis=0)
        return cleared

    def valid_moves(self):
        moves = []
        base = TETROMINOES[self.current['type']]
        for rot in range(N_ROTATIONS[self.current['type']]):
            shape = np.rot90(base, -rot)
            w = shape.shape[1]
            for x_pos in range(BOARD_WIDTH - w + 1):
                y = 0
                while not self._collides(shape, x_pos, y + 1):
                    y += 1
                if not self._collides(shape, x_pos, y):
                    moves.append((rot, x_pos))
        return moves if moves else [(0, (BOARD_WIDTH - base.shape[1]) // 2)]


    def step(self, action):
        if self.game_over:
            return self._get_state(), 0.0, True

        rot, x_pos = action
        base = TETROMINOES[self.current['type']]
        shape = np.rot90(base, -rot)

        y = 0
        while not self._collides(shape, x_pos, y + 1):
            y += 1

        for i in range(shape.shape[0]):
            for j in range(shape.shape[1]):
                if shape[i, j]:
                    self.board[y + i, x_pos + j] = 1

        lines_cleared = self._clear_lines()
        
        reward = {0: 0, 1: 100, 2: 300, 3: 500, 4: 800}[lines_cleared]
        self.score += reward
        self.step_count += 1
        
        self.current = self._spawn()
        if self._collides(self.current['shape'], self.current['x'], self.current['y']):
            self.game_over = True
            
        done = self.game_over
        return self._get_state(), reward, done

# ------------------------------------------------------------------------------
# --- Heuristic & GFN Reward Function ------------------------------------------
# ------------------------------------------------------------------------------
def score_board_state(board):
    heights = np.zeros(BOARD_WIDTH, dtype=int)
    for col in range(BOARD_WIDTH):
        if np.any(board[:, col]):
            heights[col] = BOARD_HEIGHT - np.where(board[:, col])[0][0]
        else:
            heights[col] = 0
    
    aggregate_height = np.sum(heights)
    bumpiness = np.sum(np.abs(heights[:-1] - heights[1:]))

    holes = 0
    for col in range(BOARD_WIDTH):
        col_filled = False
        for row in range(BOARD_HEIGHT):
            if board[row, col]:
                col_filled = True
            elif col_filled:
                holes += 1
    
    lines_cleared = np.sum([np.all(board[row]) for row in range(BOARD_HEIGHT)])

    w_height = 0.51
    w_holes = 0.76
    w_bumpiness = 0.18
    w_lines = 0.84
    
    score = 200.0 - w_height * aggregate_height - w_holes * holes - w_bumpiness * bumpiness + w_lines * lines_cleared**2
    
    return max(1e-6, score)

def heuristic_action(env, board, ptype):
    valid = env.valid_moves()
    if not valid:
        return (0, 0)
        
    best_score = -1e9
    best_move = valid[0]
    
    base = TETROMINOES[ptype]

    for rot, x_pos in valid:
        sim_board = board.copy()
        shape = np.rot90(base, -rot)
        
        y = 0
        while not env._collides(shape, x_pos, y + 1):
            y += 1
        
        for i in range(shape.shape[0]):
            for j in range(shape.shape[1]):
                if shape[i,j]:
                    sim_board[y + i, x_pos + j] = 1

        score = score_board_state(sim_board)

        if score > best_score:
            best_score, best_move = score, (rot, x_pos)
            
    return best_move

# ------------------------------------------------------------------------------
# --- Model Definition ---------------------------------------------------------
# ------------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        h = torch.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return torch.relu(h + x)

class GFNNet(nn.Module):
    def __init__(self):
        super().__init__()
        channels = 32
        self.conv_in = nn.Conv2d(1, channels, 3, padding=1)
        self.res1 = ResidualBlock(channels)
        self.res2 = ResidualBlock(channels)
        
        # FIXED: Replaced AdaptiveAvgPool2d with a standard AvgPool2d for ONNX compatibility.
        # This pools the (20, 10) feature map to (5, 5).
        self.pool = nn.AvgPool2d(kernel_size=(4, 2), stride=(4, 2))
        
        # FIXED: Update the input features for the linear layer to match the new pooled size (5*5*channels).
        self.fc1 = nn.Linear(5 * 5 * channels, 128)
        self.fc2 = nn.Linear(128 + len(TETROMINOES), 4 * BOARD_WIDTH)

    def forward(self, board, piece_onehot):
        if board.dim() == 2:
            x = board.unsqueeze(0).unsqueeze(0).float()
        elif board.dim() == 3:
            x = board.unsqueeze(1).float()
        else:
            x = board.float()
        x = torch.relu(self.conv_in(x))
        x = self.res1(x)
        x = self.res2(x)
        x = self.pool(x).flatten(1)
        x = torch.relu(self.fc1(x))
        p = piece_onehot.unsqueeze(0) if piece_onehot.dim() == 1 else piece_onehot
        x = torch.cat([x, p], dim=1)
        return self.fc2(x).view(-1, 4, BOARD_WIDTH)

# ------------------------------------------------------------------------------
# --- GFlowNet Class & Loss ----------------------------------------------------
# ------------------------------------------------------------------------------
class FlowNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.F = GFNNet()
        self.B = GFNNet()
        self.logZ = nn.Parameter(torch.zeros(()))
    def forward(self, b, p):
        return self.F(b, p)

def one_hot_piece(ptype, device=None):
    v = torch.zeros(len(TETROMINOES), device=device)
    v[list(TETROMINOES.keys()).index(ptype)] = 1.0
    return v

def trajectory_balance_loss(model, trajectory, log_reward, device):
    b0, p0, idx, b1, p1 = trajectory
    
    log_Pf = model.F(b0, p0).flatten()[idx]
    log_Pb = model.B(b1, p1).flatten()[idx]

    return (model.logZ + log_Pf - log_reward - log_Pb).pow(2)

# ------------------------------------------------------------------------------
# --- Main: Training & Export -------------------------------------------------
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=50000)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--imitation-epochs', type=int, default=2000)
    parser.add_argument('--eps-decay-episodes', type=int, default=5000)
    parser.add_argument('--checkpoint-path', type=str, default='gfn_path_b.pt')
    parser.add_argument('--onnx-path', type=str, default='gfn_path_b.onnx')
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    env = TetrisEnv()
    model = FlowNet().to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=250, min_lr=1e-6)

    batch_data = []
    ep_scores = deque(maxlen=100)
    ep_losses = deque(maxlen=100)

    eps_start, eps_end = 1.0, 0.05
    
    for ep in range(1, args.episodes + 1):
        if ep <= args.imitation_epochs:
            epsilon = eps_start
        elif ep <= args.imitation_epochs + args.eps_decay_episodes:
            frac = (ep - args.imitation_epochs) / args.eps_decay_episodes
            epsilon = eps_start - frac * (eps_start - eps_end)
        else:
            epsilon = eps_end

        state = env.reset()
        done = False

        while not done:
            board_tensor = torch.tensor(state['board'], device=device, dtype=torch.float32)
            piece_tensor = one_hot_piece(state['piece']['type'], device=device)
            
            valid_moves = env.valid_moves()
            heur_rot, heur_x = heuristic_action(env, state['board'], state['piece']['type'])
            
            if random.random() < epsilon:
                rot, xpos = heur_rot, heur_x
            else:
                with torch.no_grad():
                    logits = model(board_tensor, piece_tensor).squeeze(0)
                    mask = torch.full_like(logits, float('-inf'))
                    for r, x in valid_moves:
                        mask[r, x] = 0.0
                    
                    masked_logits = logits + mask
                    probs = torch.softmax(masked_logits.flatten(), dim=0)
                    
                    if torch.isnan(probs).any() or probs.sum() <= 0:
                        rot, xpos = heur_rot, heur_x
                    else:
                        idx = torch.multinomial(probs, 1).item()
                        rot, xpos = divmod(idx, BOARD_WIDTH)
            
            action = (rot, xpos)
            action_idx = rot * BOARD_WIDTH + xpos

            next_s, _, done = env.step(action)
            
            board_score = score_board_state(next_s['board'])
            
            next_board_tensor = torch.tensor(next_s['board'], device=device, dtype=torch.float32)
            next_piece_tensor = one_hot_piece(next_s['piece']['type'], device=device)
            
            trajectory_step = (board_tensor, piece_tensor, action_idx, next_board_tensor, next_piece_tensor)
            batch_data.append((trajectory_step, board_score))
            
            state = next_s

            if len(batch_data) >= args.batch_size:
                opt.zero_grad()
                
                trajectories, scores = zip(*batch_data)
                log_rewards = torch.log(torch.tensor(scores, device=device, dtype=torch.float32))
                tb_loss = torch.stack([
                    trajectory_balance_loss(model, traj, log_r, device)
                    for traj, log_r in zip(trajectories, log_rewards)
                ]).mean()

                imitation_loss = torch.tensor(0.0, device=device)
                if ep <= args.imitation_epochs:
                    ce = nn.CrossEntropyLoss()
                    b0s, p0s, idxs, _, _ = zip(*trajectories)
                    
                    logits_all = model.F(torch.stack(b0s), torch.stack(p0s)).flatten(1)
                    targets = torch.tensor(idxs, device=device, dtype=torch.long)
                    imitation_loss = ce(logits_all, targets)
                
                loss = tb_loss + imitation_loss
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                
                scheduler.step(tb_loss)
                ep_losses.append(loss.item())
                batch_data.clear()

        ep_scores.append(env.score)

        if ep % 100 == 0 or ep == 1 or ep == args.episodes:
            avg_s = np.mean(ep_scores) if ep_scores else 0.0
            avg_l = np.mean(ep_losses) if ep_losses else 0.0
            print(f"Ep {ep}/{args.episodes} | ε={epsilon:.3f} | AvgScore={avg_s:.2f} | Loss={avg_l:.4f} | LR={opt.param_groups[0]['lr']:.3e}")

    # --- SAVING AND EXPORT ---
    torch.save(model.state_dict(), args.checkpoint_path)
    print(f"✅ Saved checkpoint: {args.checkpoint_path}")

    dummy_b = torch.zeros(1, BOARD_HEIGHT, BOARD_WIDTH, device=device)
    dummy_p = torch.zeros(1, len(TETROMINOES), device=device)
    torch.onnx.export(
        model.F, (dummy_b, dummy_p), args.onnx_path,
        export_params=True, opset_version=11,
        input_names=['board', 'piece_onehot'], output_names=['logits'],
        dynamic_axes={'board': {0: 'batch'}, 'piece_onehot': {0: 'batch'}, 'logits': {0: 'batch'}}
    )
    print(f"✅ Exported ONNX model: {args.onnx_path}")
    
if __name__ == '__main__':
    main()