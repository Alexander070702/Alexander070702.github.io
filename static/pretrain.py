import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

# --- Tetromino definitions ---
TETROMINOES = {
    'I': np.array([[0,0,0,0], [1,1,1,1], [0,0,0,0], [0,0,0,0]]),
    'O': np.array([[1,1], [1,1]]),
    'T': np.array([[0,1,0], [1,1,1], [0,0,0]]),
    'S': np.array([[0,1,1], [1,1,0], [0,0,0]]),
    'Z': np.array([[1,1,0], [0,1,1], [0,0,0]]),
    'J': np.array([[1,0,0], [1,1,1], [0,0,0]]),
    'L': np.array([[0,0,1], [1,1,1], [0,0,0]]),
}
PIECE_ORDER     = ['I','O','T','S','Z','J','L']
BOARD_WIDTH     = 10
BOARD_HEIGHT    = 20
MAX_STEPS       = 2000

# --- Proper Tetris environment ---
class TetrisEnv:
    def __init__(self):
        self.width = BOARD_WIDTH
        self.height = BOARD_HEIGHT
        self.piece_order = PIECE_ORDER
        self.reset()

    def reset(self):
        self.board = np.zeros((self.height, self.width), dtype=int)
        self.next_idx   = 0
        self.step_count = 0
        self.score = 0
        self.current_piece = self._spawn_next()
        return self._get_state()

    def _spawn_next(self):
        t = self.piece_order[self.next_idx]
        self.next_idx = (self.next_idx + 1) % len(self.piece_order)
        return {'type': t,
                'shape': TETROMINOES[t].copy().tolist(),
                'x': 0, 'y': 0}

    def _get_state(self):
        return {'board': self.board.tolist(),
                'piece': {'type': self.current_piece['type'],
                          'shape': self.current_piece['shape'],
                          'x': 0, 'y': 0}}

    def _collides(self, shape, x, y):
        h, w = shape.shape
        for i in range(h):
            for j in range(w):
                if shape[i, j]:
                    xi = x + j
                    yi = y + i
                    if xi < 0 or xi >= self.width or yi < 0 or yi >= self.height:
                        return True
                    if self.board[yi, xi]:
                        return True
        return False

    def _clear_lines(self):
        new_board = []
        lines_cleared = 0
        for row in self.board:
            if all(cell == 1 for cell in row):
                lines_cleared += 1
            else:
                new_board.append(row.tolist())
        for _ in range(lines_cleared):
            new_board.insert(0, [0]*self.width)
        self.board = np.array(new_board, dtype=int)
        return lines_cleared

    def step(self, action):
        r, x_pos = action
        shape = np.array(self.current_piece['shape'])
        if r > 0:
            shape = np.rot90(shape, -r)
        y = 0
        while not self._collides(shape, x_pos, y+1):
            y += 1
        h, w = shape.shape
        for i in range(h):
            for j in range(w):
                if shape[i, j]:
                    xi = x_pos + j
                    yi = y + i
                    if 0 <= yi < self.height and 0 <= xi < self.width:
                        self.board[yi, xi] = 1
        lines = self._clear_lines()
        self.score += lines
        self.current_piece = self._spawn_next()
        self.step_count += 1
        done = (self.step_count >= MAX_STEPS)
        reward = 1.0 if done else 0.0
        return self._get_state(), reward, done

# --- GFlowNet Agent with learnable logZ ---
class GFlowNetAgent(nn.Module):
    def __init__(self, board_size, n_piece_types, hidden_dim=128):
        super().__init__()
        inp_dim  = board_size + n_piece_types
        out_dim  = 4 * BOARD_WIDTH
        self.net = nn.Sequential(
            nn.Linear(inp_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.logZ = nn.Parameter(torch.zeros(()))

    def forward(self, board, piece_onehot):
        x    = torch.cat([board.flatten().float(), piece_onehot.float()])
        logF = self.net(x).view(4, BOARD_WIDTH)
        return logF

# --- Utility ---
def one_hot(idx, size):
    v = torch.zeros(size)
    v[idx] = 1.0
    return v

# --- Training with top-3 trajectory retention ---
def train_and_export(num_episodes=500, lr=1e-3, output_path='pretrained_flows_tb.json'):
    env = TetrisEnv()
    agent = GFlowNetAgent(BOARD_WIDTH*BOARD_HEIGHT, len(PIECE_ORDER))
    optimizer = optim.Adam(agent.parameters(), lr=lr)

    # Keep only the best 3 episodes by score
    best_episodes = []  # list of tuples (score, episode_flows)
    log_flows = OrderedDict()

    for ep in range(1, num_episodes+1):
        state = env.reset()
        trajectory = []
        episode_flows = OrderedDict()
        done = False
        step = 0

        while not done:
            step += 1
            board_t = torch.tensor(state['board'])
            pt = state['piece']
            pt_idx = PIECE_ORDER.index(pt['type'])
            piece_t = one_hot(pt_idx, len(PIECE_ORDER))

            logF = agent(board_t, piece_t)
            m = logF.max()
            probs = (logF - m).exp()
            probs = probs / probs.sum()

            key = json.dumps(state, separators=(',',':'))
            flows = {f"r{r}_x{c}": float(probs[r,c].item())
                     for r in range(4) for c in range(BOARD_WIDTH)}
            episode_flows[key] = flows

            flat = probs.flatten()
            idx = torch.multinomial(flat, 1).item()
            r_act, x_act = divmod(idx, BOARD_WIDTH)
            trajectory.append((board_t, piece_t, idx))

            state, reward, done = env.step((r_act, x_act))

        # TB update
        final_reward = max(reward, 1e-8)
        logR = torch.log(torch.tensor(final_reward))
        sum_logF = torch.zeros(())
        for b,p,i in trajectory:
            sum_logF += agent(b,p).flatten()[i]
        loss = (sum_logF + logR - agent.logZ).pow(2)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

        score = env.score
        # Insert into best_episodes if qualifies
        best_episodes.append((score, episode_flows))
        # Keep top 3 by score
        best_episodes = sorted(best_episodes, key=lambda x: x[0], reverse=True)[:3]

        # Rebuild global log_flows from best_episodes
        log_flows.clear()
        for _, flows_dict in best_episodes:
            for sk, fdict in flows_dict.items():
                log_flows[sk] = fdict

        print(f"Episode {ep}/{num_episodes}, score={score}, best_scores={[b for b,_ in best_episodes]}")

    # Export only best trajectories' flows
    out = {'log_flows': log_flows, 'logZ': float(agent.logZ.item())}
    with open(output_path, 'w') as f:
        json.dump(out, f, separators=(',',':'), ensure_ascii=False)
    print(f"Saved pretrained flows (top 3 trajectories) to {output_path}")

if __name__ == '__main__':
    train_and_export(num_episodes=100, lr=1e-3)
    print("done :)")
