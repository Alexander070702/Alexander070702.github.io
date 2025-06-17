import json
import random
import torch
import copy

class TetrisGame:
    COLS = 10
    ROWS = 20
    BASE = ['I','O','T','S','Z','J','L']
    TETROMINOES = {
        'I': [[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]],
        'O': [[1,1],[1,1]],
        'T': [[0,1,0],[1,1,1],[0,0,0]],
        'S': [[0,1,1],[1,1,0],[0,0,0]],
        'Z': [[1,1,0],[0,1,1],[0,0,0]],
        'J': [[1,0,0],[1,1,1],[0,0,0]],
        'L': [[0,0,1],[1,1,1],[0,0,0]]
    }

    def __init__(self, cols=None, rows=None):
        self.cols = cols or self.COLS
        self.rows = rows or self.ROWS
        self._next_idx = 0
        self._predefined = [self.BASE[i % len(self.BASE)] for i in range(1000)]
        self.reset()

    def reset(self):
        self.board = [[0]*self.cols for _ in range(self.rows)]
        self.game_over = False
        self.n_actions = 0
        self._next_idx = 0
        self.current_piece = self.spawn_piece()

    def spawn_piece(self):
        t_type = self._predefined[self._next_idx]
        self._next_idx = (self._next_idx + 1) % len(self._predefined)
        shape = copy.deepcopy(self.TETROMINOES[t_type])
        x = (self.cols - len(shape[0])) // 2
        piece = {'type': t_type, 'shape': shape, 'x': x, 'y': 0}
        if self.collides(piece):
            self.game_over = True
        return piece

    def collides(self, p):
        for r,row in enumerate(p['shape']):
            for c,val in enumerate(row):
                if not val: continue
                x,y = p['x']+c, p['y']+r
                if x<0 or x>=self.cols or y>=self.rows or (y>=0 and self.board[y][x]):
                    return True
        return False

    def clear_lines(self):
        newb = [r for r in self.board if not all(r)]
        cleared = self.rows - len(newb)
        self.board = [[0]*self.cols for _ in range(cleared)] + newb
        return cleared

    def lock_piece(self):
        p = self.current_piece
        for r,row in enumerate(p['shape']):
            for c,val in enumerate(row):
                if val:
                    self.board[p['y']+r][p['x']+c] = 1
        self.clear_lines()
        self.current_piece = self.spawn_piece()

    def get_state_key(self):
        return json.dumps({'board': self.board, 'next': self.current_piece['type']}, separators=(',',':'))

    def get_terminal_moves(self):
        if self.game_over:
            return []
        orig = self.current_piece
        base = self.TETROMINOES[orig['type']]
        moves = []
        rots = [0] if orig['type']=='O' else [0,1,2,3]
        for rot in rots:
            shape = copy.deepcopy(base)
            for _ in range(rot):
                shape = list(map(list, zip(*shape[::-1])))
            angle = rot*90
            w = len(shape[0])
            for x in range(self.cols-w+1):
                test = {'type':orig['type'], 'shape':shape, 'x':x, 'y':0}
                if self.collides(test): continue
                y = 0
                while not self.collides({'type':orig['type'],'shape':shape,'x':x,'y':y+1}) and y < self.rows:
                    y += 1
                test['y'] = y
                key = f"{orig['type']}_r{angle}_x{x}"
                moves.append({'action_key': key, 'placement': test})
        return moves

    def step(self, placement):
        self.current_piece = placement
        self.lock_piece()
        self.n_actions += 1
        return None, None, not self.game_over

class TrajectoryBalanceAgent:
    def __init__(self, lr=0.01):
        self.log_flows = {}
        self.logZ = 0.0
        self.lr = lr

    def _ensure(self, s, a):
        self.log_flows.setdefault(s, {})
        if a not in self.log_flows[s]:
            self.log_flows[s][a] = torch.log(torch.tensor(0.5 + random.random()))

    def get_flow(self, s, a):
        self._ensure(s, a)
        return torch.exp(self.log_flows[s][a]).item()

    def sample(self, s, cands):
        for c in cands:
            self._ensure(s, c['action_key'])
        logs = torch.stack([self.log_flows[s][c['action_key']] for c in cands])
        probs = torch.softmax(logs, 0).tolist()
        idx = random.choices(range(len(cands)), weights=probs, k=1)[0]
        return cands[idx]

    def update(self, traj, reward):
        R = max(reward, 1e-2)
        logR = torch.log(torch.tensor(R))
        sum_logp = 0
        for s,a in traj:
            logs = torch.tensor(list(self.log_flows[s].values()))
            denom = torch.logsumexp(logs, 0)
            sum_logp += (self.log_flows[s][a] - denom)
        diff = sum_logp - (logR - self.logZ)
        self.logZ += self.lr * diff.item()
        for s,a in traj:
            self.log_flows[s][a] -= self.lr * diff

def run_episode(env, agent):
    env.reset()
    state_map = {}
    traj = []
    while True:
        key = env.get_state_key()
        moves = env.get_terminal_moves()
        if not moves:
            break
        # record top-3 by flow
        flows = [(agent.get_flow(key, m['action_key']), m['action_key']) for m in moves]
        flows.sort(reverse=True)
        state_map[key] = [a for _, a in flows[:3]]
        choice = agent.sample(key, moves)
        traj.append((key, choice['action_key']))
        _, _, ok = env.step(choice['placement'])
        if not ok:
            break
    return env.n_actions, traj, state_map

def main(episodes=1000, out='best.json'):
    agent = TrajectoryBalanceAgent(0.01)
    best_reward = -1
    best_map = {}
    for ep in range(1, episodes+1):
        r, tr, sm = run_episode(TetrisGame(), agent)
        if r > best_reward:
            best_reward = r
            best_map = sm
            with open(out, 'w') as f:
                json.dump(best_map, f)
            print(f"New best {best_reward} at ep {ep}")
        else:
            print(f"Ep {ep}: r={r} (best={best_reward})")
    print(f"Done. Best reward: {best_reward}")

if __name__ == '__main__':
    main()
