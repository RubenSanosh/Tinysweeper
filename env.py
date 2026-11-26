"""6x6 Minesweeper (4 mines) Gymnasium environment for TinySweeper."""

from collections import deque
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class MinesweeperEnv(gym.Env):
    """6x6 Minesweeper with 4 mines and first-click safety."""

    metadata = {"render_modes": ["ansi"]}
    SIZE = 6
    N_MINES = 4
    N_CELLS = SIZE * SIZE

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        # Flat 72-dim observation: per cell [revealed (0/1), adjacent mines (0-8)].
        self.observation_space = spaces.Box(
            low=0.0, high=8.0, shape=(self.N_CELLS * 2,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.N_CELLS)
        self.max_steps = self.N_CELLS * 2  # Prevent stalls on repeated clicks.
        self.mines = np.zeros((self.SIZE, self.SIZE), dtype=bool)
        self.counts = np.zeros((self.SIZE, self.SIZE), dtype=np.int32)
        self.revealed = np.zeros((self.SIZE, self.SIZE), dtype=bool)
        self.mines_placed = False
        self.steps = 0
        self.won = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.mines.fill(False)
        self.counts.fill(0)
        self.revealed.fill(False)
        self.mines_placed = False
        self.steps = 0
        self.won = False
        return self._obs(), {}

    def _place_mines(self, safe_r, safe_c):
        # Place mines after the first click so a 90%+ win rate is achievable
        # (random pre-placement caps theoretical max win rate at 32/36 = 88.9%).
        safe_idx = safe_r * self.SIZE + safe_c
        candidates = [i for i in range(self.N_CELLS) if i != safe_idx]
        chosen = self.np_random.choice(candidates, size=self.N_MINES, replace=False)
        for idx in chosen:
            self.mines[idx // self.SIZE, idx % self.SIZE] = True

        for r in range(self.SIZE):
            for c in range(self.SIZE):
                if self.mines[r, c]:
                    continue
                r0, r1 = max(0, r - 1), min(self.SIZE, r + 2)
                c0, c1 = max(0, c - 1), min(self.SIZE, c + 2)
                self.counts[r, c] = int(self.mines[r0:r1, c0:c1].sum())
        self.mines_placed = True

    def _obs(self):
        obs = np.zeros((self.N_CELLS, 2), dtype=np.float32)
        rev = self.revealed.reshape(-1)
        obs[rev, 0] = 1.0
        obs[rev, 1] = self.counts.reshape(-1)[rev]
        return obs.reshape(-1)

    def _flood_reveal(self, start_r, start_c):
        # BFS flood-fill matches standard Minesweeper on zero-count cells.
        newly_revealed = 0
        queue = deque([(start_r, start_c)])
        while queue:
            r, c = queue.popleft()
            if self.revealed[r, c] or self.mines[r, c]:
                continue
            self.revealed[r, c] = True
            newly_revealed += 1
            if self.counts[r, c] == 0:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = r + dr, c + dc
                        if (
                            0 <= nr < self.SIZE
                            and 0 <= nc < self.SIZE
                            and not self.revealed[nr, nc]
                        ):
                            queue.append((nr, nc))
        return newly_revealed

    def step(self, action):
        self.steps += 1
        r, c = divmod(int(action), self.SIZE)

        if not self.mines_placed:
            self._place_mines(r, c)

        # Penalize clicking an already-revealed cell so the agent stops wasting moves.
        if self.revealed[r, c]:
            truncated = self.steps >= self.max_steps
            return self._obs(), -5.0, False, truncated, {"won": False, "invalid": True}

        if self.mines[r, c]:
            self.revealed[r, c] = True
            return self._obs(), -100.0, True, False, {"won": False, "invalid": False}

        newly = self._flood_reveal(r, c)
        reward = float(newly)

        safe_total = self.N_CELLS - self.N_MINES
        if int(self.revealed.sum()) == safe_total:
            self.won = True
            reward += 50.0
            return self._obs(), reward, True, False, {"won": True, "invalid": False}

        truncated = self.steps >= self.max_steps
        return self._obs(), reward, False, truncated, {"won": False, "invalid": False}

    def render(self):
        lines = []
        for r in range(self.SIZE):
            row = []
            for c in range(self.SIZE):
                if not self.revealed[r, c]:
                    row.append(".")
                elif self.mines[r, c]:
                    row.append("*")
                else:
                    row.append(str(self.counts[r, c]))
            lines.append(" ".join(row))
        board_str = "\n".join(lines)
        if self.render_mode == "ansi":
            return board_str
        print(board_str + "\n")
        return None


if __name__ == "__main__":
    env = MinesweeperEnv()
    obs, _ = env.reset(seed=42)
    assert obs.shape == (72,) and obs.sum() == 0.0
    obs, rew, term, trunc, info = env.step(0)
    assert rew >= 1.0 and not info["invalid"]
    env.render()
    print("env.py self-check passed.")
