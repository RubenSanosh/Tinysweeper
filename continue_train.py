"""Continue training TinySweeper from the existing model.pth.

Run from the same folder as train.py / env.py:   python continue_train.py
Only overwrites model.pth when the win rate beats the current best, and backs up the old one first.
"""

import shutil

import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import get_linear_fn

from env import MinesweeperEnv
from train import evaluate_win_rate, make_dqn

STEPS = 100_000          # how many more steps to train
LR = 2e-4                # lower than the original 5e-4: we're refining, not starting over
EVAL_EVERY = 20_000
EVAL_GAMES = 500         # same seeded games each time, so comparisons are fair


class SaveBest(BaseCallback):
    def __init__(self, start_wr):
        super().__init__()
        self.best = start_wr
        self.next_eval = EVAL_EVERY

    def _on_step(self):
        if self.num_timesteps >= self.next_eval:
            self.next_eval += EVAL_EVERY
            wr = evaluate_win_rate(self.model.q_net.q_net, n_games=EVAL_GAMES)
            tag = ""
            if wr > self.best:
                self.best = wr
                torch.save(self.model.q_net.q_net.state_dict(), "model.pth")
                tag = "  <- new best, saved model.pth"
            print(f"step {self.num_timesteps:>7,}: {wr*100:.1f}% (best {self.best*100:.1f}%){tag}", flush=True)
        return True


def main():
    sd = torch.load("model.pth", map_location="cpu", weights_only=True)
    shutil.copy("model.pth", "model_before_continue.pth")

    dqn = make_dqn(MinesweeperEnv(), lr=LR)
    dqn.q_net.q_net.load_state_dict(sd)
    dqn.q_net_target.q_net.load_state_dict(sd)

    # Low exploration (we already know how to play). The schedule object is built when the
    # DQN is created, so it must be replaced -- just changing the eps attributes does nothing.
    dqn.exploration_schedule = get_linear_fn(0.05, 0.01, 0.3)
    # 1 gradient step per env step instead of 4 -> roughly 2-3x faster.
    dqn.gradient_steps = 1

    start = evaluate_win_rate(dqn.q_net.q_net, n_games=EVAL_GAMES)
    print(f"Starting win rate: {start*100:.1f}%", flush=True)
    dqn.learn(total_timesteps=STEPS, callback=SaveBest(start))
    print("Done. Best weights are in model.pth (previous copy: model_before_continue.pth).")


if __name__ == "__main__":
    main()
