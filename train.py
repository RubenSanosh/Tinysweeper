"""Train TinySweeper (5,105-param spatial Q-network) via SB3 DQN on 6x6/4 Minesweeper."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork

from env import MinesweeperEnv


class TinyConvQ(nn.Module):
    """3-layer 3x3 ConvNet (~5.1K params) over [unrevealed, revealed, count/8].

    Spatial weight sharing is what lets a 5K-param model beat a 70K flat MLP;
    local 3x3 receptive fields capture mine-counting constraints cell-by-cell.
    """

    def __init__(self, ch=16):
        super().__init__()
        self.ch = ch
        self.c1 = nn.Conv2d(3, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c3 = nn.Conv2d(ch, ch, 3, padding=1)
        self.out = nn.Conv2d(ch, 1, 1)

    def forward(self, x):
        b = x.shape[0]
        # Explicit unrevealed channel prevents hidden [0,0] from aliasing zero border padding.
        rev = x[:, 0::2].view(b, 1, 6, 6)
        num = (x[:, 1::2] / 8.0).view(b, 1, 6, 6)
        img = torch.cat([1.0 - rev, rev, num], dim=1)
        h = F.relu(self.c1(img))
        h = F.relu(self.c2(h))
        h = F.relu(self.c3(h))
        return self.out(h).view(b, 36)


class TinyLinearConvQ(nn.Module):
    """Exact nn.Linear equivalent of TinyConvQ, needed for INT8 dynamic quantization.

    torch.ao.quantization.quantize_dynamic only quantizes nn.Linear, not Conv2d.
    This is mathematically identical because a 3x3 conv is a linear map over patches.
    """

    def __init__(self, ch=16):
        super().__init__()
        self.l1 = nn.Linear(3 * 9, ch)
        self.l2 = nn.Linear(ch * 9, ch)
        self.l3 = nn.Linear(ch * 9, ch)
        self.out = nn.Linear(ch, 1)

    def _conv_lin(self, feat, lin):
        b = feat.shape[0]
        cols = F.unfold(feat, kernel_size=3, padding=1).permute(0, 2, 1)
        return lin(cols).permute(0, 2, 1).view(b, -1, 6, 6)

    def forward(self, x):
        b = x.shape[0]
        rev = x[:, 0::2].view(b, 1, 6, 6)
        num = (x[:, 1::2] / 8.0).view(b, 1, 6, 6)
        img = torch.cat([1.0 - rev, rev, num], dim=1)
        h = F.relu(self._conv_lin(img, self.l1))
        h = F.relu(self._conv_lin(h, self.l2))
        h = F.relu(self._conv_lin(h, self.l3))
        return self.out(h.permute(0, 2, 3, 1)).reshape(b, 36)


class ConvQNet(QNetwork):
    """SB3 QNetwork wrapper that swaps the flat MLP for TinyConvQ."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q_net = TinyConvQ(16)


class ConvDQNPolicy(DQNPolicy):
    def make_q_net(self):
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return ConvQNet(**net_args).to(self.device)


class MaskedDQN(DQN):
    """DQN subclass that restricts exploration and greedy picks to unrevealed cells only.

    Masking is the single biggest accuracy lever — without it the agent wastes
    ~30% of exploration clicks on cells it already knows, inflating sample complexity.
    """

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        if not deterministic and np.random.rand() < self.exploration_rate:
            obs_flat = observation.reshape(-1, 72)
            actions = [
                int(np.random.choice(np.where(r[0::2] < 0.5)[0])) for r in obs_flat
            ]
            return np.array(actions), state
        with torch.no_grad():
            obs_t = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
            if obs_t.ndim == 1:
                obs_t = obs_t.unsqueeze(0)
            q = self.q_net(obs_t)
            rev = torch.as_tensor(
                observation.reshape(-1, 72)[:, 0::2] > 0.5,
                dtype=torch.bool,
                device=self.device,
            )
            q = q.masked_fill(rev, float("-inf"))
            actions = q.argmax(dim=1).cpu().numpy()
        return (actions[0] if observation.ndim == 1 else actions), state


def select_action(q_net, obs):
    """Pick the highest-Q unrevealed cell (works for Conv and Linear variants)."""
    with torch.no_grad():
        q = q_net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0)
    revealed = torch.as_tensor(obs[0::2] > 0.5, dtype=torch.bool)
    q = q.masked_fill(revealed, float("-inf"))
    return int(q.argmax().item())


def evaluate_win_rate(q_net, n_games=500, seed=1000):
    """Evaluate a Q-network (Conv or Linear variant) and return win rate in [0, 1]."""
    env = MinesweeperEnv()
    wins = 0
    for i in range(n_games):
        obs, _ = env.reset(seed=seed + i)
        done = False
        while not done:
            action = select_action(q_net, obs)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        if info.get("won"):
            wins += 1
    return wins / n_games


class WinRateLogCallback(BaseCallback):
    """Log evaluation win rate every eval_freq steps during training."""

    def __init__(self, eval_freq=25_000, n_eval_games=300):
        super().__init__()
        self.eval_freq = eval_freq
        self.n_eval_games = n_eval_games
        self.next_eval = eval_freq

    def _on_step(self):
        if self.num_timesteps >= self.next_eval:
            wr = evaluate_win_rate(self.model.q_net.q_net, n_games=self.n_eval_games)
            print(f"Step {self.num_timesteps:>7,}: win rate = {wr * 100:.1f}%", flush=True)
            self.next_eval += self.eval_freq
        return True


def make_dqn(env, lr=1e-3):
    """Create the MaskedDQN trainer tuned for fast 6x6/4 convergence.

    Key settings vs. SB3 defaults:
    - gamma=0.5: short episodes (~10 steps); low discount avoids long-horizon noise
    - batch_size=256: larger batches stabilize Q-updates on a tiny replay buffer
    - gradient_steps=4: more updates per env step without extra env rollouts
    - exploration_fraction=0.3: longer exploration window for a hard env
    """
    return MaskedDQN(
        ConvDQNPolicy,
        env,
        learning_rate=lr,
        buffer_size=100_000,
        learning_starts=2_000,
        batch_size=256,
        gamma=0.5,
        train_freq=1,
        gradient_steps=4,
        target_update_interval=500,
        exploration_fraction=0.30,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.02,
        verbose=0,
        seed=42,
        device="cpu",
    )


def main():
    env = MinesweeperEnv()
    model = make_dqn(env, lr=5e-4)
    n_params = sum(p.numel() for p in model.q_net.q_net.parameters())
    print(f"Training MaskedDQN (TinyConvQ, {n_params:,} params) for 400,000 steps...", flush=True)

    model.learn(total_timesteps=400_000, callback=WinRateLogCallback(25_000, 400))

    final_wr = evaluate_win_rate(model.q_net.q_net, n_games=1000)
    print(f"Final baseline win rate (1,000 games): {final_wr * 100:.2f}%", flush=True)

    import shutil
    # Keep old model as backup before overwriting
    try:
        shutil.copy("model.pth", "model_fp32_prev.pth")
    except FileNotFoundError:
        pass

    torch.save(model.q_net.q_net.state_dict(), "model.pth")
    print("Saved model.pth", flush=True)


if __name__ == "__main__":
    main()
