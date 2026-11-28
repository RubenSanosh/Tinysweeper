"""
Prune + shrink TinySweeper without giving up win rate.

1. Win rate first: every stage fine-tunes at low LR / low exploration and keeps the best
   checkpoint it sees, so the 0% stage polishes the unpruned model and no stage can end
   worse than it started.
2. Prune in 10% steps while the win rate stays >= FLOOR.
3. Pack to int8 weights (per-channel scale) + zlib -> model_tiny.bin, then re-measure
   the win rate from the file itself (that is the number that ships).
"""

import copy
import os
import warnings
import zlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import get_linear_fn

from env import MinesweeperEnv
from train import TinyConvQ, evaluate_win_rate, make_dqn

torch.set_num_threads(1)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

FLOOR = 0.80                            # never ship below this
SEL = dict(n_games=500, seed=50_000)    # used to pick checkpoints
TEST = dict(n_games=2000, seed=90_000)  # reported numbers: games never used for picking


class KeepMaskCallback(BaseCallback):
    """Re-apply pruning masks every step; every `every` steps keep the best weights seen."""

    def __init__(self, masks, best_sd, best_wr, every=5_000):
        super().__init__()
        self.masks, self.best_sd, self.best_wr, self.every = masks, best_sd, best_wr, every

    def _on_step(self):
        net = self.model.q_net.q_net
        mods = dict(net.named_modules())
        with torch.no_grad():
            for name, mask in self.masks.items():
                mods[name].weight.mul_(mask)
        if self.num_timesteps % self.every == 0:
            wr = evaluate_win_rate(net, **SEL)
            if wr > self.best_wr:
                self.best_wr, self.best_sd = wr, copy.deepcopy(net.state_dict())
        return True


def apply_prune(net, cumulative_sparsity):
    """Global L1 pruning over all Conv2d to the target total sparsity; returns the masks.

    shortcut: unstructured pruning only shrinks the *file* (zeros compress in pack());
              structured channel pruning would shrink the model itself, upgrade if needed.
    """
    params = [(m, "weight") for m in net.modules() if isinstance(m, nn.Conv2d)]
    prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=cumulative_sparsity)
    masks = {}
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            masks[name] = m.weight_mask.clone()
            prune.remove(m, "weight")
    return masks


def finetune(net, masks, steps=20_000, lr=1e-4):
    """Fine-tune in a fresh DQN; returns the best-scoring weights seen (the start point counts)."""
    dqn = make_dqn(MinesweeperEnv(), lr=lr)
    dqn.q_net.q_net.load_state_dict(net.state_dict())
    dqn.q_net_target.q_net.load_state_dict(net.state_dict())
    # The eps attributes are baked into the schedule when the DQN is built, so replace the schedule.
    dqn.exploration_schedule = get_linear_fn(0.05, 0.01, 0.3)
    dqn.gradient_steps = 1  # 4 was ~3x slower for no visible gain
    cb = KeepMaskCallback(masks, copy.deepcopy(net.state_dict()), evaluate_win_rate(net, **SEL))
    dqn.learn(total_timesteps=steps, callback=cb)
    out = TinyConvQ(16)
    out.load_state_dict(cb.best_sd)
    return out.eval()


def count_nonzero(net):
    total = sum(p.numel() for p in net.parameters())
    return total, sum(int(torch.count_nonzero(p)) for p in net.parameters())


def _layers(net):
    return (net.c1, net.c2, net.c3, net.out)


def pack(net, path):
    """int8 weights (one scale per output channel), fp16 scales/biases, zlib. Pruned zeros compress away."""
    s, b, q = [], [], []
    for m in _layers(net):
        w = m.weight.detach().flatten(1)
        sc = (w.abs().amax(1) / 127).clamp_min(1e-4).half().float()
        s.append(sc)
        b.append(m.bias.detach())
        q.append((w / sc[:, None]).round().clamp(-127, 127).to(torch.int8).flatten())
    parts = (torch.cat(s).half(), torch.cat(b).half(), torch.cat(q))
    with open(path, "wb") as f:
        f.write(zlib.compress(b"".join(t.numpy().tobytes() for t in parts), 9))
    return os.path.getsize(path)


def load_packed(path):
    """Inverse of pack(): returns a regular TinyConvQ (fp32 compute, int8-rounded weights)."""
    with open(path, "rb") as f:
        blob = zlib.decompress(f.read())
    net = TinyConvQ(16)
    n = sum(m.out_channels for m in _layers(net))
    s = torch.from_numpy(np.frombuffer(blob, np.float16, n).astype(np.float32))
    b = torch.from_numpy(np.frombuffer(blob, np.float16, n, offset=2 * n).astype(np.float32))
    q = torch.from_numpy(np.frombuffer(blob, np.int8, offset=4 * n).astype(np.float32))
    i = j = 0
    for m in _layers(net):
        o, k = m.out_channels, m.weight[0].numel()
        m.weight.data = (q[j:j + o * k].view(o, k) * s[i:i + o, None]).view_as(m.weight)
        m.bias.data = b[i:i + o]
        i, j = i + o, j + o * k
    return net.eval()


def main():
    net = TinyConvQ(16)
    net.load_state_dict(torch.load("model.pth", map_location="cpu", weights_only=True))
    net.eval()
    base_wr = evaluate_win_rate(net, **TEST)
    print(f"model.pth: {base_wr*100:.1f}% on {TEST['n_games']} unseen games, floor {FLOOR*100:.0f}%\n")
    print(f"{'Prune':>6} {'Win rate':>9} {'NZ params':>12}")

    best, best_target, target = net, 0.0, 0.0
    while target <= 0.70:
        cand = copy.deepcopy(best)
        ft = finetune(cand, apply_prune(cand, target))
        wr = evaluate_win_rate(ft, **TEST)
        total, nz = count_nonzero(ft)
        print(f"{target*100:>5.0f}% {wr*100:>8.1f}% {nz:>6,}/{total:,}", flush=True)
        torch.save(ft.state_dict(), f"model_pruned{round(target*100)}pct.pth")
        if wr < FLOOR:
            print(f"  below the {FLOOR*100:.0f}% floor, stopping; keeping the {best_target*100:.0f}% model")
            break
        best, best_target = ft, target
        target = round(target + 0.10, 2)

    size = pack(best, "model_tiny.bin")
    packed_wr = evaluate_win_rate(load_packed("model_tiny.bin"), **TEST)
    print(f"\nPruning kept: {best_target*100:.0f}%  |  model_tiny.bin: {size:,} bytes (model.pth: {os.path.getsize('model.pth'):,})")
    print(f"Win rate loaded from the file: {packed_wr*100:.1f}%  (model.pth was {base_wr*100:.1f}%)")


if __name__ == "__main__":
    main()
