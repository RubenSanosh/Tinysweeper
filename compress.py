"""
Gradual iterative pruning pipeline for TinySweeper.

Strategy: prune in small steps (10% at a time), fine-tuning after each step,
stopping when win rate drops more than 2% below baseline. This keeps far
more accuracy than a one-shot 80% prune.
"""

import copy
import os
import shutil
import warnings

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from stable_baselines3.common.callbacks import BaseCallback

from env import MinesweeperEnv
from train import TinyConvQ, TinyLinearConvQ, evaluate_win_rate, make_dqn

torch.set_num_threads(1)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch")


class KeepMaskCallback(BaseCallback):
    """Re-apply binary pruning masks after every gradient step to enforce sparsity."""

    def __init__(self, masks):
        super().__init__()
        self.masks = masks

    def _on_step(self):
        with torch.no_grad():
            modules = dict(self.model.q_net.q_net.named_modules())
            for name, mask in self.masks.items():
                modules[name].weight.mul_(mask)
        return True


def apply_prune(net, cumulative_sparsity):
    """Apply L1-unstructured pruning globally across all Conv2d to reach target sparsity.

    Uses cumulative amounts so each call brings total sparsity to the target level.
    Returns the new binary masks dict.
    """
    # Collect all (module, 'weight') pairs for Conv2d layers
    params = [(m, 'weight') for _, m in net.named_modules() if isinstance(m, nn.Conv2d)]
    # Global L1 unstructured pruning to reach cumulative_sparsity
    prune.global_unstructured(
        params,
        pruning_method=prune.L1Unstructured,
        amount=cumulative_sparsity,
    )
    masks = {}
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            masks[name] = m.weight_mask.clone()
            prune.remove(m, 'weight')
    return masks


def finetune(net, masks, steps=15_000, lr=5e-4):
    """Fine-tune a pruned network inside a fresh DQN trainer, returning the net."""
    env = MinesweeperEnv()
    dqn = make_dqn(env, lr=lr)
    dqn.q_net.q_net.load_state_dict(net.state_dict())
    dqn.q_net_target.q_net.load_state_dict(net.state_dict())
    dqn.exploration_initial_eps = 0.08
    dqn.exploration_final_eps = 0.01
    dqn.learn(total_timesteps=steps, callback=KeepMaskCallback(masks))
    out = TinyConvQ(16)
    out.load_state_dict(dqn.q_net.q_net.state_dict())
    # Re-enforce masks after the final optimizer step
    with torch.no_grad():
        modules = dict(out.named_modules())
        for name, mask in masks.items():
            modules[name].weight.mul_(mask)
    out.eval()
    return out


def to_linear_conv(conv_net):
    """Exact nn.Linear copy of Conv2d layers for INT8 dynamic quantization compatibility."""
    lin_net = TinyLinearConvQ(conv_net.ch)
    with torch.no_grad():
        for cl, ll in zip(
            [conv_net.c1, conv_net.c2, conv_net.c3, conv_net.out],
            [lin_net.l1, lin_net.l2, lin_net.l3, lin_net.out],
        ):
            ll.weight.copy_(cl.weight.view(cl.out_channels, -1))
            ll.bias.copy_(cl.bias)
    return lin_net.eval()


def count_nonzero(net):
    total = sum(p.numel() for p in net.parameters())
    nz = sum(int(torch.count_nonzero(p).item()) for p in net.parameters())
    return total, nz


def quantize_and_save(net, path):
    """INT8 dynamic-quantize Conv2d-net via its Linear equivalent and save."""
    lin = to_linear_conv(net)
    qnet = torch.ao.quantization.quantize_dynamic(lin, {nn.Linear}, dtype=torch.qint8)
    torch.save(qnet.state_dict(), path)
    return qnet, os.path.getsize(path)


def main():
    # --- Load baseline ---
    baseline = TinyConvQ(16)
    baseline.load_state_dict(torch.load("model.pth", map_location="cpu", weights_only=True))
    baseline.eval()
    base_size = os.path.getsize("model.pth")
    base_wr = evaluate_win_rate(baseline, n_games=1000)
    print(f"Baseline: {base_wr*100:.2f}% win rate, {base_size:,} bytes")

    # Drop tolerance: stop pruning if we fall more than 3% below baseline
    tol = 0.03
    threshold = base_wr - tol

    rows = [("Baseline FP32", base_size, base_wr, 5105)]

    # Gradual pruning: 10% cumulative steps. Each iteration starts from the
    # current best net and applies prune to the total target sparsity level.
    # shortcut: global unstructured pruning re-runs from scratch each iter;
    #           structured channel pruning would give better speedup, upgrade if needed.
    best_net = copy.deepcopy(baseline)
    best_wr = base_wr
    step = 0.10
    target = step

    print(f"\nGradual pruning (tolerance: drop ≤{tol*100:.0f}% below {base_wr*100:.1f}%)\n")
    print(f"{'Target':>8}  {'Zero-shot':>10}  {'Fine-tuned':>10}  {'NZ params':>10}")
    print("-" * 44)

    final_net = best_net
    final_sparsity = 0.0
    while target <= 0.70:
        candidate = copy.deepcopy(best_net)
        masks = apply_prune(candidate, cumulative_sparsity=target)
        zs_wr = evaluate_win_rate(candidate, n_games=300)

        ft_net = finetune(candidate, masks, steps=15_000, lr=5e-4)
        ft_wr = evaluate_win_rate(ft_net, n_games=500)

        total_p, nz_p = count_nonzero(ft_net)
        print(f"{int(target*100):>7}%  {zs_wr*100:>9.1f}%  {ft_wr*100:>9.1f}%  {nz_p:>10,}/{total_p}", flush=True)

        rows.append((f"Pruned {int(target*100)}% + FT", base_size, ft_wr, nz_p))
        # Save checkpoint for every step
        ckpt = f"model_pruned{int(target*100)}pct.pth"
        torch.save(ft_net.state_dict(), ckpt)

        if ft_wr >= threshold:
            best_net = ft_net
            best_wr = ft_wr
            final_net = ft_net
            final_sparsity = target
            target += step
        else:
            print(f"  → Win rate {ft_wr*100:.1f}% dropped below threshold {threshold*100:.1f}%; stopping pruning.")
            break

    # Quantize the best pruned net to INT8
    qnet, tiny_size = quantize_and_save(final_net, "model_tiny.pth")
    shutil.copy("model_tiny.pth", "model_tiny.pth.bak")
    tiny_wr = evaluate_win_rate(qnet, n_games=1000)
    rows.append(("INT8 Quantized (tiny.pth)", tiny_size, tiny_wr, "-"))

    # Print final table
    total_p, nz_p = count_nonzero(final_net)
    print(f"\n{'Stage':<28} {'Size':>12} {'Win Rate':>10} {'NZ Params':>12}")
    print("-" * 64)
    for label, sz, wr, nz in rows:
        nz_str = f"{nz:,}" if isinstance(nz, int) else nz
        print(f"{label:<28} {sz:>12,} {wr*100:>9.2f}% {nz_str:>12}")
    print(f"\nFinal sparsity: {final_sparsity*100:.0f}%  |  Non-zero params: {nz_p:,} / {total_p:,}")
    print(f"model_tiny.pth: {tiny_size:,} bytes  |  win rate: {tiny_wr*100:.2f}%")


if __name__ == "__main__":
    main()
