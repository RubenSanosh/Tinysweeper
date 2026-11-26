"""Prune (80% L1 unstructured), fine-tune, and INT8-quantize model.pth -> model_tiny.pth."""

import os
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
    """Re-apply binary pruning masks after gradient steps so sparsity stays at 80%."""

    def __init__(self, masks):
        super().__init__()
        self.masks = masks

    def _on_step(self):
        with torch.no_grad():
            modules = dict(self.model.q_net.q_net.named_modules())
            for name, mask in self.masks.items():
                modules[name].weight.mul_(mask)
        return True


def to_linear_conv(conv_net):
    """Copy Conv2d weights into mathematically equivalent nn.Linear layers for INT8 quantization."""
    lin_net = TinyLinearConvQ(conv_net.ch)
    pairs = [
        (conv_net.c1, lin_net.l1),
        (conv_net.c2, lin_net.l2),
        (conv_net.c3, lin_net.l3),
        (conv_net.out, lin_net.out),
    ]
    with torch.no_grad():
        for cl, ll in pairs:
            ll.weight.copy_(cl.weight.view(cl.out_channels, -1))
            ll.bias.copy_(cl.bias)
    return lin_net.eval()


def count_nonzero_params(model):
    """Count non-zero scalar weights/biases to report effective capacity after pruning."""
    total, nonzero = 0, 0
    for p in model.parameters():
        total += p.numel()
        nonzero += int(torch.count_nonzero(p).item())
    return total, nonzero


def main():
    base_net = TinyConvQ(16)
    base_net.load_state_dict(torch.load("model.pth", map_location="cpu", weights_only=True))
    base_net.eval()
    orig_size = os.path.getsize("model.pth")
    orig_wr = evaluate_win_rate(base_net, n_games=1000)

    # Step 3a: L1 unstructured pruning at 80% sparsity across all Conv2d weights.
    masks = {}
    for name, module in base_net.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name="weight", amount=0.8)
            masks[name] = module.weight_mask.clone()
            prune.remove(module, "weight")

    pruned_wr = evaluate_win_rate(base_net, n_games=1000)

    # Fine-tune 20K steps with low exploration while preserving the 80% sparsity mask.
    env = MinesweeperEnv()
    dqn = make_dqn(env, lr=3e-4)
    dqn.q_net.q_net.load_state_dict(base_net.state_dict())
    dqn.q_net_target.q_net.load_state_dict(base_net.state_dict())
    dqn.exploration_initial_eps = 0.05
    dqn.exploration_final_eps = 0.01
    dqn.learn(total_timesteps=20_000, callback=KeepMaskCallback(masks))

    ft_net = TinyConvQ(16)
    ft_net.load_state_dict(dqn.q_net.q_net.state_dict())
    with torch.no_grad():
        modules = dict(ft_net.named_modules())
        for name, mask in masks.items():
            modules[name].weight.mul_(mask)
    ft_net.eval()
    ft_wr = evaluate_win_rate(ft_net, n_games=1000)

    # Step 3b: Convert equivalent weights to nn.Linear and apply INT8 dynamic quantization.
    lin_net = to_linear_conv(ft_net)
    tiny_net = torch.ao.quantization.quantize_dynamic(
        lin_net, {nn.Linear}, dtype=torch.qint8
    )
    tiny_wr = evaluate_win_rate(tiny_net, n_games=1000)

    torch.save(tiny_net.state_dict(), "model_tiny.pth")
    tiny_size = os.path.getsize("model_tiny.pth")

    total_p, nz_p = count_nonzero_params(ft_net)
    print("\nCompression Results (1,000 evaluation games):")
    print(f"{'Stage':<30} {'Size (bytes)':>14} {'Win Rate':>10}")
    print("-" * 56)
    print(f"{'Original FP32 (model.pth)':<30} {orig_size:>14,} {orig_wr * 100:>9.2f}%")
    print(f"{'Pruned 80% (zero-shot)':<30} {orig_size:>14,} {pruned_wr * 100:>9.2f}%")
    print(f"{'Pruned 80% + 20K fine-tune':<30} {orig_size:>14,} {ft_wr * 100:>9.2f}%")
    print(f"{'INT8 Quantized (tiny.pth)':<30} {tiny_size:>14,} {tiny_wr * 100:>9.2f}%")
    print(f"\nNon-zero parameters after 80% pruning: {nz_p:,} / {total_p:,}")


if __name__ == "__main__":
    main()
