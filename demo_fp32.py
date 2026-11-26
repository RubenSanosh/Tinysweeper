"""Watch the full-precision model.pth play (same window as demo.py).

Run from the project folder:   python demo_fp32.py
Close the window or press Esc to quit, R to reset the score.
"""

import torch

import demo
from train import TinyConvQ


def load_fp32(_path=None):
    net = TinyConvQ(16)
    net.load_state_dict(torch.load("model.pth", map_location="cpu", weights_only=True))
    net.eval()
    return net, sum(p.numel() for p in net.parameters())


demo.load_tiny_model = load_fp32

if __name__ == "__main__":
    demo.main()
