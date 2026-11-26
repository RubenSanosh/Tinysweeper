# TinySweeper

A tiny neural network that plays 6x6 Minesweeper (4 mines) with a win rate of **XX%** in about **XX KB**.

<!-- Add a screenshot or GIF of demo.py running here: ![demo](demo.gif) -->

## Results

| Model | File | Size | Parameters | Win rate |
|---|---|---|---|---|
| Full precision (FP32) | `model.pth` | 23 KB | 5,105 | XX% |
| INT8 quantized | `model_tiny.pth` | XX KB | 5,105 | XX% |

Win rates are measured over 1,000 games on fixed seeds. First click is always safe.
<!-- Optional: add a row for a random player and a simple heuristic for comparison. -->

## How it works

- **Environment** (`env.py`): a Gymnasium 6x6 Minesweeper with 4 mines. Mines are placed after the first click, so the first move can never lose.
- **Model** (`train.py`): a 3-layer, 16-channel 3x3 convolutional Q-network. Convolutions share weights across the board, which is how ~5K parameters can reason about neighbouring mine counts.
- **Training**: DQN (Stable-Baselines3) with action masking, so the agent never clicks an already revealed cell.
- **Compression** (`compress.py`): gradual pruning with fine-tuning, then INT8 dynamic quantization.

## Run it

```bash
python -m venv venv
source venv/bin/activate        # fish: source venv/bin/activate.fish
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python demo.py                  # watch the compressed model play
python demo_fp32.py             # watch the full-precision model play
```

## Train it yourself

```bash
python train.py                 # train from scratch (400k steps), writes model.pth
python continue_train.py        # keep training from model.pth, saves only improvements
python compress.py              # prune + quantize, writes model_tiny.pth
```

Or run everything with `./run_all.sh`.

## Notes

- Win rates on a few hundred games are noisy (about ±2%), so small differences between checkpoints are not meaningful.
- Minesweeper sometimes forces a guess, so no player can win every game.
