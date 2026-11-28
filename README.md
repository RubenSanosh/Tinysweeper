# TinySweeper

A tiny neural network that plays 6x6 Minesweeper (4 mines) with an **83.2% win rate** in about **23 KB** (5,105 parameters).

## Results

| Model | File | Size | Parameters | Win rate |
|---|---|---|---|---|
| Full precision (FP32) | `model.pth` | 23 KB | 5,105 | 83.2% |

Win rate was measured over 500 games on fixed seeds, with the first click always safe. Results on a few hundred games are noisy (about ±2%), so treat the number as approximate.

## Failed experiment: INT8 quantization

I also tried compressing the model with pruning and INT8 dynamic quantization (`compress.py`). The file shrank to about 11 KB, but the win rate dropped to **38.5%**, which is far too much accuracy loss. It is **not** part of the main model, and is kept only as a record of what did not work.

| Model | File | Size | Win rate |
|---|---|---|---|
| INT8 quantized (failed experiment) | `model_tiny.pth` | ~11 KB | 38.5% |

## How it works

- **Environment** (`env.py`): a Gymnasium 6x6 Minesweeper with 4 mines. Mines are placed after the first click, so the first move can never lose.
- **Model** (`train.py`): a 3-layer, 16-channel 3x3 convolutional Q-network. Convolutions share weights across the board, which is how ~5K parameters can reason about neighbouring mine counts.
- **Training**: DQN (Stable-Baselines3) with action masking, so the agent never clicks an already revealed cell.

## Run it

```bash
python -m venv venv
source venv/bin/activate        # fish: source venv/bin/activate.fish
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python demo_fp32.py             # watch the main (FP32) model play
```

## Train it yourself

```bash
python train.py                 # train from scratch (400k steps), writes model.pth
python continue_train.py        # keep training from model.pth, saves only improvements
```

Experimental compression (see the failed experiment above): `python compress.py`, then `python demo.py`.

## Notes

- Minesweeper sometimes forces a guess, so no player can win every game.
