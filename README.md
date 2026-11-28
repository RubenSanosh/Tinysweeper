# TinySweeper

A tiny convolutional Q-network that plays 6x6 Minesweeper (4 mines) at an 80.2% win rate packed into 4,716 
bytes (~4.6 KB, 5,105 parameters) within 12.3 percentage points of the ~92.5% near-optimal solver ceiling 
across random first clicks (100% is impossible due to unavoidable guess boards).

## Results

| Model | File | Size | Parameters | Win rate |
|---|---|---|---|---|
| Packed INT8 + zlib (main) | `model_tiny.bin` | 4,716 bytes (~4.6 KB) | 5,105 | 80.2% |
| Full precision (FP32 baseline) | `model.pth` | 23,473 bytes (~22.9 KB) | 5,105 | 80.8% (83.2% on 500 games) |

The packed model (`model_tiny.bin`) is evaluated on 2,000 held-out games with the first click always safe. Earlier 500-game scores have roughly ±2% evaluation noise.

## Failed experiment: dynamic INT8 quantization

An earlier attempt using pruning and dynamic INT8 quantization shrank the model to about 11 KB, but win rate dropped to **38.5%** (likely due to dynamic activation quantization). That approach was replaced by INT8 weight packing with zlib compression in `compress.py` and is not used by the main model.

## How it works

- **Environment** (`env.py`): a Gymnasium 6x6 Minesweeper environment with 4 mines. Mines are placed after the first click, so the first move is always safe.
- **Model** (`train.py`): a tiny convolutional Q-network (TinyConvQ) with three 3x3 convolutional layers (16 channels) and a 1x1 output layer (5,105 parameters total). Weight sharing across spatial positions lets a small parameter budget reason over local mine counts.
- **Training**: DQN (Stable-Baselines3) with action masking so the agent never picks an already revealed cell.
- **Compression** (`compress.py`): fine-tunes with an 80% win-rate floor across pruning steps, then packs INT8 weights with zlib into `model_tiny.bin` and loads them via `load_packed()`.

## Run it

```bash
python -m venv venv
source venv/bin/activate        # fish: source venv/bin/activate.fish
pip install torch --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
pip install -r requirements.txt

python demo.py                  # watch the packed model (model_tiny.bin) play
