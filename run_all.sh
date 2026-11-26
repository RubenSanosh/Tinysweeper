#!/usr/bin/env bash
set -euo pipefail

# Install CPU-only PyTorch first to avoid pulling multi-GB CUDA wheels.
python3 -m pip install --upgrade pip
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install -r requirements.txt

python3 train.py
python3 compress.py
python3 demo.py
