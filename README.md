# Tinysweeper

Compression Results (1,000 evaluation games):
Stage                            Size (bytes)   Win Rate
--------------------------------------------------------
Original FP32 (model.pth)              23,473     81.10%
Pruned 80% (zero-shot)                 23,473     23.90%
Pruned 80% + 20K fine-tune             23,473     38.80%
INT8 Quantized (tiny.pth)              11,317     38.50%
