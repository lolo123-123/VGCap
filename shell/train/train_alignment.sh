#!/usr/bin/env bash
set -euo pipefail

TRAIN_CSV="PATH/TO/train_alignment.csv"
VAL_CSV="PATH/TO/val_alignment.csv"
VIDEO_DIR="PATH/TO/videos"
OUTPUT_DIR="PATH/TO/outputs/alignment"

python -m vgcap.alignment.train \
  --train-csv "$TRAIN_CSV" --val-csv "$VAL_CSV" \
  --video-dir "$VIDEO_DIR" --output-dir "$OUTPUT_DIR" \
  --fusion-mode agbg --gate-init 0.55 0.25 0.20 \
  --epochs 5 --batch-size 4
