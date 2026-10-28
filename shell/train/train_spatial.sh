#!/usr/bin/env bash
set -euo pipefail

TRAIN_CSV="PATH/TO/train_spatial.csv"
VAL_CSV="PATH/TO/val_spatial.csv"
VIDEO_DIR="PATH/TO/videos"
OUTPUT_DIR="PATH/TO/outputs/spatial"

python -m vgcap.spatial.train \
  --train-csv "$TRAIN_CSV" --val-csv "$VAL_CSV" \
  --video-dir "$VIDEO_DIR" \
  --output-dir "$OUTPUT_DIR" --caption-source global \
  --vision-mode partial \
  --vision-model google/siglip2-base-patch16-224 \
  --agbg-gate-init 0.33 --epochs 5 --batch-size 4

