#!/usr/bin/env bash
set -euo pipefail

TRAIN_CSV="PATH/TO/train_temporal.csv"
VAL_CSV="PATH/TO/val_temporal.csv"
WHOLE_DIR="PATH/TO/whole_motion_features"
SEGMENT_DIR="PATH/TO/segment_motion_features"
SEGMENT_CAPTION_DIR="PATH/TO/segment_caption_features"
GLOBAL_TRAIN_DIR="PATH/TO/global_caption_features/train"
GLOBAL_VAL_DIR="PATH/TO/global_caption_features/val"
OUTPUT_DIR="PATH/TO/outputs/temporal"

python -m vgcap.temporal.train \
  --train-csv "$TRAIN_CSV" --val-csv "$VAL_CSV" \
  --whole-dir "$WHOLE_DIR" --seg-dir "$SEGMENT_DIR" \
  --segment-caption-dir "$SEGMENT_CAPTION_DIR" \
  --global-train-dir "$GLOBAL_TRAIN_DIR" --global-val-dir "$GLOBAL_VAL_DIR" \
  --output-dir "$OUTPUT_DIR" --global-gate-init 0.33 \
  --segment-gate-init 0.45 --fusion-gate-init 0.65 \
  --epochs 5 --batch-size 4
