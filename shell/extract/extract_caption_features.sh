#!/usr/bin/env bash
set -euo pipefail

python -m vgcap.temporal.extract_caption_features \
  --csv "PATH/TO/temporal.csv" \
  --global-output-dir "PATH/TO/global_caption_features" \
  --segment-output-dir "PATH/TO/segment_caption_features" \
  --mode both \
  --resume
