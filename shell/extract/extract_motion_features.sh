#!/usr/bin/env bash
set -euo pipefail

python -m vgcap.temporal.extract_motion_features \
  --csv "PATH/TO/temporal.csv" \
  --video-dir "PATH/TO/videos" \
  --whole-output-dir "PATH/TO/whole_motion_features" \
  --segment-output-dir "PATH/TO/segment_motion_features" \
  --mode both \
  --resume
