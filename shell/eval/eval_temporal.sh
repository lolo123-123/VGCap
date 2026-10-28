#!/usr/bin/env bash
set -euo pipefail

python -m vgcap.temporal.evaluate \
  --csv "PATH/TO/test_temporal.csv" \
  --whole-dir "PATH/TO/whole_motion_features" \
  --seg-dir "PATH/TO/segment_motion_features" \
  --segment-caption-dir "PATH/TO/segment_caption_features" \
  --global-dir "PATH/TO/global_caption_features/test" \
  --checkpoint "PATH/TO/temporal/best.pt" \
  --output-dir "PATH/TO/outputs/temporal/test"

