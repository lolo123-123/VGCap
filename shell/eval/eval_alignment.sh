#!/usr/bin/env bash
set -euo pipefail

python -m vgcap.alignment.evaluate \
  --csv "PATH/TO/test_alignment.csv" \
  --video-dir "PATH/TO/videos" \
  --checkpoint "PATH/TO/alignment/best.pt" \
  --output-dir "PATH/TO/outputs/alignment/test"

