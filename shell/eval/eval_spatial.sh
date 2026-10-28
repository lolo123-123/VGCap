#!/usr/bin/env bash
set -euo pipefail

python -m vgcap.spatial.evaluate \
  --csv "PATH/TO/test_spatial.csv" \
  --video-dir "PATH/TO/videos" \
  --checkpoint "PATH/TO/spatial/best.pt" \
  --output-dir "PATH/TO/outputs/spatial/test"

