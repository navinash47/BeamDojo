#!/usr/bin/env bash
# GPU RTX video of the latest Stage 1 checkpoint. Run inside isaac-lab-base.
set -euo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "$0")" && pwd)/_env.sh"
export BEAMDOJO_LOG_ROOT="${BEAMDOJO_LOG_ROOT:-/workspace/isaaclab/logs}"
exec /workspace/isaaclab/isaaclab.sh -p play_beamdojo.py \
  --headless \
  --enable_cameras \
  --video \
  --video_length 80 \
  --num_envs 4 \
  --device cuda:0
