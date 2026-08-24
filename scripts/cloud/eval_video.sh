#!/usr/bin/env bash
# GPU RTX eval video. Run inside isaac-lab-base.
set -euo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "$0")" && pwd)/_env.sh"
STAGE="${STAGE:-1}"
ROBOT="${ROBOT:-h1}"
TERRAIN="${TERRAIN:-beam}"
NUM_ENVS="${NUM_ENVS:-4}"
VIDEO_LENGTH="${VIDEO_LENGTH:-80}"
exec /workspace/isaaclab/isaaclab.sh -p play_beamdojo.py \
  --headless \
  --enable_cameras \
  --video \
  --video_length "$VIDEO_LENGTH" \
  --num_envs "$NUM_ENVS" \
  --device cuda:0 \
  --stage "$STAGE" \
  --robot "$ROBOT" \
  --terrain "$TERRAIN"
