#!/usr/bin/env bash
# Run inside isaac-lab-base. GPU-only 64-env Stage 1 smoke.
set -euo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "$0")" && pwd)/_env.sh"
export BEAMDOJO_LOG_ROOT="${BEAMDOJO_LOG_ROOT:-/workspace/isaaclab/logs}"
exec /workspace/isaaclab/isaaclab.sh -p train_beamdojo.py \
    --headless \
    --device cuda:0 \
    --num_envs 64 \
    --max_iterations 5 \
    --logger tensorboard
