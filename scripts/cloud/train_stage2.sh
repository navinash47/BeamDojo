#!/usr/bin/env bash
# Stage 2 hard beam, resume from Stage 1. Set LOAD_RUN and CHECKPOINT.
set -euo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "$0")" && pwd)/_env.sh"
NUM_ENVS="${NUM_ENVS:-1024}"
MAX_ITERS="${MAX_ITERS:-10000}"
ROBOT="${ROBOT:-h1}"
TERRAIN="${TERRAIN:-beam}"
LOAD_RUN="${LOAD_RUN:?Set LOAD_RUN to the Stage 1 run folder name}"
CHECKPOINT="${CHECKPOINT:-model_9999.pt}"
LOGGER_ARGS=(--logger tensorboard)
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  LOGGER_ARGS=(--logger wandb --log_project_name "${WANDB_PROJECT:-beamdojo}")
fi
exec /workspace/isaaclab/isaaclab.sh -p train_beamdojo.py \
  --headless \
  --device cuda:0 \
  --stage 2 \
  --robot "$ROBOT" \
  --terrain "$TERRAIN" \
  --num_envs "$NUM_ENVS" \
  --max_iterations "$MAX_ITERS" \
  --resume \
  --load_run "$LOAD_RUN" \
  --checkpoint "$CHECKPOINT" \
  "${LOGGER_ARGS[@]}"
