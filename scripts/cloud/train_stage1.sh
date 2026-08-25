#!/usr/bin/env bash
# Stage 1 dual-terrain train on the A10. 1024 envs, 10k iters, W&B if key is set.
set -euo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "$0")" && pwd)/_env.sh"
NUM_ENVS="${NUM_ENVS:-1024}"
MAX_ITERS="${MAX_ITERS:-10000}"
ROBOT="${ROBOT:-h1}"
LOGGER_ARGS=(--logger tensorboard)
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  LOGGER_ARGS=(--logger wandb --log_project_name "${WANDB_PROJECT:-beamdojo}")
  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    echo "W&B: https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT:-beamdojo}"
  else
    echo "W&B project ${WANDB_PROJECT:-beamdojo} — open https://wandb.ai (set WANDB_ENTITY for a direct link)."
  fi
else
  echo "WANDB_API_KEY unset; TensorBoard only. ssh -L 6006:localhost:6006 lambda-beamdojo"
fi
python3 -c "
import beamdojo_runtime
beamdojo_runtime.write_boot_status(
    stage=1,
    robot='${ROBOT}',
    terrain='beam',
    num_envs=int('${NUM_ENVS}'),
    max_iterations=int('${MAX_ITERS}'),
    note='train_stage1.sh: launching isaaclab.sh. Not a live W&B run yet.',
)
"
exec /workspace/isaaclab/isaaclab.sh -p train_beamdojo.py \
  --headless \
  --device cuda:0 \
  --stage 1 \
  --robot "$ROBOT" \
  --num_envs "$NUM_ENVS" \
  --max_iterations "$MAX_ITERS" \
  "${LOGGER_ARGS[@]}"
