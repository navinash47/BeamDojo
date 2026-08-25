#!/usr/bin/env bash
# Stage 2 hard beam, resume from Stage 1.
# Default: latest Stage 1 run (mtime) + highest model_<N>.pt under
# logs/rsl_rl/beamdojo_${ROBOT:-h1}_stage1. Do not pin a finished-10k filename —
# a smoke or killed 10k is model_4.pt / model_499.pt.
# Pin a run: LOAD_RUN=<stage1-run-folder> CHECKPOINT=model_499.pt
# Continue an interrupted Stage 2 run with:
#   LOAD_EXPERIMENT=beamdojo_h1_stage2 LOAD_RUN=<stage2-run> CHECKPOINT=model_XXXX.pt
set -euo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "$0")" && pwd)/_env.sh"
NUM_ENVS="${NUM_ENVS:-1024}"
MAX_ITERS="${MAX_ITERS:-10000}"
ROBOT="${ROBOT:-h1}"
TERRAIN="${TERRAIN:-beam}"
LOGGER_ARGS=(--logger tensorboard)
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  LOGGER_ARGS=(--logger wandb --log_project_name "${WANDB_PROJECT:-beamdojo}")
  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    echo "W&B: https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT:-beamdojo}"
  else
    echo "W&B project ${WANDB_PROJECT:-beamdojo} — open https://wandb.ai (set WANDB_ENTITY for a direct link)."
  fi
fi
LOAD_ARGS=()
if [[ -n "${LOAD_RUN:-}" ]]; then
  LOAD_ARGS+=(--load_run "$LOAD_RUN")
fi
if [[ -n "${CHECKPOINT:-}" ]]; then
  LOAD_ARGS+=(--checkpoint "$CHECKPOINT")
fi
LOAD_EXP_ARGS=()
if [[ -n "${LOAD_EXPERIMENT:-}" ]]; then
  LOAD_EXP_ARGS=(--load_experiment "$LOAD_EXPERIMENT")
fi
echo "Stage 2 resume: latest Stage 1 checkpoint under logs/rsl_rl/beamdojo_${ROBOT}_stage1 unless LOAD_RUN/CHECKPOINT/LOAD_EXPERIMENT are set."
python3 -c "
import beamdojo_runtime
beamdojo_runtime.write_boot_status(
    stage=2,
    robot='${ROBOT}',
    terrain='${TERRAIN}',
    num_envs=int('${NUM_ENVS}'),
    max_iterations=int('${MAX_ITERS}'),
    note='train_stage2.sh: launching isaaclab.sh. Not a live W&B run yet.',
)
"
exec /workspace/isaaclab/isaaclab.sh -p train_beamdojo.py \
  --headless \
  --device cuda:0 \
  --stage 2 \
  --robot "$ROBOT" \
  --terrain "$TERRAIN" \
  --num_envs "$NUM_ENVS" \
  --max_iterations "$MAX_ITERS" \
  --resume \
  "${LOAD_ARGS[@]}" \
  "${LOAD_EXP_ARGS[@]}" \
  "${LOGGER_ARGS[@]}"
