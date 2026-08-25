#!/usr/bin/env bash
# Shared env for Isaac Lab container scripts. GPU-only.
set -euo pipefail
export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=YES
export BEAMDOJO_LOG_ROOT="${BEAMDOJO_LOG_ROOT:-/lambda/nfs/beamdojo/logs}"
if [[ -f /workspace/beamdojo/.env.lambda ]]; then
  # shellcheck disable=SC1091
  set -a
  source /workspace/beamdojo/.env.lambda
  set +a
fi
# rsl-rl 3.0.1 WandbSummaryWriter reads WANDB_USERNAME, not WANDB_ENTITY.
# Empty WANDB_USERNAME is set-but-blank → wandb.init(entity="") fails.
if [[ -n "${WANDB_ENTITY:-}" && -z "${WANDB_USERNAME:-}" ]]; then
  export WANDB_USERNAME="$WANDB_ENTITY"
fi
if [[ -z "${WANDB_USERNAME:-}" ]]; then
  unset WANDB_USERNAME || true
fi
export WANDB_PROJECT="${WANDB_PROJECT:-beamdojo}"
# First wandb.init on a cold Lambda often exceeds the 90s default.
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-180}"
export WANDB_HTTP_TIMEOUT="${WANDB_HTTP_TIMEOUT:-60}"
export PYTHONPATH="/workspace/beamdojo:/workspace/isaaclab/source/isaaclab:/workspace/isaaclab/source/isaaclab_assets:/workspace/isaaclab/source/isaaclab_tasks:/workspace/isaaclab/source/isaaclab_rl:${PYTHONPATH:-}"
cd /workspace/beamdojo/scripts/rsl_rl
