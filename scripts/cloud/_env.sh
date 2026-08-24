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
export PYTHONPATH="/workspace/beamdojo:/workspace/isaaclab/source/isaaclab:/workspace/isaaclab/source/isaaclab_assets:/workspace/isaaclab/source/isaaclab_tasks:/workspace/isaaclab/source/isaaclab_rl:${PYTHONPATH:-}"
cd /workspace/beamdojo/scripts/rsl_rl
