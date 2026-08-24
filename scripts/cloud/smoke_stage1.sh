#!/usr/bin/env bash
# Run inside isaac-lab-base. GPU-only 64-env Stage 1 smoke.
set -euo pipefail
export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=YES
export BEAMDOJO_LOG_ROOT=/workspace/isaaclab/logs
export PYTHONPATH="/workspace/isaaclab/source/isaaclab:/workspace/isaaclab/source/isaaclab_assets:/workspace/isaaclab/source/isaaclab_tasks:/workspace/isaaclab/source/isaaclab_rl:${PYTHONPATH:-}"
cd /workspace/beamdojo/scripts/rsl_rl
exec /workspace/isaaclab/isaaclab.sh -p train_beamdojo.py \
  --headless \
  --device cuda:0 \
  --num_envs 64 \
  --max_iterations 5
