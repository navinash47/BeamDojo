#!/usr/bin/env bash
# GPU RTX video of the latest Stage 1 checkpoint. Run inside isaac-lab-base.
set -euo pipefail
export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=YES
export BEAMDOJO_LOG_ROOT=/workspace/isaaclab/logs
export PYTHONPATH="/workspace/isaaclab/source/isaaclab:/workspace/isaaclab/source/isaaclab_assets:/workspace/isaaclab/source/isaaclab_tasks:/workspace/isaaclab/source/isaaclab_rl:${PYTHONPATH:-}"
cd /workspace/beamdojo/scripts/rsl_rl
exec /workspace/isaaclab/isaaclab.sh -p play_beamdojo.py \
  --headless \
  --enable_cameras \
  --video \
  --video_length 80 \
  --num_envs 4 \
  --device cuda:0
