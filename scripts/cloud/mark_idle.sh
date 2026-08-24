#!/usr/bin/env bash
# Call this BEFORE terminating the A10 so Kingdom does not keep status=running.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${ROOT}/scripts/rsl_rl:${ROOT}:${PYTHONPATH:-}"
python3 -c 'import beamdojo_runtime as rt; rt.mark_training_idle("A10 terminating / no train running. NFS checkpoints survive; root disk does not."); print("Wrote idle training-status.json")'
