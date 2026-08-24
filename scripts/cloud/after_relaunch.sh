#!/usr/bin/env bash
# Run ON a new Lambda A10 after attach of the beamdojo NFS. Does not launch a GPU from a laptop.
set -euo pipefail
ROOT="$(cd "$(dirname "$0") && pwd)"
# shellcheck disable=SC1091
source "$ROOT/_env.sh"

if ! nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi failed. Refusing CPU host." >&2
  exit 1
fi
if ! nvidia-smi | grep -E -q 'A10|A6000|L40|RTX|GeForce'; then
  echo "Need an RT-core GPU (A10). Not A100/H100." >&2
  nvidia-smi
  exit 1
fi

if [[ ! -d /lambda/nfs/beamdojo ]]; then
  echo "Attach filesystem beamdojo at instance create. Missing /lambda/nfs/beamdojo" >&2
  exit 1
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY missing in .env.lambda. Research Lab / W&B will not get live curves." >&2
  exit 1
fi

echo "Starting Stage 1 on CUDA. Watch W&B + Kingdom Research Lab."
exec "$ROOT/train_stage1.sh"
