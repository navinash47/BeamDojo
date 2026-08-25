#!/usr/bin/env bash
# G1 Stage 1 on the A10. Same 1024-env / 10k-iter W&B path as H1.
set -euo pipefail
export ROBOT="${ROBOT:-g1}"
exec "$(cd "$(dirname "$0")" && pwd)/train_stage1.sh"
