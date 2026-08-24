#!/usr/bin/env bash
# SSH to the BeamDojo Lambda A10. IP lives in gitignored .env.lambda
# and ~/.ssh/config Host lambda-beamdojo.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
if [[ -f "$ROOT/.env.lambda" ]]; then
  # shellcheck disable=SC1091
  set -a
  source "$ROOT/.env.lambda"
  set +a
fi
if [[ -n "${LAMBDA_HOST:-}" ]]; then
  echo "Lambda A10 at ${LAMBDA_HOST} (ssh lambda-beamdojo)"
fi
exec ssh lambda-beamdojo "$@"
