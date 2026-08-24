#!/usr/bin/env bash
# Run ON a new Lambda A10 after attach of the beamdojo NFS. Does not launch a GPU from a laptop.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$ROOT/../.." && pwd)"
REF="${BEAMDOJO_GIT_REF:-cursor/dual-terrain-stage2-g1-73ce}"

# Pull the GPU-path + live W&B writer before sourcing _env.sh (that script cds).
if [[ -d "$REPO/.git" ]]; then
  echo "Syncing BeamDojo ${REF} so this A10 trains the current dual-terrain / W&B path."
  # Lambda/Docker often flags the NFS checkout as dubious ownership; fetch then fails
  # and the box trains a stale tree that still dies after Isaac boot.
  git config --global --add safe.directory "$REPO" || true
  git -C "$REPO" fetch origin "$REF" || git -C "$REPO" fetch origin || echo "[WARN] git fetch failed; using checkout as-is."
  if git -C "$REPO" show-ref --verify --quiet "refs/remotes/origin/${REF}" || git -C "$REPO" show-ref --verify --quiet "refs/heads/${REF}"; then
    git -C "$REPO" checkout "$REF" || git -C "$REPO" checkout -B "$REF" "origin/${REF}" || true
    git -C "$REPO" pull --ff-only origin "$REF" || true
  fi
fi

if ! grep -q "return RigidObjectCfg(" "$REPO/h1_cfg/scene_props.py"; then
  echo "scene_props.py is missing per-env RigidObjectCfg. Pull ${REF} before training or Stage 2 falls through the beam." >&2
  exit 1
fi
if ! grep -q "def sanitize_rsl_rl_train_cfg" "$REPO/scripts/rsl_rl/beamdojo_runtime.py"; then
  echo "beamdojo_runtime.py is missing Hydra RND sanitize. Pull ${REF} or OnPolicyRunner treats rnd_cfg={} as enabled." >&2
  exit 1
fi
if ! grep -q "def _patch_store_code_state" "$REPO/scripts/rsl_rl/beamdojo_runtime.py"; then
  echo "beamdojo_runtime.py is missing git-diff keep-alive. Pull ${REF} or dubious-ownership git status aborts learn()." >&2
  exit 1
fi
if ! grep -q "def sanitize_ep_infos_for_rsl_log" "$REPO/scripts/rsl_rl/beamdojo_runtime.py"; then
  echo "beamdojo_runtime.py is missing extras['log'] sanitize. Pull ${REF} or a per-env foothold tensor in ep_infos blanks W&B for the iter." >&2
  exit 1
fi
if ! grep -q "def apply_physx_gpu_capacity" "$REPO/h1_cfg/beamdojo_common.py"; then
  echo "beamdojo_common.py is missing PhysX GPU buffer bump. Pull ${REF} or cloned beams/stones overflow contact buffers." >&2
  exit 1
fi
if ! grep -q "PHYSX_PATCH_COUNT_BEAM" "$REPO/h1_cfg/physx_gpu.py"; then
  echo "h1_cfg/physx_gpu.py is missing A10-safe PhysX floors. Pull ${REF} or a stale 16M contact stream OOMs the A10 before W&B." >&2
  exit 1
fi
if grep -qE '["'"'"']gpu_max_rigid_contact_count["'"'"']' "$REPO/h1_cfg/physx_gpu.py"; then
  echo "physx_gpu.py must not raise gpu_max_rigid_contact_count (A10 24GB OOM). Pull ${REF}." >&2
  exit 1
fi

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

STAGE="${STAGE:-1}"
echo "Starting Stage ${STAGE} ${ROBOT:-h1} on CUDA."
echo "Live curves: Weights & Biases (printed by train_stage*.sh)."
echo "Kingdom Research Lab polls tracking/training-status.json — rsync that file to the Mac or open the cloud tunnel."
if [[ "$STAGE" == "2" ]]; then
  echo "Stage 2 loads the latest Stage 1 checkpoint. Override with LOAD_RUN, CHECKPOINT, or LOAD_EXPERIMENT."
  exec "$ROOT/train_stage2.sh"
fi
exec "$ROOT/train_stage1.sh"
