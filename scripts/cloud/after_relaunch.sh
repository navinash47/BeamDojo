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
  synced=0
  if git -C "$REPO" fetch origin "$REF"; then
    synced=1
  elif git -C "$REPO" fetch origin; then
    synced=1
  else
    echo "[WARN] git fetch failed; using checkout as-is ($(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown))."
  fi
  if [[ "$synced" -eq 1 ]]; then
    # Discard dirty *tracked* files so NFS trains this origin tree. Never git clean
    # (untracked .env.lambda and checkpoints must survive).
    if git -C "$REPO" show-ref --verify --quiet "refs/remotes/origin/${REF}"; then
      git -C "$REPO" checkout -f -B "$REF" "origin/${REF}"
    elif git -C "$REPO" show-ref --verify --quiet "refs/heads/${REF}"; then
      git -C "$REPO" checkout -f "$REF"
    fi
  fi
  echo "BeamDojo HEAD $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown) (${REF})"
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
if ! grep -q "def reassert_gpu_env_cfg" "$REPO/scripts/rsl_rl/beamdojo_runtime.py"; then
  echo "beamdojo_runtime.py is missing Hydra leftover reassert. Pull ${REF} or a restored ANYmal RayCaster / mdp.height_scan crashes gym.make." >&2
  exit 1
fi
if ! grep -q "def _patch_wandb_init_retry" "$REPO/scripts/rsl_rl/beamdojo_runtime.py"; then
  echo "beamdojo_runtime.py is missing wandb.init retry. Pull ${REF} or a single Lambda 503 falls back to TensorBoard with no live W&B page." >&2
  exit 1
fi
if ! grep -q "def write_boot_status" "$REPO/scripts/rsl_rl/beamdojo_runtime.py"; then
  echo "beamdojo_runtime.py is missing write_boot_status. Pull ${REF} or Research Lab stays Idle for the whole Isaac AppLauncher boot." >&2
  exit 1
fi
if ! grep -q "def clear_stale_distributed_env" "$REPO/scripts/rsl_rl/beamdojo_runtime.py"; then
  echo "beamdojo_runtime.py is missing leftover WORLD_SIZE clear. Pull ${REF} or rsl-rl NCCL-inits before wandb.init." >&2
  exit 1
fi
if ! grep -q "def anymal_parent_body_names" "$REPO/scripts/rsl_rl/beamdojo_runtime.py"; then
  echo "beamdojo_runtime.py is missing ANYmal body-name reassert. Pull ${REF} or leftover body=base / .*THIGH crashes gym.make." >&2
  exit 1
fi
if ! grep -q "leftover curriculum.terrain_levels" "$REPO/scripts/rsl_rl/beamdojo_runtime.py"; then
  echo "beamdojo_runtime.py is missing leftover terrain_levels clear. Pull ${REF} or parent terrain_levels_vel crashes the first reset before W&B." >&2
  exit 1
fi
if ! grep -q "def _ensure_obs_groups" "$REPO/scripts/rsl_rl/beamdojo_runtime.py"; then
  echo "beamdojo_runtime.py is missing obs_groups sanitize. Pull ${REF} or leftover MISSING obs_groups dies in OnPolicyRunner before wandb.init." >&2
  exit 1
fi
if ! grep -q "def beamdojo_obs_groups_ok" "$REPO/scripts/rsl_rl/beamdojo_runtime.py"; then
  echo "beamdojo_runtime.py is missing BeamDojo obs_groups pin. Pull ${REF} or leftover critic:['critic'] dies in OnPolicyRunner before wandb.init." >&2
  exit 1
fi
if ! grep -q "def _drop_on_policy_runner_kwarg_collisions" "$REPO/scripts/rsl_rl/beamdojo_runtime.py"; then
  echo "beamdojo_runtime.py is missing OnPolicyRunner kwarg sanitize. Pull ${REF} or leftover algorithm.device TypeErrors before wandb.init." >&2
  exit 1
fi
if ! grep -q "def _reassert_g1_joint_fullmatch" "$REPO/scripts/rsl_rl/beamdojo_runtime.py"; then
  echo "beamdojo_runtime.py is missing G1 leftover H1 regex restore. Pull ${REF} or RewardManager re.fullmatch dies at gym.make before W&B." >&2
  exit 1
fi
if ! grep -q "def _drop_h1_leftover_fingers" "$REPO/scripts/rsl_rl/beamdojo_runtime.py"; then
  echo "beamdojo_runtime.py is missing H1 leftover finger drop. Pull ${REF} or leftover G1 fingers crash resolve_matching_names on H1." >&2
  exit 1
fi
if ! grep -q "def _reassert_unitree_robot" "$REPO/scripts/rsl_rl/beamdojo_runtime.py"; then
  echo "beamdojo_runtime.py is missing Unitree robot reassert. Pull ${REF} or leftover ANYmal USD crashes gym.make before W&B." >&2
  exit 1
fi
if ! grep -q "def _reassert_contact_history" "$REPO/scripts/rsl_rl/beamdojo_runtime.py"; then
  echo "beamdojo_runtime.py is missing contact history reassert. Pull ${REF} or leftover history_length=0 dies at the first reset." >&2
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
ROBOT="${ROBOT:-h1}"
TERRAIN="${TERRAIN:-beam}"
echo "Starting Stage ${STAGE} ${ROBOT} on CUDA."
echo "Live curves: Weights & Biases (printed by train_stage*.sh)."
echo "Kingdom Research Lab polls tracking/training-status.json — rsync that file to the Mac or open the cloud tunnel."

# Isaac-free: flip Research Lab off Idle before AppLauncher (5–15 min).
python3 -c "
import beamdojo_runtime
beamdojo_runtime.write_boot_status(
    stage=int('${STAGE}'),
    robot='${ROBOT}',
    terrain='${TERRAIN}',
    note='after_relaunch: starting Isaac / Stage ${STAGE}. Not a live W&B run yet.',
)
print('[INFO] Wrote boot training-status (unknown) before isaaclab.sh.')
"

# Fail fast on a rejected key so Isaac does not boot for 15 min with no W&B page.
python3 - <<'PY'
import os
import sys
try:
    import wandb
except ImportError:
    print("[INFO] host python has no wandb; Isaac train process will init W&B.")
    sys.exit(0)
key = os.environ.get("WANDB_API_KEY", "").strip()
try:
    wandb.login(key=key, relogin=True)
except Exception as exc:
    print(f"WANDB_API_KEY rejected ({type(exc).__name__}: {exc}). Fix .env.lambda.", file=sys.stderr)
    sys.exit(1)
print("[INFO] wandb login accepted.")
PY

if [[ "$STAGE" == "2" ]]; then
  echo "Stage 2 loads the latest Stage 1 checkpoint. Override with LOAD_RUN, CHECKPOINT, or LOAD_EXPERIMENT."
  exec "$ROOT/train_stage2.sh"
fi
exec "$ROOT/train_stage1.sh"
