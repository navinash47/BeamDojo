#!/usr/bin/env bash
# Run ON the Lambda A10 after SSH. Pulls Isaac Sim/Lab and smokes BeamDojo on CUDA.
set -euo pipefail

if ! nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi failed. Refusing CPU/no-GPU host." >&2
  exit 1
fi
if ! nvidia-smi | grep -E -q 'A10|A6000|L40|RTX|GeForce'; then
  echo "GPU does not look like an RT-core card. Isaac Sim visualization needs RT cores (not A100/H100)." >&2
  nvidia-smi
  exit 1
fi

if [[ -z "${NGC_API_KEY:-}" ]]; then
  echo "Set NGC_API_KEY (docker login nvcr.io user \$oauthtoken)." >&2
  exit 1
fi

NFS_ROOT="${BEAMDOJO_NFS:-/lambda/nfs/beamdojo}"
if [[ ! -d "$NFS_ROOT" ]]; then
  echo "NFS $NFS_ROOT missing. Attach the beamdojo filesystem at instance launch." >&2
  exit 1
fi

mkdir -p "$NFS_ROOT/logs" "$NFS_ROOT/docker" "$NFS_ROOT/src"
export BEAMDOJO_LOG_ROOT="$NFS_ROOT/logs"

echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin

ISAAC_SIM_TAG="${ISAAC_SIM_TAG:-5.1.0}"
docker pull "nvcr.io/nvidia/isaac-sim:${ISAAC_SIM_TAG}"

if [[ ! -d "$NFS_ROOT/src/IsaacLab" ]]; then
  git clone --depth 1 --branch v2.3.2 https://github.com/isaac-sim/IsaacLab.git "$NFS_ROOT/src/IsaacLab"
fi

# Lambda Stack is headless (compute-only). Isaac Sim RTX needs GL/Vulkan userspace
# matching the kernel driver, then a fresh CDI spec so Docker injects those libs.
DRIVER_VER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')"
GL_PKG="libnvidia-gl-580-server"
if dpkg -s "$GL_PKG" >/dev/null 2>&1; then
  echo "Host already has $GL_PKG"
else
  echo "Installing $GL_PKG to match driver ${DRIVER_VER} (Isaac Sim RTX)."
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    "${GL_PKG}=${DRIVER_VER}-0lambda0.22.04.1" || \
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "$GL_PKG"
fi
sudo nvidia-ctk cdi generate --output=/var/run/cdi/nvidia.yaml

echo "Bootstrap ready. Recreate isaac-lab-base after CDI generate, then:"
echo "  bash /workspace/beamdojo/scripts/cloud/repair_isaac_python.sh"
echo "  bash /workspace/beamdojo/scripts/cloud/smoke_stage1.sh"
