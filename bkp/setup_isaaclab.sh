#!/usr/bin/env bash

# Setup script to connect BeamDojo to IsaacLab via conda environment env_isaaclab
# This script configures the environment to use IsaacLab packages

set -e

# Get the directory where this script is located
BEAMDOJO_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ISAACLAB_PATH="/home/lily-hcrlab/issaclab/IsaacLab"

# Check if conda environment exists
if ! conda env list | grep -q "env_isaaclab"; then
    echo "[ERROR] Conda environment 'env_isaaclab' not found."
    echo "Please create it first by running:"
    echo "  cd ${ISAACLAB_PATH}"
    echo "  ./isaaclab.sh -c env_isaaclab"
    exit 1
fi

# Activate conda environment
echo "[INFO] Activating conda environment: env_isaaclab"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate env_isaaclab

# Check if IsaacLab packages are installed
if ! python -c "import isaaclab" 2>/dev/null; then
    echo "[ERROR] IsaacLab packages are not installed in env_isaaclab."
    echo "Please install them by running:"
    echo "  cd ${ISAACLAB_PATH}"
    echo "  conda activate env_isaaclab"
    echo "  ./isaaclab.sh -i"
    exit 1
fi

# Export environment variables
export ISAACLAB_PATH="${ISAACLAB_PATH}"
export BEAMDOJO_PATH="${BEAMDOJO_PATH}"

# Initialize PYTHONPATH if it's empty
if [[ -z "${PYTHONPATH}" ]]; then
    PYTHONPATH=""
fi

# Add IsaacLab source directories to PYTHONPATH if not already there
if [[ ":$PYTHONPATH:" != *":${ISAACLAB_PATH}/source:"* ]]; then
    if [[ -z "${PYTHONPATH}" ]]; then
        PYTHONPATH="${ISAACLAB_PATH}/source"
    else
        PYTHONPATH="${ISAACLAB_PATH}/source:${PYTHONPATH}"
    fi
fi

# Add IsaacLab scripts directory to PYTHONPATH for cli_args (if needed)
if [[ ":$PYTHONPATH:" != *":${ISAACLAB_PATH}/scripts/reinforcement_learning/rsl_rl:"* ]]; then
    PYTHONPATH="${ISAACLAB_PATH}/scripts/reinforcement_learning/rsl_rl:${PYTHONPATH}"
fi

# Export the final PYTHONPATH
export PYTHONPATH

echo "[INFO] Environment configured successfully!"
echo "[INFO] ISAACLAB_PATH: ${ISAACLAB_PATH}"
echo "[INFO] BEAMDOJO_PATH: ${BEAMDOJO_PATH}"
echo "[INFO] PYTHONPATH: ${PYTHONPATH}"

# Print installed IsaacLab packages
echo ""
echo "[INFO] Installed IsaacLab packages:"
pip list | grep -i isaaclab | head -5

echo ""
echo "[INFO] To use this setup, run:"
echo "  source ${BEAMDOJO_PATH}/setup_isaaclab.sh"
echo "  python ${BEAMDOJO_PATH}/scripts/train.py --task <task_name>"

