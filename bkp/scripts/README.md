# BeamDojo

BeamDojo project connected to IsaacLab for reinforcement learning training.

## Quick Start

### 1. Setup Connection to IsaacLab

Activate the conda environment and run the setup script:

```bash
conda activate env_isaaclab
cd /home/lily-hcrlab/BeamDojo
source setup_isaaclab.sh
```

### 2. Verify Installation

Check that everything is set up correctly:

```bash
python -c "import isaaclab; import isaaclab_rl; import isaaclab_tasks; print('✓ All packages imported successfully!')"
```

### 3. Run Training

```bash
# Example: Train H1 robot on rough terrain
python scripts/train.py --task Isaac-Locomotion-Velocity-Rough-H1-v0 --max_iterations 1000

# Example: Train with custom number of environments
python scripts/train.py --task Isaac-Locomotion-Velocity-Rough-H1-v0 --num_envs 4096 --max_iterations 2000

# Example: Train with video recording
python scripts/train.py --task Isaac-Locomotion-Velocity-Rough-H1-v0 --video --max_iterations 1000
```

#### Available Task Names

- `Isaac-Locomotion-Velocity-Rough-H1-v0`: H1 humanoid robot on rough terrain (training)
- `Isaac-Locomotion-Velocity-Rough-H1-Play-v0`: H1 humanoid robot on rough terrain (play/evaluation)

For more task names, see the [IsaacLab tasks documentation](https://github.com/isaac-sim/IsaacLab).

## Environment Setup

### Prerequisites

- Conda environment `env_isaaclab` must exist
- IsaacLab packages must be installed in the conda environment

### Installation Steps

If the conda environment doesn't exist or IsaacLab packages aren't installed:

```bash
# Navigate to IsaacLab directory
cd /home/lily-hcrlab/issaclab/IsaacLab

# Create conda environment (if it doesn't exist)
./isaaclab.sh -c env_isaaclab

# Activate the environment
conda activate env_isaaclab

# Install IsaacLab packages
./isaaclab.sh -i
```

## Project Structure

```
BeamDojo/
├── scripts/
│   ├── train.py          # Main training script
│   └── cli_args.py       # CLI argument parser for RSL-RL
├── setup_isaaclab.sh     # Setup script to connect to IsaacLab
├── README.md            # This file
└── ALGORITHM_MODIFICATION_GUIDE.md  # Guide for modifying algorithms
```

## Configuration

The setup script (`setup_isaaclab.sh`) configures:

- `ISAACLAB_PATH`: Path to IsaacLab repository (`/home/lily-hcrlab/issaclab/IsaacLab`)
- `BEAMDOJO_PATH`: Path to BeamDojo repository
- `PYTHONPATH`: Adds IsaacLab source directories for imports

## Checking Installed Packages

To see what's installed in the conda environment:

```bash
conda activate env_isaaclab
pip freeze | grep -i isaac
```

Or use conda:

```bash
conda activate env_isaaclab
conda list | grep -i isaac
```

## Modifying Algorithms

See [ALGORITHM_MODIFICATION_GUIDE.md](ALGORITHM_MODIFICATION_GUIDE.md) for detailed instructions on how to modify RL algorithms.

## Troubleshooting

### Import Errors

If you encounter import errors:

1. Make sure the conda environment is activated: `conda activate env_isaaclab`
2. Run the setup script: `source setup_isaaclab.sh`
3. Verify IsaacLab packages: `pip list | grep isaaclab`

### Module Not Found: cli_args

The `cli_args.py` file is located in `scripts/cli_args.py`. Make sure you're running from the BeamDojo root directory.

### IsaacLab Packages Not Found

If IsaacLab packages are not found:

```bash
cd /home/lily-hcrlab/issaclab/IsaacLab
conda activate env_isaaclab
./isaaclab.sh -i
```

## Links

- IsaacLab: `/home/lily-hcrlab/issaclab/IsaacLab`
- Conda Environment: `env_isaaclab`
