# Setup Summary: Connecting BeamDojo to IsaacLab

## What Was Done

### 1. Created `scripts/cli_args.py`
   - Copied from IsaacLab's RSL-RL training scripts
   - Provides CLI argument parsing for RSL-RL training
   - Required by `train.py`

### 2. Created `setup_isaaclab.sh`
   - Bash script that configures the environment
   - Activates `env_isaaclab` conda environment
   - Sets up environment variables (`ISAACLAB_PATH`, `BEAMDOJO_PATH`)
   - Adds IsaacLab source directories to `PYTHONPATH`
   - Verifies IsaacLab packages are installed

### 3. Created Documentation
   - `README.md`: Quick start guide and project overview
   - `ALGORITHM_MODIFICATION_GUIDE.md`: Detailed guide on modifying RL algorithms
   - `SETUP_SUMMARY.md`: This file

### 4. Created `check_environment.py`
   - Python script to verify environment setup
   - Checks conda environment, imports, and configuration

## How to Use

### Basic Usage

```bash
# 1. Activate conda environment
conda activate env_isaaclab

# 2. Navigate to BeamDojo
cd /home/lily-hcrlab/BeamDojo

# 3. Run setup script
source setup_isaaclab.sh

# 4. Verify setup
python check_environment.py

# 5. Run training
python scripts/train.py --task <task_name> --max_iterations 1000
```

### Checking Installed Packages

```bash
conda activate env_isaaclab
pip freeze | grep -i isaac
```

Or using conda:

```bash
conda activate env_isaaclab
conda list | grep -i isaac
```

## Environment Variables Set by setup_isaaclab.sh

- `ISAACLAB_PATH`: `/home/lily-hcrlab/issaclab/IsaacLab`
- `BEAMDOJO_PATH`: `/home/lily-hcrlab/BeamDojo`
- `PYTHONPATH`: Includes:
  - `${ISAACLAB_PATH}/source` (for isaaclab, isaaclab_rl, isaaclab_tasks packages)
  - `${ISAACLAB_PATH}/scripts/reinforcement_learning/rsl_rl` (for cli_args if needed)

## Algorithm Modification Locations

### Where Algorithms Are Defined

1. **Algorithm Configuration**: 
   - `/home/lily-hcrlab/issaclab/IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py`
   - Classes: `RslRlPpoAlgorithmCfg`, `RslRlPpoActorCriticCfg`, `RslRlOnPolicyRunnerCfg`

2. **Algorithm Implementation**:
   - RSL-RL library (installed via pip: `rsl-rl-lib`)
   - IsaacLab wrapper: `/home/lily-hcrlab/issaclab/IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py`

### How to Modify Algorithms

**Option 1: Runtime Override (Easiest)**
- Modify `agent_cfg` in `train.py` after it's loaded
- Example: `agent_cfg.algorithm.learning_rate = 5e-4`

**Option 2: Custom Configuration Class (Recommended)**
- Create new config classes in `scripts/custom_algorithm.py`
- Inherit from IsaacLab's config classes
- Override parameters as needed

**Option 3: Direct Source Modification (Not Recommended)**
- Edit files in IsaacLab source directory
- Affects all projects using IsaacLab

See `ALGORITHM_MODIFICATION_GUIDE.md` for detailed examples.

## File Structure

```
BeamDojo/
├── scripts/
│   ├── train.py                    # Main training script (from IsaacLab)
│   └── cli_args.py                 # CLI argument parser (copied from IsaacLab)
├── setup_isaaclab.sh               # Environment setup script
├── check_environment.py            # Environment verification script
├── README.md                       # Quick start guide
├── ALGORITHM_MODIFICATION_GUIDE.md # Algorithm modification guide
└── SETUP_SUMMARY.md                # This file
```

## Troubleshooting

### Issue: Import errors
**Solution**: 
1. Activate conda: `conda activate env_isaaclab`
2. Run setup: `source setup_isaaclab.sh`
3. Verify: `python check_environment.py`

### Issue: cli_args not found
**Solution**: Make sure you're running from BeamDojo root directory. The `cli_args.py` is in `scripts/cli_args.py`.

### Issue: IsaacLab packages not found
**Solution**: 
```bash
cd /home/lily-hcrlab/issaclab/IsaacLab
conda activate env_isaaclab
./isaaclab.sh -i
```

## Next Steps

1. **Test the setup**: Run `python check_environment.py` to verify everything works
2. **Try training**: Run a simple training task to test the connection
3. **Modify algorithms**: Use the guide to customize algorithm parameters
4. **Check packages**: Use `pip freeze` or `conda list` to see installed packages

## Notes

- The conda environment `env_isaaclab` must exist and have IsaacLab packages installed
- IsaacLab packages are installed in editable mode from `/home/lily-hcrlab/issaclab/IsaacLab/source/`
- The setup script needs to be sourced each time you open a new terminal (or add to your shell profile)

