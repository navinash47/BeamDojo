# Algorithm Modification Guide for BeamDojo

This guide explains how to connect BeamDojo to IsaacLab and modify RL algorithms.

## Setup Connection to IsaacLab

### 1. Activate the Conda Environment

First, activate the IsaacLab conda environment:

```bash
conda activate env_isaaclab
```

### 2. Run the Setup Script

Source the setup script to configure the environment:

```bash
cd /home/lily-hcrlab/BeamDojo
source setup_isaaclab.sh
```

This script:
- Activates the `env_isaaclab` conda environment
- Sets up `ISAACLAB_PATH` and `BEAMDOJO_PATH` environment variables
- Adds IsaacLab source directories to `PYTHONPATH`
- Verifies that IsaacLab packages are installed

### 3. Verify Installation

Check that IsaacLab packages are accessible:

```bash
python -c "import isaaclab; import isaaclab_rl; import isaaclab_tasks; print('All packages imported successfully!')"
```

## Understanding the Algorithm Structure

### RSL-RL Algorithm Components

The RSL-RL algorithm in IsaacLab consists of several key components:

1. **Algorithm Configuration** (`RslRlPpoAlgorithmCfg`):
   - Located in: `/home/lily-hcrlab/issaclab/IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py`
   - Contains hyperparameters like learning rate, gamma, entropy coefficient, etc.

2. **Policy Configuration** (`RslRlPpoActorCriticCfg`):
   - Located in: `/home/lily-hcrlab/issaclab/IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py`
   - Defines network architecture (hidden dimensions, activation functions, etc.)

3. **Runner Configuration** (`RslRlOnPolicyRunnerCfg`):
   - Located in: `/home/lily-hcrlab/issaclab/IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py`
   - Contains training loop settings (num_steps_per_env, max_iterations, etc.)

### Where to Modify Algorithms

#### Option 1: Modify IsaacLab Source Directly (Not Recommended)

You can directly edit files in:
- `/home/lily-hcrlab/issaclab/IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py`
- `/home/lily-hcrlab/issaclab/IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py`

**Warning**: This will affect all projects using IsaacLab.

#### Option 2: Create Custom Algorithm Configuration (Recommended)

1. Create a new file in BeamDojo: `scripts/custom_algorithm.py`

```python
from isaaclab_rl.rsl_rl import (
    RslRlPpoAlgorithmCfg,
    RslRlPpoActorCriticCfg,
    RslRlOnPolicyRunnerCfg,
)

# Custom algorithm configuration
@configclass
class CustomPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Custom PPO algorithm with modified hyperparameters."""
    
    learning_rate: float = 5e-4  # Modified from default
    entropy_coef: float = 0.01   # Modified from default
    # Add your custom parameters here

# Custom policy configuration
@configclass
class CustomActorCriticCfg(RslRlPpoActorCriticCfg):
    """Custom actor-critic network."""
    
    actor_hidden_dims: list[int] = [512, 256, 128]  # Modified architecture
    # Add your custom parameters here
```

2. Modify `train.py` to use your custom configuration:

```python
from scripts.custom_algorithm import CustomPpoAlgorithmCfg, CustomActorCriticCfg

# In the main function, override the agent_cfg:
agent_cfg.policy = CustomActorCriticCfg()
agent_cfg.algorithm = CustomPpoAlgorithmCfg()
```

#### Option 3: Override Configuration at Runtime

Modify the configuration in `train.py` after loading:

```python
@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    # Modify algorithm hyperparameters
    agent_cfg.algorithm.learning_rate = 5e-4
    agent_cfg.algorithm.entropy_coef = 0.01
    
    # Modify policy architecture
    agent_cfg.policy.actor_hidden_dims = [512, 256, 128]
    
    # Continue with training...
```

## Key Algorithm Parameters

### PPO Algorithm Parameters (`RslRlPpoAlgorithmCfg`)

- `learning_rate`: Learning rate for the optimizer (typically 1e-4 to 5e-4)
- `gamma`: Discount factor (typically 0.99)
- `lam`: GAE lambda parameter (typically 0.95)
- `entropy_coef`: Entropy regularization coefficient (typically 0.01)
- `value_loss_coef`: Value function loss coefficient (typically 0.5)
- `clip_param`: PPO clipping parameter (typically 0.2)
- `num_learning_epochs`: Number of optimization epochs per update
- `num_mini_batches`: Number of mini-batches per update

### Policy Parameters (`RslRlPpoActorCriticCfg`)

- `actor_hidden_dims`: List of hidden layer dimensions for actor network
- `critic_hidden_dims`: List of hidden layer dimensions for critic network
- `activation`: Activation function ("elu", "relu", "tanh", etc.)
- `init_noise_std`: Initial action noise standard deviation
- `actor_obs_normalization`: Whether to normalize observations for actor
- `critic_obs_normalization`: Whether to normalize observations for critic

### Runner Parameters (`RslRlOnPolicyRunnerCfg`)

- `num_steps_per_env`: Number of steps to collect per environment per update
- `max_iterations`: Maximum number of training iterations
- `save_interval`: How often to save checkpoints
- `device`: Device to run on ("cuda:0", "cpu", etc.)

## Checking Current Configuration

To see what packages are installed in the conda environment:

```bash
conda activate env_isaaclab
pip freeze | grep -i isaac
```

To check the Python path:

```bash
python -c "import sys; print('\n'.join(sys.path))"
```

## Example: Modifying Learning Rate

Here's a complete example of modifying the learning rate:

1. In `train.py`, add after line 111:

```python
# Override learning rate
agent_cfg.algorithm.learning_rate = 5e-4  # Your custom value
```

2. Run training:

```bash
source setup_isaaclab.sh
python scripts/train.py --task <your_task> --max_iterations 1000
```

## Troubleshooting

### Import Errors

If you get import errors:
1. Make sure `env_isaaclab` is activated: `conda activate env_isaaclab`
2. Run the setup script: `source setup_isaaclab.sh`
3. Verify IsaacLab packages are installed: `pip list | grep isaaclab`

### Module Not Found: cli_args

The `cli_args.py` file is now in `scripts/cli_args.py`. Make sure you're running from the BeamDojo directory.

### Algorithm Configuration Not Found

Make sure the task configuration includes an RSL-RL entry point. Check task configs in:
`/home/lily-hcrlab/issaclab/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/`

## Additional Resources

- IsaacLab RSL-RL documentation: Check the source files in `isaaclab_rl/rsl_rl/`
- RSL-RL library: https://github.com/leggedrobotics/rsl_rl
- IsaacLab documentation: https://isaac-sim.github.io/IsaacLab/

