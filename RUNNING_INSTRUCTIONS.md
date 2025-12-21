# ElegantRL Running Instructions

This guide provides comprehensive instructions for setting up and running ElegantRL.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Running Examples](#running-examples)
5. [Using ElegantRL Helloworld](#using-elegantrl-helloworld)
6. [Training Your Own Agent](#training-your-own-agent)
7. [Running Tests](#running-tests)
8. [Optional Dependencies](#optional-dependencies)
9. [Troubleshooting](#troubleshooting)

## System Requirements

- **Python**: 3.8 or higher (Python 3.11+ recommended)
- **OS**: Linux, macOS, or Windows
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended for faster training)
- **RAM**: Minimum 8GB (16GB+ recommended for larger simulations)

## Installation

### Option 1: Install from PyPI (Recommended)

```bash
pip install elegantrl
```

### Option 2: Install from Source

1. **Clone the repository:**

```bash
git clone https://github.com/AI4Finance-Foundation/ElegantRL.git
cd ElegantRL
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Install ElegantRL in development mode:**

```bash
pip install -e .
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import gymnasium; print(f'Gymnasium version: {gymnasium.__version__}')"
```

## Quick Start

### Running Your First Training Session

The easiest way to get started is with the HelloWorld examples:

```bash
cd helloworld
python erl_tutorial_PPO.py
```

This will train a PPO agent on the Pendulum environment. The training process will:
- Display training progress in the console
- Save checkpoints to `./Pendulum_PPO_0/` directory
- Prompt you to render the trained agent after training completes

## Running Examples

ElegantRL provides several example scripts in the `examples/` directory:

### 1. Continuous Control Tasks

**PPO on LunarLanderContinuous:**
```bash
python examples/tutorial_LunarLanderContinous-v2.py
```

**TD3/SAC on continuous environments:**
```bash
python examples/demo_DDPG_TD3_SAC.py
```

**Hopper-v3 (MuJoCo):**
```bash
python examples/tutorial_Hopper-v3.py
```

### 2. Discrete Action Tasks

**DQN variants:**
```bash
python examples/demo_DQN_variants.py
```

**PPO/A2C on discrete environments:**
```bash
python examples/demo_A2C_PPO_discrete.py
```

### 3. Advanced Examples

**With Prioritized Experience Replay:**
```bash
python examples/demo_DDPG_TD3_SAC_with_PER.py
```

**Financial Trading (FinRL integration):**
```bash
python examples/demo_FinRL_ElegantRL_China_A_shares.py
```

## Using ElegantRL Helloworld

The `helloworld/` directory contains a lightweight tutorial version of ElegantRL.

### Structure

- `erl_config.py` - Configuration and hyperparameters
- `erl_agent.py` - DRL algorithms (PPO, DDPG, etc.)
- `erl_net.py` - Neural network architectures
- `erl_run.py` - Training loop
- `erl_env.py` - Environment wrappers

### Customize Your Training

Edit the training script to modify hyperparameters:

```python
from erl_config import Config
from erl_agent import AgentPPO

# Configure your agent
args = Config(agent_class, env_class, env_args)
args.break_step = int(2e5)      # Total training steps
args.net_dims = [64, 32]        # Network architecture
args.gamma = 0.97               # Discount factor
args.repeat_times = 16          # Update iterations per step
args.gpu_id = 0                 # GPU ID (-1 for CPU)

# Start training
train_agent(args)
```

## Training Your Own Agent

### Basic Template

```python
import gymnasium as gym
from elegantrl.agents import AgentPPO
from elegantrl.train.config import Config
from elegantrl.train.run import train_agent

# Define environment
env_class = gym.make
env_args = {
    'env_name': 'YourEnvironment-v0',
    'state_dim': 8,    # Observation space dimension
    'action_dim': 2,   # Action space dimension
    'if_discrete': False
}

# Configure agent
agent_class = AgentPPO
args = Config(agent_class, env_class, env_args)
args.gpu_id = 0
args.break_step = int(1e6)

# Train
train_agent(args)
```

### Available Agents

- **Continuous actions**: DDPG, TD3, SAC, PPO, REDQ
- **Discrete actions**: DQN, DoubleDQN, D3QN
- **Multi-agent**: QMIX, VDN, MADDPG, MAPPO, MATD3

## Running Tests

ElegantRL uses Python's built-in `unittest` framework.

### Run all tests:

```bash
python -m unittest discover
```

### Run specific test file:

```bash
python -m unittest unit_tests/test_training_agents.py
```

### Run specific test class or method:

```bash
python -m unittest unit_tests.test_training_agents.TestTrainingAgents
python -m unittest unit_tests.test_training_agents.TestTrainingAgents.test_ppo_training
```

**Note**: Some tests require Isaac Gym. Tests will fail if Isaac Gym is not installed.

## Optional Dependencies

### PyBullet (Free MuJoCo Alternative)

```bash
pip install pybullet>=3.2.0
```

### Box2D (For BipedalWalker, etc.)

```bash
pip install Box2D>=2.3.10
```

### Weights & Biases (Experiment Tracking)

```bash
pip install wandb>=0.13.0
```

### Isaac Gym (Massively Parallel Simulations)

1. Download from [NVIDIA Isaac Gym](https://developer.nvidia.com/isaac-gym)
2. Follow the installation instructions in the downloaded package

### StarCraft II Environment

```bash
bash ./elegantrl/envs/installsc2.sh
pip install -r sc2_requirements.txt
```

## GPU Support

### CUDA Setup

Ensure you have CUDA installed. Check compatibility:

```bash
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Using Multiple GPUs

ElegantRL supports multi-GPU training:

```python
args.gpu_id = 0  # Use GPU 0
# Or for multiple GPUs, use separate processes
```

## Monitoring Training

### Console Output

Training progress is displayed in real-time:
```
| Step: 10000 | Reward: 150.32 | Loss: 0.045 | Time: 120s
```

### Weights & Biases Integration

```python
import wandb
wandb.init(project="elegantrl-training")

# In your training loop, log metrics
wandb.log({"reward": avg_reward, "loss": loss})
```

### Saved Models

Trained models are saved in: `./EnvName_AgentName_GPU_ID/`

Files:
- `actor.pth` - Trained policy network
- `critic.pth` - Value network (if applicable)
- `recorder.npy` - Training statistics

## Troubleshooting

### Common Issues

**1. Import Error: "No module named 'elegantrl'"**
```bash
# Make sure you're in the right directory or install with pip
pip install -e .
```

**2. CUDA Out of Memory**
```python
# Reduce batch size or network dimensions
args.batch_size = 128  # Reduce from default
args.net_dims = [128, 128]  # Smaller networks
```

**3. Gym/Gymnasium Compatibility**
- ElegantRL now uses `gymnasium` (the maintained fork of gym)
- If you encounter old gym environments, install: `pip install gymnasium[classic-control]`

**4. Rendering Issues**
```bash
# For headless servers, use virtual display
sudo apt-get install xvfb
xvfb-run -a python your_script.py
```

**5. "th" module not found (Old installations)**
- This was a bug in older setup.py
- Update to the latest version or install from source

### Getting Help

- **Documentation**: https://elegantrl.readthedocs.io
- **GitHub Issues**: https://github.com/AI4Finance-Foundation/ElegantRL/issues
- **Discord**: https://discord.gg/trsr8SXpW5

## Performance Tips

1. **Use GPU**: Training is 10-100x faster on GPU
2. **Adjust batch size**: Larger batches use more memory but can be more efficient
3. **Tune hyperparameters**: Start with default values, then adjust based on performance
4. **Use vectorized environments**: For massively parallel simulations (Isaac Gym)
5. **Monitor GPU utilization**: `nvidia-smi -l 1` to check GPU usage

## Next Steps

- Read the [tutorials](https://elegantrl.readthedocs.io/en/latest/tutorial/tutorial.html)
- Explore [FinRL integration](https://github.com/AI4Finance-Foundation/FinRL) for financial applications
- Check out [ElegantRL-Podracer](https://elegantrl.readthedocs.io/en/latest/tutorial/elegantrl-podracer.html) for cloud-native deployment
- Join the community on [Discord](https://discord.gg/trsr8SXpW5)

## Citation

If you use ElegantRL in your research, please cite:

```bibtex
@misc{erl,
  author = {Liu, Xiao-Yang and Li, Zechu and Zhu, Ming and Wang, Zhaoran and Zheng, Jiahao},
  title = {{ElegantRL}: Massively Parallel Framework for Cloud-native Deep Reinforcement Learning},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/AI4Finance-Foundation/ElegantRL}},
}
```
