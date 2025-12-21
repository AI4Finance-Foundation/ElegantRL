# User-Friendly Interface for ElegantRL

## Overview

This branch adds a simplified, production-ready interface for integrating ElegantRL into your games. The new interface separates concerns into two main classes:

1. **`Evaluator`** - Load trained models and get actions (inference)
2. **`DataSaver`** - Collect gameplay data and save for training

## Key Features

### Evaluator Class

**Purpose:** Simple inference interface for deployed models

**Features:**
- Load trained model with one line
- Get actions with `get_action(state, valid_mask)`
- Automatic device management (CPU/GPU)
- Optional action probabilities and value estimates
- Support for deterministic and stochastic policies

**Example:**
```python
from evaluator import Evaluator

# Load model
evaluator = Evaluator('trained_model.pth')

# Get action during gameplay
action = evaluator.get_action(state, valid_mask, deterministic=True)
```

### DataSaver Class

**Purpose:** Collect and save training data from gameplay

**Features:**
- **Automatic episode IDs** - Unique ID generation with persistent counter
- **Configurable save frequency** - Save every N episodes to control disk I/O
- **Reward propagation** - Gamma parameter for credit assignment to previous actions
- **Multi-agent support** - Per-player reward assignment
- **Automatic batching** - Efficient disk storage in batches
- **Easy loading** - Load episodes and convert to tensors for training

**Example:**
```python
from data_saver import DataSaver

# Initialize
saver = DataSaver(save_dir='./training_data', save_frequency=10)

# Start episode
episode_id = saver.new_episode()

# Collect transitions
saver.add_transition(state, action, logprob, player=0)

# Set final reward (gamma=0.99 for reward propagation)
saver.set_reward(reward=1.0, gamma=0.99, player=0)

# Force save
saver.flush()
```

## Architecture

### Data Flow

```
Gameplay → DataSaver → Disk Storage → Load Episodes → Train → Save Model → Evaluator
```

**Phase 1: Data Collection**
```python
saver = DataSaver(save_dir='./data', save_frequency=10)

for game in games:
    saver.new_episode()

    for step in game:
        saver.add_transition(state, action, logprob)

    saver.set_reward(final_reward, gamma=0.99)

saver.flush()
```

**Phase 2: Training**
```python
# Load collected data
episodes = DataSaver.load_episodes('./data')
buffer_data = DataSaver.convert_to_tensors(episodes)

# Train
agent.update_net(buffer_data)
agent.save_model('trained_model.pth')
```

**Phase 3: Deployment**
```python
# Use in production
evaluator = Evaluator('trained_model.pth')
action = evaluator.get_action(state, valid_mask)
```

## Key Innovations

### 1. Persistent Episode Counter

DataSaver maintains a counter in `.episode_counter` file:
- Generates unique IDs across sessions
- No ID collisions
- Easy to track total episodes

### 2. Reward Propagation with Gamma

Control how rewards credit previous actions:

```python
# gamma=0.0: Only last action gets reward
saver.set_reward(reward=1.0, gamma=0.0)

# gamma=0.99: Strong propagation (last action: 1.0, previous: 0.99, etc.)
saver.set_reward(reward=1.0, gamma=0.99)
```

This is critical for games where early moves matter!

### 3. Flexible Save Frequency

Balance between data safety and performance:

```python
# Save after every episode (safest, slower)
saver = DataSaver(save_frequency=1)

# Save every 50 episodes (faster, more risk)
saver = DataSaver(save_frequency=50)
```

### 4. Multi-Agent Support

Assign different rewards to different players:

```python
# Player 0 wins
saver.set_reward(reward=1.0, gamma=0.99, player=0)
saver.set_reward(reward=-1.0, gamma=0.99, player=1)
```

## Usage Examples

### Example 1: Collect Data from Gameplay

```python
from game import TicTacToe
from agent import TicTacToeAgent
from data_saver import DataSaver

agent = TicTacToeAgent()
saver = DataSaver('./data', save_frequency=10)

for game_num in range(100):
    game = TicTacToe()
    state = game.reset()

    episode_id = saver.new_episode()

    while not done:
        valid_mask = game.get_valid_actions_mask()
        action = agent.get_action(state, valid_mask)
        logprob = get_logprob(action)  # From your agent

        saver.add_transition(state, action, logprob)

        state, reward, done, info = game.step(action)

    saver.set_reward(reward, gamma=0.99)

saver.flush()
print(f"Collected {saver.get_stats()['total_episodes']} episodes")
```

### Example 2: Train from Saved Data

```python
from data_saver import DataSaver
from agent import TicTacToeAgent
import torch

# Load data
episodes = DataSaver.load_episodes('./data')
print(f"Loaded {len(episodes)} episodes")

# Convert to tensors
buffer_data = DataSaver.convert_to_tensors(episodes, device='cuda')

# Train
agent = TicTacToeAgent()
agent.agent.last_state = torch.zeros((1, 9))

for iteration in range(10):
    obj_critic, obj_actor, obj_entropy = agent.agent.update_net(buffer_data)
    print(f"Iteration {iteration}: critic={obj_critic:.4f}")

agent.save_model('trained_model.pth')
```

### Example 3: Deploy Trained Model

```python
from evaluator import Evaluator
from game import TicTacToe

# Load trained model
evaluator = Evaluator('trained_model.pth')

# Play game
game = TicTacToe()
state = game.reset()

while not done:
    valid_mask = game.get_valid_actions_mask()

    # Get best action
    action = evaluator.get_action(state, valid_mask, deterministic=True)

    # Show reasoning
    probs = evaluator.get_action_probs(state, valid_mask)
    value = evaluator.get_value(state)
    print(f"Action: {action}, Value: {value:.3f}")

    state, reward, done, info = game.step(action)
```

## Simplified Scripts

### simple_train.py

All-in-one training script with three modes:

```bash
# Mode 1: Just collect data
python simple_train.py --mode collect --games 100 --data-dir ./my_data

# Mode 2: Just train from existing data
python simple_train.py --mode train --data-dir ./my_data --save-model my_model.pth

# Mode 3: Iterative collect + train
python simple_train.py --mode both --iterations 10 --games 50
```

### simple_eval.py

Easy evaluation:

```bash
# Evaluate vs random
python simple_eval.py --model my_model.pth --mode vs_random --num-games 100

# Play against model
python simple_eval.py --model my_model.pth --mode vs_human
```

## Comparison: Simple vs Original

### Original Interface
```python
# Complex: Manual data formatting
trainer = SelfPlayTrainer()
buffer_data, stats = trainer.collect_trajectory_data(100)
obj_critic, obj_actor = agent.update_net(buffer_data)
```

**Pros:**
- Full control over data collection
- Integrated training loop

**Cons:**
- Tightly coupled (can't easily separate collection and training)
- All data in memory
- No persistence across sessions
- Complex to integrate into existing games

### Simple Interface
```python
# Easy: Just save transitions
saver = DataSaver('./data', save_frequency=10)
saver.new_episode()
saver.add_transition(state, action, logprob)
saver.set_reward(reward, gamma=0.99)
saver.flush()

# Train later
episodes = DataSaver.load_episodes('./data')
buffer_data = DataSaver.convert_to_tensors(episodes)
agent.update_net(buffer_data)
```

**Pros:**
- Decoupled (collect data anytime, train later)
- Persistent storage
- Easy to integrate into existing games
- Flexible reward assignment
- Production-ready

**Cons:**
- Slightly more code for simple cases
- Need to manage data directory

## Files Created

```
examples/tictactoe/
├── evaluator.py             # Evaluator class
├── data_saver.py            # DataSaver class
├── simple_train.py          # Simplified training script
├── simple_eval.py           # Simplified evaluation script
└── SIMPLE_INTERFACE.md      # Detailed documentation
```

## Pull Request

Branch: `claude/user-friendly-interface-XluTa`

Create PR: https://github.com/Battlecode2026/ElegantRL/pull/new/claude/user-friendly-interface-XluTa

## Benefits for Users

1. **Easier Integration** - Drop Evaluator/DataSaver into existing games
2. **Flexible Training** - Collect data whenever, train offline
3. **Production Ready** - Persistent storage, error handling
4. **Multi-Agent** - Per-player rewards out of the box
5. **Reward Shaping** - Gamma parameter for credit assignment
6. **Clear Separation** - Inference vs data collection vs training

## Next Steps

1. Test with your game
2. Adjust save_frequency based on needs
3. Experiment with gamma for reward propagation
4. Use Evaluator for deployed models

See **SIMPLE_INTERFACE.md** in examples/tictactoe/ for complete API documentation and examples!
