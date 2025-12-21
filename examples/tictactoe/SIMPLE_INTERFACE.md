# Simple User-Friendly Interface for ElegantRL

This directory provides a simplified interface for using ElegantRL in your games, with two main classes:

1. **`Evaluator`** - Load a trained model and get actions (for evaluation/gameplay)
2. **`DataSaver`** - Collect gameplay data and save it for training

## Quick Start

### 1. Collect Data from Gameplay

```python
from game import TicTacToe
from agent import TicTacToeAgent
from data_saver import DataSaver

# Initialize
agent = TicTacToeAgent()  # Untrained agent
saver = DataSaver(save_dir='./my_data', save_frequency=10)

# Play a game
game = TicTacToe()
state = game.reset()

episode_id = saver.new_episode()

while not done:
    action = agent.get_action(state, valid_mask)
    logprob = calculate_logprob(action)  # Get from your agent

    # Save transition
    saver.add_transition(state, action, logprob)

    state, reward, done, info = game.step(action)

# Set final reward (with optional gamma for reward propagation)
saver.set_reward(reward=1.0, gamma=0.99)

# Flush to disk
saver.flush()
```

### 2. Train from Collected Data

```python
from data_saver import DataSaver
from agent import TicTacToeAgent

# Load data
episodes = DataSaver.load_episodes('./my_data')
buffer_data = DataSaver.convert_to_tensors(episodes)

# Train
agent = TicTacToeAgent()
agent.agent.last_state = torch.zeros((1, 9))
agent.agent.update_net(buffer_data)

# Save trained model
agent.save_model('./trained_model.pth')
```

### 3. Use Trained Model for Evaluation

```python
from evaluator import Evaluator

# Load trained model
evaluator = Evaluator(model_path='./trained_model.pth')

# Get actions during gameplay
action = evaluator.get_action(state, valid_mask, deterministic=True)
```

---

## Class: `Evaluator`

Simple interface for loading a trained model and getting actions.

### Initialization

```python
from evaluator import Evaluator

evaluator = Evaluator(
    model_path='checkpoints/best_model.pth',  # Path to trained model
    state_dim=9,                               # State dimension
    action_dim=9,                              # Action dimension
    gpu_id=-1                                  # GPU ID (-1 for CPU)
)
```

### Getting Actions

```python
# Deterministic action (best action)
action = evaluator.get_action(state, valid_mask, deterministic=True)

# Stochastic action (sample from distribution)
action = evaluator.get_action(state, valid_mask, deterministic=False, temperature=1.0)

# Get full probability distribution
probs = evaluator.get_action_probs(state, valid_mask)

# Get state value estimate
value = evaluator.get_value(state)
```

### Example: Play Against Human

```python
from game import TicTacToe
from evaluator import Evaluator

evaluator = Evaluator('trained_model.pth')
game = TicTacToe()
state = game.reset()

while not done:
    if current_player == AI:
        action = evaluator.get_action(state, valid_mask)
    else:
        action = int(input("Your move (0-8): "))

    state, reward, done, info = game.step(action)
```

---

## Class: `DataSaver`

Collects gameplay data and saves it for training.

### Initialization

```python
from data_saver import DataSaver

saver = DataSaver(
    save_dir='./training_data',  # Where to save data
    save_frequency=10,             # Save every N episodes
    state_dim=9,                   # State dimension
    action_dim=1                   # Action dimension (1 for discrete)
)
```

### Collecting Data

```python
# Start new episode
episode_id = saver.new_episode()

# Add transitions during gameplay
for each step:
    saver.add_transition(
        state=state,           # Current state (numpy array)
        action=action,         # Action taken (int)
        logprob=logprob,      # Log probability (optional)
        player=player_id,      # Player ID (optional, for multi-agent)
        metadata={}            # Any additional data (optional)
    )

# Set reward at end
saver.set_reward(
    reward=1.0,      # Final reward
    gamma=0.99,      # Discount factor for reward propagation
    player=0         # Player ID (optional)
)

# Force save remaining data
saver.flush()
```

### Reward Propagation with Gamma

The `gamma` parameter controls how rewards are distributed across previous actions:

```python
# gamma=0.0: Only last action gets reward
saver.set_reward(reward=1.0, gamma=0.0)
# Last action: 1.0
# All others: 0.0

# gamma=0.9: Rewards propagate backward
saver.set_reward(reward=1.0, gamma=0.9)
# Last action: 1.0
# Second-to-last: 0.9
# Third-to-last: 0.81
# Fourth-to-last: 0.729
# etc.

# gamma=0.99: Strong reward propagation
saver.set_reward(reward=1.0, gamma=0.99)
# Rewards decay slowly, crediting earlier actions
```

### Loading Saved Data

```python
from data_saver import DataSaver

# Load all episodes
episodes = DataSaver.load_episodes('./training_data')

# Load specific episode range
episodes = DataSaver.load_episodes('./training_data', episode_range=(100, 200))

# Convert to tensors for training
buffer_data = DataSaver.convert_to_tensors(episodes, device='cuda')
# Returns: (states, actions, logprobs, rewards, undones, unmasks)
```

### Data Format

Saved data structure:
```python
{
    'episode_id': 1,
    'num_transitions': 8,
    'final_reward': 1.0,
    'gamma': 0.99,
    'timestamp': '2025-01-15T10:30:00',
    'transitions': [
        {
            'state': np.array([...]),
            'action': 4,
            'logprob': -2.1,
            'player': 0,
            'reward': 0.95,  # Computed based on gamma
            'metadata': {}
        },
        # ... more transitions
    ]
}
```

---

## Complete Examples

### Example 1: Simple Collection and Training

```bash
# Collect 100 games of data
python simple_train.py --mode collect --games 100 --data-dir ./my_data

# Train from collected data
python simple_train.py --mode train --data-dir ./my_data --save-model my_model.pth

# Or do both iteratively
python simple_train.py --mode both --iterations 10 --games 50
```

### Example 2: Evaluation

```bash
# Evaluate against random opponent
python simple_eval.py --model my_model.pth --mode vs_random --num-games 100

# Play against the model
python simple_eval.py --model my_model.pth --mode vs_human
```

### Example 3: Custom Game Integration

```python
from data_saver import DataSaver
from evaluator import Evaluator

# Your custom game
class MyGame:
    def reset(self): ...
    def step(self, action): ...
    def get_valid_actions(self): ...

# Collect data
saver = DataSaver(save_dir='./my_game_data', save_frequency=20)

for game_num in range(100):
    game = MyGame()
    state = game.reset()

    episode_id = saver.new_episode()

    while not done:
        action = select_action(state)  # Your action selection
        logprob = get_logprob(action)  # Your logprob calculation

        saver.add_transition(state, action, logprob)

        state, reward, done, info = game.step(action)

    # Set reward with gamma for credit assignment
    saver.set_reward(reward, gamma=0.95)

saver.flush()

# Train
episodes = DataSaver.load_episodes('./my_game_data')
buffer_data = DataSaver.convert_to_tensors(episodes)
# ... train your agent
```

---

## Key Features

### Automatic Episode IDs
- DataSaver automatically generates unique episode IDs
- Counter saved in `.episode_counter` file
- Persistent across sessions

### Flexible Reward Assignment
- Set rewards at episode end
- Control reward propagation with `gamma`
- Support for multi-agent games (per-player rewards)

### Efficient Storage
- Data saved in batches (controlled by `save_frequency`)
- Pickle format for full data
- JSON metadata for easy inspection
- Automatic filename generation with timestamps

### Easy Training Pipeline
1. Collect data during gameplay
2. Save to disk periodically
3. Load all data at once
4. Convert to tensors
5. Train your agent

---

## Comparison: Simple vs Original Interface

### Original Interface (train.py)
```python
# Complex: Manual data collection and formatting
trainer = SelfPlayTrainer()
buffer_data = trainer.collect_trajectory_data(...)
obj_critic, obj_actor = agent.update_net(buffer_data)
```

### Simple Interface (simple_train.py)
```python
# Easy: Just save transitions and set rewards
saver = DataSaver(save_dir='./data')
saver.new_episode()
saver.add_transition(state, action, logprob)
saver.set_reward(reward, gamma=0.99)
saver.flush()

# Train later
episodes = DataSaver.load_episodes('./data')
buffer_data = DataSaver.convert_to_tensors(episodes)
agent.update_net(buffer_data)
```

---

## Tips and Best Practices

### 1. Reward Propagation (Gamma)
- **gamma=0.0**: Use when only the final action matters (e.g., checkmating in chess)
- **gamma=0.9-0.95**: Moderate credit assignment (good default)
- **gamma=0.99**: Strong credit assignment (use for long episodes)

### 2. Save Frequency
- Higher frequency = more disk writes, but safer (less data loss)
- Lower frequency = fewer disk writes, more in memory
- Recommended: 10-50 episodes

### 3. Multi-Agent Games
```python
# Set different rewards for each player
saver.set_reward(reward=1.0, gamma=0.99, player=0)  # Winner
saver.set_reward(reward=-1.0, gamma=0.99, player=1)  # Loser
```

### 4. Metadata
```python
# Store additional info for analysis
saver.add_transition(
    state, action, logprob,
    metadata={
        'time_taken': 0.5,
        'depth': 3,
        'board_complexity': 15
    }
)
```

### 5. Loading Specific Episodes
```python
# Load only recent episodes for training
recent_episodes = DataSaver.load_episodes('./data', episode_range=(900, 1000))
```

---

## File Structure

```
training_data/
├── .episode_counter                     # Episode counter (persistent)
├── episodes_000001_to_000010_*.pkl     # Episode data
├── episodes_000001_to_000010_*_meta.json  # Metadata (human-readable)
├── episodes_000011_to_000020_*.pkl
├── episodes_000011_to_000020_*_meta.json
└── ...
```

---

## Advanced Usage

### Custom Agent Integration

```python
from evaluator import Evaluator
from my_custom_agent import MyAgent

# Use custom agent with Evaluator
evaluator = Evaluator(
    model_path='model.pth',
    agent_class=MyAgent,  # Your custom agent class
    state_dim=64,
    action_dim=16
)
```

### Batch Training

```python
# Collect data over multiple sessions
for session in range(10):
    # Session 1: Collect 100 episodes
    saver = DataSaver('./data', save_frequency=10)
    collect_games(saver, num_games=100)
    saver.flush()

# Train on all collected data
all_episodes = DataSaver.load_episodes('./data')
print(f"Training on {len(all_episodes)} episodes")

buffer_data = DataSaver.convert_to_tensors(all_episodes)
agent.update_net(buffer_data)
```

---

## Troubleshooting

**Q: DataSaver not saving?**
- Call `saver.flush()` at the end
- Check `save_frequency` (higher = less frequent saves)

**Q: Episode IDs not incrementing?**
- Check `.episode_counter` file exists and is writable
- Call `saver.flush()` to save counter

**Q: Out of memory during training?**
- Load episodes in batches: `episode_range=(start, end)`
- Reduce number of episodes loaded at once

**Q: Logprobs missing?**
- Set `logprob=None` in `add_transition()` if not using PPO
- Or calculate later during data loading

---

For complete examples, see:
- `simple_train.py` - Data collection and training
- `simple_eval.py` - Evaluation with Evaluator
- Original `train.py` - Full-featured training loop (for comparison)
