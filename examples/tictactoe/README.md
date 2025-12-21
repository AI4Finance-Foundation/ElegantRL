# Tic-Tac-Toe with ElegantRL

Complete example of using ElegantRL for game playing, demonstrating:
- Custom game environment implementation
- Self-play training
- Multiple evaluation modes
- Inference during gameplay

## Quick Start

### 1. Train an Agent

**Basic training (self-play):**
```bash
cd examples/tictactoe
python train.py --iterations 20 --games 100
```

**Train against random opponent:**
```bash
python train.py --iterations 20 --games 100 --opponent random
```

**Advanced training options:**
```bash
python train.py \
    --iterations 50 \
    --games 200 \
    --updates 1000 \
    --opponent self \
    --gpu 0 \
    --save-dir ./my_checkpoints
```

### 2. Evaluate Trained Agent

**Evaluate against random opponent:**
```bash
python evaluate.py --model checkpoints/best_model.pth --mode vs_random --num-games 100
```

**Play against the agent interactively:**
```bash
python evaluate.py --model checkpoints/best_model.pth --mode vs_human
```

**Watch agent play against itself:**
```bash
python evaluate.py --model checkpoints/best_model.pth --mode watch --num-games 5
```

---

## File Structure

```
tictactoe/
├── game.py          # Tic-Tac-Toe game environment
├── agent.py         # RL agent wrapper with action masking
├── train.py         # Training script
├── evaluate.py      # Evaluation script
└── README.md        # This file
```

---

## Training Details

### Training Process

1. **Data Collection**: Agent plays games (self-play or vs random)
2. **Reward Assignment**: +1 for wins, -1 for losses, 0 for draws
3. **Buffer Storage**: Transitions stored in replay buffer
4. **Network Updates**: PPO agent updates policy and value networks
5. **Evaluation**: Periodic evaluation against random opponent
6. **Checkpointing**: Best model saved based on win rate

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--iterations` | 20 | Number of training iterations |
| `--games` | 100 | Games to play per iteration |
| `--updates` | 500 | Gradient updates per iteration |
| `--opponent` | self | Opponent type: 'self' or 'random' |
| `--gpu` | 0 | GPU ID (-1 for CPU) |
| `--save-dir` | ./checkpoints | Directory for saving models |

### Expected Performance

After 20 iterations (~2000 games):
- **vs Random**: ~80-95% win rate
- **Self-play**: Should reach optimal play (mostly draws)

---

## Evaluation Modes

### 1. vs_random

Evaluate agent against random opponent:
```bash
python evaluate.py \
    --model checkpoints/best_model.pth \
    --mode vs_random \
    --num-games 100 \
    --verbose  # Optional: show each game
```

**Output:**
```
Games: 100/100 | Wins:  92 | Losses:   3 | Draws:   5 | Win Rate: 92.0%
```

### 2. vs_human

Play interactively against the agent:
```bash
python evaluate.py \
    --model checkpoints/best_model.pth \
    --mode vs_human \
    --agent-player 0  # Agent plays first (X)
```

**Example game:**
```
Tic-Tac-Toe: Human vs Agent
You are Player 1 (O)
Agent is Player 0 (X)

Board positions:
  0 1 2
  3 4 5
  6 7 8

Current board:
  . . .
  . . .
  . . .

Agent is thinking...
Agent chooses position 4
Position probabilities: [0.05 0.05 0.05 0.05 0.6 0.05 0.05 0.05 0.05]
State value estimate: 0.123

Current board:
  . . .
  . X .
  . . .

Your turn!
Enter your move (0-8): 0
```

### 3. watch

Watch agent play against itself:
```bash
python evaluate.py \
    --model checkpoints/best_model.pth \
    --mode watch \
    --num-games 5
```

Shows complete games with move-by-move analysis.

---

## Architecture

### Game Environment (`game.py`)

```python
class TicTacToe:
    def reset(self) -> np.ndarray
        """Return initial state (9-element array)"""

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]
        """Execute action, return (state, reward, done, info)"""

    def get_valid_actions_mask(self) -> np.ndarray
        """Return binary mask for valid actions"""

    def render(self)
        """Print board to console"""
```

**State representation:**
- 9-element numpy array
- Values: 1 (current player), -1 (opponent), 0 (empty)
- Automatically flipped based on current player perspective

**Actions:**
- Integers 0-8 representing board positions
- Layout:
  ```
  0 1 2
  3 4 5
  6 7 8
  ```

**Rewards:**
- +1: Win
- -1: Loss
- 0: Draw or ongoing game

### Agent (`agent.py`)

```python
class TicTacToeAgent:
    def get_action(state, valid_mask, temperature=1.0, deterministic=False) -> int
        """Get action with masking and temperature control"""

    def get_action_probs(state, valid_mask, temperature=1.0) -> np.ndarray
        """Get action probability distribution"""

    def get_value(state) -> float
        """Get state value estimate"""

    def save_model(path)
        """Save trained model"""

    def load_model(path)
        """Load trained model"""
```

**Features:**
- **Action Masking**: Prevents invalid moves
- **Temperature**: Controls exploration (higher = more random)
- **Deterministic Mode**: For evaluation
- **Value Estimation**: For debugging/analysis

### Network Architecture

```python
Actor (Policy Network):
- Input: State (9 dimensions)
- Hidden: [128, 128]
- Output: Action logits (9 dimensions)

Critic (Value Network):
- Input: State (9 dimensions)
- Hidden: [128, 128]
- Output: State value (1 dimension)
```

---

## Using This Code for Your Own Game

### Step 1: Implement Game Environment

```python
class MyGame:
    def reset(self):
        """Return initial state"""
        return np.array([...])  # Your state representation

    def step(self, action):
        """Execute action, return (state, reward, done, info)"""
        # Your game logic here
        return next_state, reward, done, info

    def get_valid_actions_mask(self):
        """Return mask of valid actions (optional but recommended)"""
        return np.ones(action_dim)  # All valid if not implemented
```

### Step 2: Update Agent Configuration

In `agent.py`, modify:
```python
self.agent = AgentPPO(
    net_dims=[256, 256, 256],  # Adjust network size
    state_dim=YOUR_STATE_DIM,
    action_dim=YOUR_ACTION_DIM,
    gpu_id=gpu_id
)
```

### Step 3: Modify Training Script

Update `train.py` to use your game:
```python
from my_game import MyGame

# In SelfPlayTrainer.__init__:
self.buffer = ReplayBuffer(
    max_size=100_000,
    state_dim=YOUR_STATE_DIM,
    action_dim=YOUR_ACTION_DIM,  # Set to 1 for discrete actions
    gpu_id=gpu_id
)

# In play_selfplay_game:
game = MyGame()  # Use your game
```

### Step 4: Train and Evaluate

```bash
python train.py --iterations 50
python evaluate.py --model checkpoints/best_model.pth --mode vs_random
```

---

## Advanced Features

### Custom Reward Shaping

Modify `train.py` to add intermediate rewards:
```python
# In collect_data(), when creating transitions:
if winner == player:
    reward = 1.0
    # Add bonus for faster wins
    reward += 0.1 * (9 - len(game_data)) / 9
elif winner != -1:
    reward = -1.0
else:
    reward = 0.0
    # Small penalty for draws
    reward -= 0.05
```

### Temperature Scheduling

Control exploration over time:
```python
# In train():
temperature = max(0.5, 1.0 - iteration / num_iterations)
```

### Curriculum Learning

Start training against weak opponents:
```python
# Iteration 0-5: vs random
# Iteration 6-10: 50% self-play, 50% random
# Iteration 11+: 100% self-play
```

---

## Troubleshooting

### Agent not improving

**Solutions:**
1. Increase training iterations
2. Increase games per iteration
3. Try different opponents (self-play vs random)
4. Adjust network size
5. Check temperature scheduling

### Agent plays invalid moves

**Should not happen** due to action masking, but if it does:
1. Check `get_valid_actions_mask()` implementation
2. Verify mask is applied in `agent.get_action()`
3. Add assertion: `assert valid_mask[action] == 1`

### Training is slow

**Solutions:**
1. Use GPU: `--gpu 0`
2. Reduce `--updates` per iteration
3. Reduce `--games` per iteration
4. Use smaller network: `net_dims=[64, 64]`

---

## Extending This Example

### Multi-Agent Training

- Train different agents with different strategies
- Run tournaments between agents
- Implement ELO rating system

### MCTS Integration

Add Monte Carlo Tree Search for stronger play:
```python
def get_action_with_mcts(state, valid_mask, num_simulations=100):
    # Run MCTS simulations
    # Use agent.get_value() to evaluate leaf nodes
    # Return best action from search
```

### More Complex Games

- **Connect Four**: 7x6 board, gravity
- **Chess**: ~64 state dims, complex rules
- **Go**: 19x19 board, large action space

---

## Performance Benchmarks

Tested on:
- CPU: Intel i7-10700K
- GPU: NVIDIA RTX 3080
- PyTorch 2.0

| Configuration | Games/sec | Training Time |
|---------------|-----------|---------------|
| CPU only | ~50 | ~30 min |
| GPU (RTX 3080) | ~200 | ~8 min |

For 20 iterations × 100 games = 2000 games total

---

## References

- [ElegantRL Documentation](https://elegantrl.readthedocs.io)
- [AlphaZero Paper](https://arxiv.org/abs/1712.01815)
- [PPO Paper](https://arxiv.org/abs/1707.06347)

---

## Questions?

- **GitHub Issues**: https://github.com/AI4Finance-Foundation/ElegantRL/issues
- **Discord**: https://discord.gg/trsr8SXpW5
- **Documentation**: https://elegantrl.readthedocs.io

Happy Training! 🎮🤖
