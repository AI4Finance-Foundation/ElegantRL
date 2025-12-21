# ElegantRL vs AlphaZero: Architecture Comparison

## Summary

**ElegantRL provides most of the infrastructure you need for AlphaZero-style training**, but it's designed for **model-free RL** rather than **model-based search**. This document explains what's included and what you'd need to add.

---

## ✅ What ElegantRL Provides

### 1. **Agent Inference** (Getting actions from states)
- ✅ **Actor networks** with `forward()` and `get_action()` methods
- ✅ **Deterministic inference** for evaluation
- ✅ **Stochastic inference** with exploration for training
- ✅ **GPU acceleration** for fast batch inference
- ✅ **State normalization** built-in

**Files:**
- `elegantrl/agents/AgentBase.py` - Base actor/critic interface
- `elegantrl/agents/AgentPPO.py` - On-policy example
- `elegantrl/agents/AgentSAC.py` - Off-policy example

### 2. **Replay Buffer** (Data storage and sampling)
- ✅ **GPU-based storage** for fast sampling
- ✅ **Save/load functionality** for offline training
- ✅ **Vectorized environments** support
- ✅ **Prioritized Experience Replay** (PER)
- ✅ **Millions of transitions** capacity

**Files:**
- `elegantrl/train/replay_buffer.py` - Full implementation

### 3. **Worker/Learner Architecture** (Parallel training)
- ✅ **Multi-process data collection** (workers)
- ✅ **Centralized learning** (learner)
- ✅ **Multi-GPU support** for scaling
- ✅ **Asynchronous training** pipeline
- ✅ **Evaluator process** for periodic testing

**Files:**
- `elegantrl/train/run.py` - Worker/learner/evaluator processes

### 4. **Offline RL Support** (Collect data → Train offline)
- ✅ **Save collected data** to disk
- ✅ **Load and train** on pre-collected data
- ✅ **Separate data collection from training**

---

## ❌ What's Missing for AlphaZero

### 1. **Monte Carlo Tree Search (MCTS)**
AlphaZero uses MCTS to search during inference:
```
For each move:
  1. Run MCTS simulations (800-1600 iterations)
  2. Use neural network to evaluate leaf nodes
  3. Backpropagate values through tree
  4. Select move based on visit counts
```

**ElegantRL alternative:**
- Standard RL agents (PPO, SAC) use direct policy networks
- No tree search during action selection
- Faster but potentially less strategic

**To add MCTS:**
```python
class MCTSAgent:
    def __init__(self, network, num_simulations=800):
        self.network = network
        self.num_simulations = num_simulations

    def get_action(self, state):
        root = MCTSNode(state)
        for _ in range(self.num_simulations):
            node = self.select(root)
            value = self.network.evaluate(node.state)
            self.backpropagate(node, value)
        return root.best_action()
```

### 2. **Dual-Head Network** (Policy + Value)
AlphaZero networks output both:
- **Policy head**: Action probabilities
- **Value head**: State value estimate

**ElegantRL has this partially:**
- Separate `actor` (policy) and `critic` (value) networks
- But not combined in single forward pass

**To adapt:**
```python
class AlphaZeroNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(...)  # Shared layers
        self.policy_head = nn.Linear(256, action_dim)
        self.value_head = nn.Linear(256, 1)

    def forward(self, state):
        features = self.shared(state)
        policy = F.softmax(self.policy_head(features), dim=-1)
        value = torch.tanh(self.value_head(features))
        return policy, value
```

### 3. **Self-Play Infrastructure**
AlphaZero trains by:
1. Current agent plays against itself
2. Generates game trajectories
3. Stores entire games with final outcomes
4. Trains on historical self-play data

**ElegantRL has:**
- Generic replay buffer (but no game-specific logic)
- No built-in self-play loop

**Solution:**
- See `examples/example_selfplay_game.py` (provided)
- Implement `TwoPlayerGame` interface
- Use `SelfPlayDataCollector` class

### 4. **Resignation and Temperature**
AlphaZero features:
- **Temperature**: Controls exploration during move selection
- **Resignation**: Agent gives up if value drops too low
- **Arena evaluation**: New model plays old model to determine improvement

**Can be added:**
```python
def get_action_with_temperature(state, temperature=1.0):
    policy, value = network(state)

    # Resignation
    if value < -0.9:
        return RESIGN_ACTION

    # Temperature scaling
    policy = policy ** (1 / temperature)
    policy = policy / policy.sum()

    return sample_from(policy)
```

---

## 📊 Side-by-Side Comparison

| Feature | AlphaZero | ElegantRL | Gap |
|---------|-----------|-----------|-----|
| **Neural Network** | | | |
| Policy output | ✅ | ✅ | None |
| Value output | ✅ | ✅ | None |
| Combined forward pass | ✅ | ⚠️ | Separate networks |
| **Action Selection** | | | |
| MCTS search | ✅ | ❌ | Must implement |
| Direct policy | ❌ | ✅ | Different approach |
| Temperature scaling | ✅ | ⚠️ | Easy to add |
| **Training** | | | |
| Self-play | ✅ | ⚠️ | Manual implementation |
| Replay buffer | ✅ | ✅ | None |
| Offline training | ✅ | ✅ | None |
| Multi-GPU | ✅ | ✅ | None |
| **Data Collection** | | | |
| Parallel workers | ✅ | ✅ | None |
| Game outcome labels | ✅ | ⚠️ | Manual implementation |
| Historical data window | ✅ | ⚠️ | Buffer size control |
| **Evaluation** | | | |
| Arena tournaments | ✅ | ⚠️ | Easy to add |
| ELO rating | ✅ | ❌ | Must implement |
| Deterministic play | ✅ | ✅ | None |

---

## 🛠️ Implementation Roadmap

### Your Use Case: Game Integration

You want to:
1. **Call function to get actions** during game evaluation
2. **Collect data** from game runs
3. **Train offline** on collected data

### ✅ **This is 100% Supported**

**Workflow:**

#### **Step 1: Define Your Game**
```python
class MyGame:
    def reset(self):
        """Return initial state"""
        return np.array([...])

    def step(self, action):
        """Return (next_state, reward, done, info)"""
        return next_state, reward, done, {}

    def get_valid_actions(self):
        """Optional: mask invalid moves"""
        return np.ones(action_dim)
```

#### **Step 2: Collect Data**
```python
from elegantrl.train.replay_buffer import ReplayBuffer
from examples.example_game_integration import GameAgent, GameDataCollector

# Initialize
agent = GameAgent(state_dim, action_dim)
collector = GameDataCollector(state_dim, action_dim)

# Play games and collect data
for episode in range(1000):
    collector.collect_episode(game, agent)

# Save data
collector.save_data('./my_game_data')
```

#### **Step 3: Train Offline**
```python
from examples.example_game_integration import GameTrainer

# Load data
trainer = GameTrainer(state_dim, action_dim)
buffer = ReplayBuffer(...)
buffer.save_or_load_history('./my_game_data', if_save=False)

# Train
trainer.train(buffer, num_updates=10000)
trainer.save_model('./trained_agent.pth')
```

#### **Step 4: Inference in Game**
```python
# Load trained agent
agent = GameAgent(state_dim, action_dim, model_path='./trained_agent.pth')

# In game loop
state = game.get_current_state()
action = agent.get_action(state, deterministic=True)
```

---

## 🎯 Quick Start for Your Use Case

### **For Simple RL (Recommended to Start):**

1. Use examples:
   - `examples/example_game_integration.py` - Basic workflow
   - Implements: data collection → offline training → evaluation

2. Choose agent:
   - **PPO**: Good for general games, stable
   - **SAC**: Sample efficient, continuous actions
   - **DQN**: Discrete actions, simpler

3. Customize:
   - Implement your game's `reset()` and `step()` methods
   - Define state/action dimensions
   - Run the example workflow

### **For Self-Play Games (AlphaZero-style):**

1. Use examples:
   - `examples/example_selfplay_game.py` - Self-play workflow
   - Implements: self-play → data collection → training

2. Implement `TwoPlayerGame` interface:
   ```python
   class MyGame(TwoPlayerGame):
       def reset(self): ...
       def step(self, action): ...
       def get_valid_actions(self): ...
       def get_current_player(self): ...
   ```

3. Run iterative training:
   ```python
   for iteration in range(10):
       # Generate self-play data
       # Train on collected data
       # Evaluate new model
   ```

### **For Full AlphaZero (Advanced):**

You'll need to add:
1. **MCTS class** for tree search
2. **Combined policy-value network**
3. **Arena evaluation** (new vs old model)
4. **Temperature scheduling**

Libraries to consider:
- [alpha-zero-general](https://github.com/suragnair/alpha-zero-general)
- [EfficientZero](https://github.com/YeWR/EfficientZero)

Or build on ElegantRL's infrastructure with custom MCTS.

---

## 📚 Recommended Reading

### ElegantRL Documentation
- [Running Instructions](../RUNNING_INSTRUCTIONS.md)
- [Agent Interface](../elegantrl/agents/AgentBase.py)
- [Replay Buffer](../elegantrl/train/replay_buffer.py)
- [Training Loop](../elegantrl/train/run.py)

### AlphaZero Resources
- [AlphaZero Paper](https://arxiv.org/abs/1712.01815)
- [AlphaGo Zero Paper](https://www.nature.com/articles/nature24270)
- [MCTS Survey](https://ieeexplore.ieee.org/document/6145622)

---

## 🤝 Community Support

For help implementing AlphaZero-style training on ElegantRL:
- **GitHub Issues**: https://github.com/AI4Finance-Foundation/ElegantRL/issues
- **Discord**: https://discord.gg/trsr8SXpW5

---

## ✅ Conclusion

**Can you use ElegantRL for your use case?**

**YES** - ElegantRL provides:
- ✅ Action inference from states
- ✅ Data collection infrastructure
- ✅ Offline training on collected data
- ✅ Multi-process/multi-GPU scaling
- ✅ Save/load functionality

**However:**
- ⚠️ Not a drop-in AlphaZero replacement
- ⚠️ No MCTS (but can be added)
- ⚠️ Standard RL agents (not game-specific)

**Recommendation:**
1. **Start with standard RL** (PPO/SAC) using provided examples
2. **Evaluate performance** on your game
3. **Add MCTS later** if needed for strategic depth

ElegantRL gives you a production-ready foundation for RL-based game agents!
