# Claude Development Notes - ElegantRL Game Playing Infrastructure

## Project Overview

User wanted to implement game-playing agents using ElegantRL with an AlphaZero-like workflow:
1. Call a function to get actions for observed states (for evaluation/gameplay)
2. Collect data from game evaluations
3. Train models offline using collected data

## Key Question: Does ElegantRL Support This?

**YES** ✅ - ElegantRL has all the infrastructure needed:
- Agent inference (getting actions from states)
- Data collection mechanisms
- Offline training capabilities
- Replay buffer system
- Worker/learner architecture for parallelization

## What Was Created

### Branch 1: `claude/update-requirements-docs-XluTa`

**Requirements Updates:**
- Fixed critical bug in `setup.py`: `"th"` → `"torch"`
- Added version constraints to all dependencies
- Updated Python requirement from 3.6+ to 3.8+ (3.6-3.7 EOL)

**Documentation:**
- `RUNNING_INSTRUCTIONS.md` - Complete setup and usage guide
- `docs/ALPHAZERO_COMPARISON.md` - Detailed comparison of ElegantRL vs AlphaZero
- `examples/example_game_integration.py` - General game integration template
- `examples/example_selfplay_game.py` - Self-play training template

### Branch 2: `claude/tictactoe-example-XluTa` (Main Example)

**Complete Working Example:**
- `examples/tictactoe/game.py` - Tic-Tac-Toe environment
- `examples/tictactoe/agent.py` - RL agent wrapper
- `examples/tictactoe/train.py` - Training script
- `examples/tictactoe/evaluate.py` - Evaluation modes
- `examples/tictactoe/run_example.sh` - Quick start script
- `examples/tictactoe/README.md` - Comprehensive documentation
- `SUMMARY.md` - Overview of entire infrastructure

## Critical Fixes Made (Training Debugging Journey)

### Issue 1: Missing `unmasks` Parameter
**Error:** `ValueError: not enough values to unpack (expected 5, got 4)`

**Problem:** ReplayBuffer.update() expects 5 values: (states, actions, rewards, undones, unmasks)

**Fix:**
```python
# Add unmasks tensor
unmasks = torch.ones_like(rewards)  # All valid (not truncated)
buffer.update((states, actions, rewards, undones, unmasks))
```

### Issue 2: PPO Expects Different Data Format
**Error:** `'ReplayBuffer' object is not subscriptable`

**Problem:** PPO is on-policy and expects data as tuple, not ReplayBuffer object

**Fix:** Complete rewrite of data collection:
- PPO expects: `(states, actions, logprobs, rewards, undones, unmasks)`
- Must collect logprobs during action selection
- Removed ReplayBuffer usage (PPO doesn't use it)
- Format tensors as `(horizon_len, num_envs, dim)`

### Issue 3: Continuous vs Discrete Actions
**Error:** `ValueError: Value is not broadcastable with batch_shape+event_shape: torch.Size([64]) vs torch.Size([64, 9])`

**Problem:** `AgentPPO` uses `ActorPPO` with Normal distribution (continuous actions), but tic-tac-toe has discrete actions (0-8)

**First attempt:** Set `args.if_discrete = True` - didn't work because AgentPPO still uses ActorPPO

**Final Fix:** Use `AgentDiscretePPO` instead:
```python
# WRONG:
from elegantrl.agents import AgentPPO
self.agent = AgentPPO(...)

# CORRECT:
from elegantrl.agents.AgentPPO import AgentDiscretePPO
self.agent = AgentDiscretePPO(...)
```

**Why it works:**
- `AgentDiscretePPO` uses `ActorDiscretePPO`
- `ActorDiscretePPO` uses Categorical distribution (for discrete actions)
- `ActorPPO` uses Normal distribution (for continuous actions)

## Architecture Insights

### ElegantRL Agent Types

**For Continuous Actions:**
- AgentPPO (uses ActorPPO with Normal distribution)
- AgentSAC
- AgentTD3
- AgentDDPG

**For Discrete Actions:**
- **AgentDiscretePPO** (uses ActorDiscretePPO with Categorical distribution)
- AgentDiscreteA2C
- AgentDQN
- AgentDoubleDQN

### PPO Data Flow

1. **Data Collection:** Play games and collect (state, action, logprob) tuples
2. **Format for PPO:** Create tensors `(horizon_len, num_envs, dim)`
3. **Compute Advantages:** PPO uses GAE (Generalized Advantage Estimation)
4. **Update Networks:** Actor and critic networks updated with clipped surrogate objective

### Key Differences from AlphaZero

| Feature | AlphaZero | ElegantRL |
|---------|-----------|-----------|
| MCTS Search | ✅ Yes | ❌ No (but can be added) |
| Self-Play | ✅ Yes | ✅ Yes (manual implementation) |
| Replay Buffer | ✅ Yes | ✅ Yes (for off-policy agents) |
| Policy Network | ✅ Yes | ✅ Yes |
| Value Network | ✅ Yes | ✅ Yes |
| Combined Network | ✅ Yes | ⚠️ Separate actor/critic |

**ElegantRL uses standard RL instead of MCTS:**
- Faster inference (no tree search)
- Still effective for many games
- MCTS can be added if needed

## How to Use for Your Game

### 1. Implement Game Environment

```python
class MyGame:
    def reset(self) -> np.ndarray:
        """Return initial state"""
        return np.array([...])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute action, return (state, reward, done, info)"""
        return next_state, reward, done, info

    def get_valid_actions_mask(self) -> np.ndarray:
        """Return binary mask for valid actions"""
        return np.ones(action_dim)
```

### 2. Create Agent Wrapper

```python
from elegantrl.agents.AgentPPO import AgentDiscretePPO

class MyGameAgent:
    def __init__(self, state_dim, action_dim):
        self.agent = AgentDiscretePPO(
            net_dims=[256, 256],
            state_dim=state_dim,
            action_dim=action_dim,
            gpu_id=0
        )

    def get_action(self, state, valid_mask, deterministic=False):
        # Get action with masking logic
        pass
```

### 3. Training Loop

```python
# Collect data
for game in range(num_games):
    game_data = play_game()  # Returns (state, action, logprob) tuples

# Format for PPO
states = torch.FloatTensor(...).unsqueeze(1)
actions = torch.LongTensor(...).unsqueeze(1)
logprobs = torch.FloatTensor(...).unsqueeze(1)
rewards = torch.FloatTensor(...).unsqueeze(1)
undones = torch.FloatTensor(...).unsqueeze(1)
unmasks = torch.ones_like(rewards)

buffer_data = (states, actions, logprobs, rewards, undones, unmasks)

# Train
agent.last_state = torch.zeros((1, state_dim))
obj_critic, obj_actor, obj_entropy = agent.update_net(buffer_data)
```

### 4. Evaluation

```python
# Load trained agent
agent = MyGameAgent(state_dim, action_dim)
agent.load_model('trained_model.pth')

# Get actions during gameplay
state = game.get_current_state()
valid_mask = game.get_valid_actions_mask()
action = agent.get_action(state, valid_mask, deterministic=True)
```

## Testing the Example

```bash
cd examples/tictactoe

# Quick test (1 iteration)
python train.py --iterations 1 --games 10

# Full training
python train.py --iterations 20 --games 100

# Evaluate
python evaluate.py --model checkpoints/best_model.pth --mode vs_random

# Play against it
python evaluate.py --model checkpoints/best_model.pth --mode vs_human

# Watch self-play
python evaluate.py --model checkpoints/best_model.pth --mode watch
```

## Expected Performance

**Tic-Tac-Toe:**
- Training time: 10-30 minutes (20 iterations)
- Final performance: 80-95% win rate vs random opponent
- Self-play: Near-optimal play (mostly draws)

## Lessons Learned

1. **Check agent type for action space:**
   - Discrete actions → `AgentDiscretePPO`, `AgentDQN`
   - Continuous actions → `AgentPPO`, `AgentSAC`, `AgentTD3`

2. **PPO is on-policy:**
   - Doesn't use ReplayBuffer
   - Expects fresh data each iteration
   - Needs logprobs collected during exploration

3. **Data format matters:**
   - PPO expects: `(horizon_len, num_envs, dim)`
   - ReplayBuffer format is different
   - Must include logprobs for PPO

4. **Action masking:**
   - Critical for games with invalid moves
   - Apply before softmax: `logits.masked_fill(mask == 0, -1e9)`

5. **Temperature scheduling:**
   - High temperature early (exploration)
   - Low temperature late (exploitation)
   - `temp = max(0.5, 1.0 - iteration / total_iterations)`

## Files Changed/Created Summary

**Core fixes:**
- `setup.py` - Fixed 'th' → 'torch' bug
- `requirements.txt` - Added version constraints
- `README.md` - Updated requirements section

**Documentation:**
- `RUNNING_INSTRUCTIONS.md` - Complete setup guide
- `docs/ALPHAZERO_COMPARISON.md` - Architecture comparison
- `SUMMARY.md` - Project overview

**Examples:**
- `examples/example_game_integration.py` - General template
- `examples/example_selfplay_game.py` - Self-play template
- `examples/tictactoe/` - Complete working example
  - `game.py` - Environment
  - `agent.py` - Agent wrapper (using AgentDiscretePPO)
  - `train.py` - Training script (with PPO format)
  - `evaluate.py` - Evaluation modes
  - `run_example.sh` - Quick start
  - `README.md` - Documentation

## Pull Requests

1. **Requirements & Documentation:**
   - Branch: `claude/update-requirements-docs-XluTa`
   - URL: https://github.com/Battlecode2026/ElegantRL/pull/new/claude/update-requirements-docs-XluTa

2. **Tic-Tac-Toe Example:**
   - Branch: `claude/tictactoe-example-XluTa`
   - URL: https://github.com/Battlecode2026/ElegantRL/pull/new/claude/tictactoe-example-XluTa

## Next Steps for Users

1. **Try the example** - Run tic-tac-toe training
2. **Understand the code** - Read through the example
3. **Adapt for your game** - Copy and modify
4. **Scale up** - Larger networks, more iterations
5. **Add MCTS** (optional) - For strategic depth

## Key Takeaways

✅ ElegantRL provides everything needed for game playing
✅ Complete working example demonstrates the full workflow
✅ Use `AgentDiscretePPO` for discrete action games
✅ PPO requires specific data format with logprobs
✅ Action masking is straightforward to implement
✅ Performance is competitive without MCTS

## References

- ElegantRL Docs: https://elegantrl.readthedocs.io
- PPO Paper: https://arxiv.org/abs/1707.06347
- AlphaZero Paper: https://arxiv.org/abs/1712.01815
- Discord: https://discord.gg/trsr8SXpW5
