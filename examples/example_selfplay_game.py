"""
Example: Self-Play Training for Games (AlphaZero-like workflow)

Note: This implements the DATA COLLECTION and TRAINING parts of AlphaZero,
but does NOT include MCTS search. For full AlphaZero:
- Add MCTS search during action selection
- Use neural network to guide MCTS (policy + value heads)
- This example uses standard RL agents (PPO/SAC) instead
"""

import torch
import numpy as np
from elegantrl.agents import AgentPPO
from elegantrl.train.replay_buffer import ReplayBuffer
from typing import List, Tuple


class SelfPlayAgent:
    """Agent for self-play games (e.g., Chess, Go, custom board games)"""

    def __init__(self, state_dim, action_dim, model_path=None):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Initialize agent (can use PPO, SAC, etc.)
        self.agent = AgentPPO(
            net_dims=[256, 256, 256],  # Deeper network for complex games
            state_dim=state_dim,
            action_dim=action_dim,
            gpu_id=0
        )

        if model_path:
            self.load_model(model_path)

    def get_action(self, state, valid_actions_mask=None, temperature=1.0):
        """
        Get action with optional masking for invalid moves

        Args:
            state: Current game state
            valid_actions_mask: Binary mask for valid actions (1=valid, 0=invalid)
            temperature: Exploration temperature (higher = more random)

        Returns:
            action_idx: Selected action index
            action_probs: Action probabilities (for training)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)

        with torch.no_grad():
            # Get action distribution
            if hasattr(self.agent.act, 'get_action_logprob'):
                # PPO-style actor
                action_mean = self.agent.act.net(state_tensor)

                # Apply temperature
                action_logits = action_mean / temperature

                # Mask invalid actions
                if valid_actions_mask is not None:
                    mask_tensor = torch.FloatTensor(valid_actions_mask).to(self.agent.device)
                    action_logits = action_logits.masked_fill(mask_tensor == 0, -1e9)

                # Sample from distribution
                action_probs = torch.softmax(action_logits, dim=-1)
                action_idx = torch.multinomial(action_probs, 1).item()

                return action_idx, action_probs.cpu().numpy()[0]
            else:
                # Deterministic actor (DDPG/TD3/SAC)
                action = self.agent.act.forward(state_tensor)
                return action.cpu().numpy()[0], None

    def get_value(self, state):
        """Get state value estimate"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)

        with torch.no_grad():
            value = self.agent.cri(state_tensor)

        return value.cpu().numpy()[0]

    def save_model(self, path):
        torch.save({
            'actor': self.agent.act.state_dict(),
            'critic': self.agent.cri.state_dict()
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.agent.device)
        self.agent.act.load_state_dict(checkpoint['actor'])
        self.agent.cri.load_state_dict(checkpoint['critic'])


class TwoPlayerGame:
    """
    Base class for two-player games
    Implement this interface for your custom game
    """

    def reset(self):
        """Reset game, return initial state"""
        raise NotImplementedError

    def step(self, action):
        """
        Execute action, return (next_state, reward, done, info)

        Reward convention for self-play:
        - Positive: current player winning
        - Negative: current player losing
        - Zero: neutral/ongoing
        """
        raise NotImplementedError

    def get_valid_actions(self):
        """Return mask of valid actions (1=valid, 0=invalid)"""
        raise NotImplementedError

    def get_current_player(self):
        """Return current player (0 or 1)"""
        raise NotImplementedError

    def clone(self):
        """Return deep copy of game state"""
        raise NotImplementedError


class SelfPlayDataCollector:
    """Collect data from self-play games"""

    def __init__(self, state_dim, action_dim, buffer_size=1_000_000):
        self.buffer = ReplayBuffer(
            max_size=buffer_size,
            state_dim=state_dim,
            action_dim=action_dim,
            gpu_id=0
        )

    def play_game(self, game: TwoPlayerGame, agent1: SelfPlayAgent,
                  agent2: SelfPlayAgent = None, temperature=1.0):
        """
        Play one self-play game

        Args:
            game: Game environment
            agent1: First agent
            agent2: Second agent (if None, agent1 plays against itself)
            temperature: Exploration temperature

        Returns:
            winner: 0, 1, or -1 (draw)
            game_data: List of (state, action, player) tuples
        """
        if agent2 is None:
            agent2 = agent1  # Self-play

        state = game.reset()
        game_data = []
        done = False

        while not done:
            current_player = game.get_current_player()
            agent = agent1 if current_player == 0 else agent2

            # Get valid actions
            valid_mask = game.get_valid_actions()

            # Select action
            action, probs = agent.get_action(state, valid_mask, temperature)

            # Store (state, action, player) for later processing
            game_data.append((state.copy(), action, current_player))

            # Execute action
            next_state, reward, done, info = game.step(action)
            state = next_state

        # Determine winner from info
        winner = info.get('winner', -1)  # -1 = draw

        return winner, game_data

    def collect_selfplay_games(self, game: TwoPlayerGame, agent: SelfPlayAgent,
                                num_games=100, temperature=1.0):
        """
        Collect data from multiple self-play games

        Args:
            game: Game environment
            agent: Agent to play against itself
            num_games: Number of games to play
            temperature: Exploration temperature

        Returns:
            stats: Dict with win/loss/draw counts
        """
        stats = {'wins_p0': 0, 'wins_p1': 0, 'draws': 0}
        all_transitions = []

        for game_idx in range(num_games):
            winner, game_data = self.play_game(game, agent, None, temperature)

            # Update stats
            if winner == 0:
                stats['wins_p0'] += 1
            elif winner == 1:
                stats['wins_p1'] += 1
            else:
                stats['draws'] += 1

            # Process game data into transitions
            game_length = len(game_data)

            for step_idx, (state, action, player) in enumerate(game_data):
                # Assign reward based on game outcome
                if winner == -1:  # Draw
                    reward = 0.0
                elif winner == player:  # Win
                    reward = 1.0
                else:  # Loss
                    reward = -1.0

                # Optional: Discount based on how far from end
                # steps_from_end = game_length - step_idx
                # reward = reward * (0.99 ** steps_from_end)

                # Get next state
                if step_idx < game_length - 1:
                    next_state = game_data[step_idx + 1][0]
                    done = 0.0
                else:
                    next_state = state  # Terminal state
                    done = 1.0

                all_transitions.append((state, action, reward, next_state, done))

            if (game_idx + 1) % 10 == 0:
                print(f"Game {game_idx+1}/{num_games}: "
                      f"P0 Wins={stats['wins_p0']}, P1 Wins={stats['wins_p1']}, "
                      f"Draws={stats['draws']}, Transitions={len(all_transitions)}")

        # Add transitions to buffer
        self._add_transitions_to_buffer(all_transitions)

        return stats

    def _add_transitions_to_buffer(self, transitions: List[Tuple]):
        """Convert transitions to buffer format"""
        if not transitions:
            return

        states = []
        actions = []
        rewards = []
        undones = []

        for state, action, reward, next_state, done in transitions:
            states.append(state)
            # Convert action to one-hot or continuous format as needed
            actions.append(action if isinstance(action, (list, np.ndarray)) else [action])
            rewards.append(reward)
            undones.append(1.0 - done)

        # Convert to tensors
        states_t = torch.FloatTensor(states).unsqueeze(1)  # (T, 1, state_dim)
        actions_t = torch.FloatTensor(actions).unsqueeze(1)  # (T, 1, action_dim)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1)  # (T, 1)
        undones_t = torch.FloatTensor(undones).unsqueeze(1)  # (T, 1)

        buffer_items = (states_t, actions_t, rewards_t, undones_t)
        self.buffer.update(buffer_items)

    def save_data(self, save_dir='./selfplay_data'):
        """Save collected data"""
        self.buffer.save_or_load_history(cwd=save_dir, if_save=True)
        print(f"Saved {self.buffer.cur_size} transitions to {save_dir}")

    def load_data(self, save_dir='./selfplay_data'):
        """Load collected data"""
        self.buffer.save_or_load_history(cwd=save_dir, if_save=False)
        print(f"Loaded {self.buffer.cur_size} transitions from {save_dir}")


class SelfPlayTrainer:
    """Iterative self-play training loop (AlphaZero-style)"""

    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

    def train_iteration(self, game: TwoPlayerGame, agent: SelfPlayAgent,
                        num_selfplay_games=100, num_training_updates=1000,
                        save_dir='./iteration_data'):
        """
        One iteration of AlphaZero-style training

        1. Generate self-play games
        2. Collect training data
        3. Train agent on collected data
        4. Evaluate agent

        Args:
            game: Game environment
            agent: Current agent
            num_selfplay_games: Number of self-play games per iteration
            num_training_updates: Number of gradient updates per iteration
            save_dir: Directory to save data

        Returns:
            agent: Updated agent
        """
        print("\n=== Generating Self-Play Data ===")

        # Collect self-play data
        collector = SelfPlayDataCollector(self.state_dim, self.action_dim)
        stats = collector.collect_selfplay_games(
            game=game,
            agent=agent,
            num_games=num_selfplay_games,
            temperature=1.0  # Higher temperature = more exploration
        )

        print(f"Self-play stats: {stats}")

        # Save data (optional)
        collector.save_data(save_dir)

        print("\n=== Training on Self-Play Data ===")

        # Train agent
        for update in range(num_training_updates):
            obj_critic, obj_actor = agent.agent.update_net(collector.buffer)

            if (update + 1) % 100 == 0:
                print(f"Update {update+1}/{num_training_updates}: "
                      f"Critic={obj_critic:.4f}, Actor={obj_actor:.4f}")

        return agent


# ========================================
# Example: Dummy Tic-Tac-Toe Game
# ========================================

class TicTacToe(TwoPlayerGame):
    """Simple Tic-Tac-Toe implementation"""

    def __init__(self):
        self.board = np.zeros(9, dtype=np.float32)  # 0=empty, 1=player0, -1=player1
        self.current_player = 0

    def reset(self):
        self.board = np.zeros(9, dtype=np.float32)
        self.current_player = 0
        return self.board.copy()

    def step(self, action):
        """Action is index 0-8"""
        # Place mark
        self.board[action] = 1.0 if self.current_player == 0 else -1.0

        # Check win
        winner = self._check_winner()
        done = winner is not None or np.all(self.board != 0)

        # Reward from current player's perspective
        if winner == self.current_player:
            reward = 1.0
        elif winner is not None:
            reward = -1.0
        else:
            reward = 0.0

        # Switch player
        self.current_player = 1 - self.current_player

        info = {'winner': winner if winner is not None else -1}

        return self.board.copy(), reward, done, info

    def get_valid_actions(self):
        """Return mask of valid actions"""
        return (self.board == 0).astype(np.float32)

    def get_current_player(self):
        return self.current_player

    def clone(self):
        new_game = TicTacToe()
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        return new_game

    def _check_winner(self):
        """Check if there's a winner"""
        # Check rows, columns, diagonals
        lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]  # Diagonals
        ]

        for line in lines:
            if np.abs(np.sum(self.board[line])) == 3:
                return 0 if self.board[line[0]] == 1 else 1

        return None


# ========================================
# Main Training Loop
# ========================================

if __name__ == "__main__":
    # Game configuration
    STATE_DIM = 9  # Tic-Tac-Toe board
    ACTION_DIM = 9  # 9 possible moves

    game = TicTacToe()
    agent = SelfPlayAgent(STATE_DIM, ACTION_DIM)
    trainer = SelfPlayTrainer(STATE_DIM, ACTION_DIM)

    # Training loop
    num_iterations = 5

    for iteration in range(num_iterations):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration + 1}/{num_iterations}")
        print(f"{'='*60}")

        # Train one iteration
        agent = trainer.train_iteration(
            game=game,
            agent=agent,
            num_selfplay_games=50,
            num_training_updates=500,
            save_dir=f'./tictactoe_iter_{iteration}'
        )

        # Save checkpoint
        agent.save_model(f'./tictactoe_agent_iter_{iteration}.pth')

    print("\n=== Training Complete! ===")
