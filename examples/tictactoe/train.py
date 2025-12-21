"""
Training script for Tic-Tac-Toe agent

Trains agent using self-play with ElegantRL
FIXED VERSION: Works with PPO's on-policy requirements
"""

import argparse
import os
import numpy as np
import torch
from typing import Tuple, List

from game import TicTacToe, RandomPlayer
from agent import TicTacToeAgent


class SelfPlayTrainer:
    """Self-play training for Tic-Tac-Toe"""

    def __init__(self, save_dir='./checkpoints', gpu_id=0):
        self.save_dir = save_dir
        self.gpu_id = gpu_id
        self.device = torch.device(f"cuda:{gpu_id}" if gpu_id >= 0 and torch.cuda.is_available() else "cpu")
        os.makedirs(save_dir, exist_ok=True)

        # Initialize agent
        self.agent = TicTacToeAgent(gpu_id=gpu_id)

    def collect_trajectory_data(self, num_games=100, opponent='self', temperature=1.0):
        """
        Collect trajectory data for PPO training

        Returns:
            Tuple of (states, actions, logprobs, rewards, undones, unmasks) for PPO
        """
        all_states = []
        all_actions = []
        all_logprobs = []
        all_rewards = []
        all_undones = []
        all_unmasks = []

        stats = {'wins': 0, 'losses': 0, 'draws': 0}

        for game_idx in range(num_games):
            # Play one game
            if opponent == 'self':
                game_data, winner = self.play_selfplay_game(temperature)
            else:
                agent_player = game_idx % 2
                game_data, winner = self.play_against_random(agent_player, temperature)

            # Update stats
            if winner == -1:
                stats['draws'] += 1
            elif opponent == 'self':
                stats['wins'] += 1 if winner == 0 else 0
                stats['losses'] += 1 if winner == 1 else 0
            else:
                agent_player = game_idx % 2
                if winner == agent_player:
                    stats['wins'] += 1
                elif winner != -1:
                    stats['losses'] += 1
                else:
                    stats['draws'] += 1

            # Process game data
            for i, (state, action, logprob, player) in enumerate(game_data):
                # Assign reward based on game outcome
                if winner == -1:  # Draw
                    reward = 0.0
                elif winner == player:  # Win
                    reward = 1.0
                else:  # Loss
                    reward = -1.0

                # All positions are terminal (game is over) except intermediate steps
                is_done = (i == len(game_data) - 1)

                all_states.append(state)
                all_actions.append(action)
                all_logprobs.append(logprob)
                all_rewards.append(reward)
                all_undones.append(0.0 if is_done else 1.0)
                all_unmasks.append(1.0)  # No truncation

        # Convert to tensors in PPO format: (horizon_len, num_envs, dim)
        states = torch.FloatTensor(np.array(all_states)).unsqueeze(1).to(self.device)  # (T, 1, 9)
        actions = torch.LongTensor(all_actions).unsqueeze(1).to(self.device)  # (T, 1)
        logprobs = torch.FloatTensor(all_logprobs).unsqueeze(1).to(self.device)  # (T, 1)
        rewards = torch.FloatTensor(all_rewards).unsqueeze(1).to(self.device)  # (T, 1)
        undones = torch.FloatTensor(all_undones).unsqueeze(1).to(self.device)  # (T, 1)
        unmasks = torch.FloatTensor(all_unmasks).unsqueeze(1).to(self.device)  # (T, 1)

        return (states, actions, logprobs, rewards, undones, unmasks), stats

    def play_selfplay_game(self, temperature=1.0) -> Tuple[list, int]:
        """
        Play one self-play game, storing logprobs

        Returns:
            game_data: List of (state, action, logprob, player) tuples
            winner: 0, 1, or -1 (draw)
        """
        game = TicTacToe()
        state = game.reset()
        game_data = []
        done = False

        while not done:
            current_player = game.current_player
            valid_mask = game.get_valid_actions_mask()

            # Get action and logprob from agent
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, logprob = self._get_action_logprob(state_tensor, valid_mask, temperature)

            # Store (state, action, logprob, player)
            game_data.append((state.copy(), action, logprob, current_player))

            # Execute action
            next_state, reward, done, info = game.step(action)
            state = next_state

        winner = info['winner']
        return game_data, winner

    def play_against_random(self, agent_player=0, temperature=1.0) -> Tuple[list, int]:
        """
        Play agent against random player

        Args:
            agent_player: Which player is the agent (0 or 1)
            temperature: Exploration temperature

        Returns:
            game_data: List of (state, action, logprob, player) tuples for agent moves only
            winner: 0, 1, or -1 (draw)
        """
        game = TicTacToe()
        random_player = RandomPlayer()
        state = game.reset()
        game_data = []
        done = False

        while not done:
            current_player = game.current_player
            valid_mask = game.get_valid_actions_mask()

            if current_player == agent_player:
                # Agent's turn - get action with logprob
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action, logprob = self._get_action_logprob(state_tensor, valid_mask, temperature)
                game_data.append((state.copy(), action, logprob, current_player))
            else:
                # Random player's turn
                action = random_player.get_action(state, valid_mask)

            # Execute action
            next_state, reward, done, info = game.step(action)
            state = next_state

        winner = info['winner']
        return game_data, winner

    def _get_action_logprob(self, state_tensor, valid_mask, temperature=1.0):
        """Get action and its log probability"""
        with torch.no_grad():
            # Get action logits
            action_logits = self.agent.get_agent().act.net(state_tensor)
            action_logits = action_logits / temperature

            # Mask invalid actions
            mask_tensor = torch.FloatTensor(valid_mask).unsqueeze(0).to(self.device)
            action_logits = action_logits.masked_fill(mask_tensor == 0, -1e9)

            # Sample action
            action_probs = torch.softmax(action_logits, dim=-1)
            action = torch.multinomial(action_probs, 1).item()

            # Calculate log probability
            logprob = torch.log(action_probs[0, action] + 1e-8).item()

        return action, logprob

    def train(self, num_iterations=10, games_per_iteration=100,
              training_updates_per_iteration=None, opponent='self', eval_interval=2):
        """
        Main training loop

        Args:
            num_iterations: Number of training iterations
            games_per_iteration: Number of games to play per iteration
            training_updates_per_iteration: Number of PPO updates (None = auto based on data size)
            opponent: 'self' or 'random'
            eval_interval: Evaluate every N iterations
        """
        print(f"\n{'='*60}")
        print(f"Starting Tic-Tac-Toe Training (PPO)")
        print(f"{'='*60}")
        print(f"Iterations: {num_iterations}")
        print(f"Games per iteration: {games_per_iteration}")
        print(f"Opponent: {opponent}")
        print(f"{'='*60}\n")

        best_win_rate = 0.0

        for iteration in range(num_iterations):
            print(f"\n{'='*60}")
            print(f"Iteration {iteration + 1}/{num_iterations}")
            print(f"{'='*60}")

            # Collect data
            print(f"\nCollecting data from {games_per_iteration} games...")
            temperature = max(0.5, 1.0 - iteration / num_iterations)  # Decay temperature
            buffer_data, stats = self.collect_trajectory_data(games_per_iteration, opponent, temperature)

            print(f"Game stats: Wins={stats['wins']}, Losses={stats['losses']}, "
                  f"Draws={stats['draws']}, Win Rate={stats['wins']/games_per_iteration:.2%}")
            print(f"Collected {buffer_data[0].shape[0]} transitions")

            # Train with PPO
            print(f"\nTraining with PPO...")
            # Set last_state for advantage calculation (use a zero state as placeholder)
            self.agent.get_agent().last_state = torch.zeros((1, 9), device=self.device)

            obj_critic, obj_actor, obj_entropy = self.agent.get_agent().update_net(buffer_data)

            print(f"  Critic Loss={obj_critic:.4f}, Actor Loss={obj_actor:.4f}, Entropy={obj_entropy:.4f}")

            # Evaluate against random opponent
            if (iteration + 1) % eval_interval == 0:
                print(f"\nEvaluating against random opponent...")
                eval_stats = self.evaluate_vs_random(num_games=100)

                win_rate = eval_stats['wins'] / 100
                print(f"Evaluation: Wins={eval_stats['wins']}, Losses={eval_stats['losses']}, "
                      f"Draws={eval_stats['draws']}, Win Rate={win_rate:.2%}")

                # Save best model
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    save_path = os.path.join(self.save_dir, f'best_model.pth')
                    self.agent.save_model(save_path)
                    print(f"New best model saved! Win rate: {win_rate:.2%}")

            # Save checkpoint
            if (iteration + 1) % 5 == 0:
                save_path = os.path.join(self.save_dir, f'checkpoint_iter_{iteration+1}.pth')
                self.agent.save_model(save_path)

        # Save final model
        final_path = os.path.join(self.save_dir, 'final_model.pth')
        self.agent.save_model(final_path)

        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best win rate: {best_win_rate:.2%}")
        print(f"{'='*60}\n")

    def evaluate_vs_random(self, num_games=100):
        """Evaluate trained agent against random opponent"""
        random_player = RandomPlayer()
        stats = {'wins': 0, 'losses': 0, 'draws': 0}

        for game_idx in range(num_games):
            game = TicTacToe()
            state = game.reset()
            done = False

            # Agent plays as player 0 half the time, player 1 half the time
            agent_player = game_idx % 2

            while not done:
                current_player = game.current_player
                valid_mask = game.get_valid_actions_mask()

                if current_player == agent_player:
                    # Agent's turn (deterministic)
                    action = self.agent.get_action(state, valid_mask, deterministic=True)
                else:
                    # Random player's turn
                    action = random_player.get_action(state, valid_mask)

                next_state, reward, done, info = game.step(action)
                state = next_state

            # Update stats
            winner = info['winner']
            if winner == -1:
                stats['draws'] += 1
            elif winner == agent_player:
                stats['wins'] += 1
            else:
                stats['losses'] += 1

        return stats


def main():
    parser = argparse.ArgumentParser(description='Train Tic-Tac-Toe agent')
    parser.add_argument('--iterations', type=int, default=20, help='Number of training iterations')
    parser.add_argument('--games', type=int, default=100, help='Games per iteration')
    parser.add_argument('--opponent', type=str, default='self', choices=['self', 'random'],
                        help='Opponent type: self-play or random')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID (-1 for CPU)')
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    args = parser.parse_args()

    # Create trainer
    trainer = SelfPlayTrainer(save_dir=args.save_dir, gpu_id=args.gpu)

    # Train
    trainer.train(
        num_iterations=args.iterations,
        games_per_iteration=args.games,
        opponent=args.opponent
    )


if __name__ == '__main__':
    main()
