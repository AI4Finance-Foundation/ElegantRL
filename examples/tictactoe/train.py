"""
Training script for Tic-Tac-Toe agent

Trains agent using self-play with ElegantRL
"""

import argparse
import os
import numpy as np
import torch
from typing import Tuple, List
from elegantrl.train.replay_buffer import ReplayBuffer

from game import TicTacToe, RandomPlayer
from agent import TicTacToeAgent


class SelfPlayTrainer:
    """Self-play training for Tic-Tac-Toe"""

    def __init__(self, save_dir='./checkpoints', gpu_id=0):
        self.save_dir = save_dir
        self.gpu_id = gpu_id
        os.makedirs(save_dir, exist_ok=True)

        # Initialize agent and buffer
        self.agent = TicTacToeAgent(gpu_id=gpu_id)
        self.buffer = ReplayBuffer(
            max_size=100_000,
            state_dim=9,
            action_dim=1,  # Discrete action (just index)
            gpu_id=gpu_id
        )

    def play_selfplay_game(self, temperature=1.0) -> Tuple[list, int]:
        """
        Play one self-play game

        Returns:
            game_data: List of (state, action, player) tuples
            winner: 0, 1, or -1 (draw)
        """
        game = TicTacToe()
        state = game.reset()
        game_data = []
        done = False

        while not done:
            current_player = game.current_player
            valid_mask = game.get_valid_actions_mask()

            # Get action from agent
            action = self.agent.get_action(state, valid_mask, temperature=temperature)

            # Store (state, action, player)
            game_data.append((state.copy(), action, current_player))

            # Execute action
            next_state, reward, done, info = game.step(action)
            state = next_state

        winner = info['winner']
        return game_data, winner

    def play_against_random(self, agent_player=0) -> Tuple[list, int]:
        """
        Play agent against random player

        Args:
            agent_player: Which player is the agent (0 or 1)

        Returns:
            game_data: List of (state, action) tuples for agent moves only
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
                # Agent's turn
                action = self.agent.get_action(state, valid_mask, temperature=1.0)
                game_data.append((state.copy(), action, current_player))
            else:
                # Random player's turn
                action = random_player.get_action(state, valid_mask)

            # Execute action
            next_state, reward, done, info = game.step(action)
            state = next_state

        winner = info['winner']
        return game_data, winner

    def collect_data(self, num_games=100, opponent='self', temperature=1.0):
        """
        Collect training data from games

        Args:
            num_games: Number of games to play
            opponent: 'self' for self-play, 'random' for random opponent
            temperature: Exploration temperature
        """
        all_transitions = []
        stats = {'wins': 0, 'losses': 0, 'draws': 0}

        for game_idx in range(num_games):
            # Play game
            if opponent == 'self':
                game_data, winner = self.play_selfplay_game(temperature)
            else:
                # Alternate which player the agent is
                agent_player = game_idx % 2
                game_data, winner = self.play_against_random(agent_player)

            # Update stats (for agent as player 0 in self-play)
            if winner == -1:
                stats['draws'] += 1
            elif opponent == 'self':
                if winner == 0:
                    stats['wins'] += 1
                else:
                    stats['losses'] += 1
            else:
                # Agent vs random
                agent_player = game_idx % 2
                if winner == agent_player:
                    stats['wins'] += 1
                elif winner != -1:
                    stats['losses'] += 1

            # Convert game data to transitions
            for state, action, player in game_data:
                # Assign reward based on game outcome
                if winner == -1:  # Draw
                    reward = 0.0
                elif winner == player:  # Win
                    reward = 1.0
                else:  # Loss
                    reward = -1.0

                all_transitions.append((state, action, reward))

        # Add to buffer
        if all_transitions:
            # Convert to numpy first for better performance
            states_np = np.array([t[0] for t in all_transitions])
            actions_np = np.array([[t[1]] for t in all_transitions])
            rewards_np = np.array([t[2] for t in all_transitions])

            states = torch.FloatTensor(states_np).unsqueeze(1)
            actions = torch.FloatTensor(actions_np).unsqueeze(1)
            rewards = torch.FloatTensor(rewards_np).unsqueeze(1)
            undones = torch.ones_like(rewards)  # All non-terminal for simplicity
            unmasks = torch.ones_like(rewards)  # All valid (not truncated)

            self.buffer.update((states, actions, rewards, undones, unmasks))

        return stats

    def train(self, num_iterations=10, games_per_iteration=100,
              training_updates=500, opponent='self', eval_interval=2):
        """
        Main training loop

        Args:
            num_iterations: Number of training iterations
            games_per_iteration: Number of games to play per iteration
            training_updates: Number of gradient updates per iteration
            opponent: 'self' or 'random'
            eval_interval: Evaluate every N iterations
        """
        print(f"\n{'='*60}")
        print(f"Starting Tic-Tac-Toe Training")
        print(f"{'='*60}")
        print(f"Iterations: {num_iterations}")
        print(f"Games per iteration: {games_per_iteration}")
        print(f"Training updates: {training_updates}")
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
            stats = self.collect_data(games_per_iteration, opponent, temperature)

            print(f"Game stats: Wins={stats['wins']}, Losses={stats['losses']}, "
                  f"Draws={stats['draws']}, Win Rate={stats['wins']/games_per_iteration:.2%}")
            print(f"Buffer size: {self.buffer.cur_size}")

            # Train
            if self.buffer.cur_size > 1000:
                print(f"\nTraining for {training_updates} updates...")
                for update in range(training_updates):
                    obj_critic, obj_actor = self.agent.get_agent().update_net(self.buffer)

                    if (update + 1) % 100 == 0:
                        print(f"  Update {update+1}/{training_updates}: "
                              f"Critic={obj_critic:.4f}, Actor={obj_actor:.4f}")

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
    parser.add_argument('--updates', type=int, default=500, help='Training updates per iteration')
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
        training_updates=args.updates,
        opponent=args.opponent
    )


if __name__ == '__main__':
    main()
