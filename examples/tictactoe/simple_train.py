"""
Simplified training script using the user-friendly DataSaver interface

This demonstrates how to:
1. Collect data from gameplay using DataSaver
2. Load saved data
3. Train a model
"""

import argparse
import numpy as np
import torch
from pathlib import Path

from game import TicTacToe, RandomPlayer
from agent import TicTacToeAgent
from data_saver import DataSaver


def collect_selfplay_data(agent: TicTacToeAgent, saver: DataSaver,
                          num_games: int = 100, temperature: float = 1.0):
    """
    Collect self-play data using DataSaver

    Args:
        agent: Agent to play with
        saver: DataSaver instance
        num_games: Number of games to play
        temperature: Exploration temperature
    """
    device = agent.device
    stats = {'wins_p0': 0, 'wins_p1': 0, 'draws': 0}

    for game_idx in range(num_games):
        game = TicTacToe()
        state = game.reset()

        # Start new episode
        episode_id = saver.new_episode()

        done = False
        while not done:
            current_player = game.current_player
            valid_mask = game.get_valid_actions_mask()

            # Get action and logprob
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action_logits = agent.agent.act.net(state_tensor)
            action_logits = action_logits / temperature

            # Mask invalid actions
            mask_tensor = torch.FloatTensor(valid_mask).unsqueeze(0).to(device)
            action_logits = action_logits.masked_fill(mask_tensor == 0, -1e9)

            # Sample action
            action_probs = torch.softmax(action_logits, dim=-1)
            action = torch.multinomial(action_probs, 1).item()
            logprob = torch.log(action_probs[0, action] + 1e-8).item()

            # Add transition to DataSaver
            saver.add_transition(state, action, logprob, player=current_player)

            # Execute action
            next_state, reward, done, info = game.step(action)
            state = next_state

        # Game finished - set rewards
        winner = info['winner']

        if winner == -1:
            stats['draws'] += 1
            # Draw - both players get 0 reward
            saver.set_reward(0.0, gamma=0.0, player=0)
            saver.set_reward(0.0, gamma=0.0, player=1)
        else:
            # Winner gets +1, loser gets -1
            if winner == 0:
                stats['wins_p0'] += 1
            else:
                stats['wins_p1'] += 1

            saver.set_reward(1.0, gamma=0.0, player=winner)
            saver.set_reward(-1.0, gamma=0.0, player=1 - winner)

        if (game_idx + 1) % 10 == 0:
            print(f"Collected {game_idx + 1}/{num_games} games | "
                  f"P0: {stats['wins_p0']}, P1: {stats['wins_p1']}, Draws: {stats['draws']}")

    return stats


def train_from_data(data_dir: str, agent: TicTacToeAgent,
                    num_updates: int = 500):
    """
    Load data and train agent

    Args:
        data_dir: Directory containing saved episodes
        agent: Agent to train
        num_updates: Number of training updates
    """
    # Load all episodes
    print(f"\nLoading episodes from {data_dir}...")
    episodes = DataSaver.load_episodes(data_dir)
    print(f"Loaded {len(episodes)} episodes")

    if len(episodes) == 0:
        print("No data to train on!")
        return

    # Convert to tensors
    print("Converting to tensors...")
    buffer_data = DataSaver.convert_to_tensors(episodes, device=agent.device)

    print(f"Training data: {buffer_data[0].shape[0]} transitions")

    # Set last_state for PPO
    agent.agent.last_state = torch.zeros((1, 9), device=agent.device)

    # Train
    print(f"Training for {num_updates} updates...")
    obj_critic, obj_actor, obj_entropy = agent.agent.update_net(buffer_data)

    print(f"Training complete!")
    print(f"  Critic Loss: {obj_critic:.4f}")
    print(f"  Actor Loss: {obj_actor:.4f}")
    print(f"  Entropy: {obj_entropy:.4f}")

    return obj_critic, obj_actor, obj_entropy


def main():
    parser = argparse.ArgumentParser(description='Simple training with DataSaver')
    parser.add_argument('--mode', type=str, default='collect',
                       choices=['collect', 'train', 'both'],
                       help='Mode: collect data, train from data, or both')
    parser.add_argument('--games', type=int, default=100,
                       help='Number of games to collect')
    parser.add_argument('--data-dir', type=str, default='./training_data',
                       help='Directory for training data')
    parser.add_argument('--save-frequency', type=int, default=10,
                       help='Save data every N episodes')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to load existing model')
    parser.add_argument('--save-model', type=str, default='./simple_model.pth',
                       help='Path to save trained model')
    parser.add_argument('--iterations', type=int, default=5,
                       help='Number of collect-train iterations (for both mode)')
    parser.add_argument('--gpu', type=int, default=-1,
                       help='GPU ID (-1 for CPU)')
    args = parser.parse_args()

    # Initialize agent
    print("Initializing agent...")
    agent = TicTacToeAgent(
        gpu_id=args.gpu,
        model_path=args.model_path if args.model_path else None
    )

    if args.mode == 'collect':
        # Just collect data
        print(f"\n{'='*60}")
        print("MODE: Data Collection")
        print(f"{'='*60}\n")

        saver = DataSaver(
            save_dir=args.data_dir,
            save_frequency=args.save_frequency
        )

        stats = collect_selfplay_data(agent, saver, num_games=args.games)
        saver.flush()

        print(f"\nCollection complete!")
        print(f"Stats: {stats}")
        print(f"Data saved to: {args.data_dir}")

    elif args.mode == 'train':
        # Just train from existing data
        print(f"\n{'='*60}")
        print("MODE: Training from Saved Data")
        print(f"{'='*60}\n")

        train_from_data(args.data_dir, agent)

        # Save model
        agent.save_model(args.save_model)
        print(f"\nModel saved to: {args.save_model}")

    elif args.mode == 'both':
        # Iterative collect + train
        print(f"\n{'='*60}")
        print("MODE: Iterative Collect + Train")
        print(f"{'='*60}\n")

        for iteration in range(args.iterations):
            print(f"\n{'='*60}")
            print(f"Iteration {iteration + 1}/{args.iterations}")
            print(f"{'='*60}\n")

            # Collect data
            print("Collecting data...")
            saver = DataSaver(
                save_dir=args.data_dir,
                save_frequency=args.save_frequency
            )

            temperature = max(0.5, 1.0 - iteration / args.iterations)
            stats = collect_selfplay_data(agent, saver, num_games=args.games,
                                        temperature=temperature)
            saver.flush()

            print(f"Collection stats: {stats}")

            # Train
            print("\nTraining...")
            train_from_data(args.data_dir, agent)

            # Save checkpoint
            checkpoint_path = f"{args.save_model}.iter_{iteration+1}"
            agent.save_model(checkpoint_path)
            print(f"Checkpoint saved to: {checkpoint_path}")

        # Save final model
        agent.save_model(args.save_model)
        print(f"\nFinal model saved to: {args.save_model}")


if __name__ == '__main__':
    main()
