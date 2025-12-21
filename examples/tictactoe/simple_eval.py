"""
Simplified evaluation script using the Evaluator interface

This demonstrates how to use the Evaluator class for easy inference.
"""

import argparse
import numpy as np
from game import TicTacToe, RandomPlayer, HumanPlayer
from evaluator import Evaluator


def evaluate_vs_random(evaluator: Evaluator, num_games: int = 100):
    """
    Evaluate against random opponent using Evaluator

    Args:
        evaluator: Evaluator instance
        num_games: Number of games to play

    Returns:
        stats: Win/loss/draw statistics
    """
    random_player = RandomPlayer()
    stats = {'wins': 0, 'losses': 0, 'draws': 0}

    for game_idx in range(num_games):
        game = TicTacToe()
        state = game.reset()
        done = False

        # Alternate which player the evaluator is
        evaluator_player = game_idx % 2

        while not done:
            current_player = game.current_player
            valid_mask = game.get_valid_actions_mask()

            if current_player == evaluator_player:
                # Evaluator's turn
                action = evaluator.get_action(state, valid_mask, deterministic=True)
            else:
                # Random player's turn
                action = random_player.get_action(state, valid_mask)

            next_state, reward, done, info = game.step(action)
            state = next_state

        # Update stats
        winner = info['winner']
        if winner == -1:
            stats['draws'] += 1
        elif winner == evaluator_player:
            stats['wins'] += 1
        else:
            stats['losses'] += 1

        if (game_idx + 1) % 10 == 0:
            print(f"Games: {game_idx+1}/{num_games} | "
                  f"Wins: {stats['wins']}, Losses: {stats['losses']}, Draws: {stats['draws']}")

    return stats


def play_vs_human(evaluator: Evaluator, evaluator_player: int = 0):
    """
    Play interactive game against human using Evaluator

    Args:
        evaluator: Evaluator instance
        evaluator_player: Which player is the evaluator (0 or 1)
    """
    game = TicTacToe()
    human = HumanPlayer()
    state = game.reset()
    done = False

    print(f"\n{'='*60}")
    print(f"Tic-Tac-Toe: Human vs Evaluator")
    print(f"{'='*60}")
    print(f"You are Player {1 - evaluator_player} ({'X' if 1 - evaluator_player == 0 else 'O'})")
    print(f"Evaluator is Player {evaluator_player} ({'X' if evaluator_player == 0 else 'O'})")
    print(f"\nBoard positions:")
    print("  0 1 2")
    print("  3 4 5")
    print("  6 7 8")
    print(f"{'='*60}\n")

    game.render()

    while not done:
        current_player = game.current_player
        valid_mask = game.get_valid_actions_mask()

        if current_player == evaluator_player:
            # Evaluator's turn
            print("Evaluator is thinking...")
            action = evaluator.get_action(state, valid_mask, deterministic=True)

            # Show evaluator's reasoning
            probs = evaluator.get_action_probs(state, valid_mask)
            value = evaluator.get_value(state)
            print(f"Evaluator chooses position {action}")
            print(f"Position probabilities: {np.round(probs, 3)}")
            print(f"State value estimate: {value:.3f}")
        else:
            # Human's turn
            print("Your turn!")
            action = human.get_action(state, valid_mask)

        next_state, reward, done, info = game.step(action)
        state = next_state

        game.render()

    # Game over
    winner = info['winner']
    print(f"\n{'='*60}")
    if winner == -1:
        print("GAME OVER: DRAW!")
    elif winner == evaluator_player:
        print("GAME OVER: EVALUATOR WINS!")
    else:
        print("GAME OVER: YOU WIN!")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Simple evaluation with Evaluator')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--mode', type=str, default='vs_random',
                       choices=['vs_random', 'vs_human'],
                       help='Evaluation mode')
    parser.add_argument('--num-games', type=int, default=100,
                       help='Number of games (for vs_random mode)')
    parser.add_argument('--evaluator-player', type=int, default=0,
                       choices=[0, 1],
                       help='Which player is the evaluator (for vs_human mode)')
    parser.add_argument('--gpu', type=int, default=-1,
                       help='GPU ID (-1 for CPU)')
    args = parser.parse_args()

    # Initialize evaluator
    print(f"Initializing evaluator...")
    evaluator = Evaluator(
        model_path=args.model,
        gpu_id=args.gpu
    )
    print()

    if args.mode == 'vs_random':
        print(f"{'='*60}")
        print(f"Evaluating vs Random ({args.num_games} games)")
        print(f"{'='*60}\n")

        stats = evaluate_vs_random(evaluator, args.num_games)

        print(f"\n{'='*60}")
        print(f"Results")
        print(f"{'='*60}")
        print(f"Wins:   {stats['wins']:3d} ({stats['wins']/args.num_games:.1%})")
        print(f"Losses: {stats['losses']:3d} ({stats['losses']/args.num_games:.1%})")
        print(f"Draws:  {stats['draws']:3d} ({stats['draws']/args.num_games:.1%})")
        print(f"{'='*60}\n")

    elif args.mode == 'vs_human':
        play_vs_human(evaluator, args.evaluator_player)


if __name__ == '__main__':
    main()
