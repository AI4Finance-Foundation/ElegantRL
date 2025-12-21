"""
Evaluation script for Tic-Tac-Toe agent

Modes:
1. vs_random: Play against random opponent
2. vs_human: Play against human player
3. watch: Watch agent play against itself
"""

import argparse
import numpy as np
from game import TicTacToe, RandomPlayer, HumanPlayer
from agent import TicTacToeAgent


def evaluate_vs_random(agent: TicTacToeAgent, num_games=100, verbose=False):
    """
    Evaluate agent against random opponent

    Args:
        agent: Trained agent
        num_games: Number of games to play
        verbose: Print detailed game information

    Returns:
        stats: Dictionary with win/loss/draw counts
    """
    random_player = RandomPlayer()
    stats = {'wins': 0, 'losses': 0, 'draws': 0}

    for game_idx in range(num_games):
        game = TicTacToe()
        state = game.reset()
        done = False

        # Alternate which player the agent is
        agent_player = game_idx % 2

        if verbose:
            print(f"\n{'='*40}")
            print(f"Game {game_idx + 1}/{num_games}")
            print(f"Agent is Player {agent_player} ({'X' if agent_player == 0 else 'O'})")
            print(f"{'='*40}")
            game.render()

        while not done:
            current_player = game.current_player
            valid_mask = game.get_valid_actions_mask()

            if current_player == agent_player:
                # Agent's turn (deterministic for evaluation)
                action = agent.get_action(state, valid_mask, deterministic=True)
                if verbose:
                    probs = agent.get_action_probs(state, valid_mask)
                    print(f"Agent (Player {current_player}) chooses position {action}")
                    print(f"Action probabilities: {probs}")
            else:
                # Random player's turn
                action = random_player.get_action(state, valid_mask)
                if verbose:
                    print(f"Random (Player {current_player}) chooses position {action}")

            next_state, reward, done, info = game.step(action)
            state = next_state

            if verbose:
                game.render()

        # Update stats
        winner = info['winner']
        if winner == -1:
            stats['draws'] += 1
            if verbose:
                print("Result: DRAW")
        elif winner == agent_player:
            stats['wins'] += 1
            if verbose:
                print(f"Result: AGENT WINS!")
        else:
            stats['losses'] += 1
            if verbose:
                print(f"Result: AGENT LOSES")

        # Print running stats every 10 games
        if not verbose and (game_idx + 1) % 10 == 0:
            games_played = game_idx + 1
            win_rate = stats['wins'] / games_played
            print(f"Games: {games_played:3d}/{num_games} | "
                  f"Wins: {stats['wins']:3d} | Losses: {stats['losses']:3d} | "
                  f"Draws: {stats['draws']:3d} | Win Rate: {win_rate:.1%}")

    return stats


def play_vs_human(agent: TicTacToeAgent, agent_player=0):
    """
    Play interactive game against human

    Args:
        agent: Trained agent
        agent_player: Which player is the agent (0 or 1)
    """
    game = TicTacToe()
    human = HumanPlayer()
    state = game.reset()
    done = False

    print(f"\n{'='*40}")
    print(f"Tic-Tac-Toe: Human vs Agent")
    print(f"{'='*40}")
    print(f"You are Player {1 - agent_player} ({'X' if 1 - agent_player == 0 else 'O'})")
    print(f"Agent is Player {agent_player} ({'X' if agent_player == 0 else 'O'})")
    print(f"\nBoard positions:")
    print("  0 1 2")
    print("  3 4 5")
    print("  6 7 8")
    print(f"{'='*40}\n")

    game.render()

    while not done:
        current_player = game.current_player
        valid_mask = game.get_valid_actions_mask()

        if current_player == agent_player:
            # Agent's turn
            print("Agent is thinking...")
            action = agent.get_action(state, valid_mask, deterministic=True)

            # Show agent's reasoning
            probs = agent.get_action_probs(state, valid_mask)
            value = agent.get_value(state)
            print(f"Agent chooses position {action}")
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
    print(f"\n{'='*40}")
    if winner == -1:
        print("GAME OVER: DRAW!")
    elif winner == agent_player:
        print("GAME OVER: AGENT WINS!")
    else:
        print("GAME OVER: YOU WIN!")
    print(f"{'='*40}\n")


def watch_selfplay(agent: TicTacToeAgent, num_games=5, delay=0.5):
    """
    Watch agent play against itself

    Args:
        agent: Trained agent
        num_games: Number of games to watch
        delay: Delay between moves (seconds)
    """
    import time

    stats = {'p0_wins': 0, 'p1_wins': 0, 'draws': 0}

    for game_idx in range(num_games):
        game = TicTacToe()
        state = game.reset()
        done = False

        print(f"\n{'='*40}")
        print(f"Self-Play Game {game_idx + 1}/{num_games}")
        print(f"{'='*40}\n")

        game.render()

        while not done:
            current_player = game.current_player
            valid_mask = game.get_valid_actions_mask()

            # Agent plays both sides
            action = agent.get_action(state, valid_mask, deterministic=True)
            probs = agent.get_action_probs(state, valid_mask)

            print(f"Player {current_player} ({'X' if current_player == 0 else 'O'}) "
                  f"chooses position {action}")
            print(f"Action probabilities: {np.round(probs, 3)}")

            next_state, reward, done, info = game.step(action)
            state = next_state

            game.render()
            time.sleep(delay)

        # Update stats
        winner = info['winner']
        if winner == -1:
            stats['draws'] += 1
            print("Result: DRAW")
        elif winner == 0:
            stats['p0_wins'] += 1
            print("Result: Player 0 (X) WINS")
        else:
            stats['p1_wins'] += 1
            print("Result: Player 1 (O) WINS")

    # Print overall stats
    print(f"\n{'='*40}")
    print(f"Self-Play Summary")
    print(f"{'='*40}")
    print(f"Player 0 wins: {stats['p0_wins']}")
    print(f"Player 1 wins: {stats['p1_wins']}")
    print(f"Draws: {stats['draws']}")
    print(f"{'='*40}\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Tic-Tac-Toe agent')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--mode', type=str, default='vs_random',
                        choices=['vs_random', 'vs_human', 'watch'],
                        help='Evaluation mode')
    parser.add_argument('--num-games', type=int, default=100,
                        help='Number of games (for vs_random and watch modes)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed game information')
    parser.add_argument('--agent-player', type=int, default=0, choices=[0, 1],
                        help='Which player is the agent (for vs_human mode)')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU ID (-1 for CPU)')
    args = parser.parse_args()

    # Load agent
    print(f"Loading model from {args.model}...")
    agent = TicTacToeAgent(gpu_id=args.gpu, model_path=args.model)
    print("Model loaded successfully!\n")

    # Run evaluation
    if args.mode == 'vs_random':
        print(f"Evaluating against random opponent ({args.num_games} games)...\n")
        stats = evaluate_vs_random(agent, args.num_games, args.verbose)

        print(f"\n{'='*60}")
        print(f"Evaluation Results (vs Random)")
        print(f"{'='*60}")
        print(f"Games played: {args.num_games}")
        print(f"Wins:         {stats['wins']:4d} ({stats['wins']/args.num_games:.1%})")
        print(f"Losses:       {stats['losses']:4d} ({stats['losses']/args.num_games:.1%})")
        print(f"Draws:        {stats['draws']:4d} ({stats['draws']/args.num_games:.1%})")
        print(f"{'='*60}\n")

    elif args.mode == 'vs_human':
        play_vs_human(agent, args.agent_player)

    elif args.mode == 'watch':
        watch_selfplay(agent, args.num_games)


if __name__ == '__main__':
    main()
