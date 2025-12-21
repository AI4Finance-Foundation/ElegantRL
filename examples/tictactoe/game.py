"""
Tic-Tac-Toe Game Environment
"""

import numpy as np
from typing import Optional, Tuple


class TicTacToe:
    """
    Tic-Tac-Toe game environment

    State representation: 9-element array
    - 0: empty
    - 1: player 0 (X)
    - -1: player 1 (O)

    Actions: integers 0-8 representing board positions
    """

    def __init__(self):
        self.board = np.zeros(9, dtype=np.float32)
        self.current_player = 0

    def reset(self) -> np.ndarray:
        """Reset game to initial state"""
        self.board = np.zeros(9, dtype=np.float32)
        self.current_player = 0
        return self.get_state()

    def get_state(self) -> np.ndarray:
        """
        Get current state from current player's perspective

        Returns normalized board where:
        - 1: current player's pieces
        - -1: opponent's pieces
        - 0: empty
        """
        if self.current_player == 0:
            return self.board.copy()
        else:
            return -self.board.copy()  # Flip perspective for player 1

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute action

        Args:
            action: Position 0-8

        Returns:
            next_state: Next state
            reward: Reward from current player's perspective
            done: Whether game is over
            info: Additional info including winner
        """
        if not self.is_valid_action(action):
            # Invalid move - game over with penalty
            return self.get_state(), -10.0, True, {'winner': 1 - self.current_player, 'invalid': True}

        # Place piece
        piece = 1.0 if self.current_player == 0 else -1.0
        self.board[action] = piece

        # Check for winner
        winner = self.check_winner()
        done = winner is not None or self.is_full()

        # Calculate reward from current player's perspective
        if winner == self.current_player:
            reward = 1.0  # Win
        elif winner is not None:
            reward = -1.0  # Loss
        elif done:
            reward = 0.0  # Draw
        else:
            reward = 0.0  # Game continues

        # Switch player
        self.current_player = 1 - self.current_player

        info = {
            'winner': winner if winner is not None else -1,
            'invalid': False
        }

        return self.get_state(), reward, done, info

    def is_valid_action(self, action: int) -> bool:
        """Check if action is valid"""
        return 0 <= action < 9 and self.board[action] == 0

    def get_valid_actions_mask(self) -> np.ndarray:
        """Get mask of valid actions (1=valid, 0=invalid)"""
        return (self.board == 0).astype(np.float32)

    def is_full(self) -> bool:
        """Check if board is full"""
        return np.all(self.board != 0)

    def check_winner(self) -> Optional[int]:
        """
        Check if there's a winner

        Returns:
            0: player 0 wins
            1: player 1 wins
            None: no winner yet
        """
        # Winning lines (rows, columns, diagonals)
        lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]  # Diagonals
        ]

        for line in lines:
            sum_line = np.sum(self.board[line])
            if sum_line == 3:
                return 0  # Player 0 wins
            elif sum_line == -3:
                return 1  # Player 1 wins

        return None

    def render(self):
        """Print board to console"""
        symbols = {0: '.', 1: 'X', -1: 'O'}
        print("\nCurrent board:")
        for i in range(3):
            row = [symbols[self.board[i*3 + j]] for j in range(3)]
            print(f"  {' '.join(row)}")
        print()

    def clone(self):
        """Create a copy of current game state"""
        new_game = TicTacToe()
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        return new_game


class RandomPlayer:
    """Random player for baseline comparison"""

    def get_action(self, state: np.ndarray, valid_mask: np.ndarray) -> int:
        """Select random valid action"""
        valid_actions = np.where(valid_mask == 1)[0]
        return np.random.choice(valid_actions)


class HumanPlayer:
    """Human player for interactive play"""

    def get_action(self, state: np.ndarray, valid_mask: np.ndarray) -> int:
        """Get action from human input"""
        while True:
            try:
                action = int(input("Enter your move (0-8): "))
                if valid_mask[action] == 1:
                    return action
                else:
                    print("Invalid move! Position already taken.")
            except (ValueError, IndexError):
                print("Invalid input! Enter a number between 0 and 8.")
