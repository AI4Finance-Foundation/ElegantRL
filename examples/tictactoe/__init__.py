"""
Tic-Tac-Toe Example for ElegantRL

This package demonstrates how to use ElegantRL for game playing with:
- Custom game environment
- Self-play training
- Evaluation modes (vs random, vs human, watch)
"""

from .game import TicTacToe, RandomPlayer, HumanPlayer
from .agent import TicTacToeAgent

__all__ = ['TicTacToe', 'RandomPlayer', 'HumanPlayer', 'TicTacToeAgent']
