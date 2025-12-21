"""
User-friendly Evaluator class for getting actions from trained models

This class provides a simple interface for loading a trained model
and getting actions during gameplay/evaluation.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Union


class Evaluator:
    """
    Simple evaluator for getting actions from a trained model

    Usage:
        evaluator = Evaluator(model_path='checkpoints/best_model.pth')
        action = evaluator.get_action(state, valid_mask)
    """

    def __init__(self, model_path: str, agent_class=None, state_dim: int = 9,
                 action_dim: int = 9, gpu_id: int = -1):
        """
        Initialize evaluator with a trained model

        Args:
            model_path: Path to saved model (.pth file)
            agent_class: Agent class to use (if None, will try to infer from model)
            state_dim: State dimension
            action_dim: Action dimension
            gpu_id: GPU ID (-1 for CPU)
        """
        self.model_path = Path(model_path)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(f"cuda:{gpu_id}" if gpu_id >= 0 and torch.cuda.is_available() else "cpu")

        # Load model
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # If agent_class not provided, try to import from context
        if agent_class is None:
            # Try to import TicTacToeAgent as default
            try:
                from agent import TicTacToeAgent
                agent_class = TicTacToeAgent
            except ImportError:
                raise ValueError("agent_class must be provided if not using TicTacToeAgent")

        # Initialize agent
        self.agent = agent_class(
            state_dim=state_dim,
            action_dim=action_dim,
            gpu_id=gpu_id,
            model_path=str(model_path)
        )

        print(f"Evaluator loaded model from {model_path}")

    def get_action(self, state: np.ndarray, valid_mask: Optional[np.ndarray] = None,
                   deterministic: bool = True, temperature: float = 1.0) -> int:
        """
        Get action for given state

        Args:
            state: Current state (numpy array)
            valid_mask: Binary mask for valid actions (1=valid, 0=invalid)
                       If None, all actions are considered valid
            deterministic: If True, select best action; if False, sample
            temperature: Exploration temperature (higher = more random)
                        Only used if deterministic=False

        Returns:
            action: Selected action (integer)
        """
        if valid_mask is None:
            valid_mask = np.ones(self.action_dim)

        action = self.agent.get_action(
            state=state,
            valid_mask=valid_mask,
            temperature=temperature,
            deterministic=deterministic
        )

        return action

    def get_action_probs(self, state: np.ndarray, valid_mask: Optional[np.ndarray] = None,
                         temperature: float = 1.0) -> np.ndarray:
        """
        Get action probability distribution

        Args:
            state: Current state
            valid_mask: Binary mask for valid actions
            temperature: Temperature for probability scaling

        Returns:
            probs: Action probabilities (numpy array)
        """
        if valid_mask is None:
            valid_mask = np.ones(self.action_dim)

        probs = self.agent.get_action_probs(
            state=state,
            valid_mask=valid_mask,
            temperature=temperature
        )

        return probs

    def get_value(self, state: np.ndarray) -> float:
        """
        Get state value estimate

        Args:
            state: Current state

        Returns:
            value: Estimated state value
        """
        return self.agent.get_value(state)
