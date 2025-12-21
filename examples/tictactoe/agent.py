"""
RL Agent wrapper for Tic-Tac-Toe
"""

import torch
import numpy as np
from elegantrl.agents import AgentPPO
from elegantrl.train.config import Config


class TicTacToeAgent:
    """
    Reinforcement Learning agent for Tic-Tac-Toe

    Uses ElegantRL's PPO agent with custom action masking
    """

    def __init__(self, state_dim=9, action_dim=9, gpu_id=0, model_path=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(f"cuda:{gpu_id}" if gpu_id >= 0 and torch.cuda.is_available() else "cpu")

        # Configure for discrete actions
        args = Config()
        args.if_discrete = True  # Tic-Tac-Toe has discrete actions (0-8)

        # Initialize PPO agent
        self.agent = AgentPPO(
            net_dims=[128, 128],  # 2-layer network
            state_dim=state_dim,
            action_dim=action_dim,
            gpu_id=gpu_id,
            args=args
        )

        # Load model if provided
        if model_path:
            self.load_model(model_path)

    def get_action(self, state: np.ndarray, valid_mask: np.ndarray,
                   temperature=1.0, deterministic=False) -> int:
        """
        Get action for given state

        Args:
            state: Current game state (9-element array)
            valid_mask: Binary mask for valid actions
            temperature: Exploration temperature (higher = more random)
            deterministic: If True, select best action; if False, sample

        Returns:
            action: Selected action index (0-8)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Get action logits from actor network
            action_logits = self.agent.act.net(state_tensor)

            # Apply temperature
            action_logits = action_logits / temperature

            # Mask invalid actions
            mask_tensor = torch.FloatTensor(valid_mask).unsqueeze(0).to(self.device)
            action_logits = action_logits.masked_fill(mask_tensor == 0, -1e9)

            # Get action
            if deterministic:
                action = torch.argmax(action_logits, dim=-1).item()
            else:
                action_probs = torch.softmax(action_logits, dim=-1)
                action = torch.multinomial(action_probs, 1).item()

        return action

    def get_action_probs(self, state: np.ndarray, valid_mask: np.ndarray,
                         temperature=1.0) -> np.ndarray:
        """Get action probability distribution"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_logits = self.agent.act.net(state_tensor)
            action_logits = action_logits / temperature

            # Mask invalid actions
            mask_tensor = torch.FloatTensor(valid_mask).unsqueeze(0).to(self.device)
            action_logits = action_logits.masked_fill(mask_tensor == 0, -1e9)

            action_probs = torch.softmax(action_logits, dim=-1)

        return action_probs.cpu().numpy()[0]

    def get_value(self, state: np.ndarray) -> float:
        """Get state value estimate"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            value = self.agent.cri(state_tensor)

        return value.cpu().item()

    def save_model(self, path: str):
        """Save trained model"""
        torch.save({
            'actor': self.agent.act.state_dict(),
            'critic': self.agent.cri.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.agent.act.load_state_dict(checkpoint['actor'])
        self.agent.cri.load_state_dict(checkpoint['critic'])
        print(f"Model loaded from {path}")

    def get_agent(self):
        """Get underlying ElegantRL agent (for training)"""
        return self.agent
