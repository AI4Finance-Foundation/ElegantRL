"""
User-friendly DataSaver class for collecting training data

This class provides a simple interface for collecting game data
and saving it for later training.
"""

import os
import json
import pickle
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime


class DataSaver:
    """
    Data collection and saving for training

    Usage:
        saver = DataSaver(save_dir='./training_data', save_frequency=10)

        # Start new episode
        episode_id = saver.new_episode()

        # Add transitions during gameplay
        saver.add_transition(state, action, logprob)

        # Set reward at end (with optional gamma for reward propagation)
        saver.set_reward(reward, gamma=0.99)
    """

    def __init__(self, save_dir: str = './training_data', save_frequency: int = 10,
                 state_dim: int = 9, action_dim: int = 1):
        """
        Initialize DataSaver

        Args:
            save_dir: Directory to save training data
            save_frequency: Save data every N episodes
            state_dim: State dimension
            action_dim: Action dimension (1 for discrete actions)
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.save_frequency = save_frequency
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Counter file to track episode IDs
        self.counter_file = self.save_dir / '.episode_counter'
        self.counter = self._load_counter()

        # Current episode data
        self.current_episode_id = None
        self.current_transitions = []

        # Buffer for unsaved episodes
        self.episode_buffer = []

        print(f"DataSaver initialized (save_dir={save_dir}, save_frequency={save_frequency})")
        print(f"Starting from episode {self.counter}")

    def _load_counter(self) -> int:
        """Load episode counter from file"""
        if self.counter_file.exists():
            with open(self.counter_file, 'r') as f:
                return int(f.read().strip())
        return 0

    def _save_counter(self):
        """Save episode counter to file"""
        with open(self.counter_file, 'w') as f:
            f.write(str(self.counter))

    def new_episode(self) -> int:
        """
        Start a new episode

        Returns:
            episode_id: Unique episode ID
        """
        # Save previous episode if exists
        if self.current_episode_id is not None and len(self.current_transitions) > 0:
            print(f"Warning: Starting new episode before setting reward for episode {self.current_episode_id}")
            self.set_reward(0.0)  # Default to 0 reward

        # Increment counter and get new ID
        self.counter += 1
        self.current_episode_id = self.counter
        self.current_transitions = []

        # Save counter periodically
        if self.counter % self.save_frequency == 0:
            self._save_counter()

        return self.current_episode_id

    def add_transition(self, state: np.ndarray, action: int, logprob: Optional[float] = None,
                       player: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Add a transition to current episode

        Args:
            state: Current state
            action: Action taken
            logprob: Log probability of action (optional, can be set later)
            player: Player ID for multi-agent games (optional)
            metadata: Additional metadata (optional)
        """
        if self.current_episode_id is None:
            raise ValueError("Must call new_episode() before adding transitions")

        transition = {
            'state': state.copy() if isinstance(state, np.ndarray) else state,
            'action': action,
            'logprob': logprob,
            'player': player,
            'metadata': metadata or {}
        }

        self.current_transitions.append(transition)

    def set_reward(self, reward: float, gamma: float = 0.0, player: Optional[int] = None):
        """
        Set reward for current episode and optionally propagate to previous steps

        Args:
            reward: Final reward for the episode
            gamma: Discount factor for reward propagation (0 = no propagation)
                  If gamma > 0, rewards are propagated backward:
                  - Last transition gets reward
                  - Second-to-last gets reward * gamma
                  - Third-to-last gets reward * gamma^2
                  - etc.
            player: Player ID (for multi-agent games, reward only applies to this player's transitions)
        """
        if self.current_episode_id is None:
            raise ValueError("Must call new_episode() before setting reward")

        if len(self.current_transitions) == 0:
            print(f"Warning: No transitions in episode {self.current_episode_id}")
            return

        # Calculate rewards for each transition
        num_transitions = len(self.current_transitions)

        for i, transition in enumerate(self.current_transitions):
            # Check if this transition belongs to the specified player
            if player is not None and transition.get('player') != player:
                transition['reward'] = 0.0  # Opponent gets 0 or opposite reward
                continue

            # Calculate discounted reward based on distance from end
            steps_from_end = num_transitions - 1 - i
            if gamma > 0:
                discounted_reward = reward * (gamma ** steps_from_end)
            else:
                # Only last transition gets reward
                discounted_reward = reward if i == num_transitions - 1 else 0.0

            transition['reward'] = discounted_reward

        # Add episode to buffer
        episode_data = {
            'episode_id': self.current_episode_id,
            'num_transitions': len(self.current_transitions),
            'final_reward': reward,
            'gamma': gamma,
            'timestamp': datetime.now().isoformat(),
            'transitions': self.current_transitions
        }

        self.episode_buffer.append(episode_data)

        # Save if buffer is full
        if len(self.episode_buffer) >= self.save_frequency:
            self._save_buffer()

        # Reset current episode
        self.current_episode_id = None
        self.current_transitions = []

    def _save_buffer(self):
        """Save buffered episodes to disk"""
        if len(self.episode_buffer) == 0:
            return

        # Create filename with timestamp and episode range
        first_ep = self.episode_buffer[0]['episode_id']
        last_ep = self.episode_buffer[-1]['episode_id']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'episodes_{first_ep:06d}_to_{last_ep:06d}_{timestamp}.pkl'

        filepath = self.save_dir / filename

        # Save as pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.episode_buffer, f)

        print(f"Saved {len(self.episode_buffer)} episodes to {filename}")

        # Also save metadata as JSON for easy inspection
        metadata_file = self.save_dir / filename.replace('.pkl', '_meta.json')
        metadata = {
            'num_episodes': len(self.episode_buffer),
            'episode_range': [first_ep, last_ep],
            'total_transitions': sum(ep['num_transitions'] for ep in self.episode_buffer),
            'timestamp': timestamp,
            'episodes': [
                {
                    'episode_id': ep['episode_id'],
                    'num_transitions': ep['num_transitions'],
                    'final_reward': ep['final_reward'],
                    'gamma': ep['gamma']
                }
                for ep in self.episode_buffer
            ]
        }

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Clear buffer
        self.episode_buffer = []

        # Save counter
        self._save_counter()

    def flush(self):
        """Force save any remaining buffered episodes"""
        if len(self.episode_buffer) > 0:
            self._save_buffer()
        self._save_counter()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about collected data

        Returns:
            stats: Dictionary with statistics
        """
        return {
            'total_episodes': self.counter,
            'buffered_episodes': len(self.episode_buffer),
            'current_episode_transitions': len(self.current_transitions) if self.current_episode_id else 0,
            'save_frequency': self.save_frequency,
            'save_dir': str(self.save_dir)
        }

    @staticmethod
    def load_episodes(save_dir: str, episode_range: Optional[Tuple[int, int]] = None) -> List[Dict]:
        """
        Load saved episodes from directory

        Args:
            save_dir: Directory containing saved episodes
            episode_range: Optional tuple (start_ep, end_ep) to filter episodes

        Returns:
            episodes: List of episode dictionaries
        """
        save_dir = Path(save_dir)

        # Find all pickle files
        pkl_files = sorted(save_dir.glob('episodes_*.pkl'))

        all_episodes = []
        for pkl_file in pkl_files:
            with open(pkl_file, 'rb') as f:
                episodes = pickle.load(f)

                # Filter by episode range if specified
                if episode_range:
                    start_ep, end_ep = episode_range
                    episodes = [ep for ep in episodes
                               if start_ep <= ep['episode_id'] <= end_ep]

                all_episodes.extend(episodes)

        return all_episodes

    @staticmethod
    def convert_to_tensors(episodes: List[Dict], device='cpu') -> Tuple[torch.Tensor, ...]:
        """
        Convert loaded episodes to PyTorch tensors for training

        Args:
            episodes: List of episode dictionaries from load_episodes()
            device: Device to place tensors on

        Returns:
            (states, actions, logprobs, rewards, undones, unmasks)
        """
        all_states = []
        all_actions = []
        all_logprobs = []
        all_rewards = []
        all_undones = []
        all_unmasks = []

        for episode in episodes:
            transitions = episode['transitions']
            num_transitions = len(transitions)

            for i, trans in enumerate(transitions):
                all_states.append(trans['state'])
                all_actions.append(trans['action'])
                all_logprobs.append(trans.get('logprob', 0.0))
                all_rewards.append(trans.get('reward', 0.0))

                # Last transition is terminal
                is_done = (i == num_transitions - 1)
                all_undones.append(0.0 if is_done else 1.0)
                all_unmasks.append(1.0)  # No truncation

        # Convert to tensors
        states = torch.FloatTensor(np.array(all_states)).unsqueeze(1).to(device)
        actions = torch.LongTensor(all_actions).unsqueeze(1).to(device)
        logprobs = torch.FloatTensor(all_logprobs).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(all_rewards).unsqueeze(1).to(device)
        undones = torch.FloatTensor(all_undones).unsqueeze(1).to(device)
        unmasks = torch.FloatTensor(all_unmasks).unsqueeze(1).to(device)

        return states, actions, logprobs, rewards, undones, unmasks
