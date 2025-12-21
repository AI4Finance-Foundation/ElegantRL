"""
Example: Using ElegantRL for Custom Game Integration
Demonstrates offline RL workflow: collect data from game → train offline → evaluate
"""

import torch
import numpy as np
from elegantrl.agents import AgentPPO, AgentSAC
from elegantrl.train.replay_buffer import ReplayBuffer


class GameAgent:
    """Wrapper for using ElegantRL agent in your custom game"""

    def __init__(self, state_dim, action_dim, agent_class=AgentPPO, model_path=None):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Initialize agent
        self.agent = agent_class(
            net_dims=[256, 256],
            state_dim=state_dim,
            action_dim=action_dim,
            gpu_id=0  # -1 for CPU
        )

        # Load pre-trained model if provided
        if model_path:
            self.load_model(model_path)

    def get_action(self, observation, deterministic=True):
        """
        Get action for given observation

        Args:
            observation: Game state (numpy array or list)
            deterministic: If True, use deterministic policy (for evaluation)
                         If False, add exploration noise (for training)

        Returns:
            action: numpy array of shape (action_dim,)
        """
        state = torch.FloatTensor(observation).unsqueeze(0).to(self.agent.device)

        with torch.no_grad():
            if deterministic:
                # For evaluation - no exploration
                action = self.agent.act.forward(state)
            else:
                # For training - with exploration
                action = self.agent.explore_action(state)

        return action.cpu().numpy()[0]

    def save_model(self, path):
        """Save trained model"""
        torch.save({
            'actor': self.agent.act.state_dict(),
            'critic': self.agent.cri.state_dict()
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Load trained model"""
        checkpoint = torch.load(path, map_location=self.agent.device)
        self.agent.act.load_state_dict(checkpoint['actor'])
        self.agent.cri.load_state_dict(checkpoint['critic'])
        print(f"Model loaded from {path}")


class GameDataCollector:
    """Collect experience data from game episodes"""

    def __init__(self, state_dim, action_dim, max_buffer_size=1_000_000):
        self.buffer = ReplayBuffer(
            max_size=max_buffer_size,
            state_dim=state_dim,
            action_dim=action_dim,
            gpu_id=0
        )

    def collect_episode(self, game_env, agent, max_steps=1000):
        """
        Play one episode and collect transitions

        Args:
            game_env: Your custom game environment with reset() and step() methods
            agent: GameAgent instance
            max_steps: Maximum steps per episode

        Returns:
            total_reward: Episode return
        """
        state = game_env.reset()
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_dones = []

        total_reward = 0

        for step in range(max_steps):
            # Get action from agent (with exploration)
            action = agent.get_action(state, deterministic=False)

            # Step in environment
            next_state, reward, done, info = game_env.step(action)

            # Store transition
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_dones.append(1.0 if done else 0.0)

            total_reward += reward
            state = next_state

            if done:
                break

        # Convert to tensors and add to buffer
        states = torch.FloatTensor(episode_states).unsqueeze(1)  # (T, 1, state_dim)
        actions = torch.FloatTensor(episode_actions).unsqueeze(1)  # (T, 1, action_dim)
        rewards = torch.FloatTensor(episode_rewards).unsqueeze(1)  # (T, 1)
        undones = torch.FloatTensor([1.0 - d for d in episode_dones]).unsqueeze(1)  # (T, 1)

        buffer_items = (states, actions, rewards, undones)
        self.buffer.update(buffer_items)

        return total_reward

    def collect_multiple_episodes(self, game_env, agent, num_episodes=100):
        """Collect data from multiple episodes"""
        rewards = []

        for ep in range(num_episodes):
            ep_reward = self.collect_episode(game_env, agent)
            rewards.append(ep_reward)

            if (ep + 1) % 10 == 0:
                avg_reward = np.mean(rewards[-10:])
                print(f"Episode {ep+1}/{num_episodes}, Avg Reward: {avg_reward:.2f}, "
                      f"Buffer Size: {self.buffer.cur_size}")

        return rewards

    def save_data(self, save_dir='./game_data'):
        """Save collected data to disk"""
        self.buffer.save_or_load_history(cwd=save_dir, if_save=True)
        print(f"Saved {self.buffer.cur_size} transitions to {save_dir}")

    def load_data(self, save_dir='./game_data'):
        """Load previously collected data"""
        self.buffer.save_or_load_history(cwd=save_dir, if_save=False)
        print(f"Loaded {self.buffer.cur_size} transitions from {save_dir}")


class GameTrainer:
    """Train agent offline on collected data"""

    def __init__(self, state_dim, action_dim, agent_class=AgentPPO):
        self.agent = agent_class(
            net_dims=[256, 256],
            state_dim=state_dim,
            action_dim=action_dim,
            gpu_id=0
        )

    def train(self, buffer, num_updates=10000, batch_size=512, log_interval=100):
        """
        Train agent on collected data

        Args:
            buffer: ReplayBuffer with collected transitions
            num_updates: Number of gradient updates
            batch_size: Batch size for training
            log_interval: Print stats every N updates
        """
        print(f"Starting training with {buffer.cur_size} transitions...")

        for update in range(num_updates):
            # Update networks
            obj_critic, obj_actor = self.agent.update_net(buffer)

            if (update + 1) % log_interval == 0:
                print(f"Update {update+1}/{num_updates}: "
                      f"Critic Loss={obj_critic:.4f}, Actor Loss={obj_actor:.4f}")

        print("Training complete!")

    def save_model(self, path):
        """Save trained model"""
        torch.save({
            'actor': self.agent.act.state_dict(),
            'critic': self.agent.cri.state_dict()
        }, path)
        print(f"Model saved to {path}")


# ========================================
# Example Usage
# ========================================

if __name__ == "__main__":
    # Your game configuration
    STATE_DIM = 8    # e.g., board state representation
    ACTION_DIM = 2   # e.g., move coordinates

    # Example: Dummy game environment (replace with your actual game)
    class DummyGameEnv:
        def reset(self):
            return np.random.randn(STATE_DIM)

        def step(self, action):
            next_state = np.random.randn(STATE_DIM)
            reward = np.random.randn()
            done = np.random.rand() < 0.1  # 10% chance to end
            info = {}
            return next_state, reward, done, info


    # ============================================
    # PHASE 1: Collect data from game evaluation
    # ============================================
    print("\n=== PHASE 1: Data Collection ===")

    game_env = DummyGameEnv()
    agent = GameAgent(STATE_DIM, ACTION_DIM)
    collector = GameDataCollector(STATE_DIM, ACTION_DIM)

    # Collect data from 100 game episodes
    rewards = collector.collect_multiple_episodes(game_env, agent, num_episodes=100)

    # Save collected data
    collector.save_data(save_dir='./my_game_data')


    # ============================================
    # PHASE 2: Train offline on collected data
    # ============================================
    print("\n=== PHASE 2: Offline Training ===")

    # Initialize trainer
    trainer = GameTrainer(STATE_DIM, ACTION_DIM, agent_class=AgentPPO)

    # Load collected data
    new_buffer = ReplayBuffer(
        max_size=1_000_000,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        gpu_id=0
    )
    new_buffer.save_or_load_history(cwd='./my_game_data', if_save=False)

    # Train on collected data
    trainer.train(new_buffer, num_updates=5000)

    # Save trained model
    trainer.save_model('./trained_game_agent.pth')


    # ============================================
    # PHASE 3: Evaluate trained agent
    # ============================================
    print("\n=== PHASE 3: Evaluation ===")

    # Load trained agent
    eval_agent = GameAgent(STATE_DIM, ACTION_DIM, model_path='./trained_game_agent.pth')

    # Evaluate for 10 episodes
    eval_rewards = []
    for ep in range(10):
        state = game_env.reset()
        ep_reward = 0
        done = False

        while not done:
            # Get deterministic action
            action = eval_agent.get_action(state, deterministic=True)
            next_state, reward, done, info = game_env.step(action)
            ep_reward += reward
            state = next_state

        eval_rewards.append(ep_reward)
        print(f"Eval Episode {ep+1}: Reward = {ep_reward:.2f}")

    print(f"\nAverage Evaluation Reward: {np.mean(eval_rewards):.2f}")
