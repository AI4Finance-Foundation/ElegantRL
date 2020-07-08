import os

from sac_d_zoo import SacdAgent
from sac_d_env import make_pytorch_env

"""
Soft Actor-Critic for Discrete Action Settings
https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch
bad+ (why should I install nn_builder and TensorFlow2 in a PyTorch implement?)

https://github.com/ku2482/sac-discrete.pytorch
good--
"""


def run():
    config = {
        'num_steps': 300000,
        'batch_size': 64,
        'lr': 0.0003,
        'memory_size': 300000,
        'gamma': 0.99,
        'multi_step': 1,
        'target_entropy_ratio': 0.98,
        'start_steps': 20000,
        'update_interval': 4,
        'target_update_interval': 8000,
        'use_per': False,
        'dueling_net': False,
        'num_eval_steps': 2 ** 12,
        'max_episode_steps': 27000,
        'log_interval': 10,
        'eval_interval': 5000,
    }
    env_id = 'MsPacmanNoFrameskip-v4'

    # Create environments.
    env = make_pytorch_env(env_id, clip_rewards=False)
    test_env = make_pytorch_env(
        env_id, episode_life=False, clip_rewards=False)

    log_dir = os.path.join('logs', env_id, f'sac_discrete')

    agent = SacdAgent(
        env=env, test_env=test_env, log_dir=log_dir, cuda=False,
        seed=0, **config)
    agent.run()


if __name__ == '__main__':
    run()
