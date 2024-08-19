import os
import sys
import gymnasium as gym

from erl_config import Config, get_gym_env_args
from erl_agent import AgentDQN
from erl_run import train_agent, valid_agent

gym.logger.set_level(40)  # Block warning


def train_dqn_for_cartpole(gpu_id=0):
    agent_class = AgentDQN  # DRL algorithm
    env_class = gym.make
    env_args = {
        'env_name': 'CartPole-v0',  # A pole is attached by an un-actuated joint to a cart.
        # Reward: keep the pole upright, a reward of `+1` for every step taken

        'state_dim': 4,  # (CartPosition, CartVelocity, PoleAngle, PoleAngleVelocity)
        'action_dim': 2,  # (Push cart to the left, Push cart to the right)
        'if_discrete': True,  # discrete action space
    }
    get_gym_env_args(env=gym.make('CartPole-v0'), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `erl_config.py Arguments()` for hyperparameter explanation
    args.break_step = int(1e5)  # break training if 'total_step > break_step'
    args.net_dims = [64, 32]  # the middle layer dimension of MultiLayer Perceptron
    args.gamma = 0.95  # discount factor of future rewards

    args.gpu_id = gpu_id  # the ID of single GPU, -1 means CPU
    train_agent(args)
    if input("| Press 'y' to load actor.pth and render:") == 'y':
        actor_name = sorted([s for s in os.listdir(args.cwd) if s[-4:] == '.pth'])[-1]
        actor_path = f"{args.cwd}/{actor_name}"
        valid_agent(env_class, env_args, args.net_dims, agent_class, actor_path)


def train_dqn_for_lunar_lander(gpu_id=0):
    agent_class = AgentDQN  # DRL algorithm
    env_class = gym.make
    env_args = {
        'env_name': 'LunarLander-v2',  # A lander learns to land on a landing pad and using as little fuel as possible
        # Reward: Lander moves to the landing pad and come rest +100; lander crashes -100.
        # Reward: Lander moves to landing pad get positive reward, move away gets negative reward.
        # Reward: Firing the main engine -0.3,  side engine -0.03 each frame.

        'state_dim': 8,  # coordinates xy, linear velocities xy, angle, angular velocity, two booleans
        'action_dim': 4,  # do nothing, fire left engine, fire main engine, fire right engine.
        'if_discrete': True  # discrete action space
    }
    get_gym_env_args(env=gym.make('LunarLander-v2'), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `erl_config.py Arguments()` for hyperparameter explanation
    args.break_step = int(4e5)  # break training if 'total_step > break_step'
    args.explore_rate = 0.1  # the probability of choosing action randomly in epsilon-greedy
    args.net_dims = [128, 64]  # the middle layer dimension of Fully Connected Network

    args.gpu_id = gpu_id  # the ID of single GPU, -1 means CPU
    train_agent(args)
    if input("| Press 'y' to load actor.pth and render:") == 'y':
        actor_name = sorted([s for s in os.listdir(args.cwd) if s[-4:] == '.pth'])[-1]
        actor_path = f"{args.cwd}/{actor_name}"
        valid_agent(env_class, env_args, args.net_dims, agent_class, actor_path)


if __name__ == "__main__":
    GPU_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    train_dqn_for_cartpole(gpu_id=GPU_ID)
    """
    | Arguments Remove cwd: ./CartPole-v1_DQN_0
    | Evaluator:
    | `step`: Number of samples, or total training steps, or running times of `env.step()`.
    | `time`: Time spent from the start of training to this moment.
    | `avgR`: Average value of cumulative rewards, which is the sum of rewards in an episode.
    | `stdR`: Standard dev of cumulative rewards, which is the sum of rewards in an episode.
    | `avgS`: Average of steps in an episode.
    | `objC`: Objective of Critic network. Or call it loss function of critic network.
    | `objA`: Objective of Actor network. It is the average Q value of the critic network.
    |     step      time  |     avgR    stdR    avgS  |     objC      objA
    | 1.02e+04        19  |    17.31    2.11      17  |     0.92     19.77
    | 2.05e+04        39  |     9.47    0.71       9  |     0.93     23.96
    | 3.07e+04        66  |   191.25   18.10     191  |     1.38     31.52
    | 4.10e+04        98  |   212.41   16.34     212  |     0.65     21.52
    | 5.12e+04       141  |   183.41   10.96     183  |     0.47     21.10
    | 6.14e+04       184  |   171.94    8.44     172  |     0.34     20.48
    | 7.17e+04       233  |   173.00    8.85     173  |     0.27     19.88
    | 8.19e+04       290  |   115.84    3.61     116  |     0.24     19.95
    | 9.22e+04       349  |   128.44    5.99     128  |     0.19     19.80
    """

    train_dqn_for_lunar_lander(gpu_id=GPU_ID)
