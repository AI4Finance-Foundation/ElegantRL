import os
import sys

from erl_config import Config, get_gym_env_args
from erl_agent import AgentDDPG
from erl_run import train_agent, valid_agent
from erl_env import PendulumEnv


def train_ddpg_for_pendulum(gpu_id=0):
    agent_class = AgentDDPG  # DRL algorithm
    env_class = PendulumEnv  # run a custom env: PendulumEnv, which based on OpenAI pendulum
    env_args = {
        'env_name': 'Pendulum',  # Apply torque on the free end to swing a pendulum into an upright position
        # Reward: r = -(theta + 0.1 * theta_dt + 0.001 * torque)

        'state_dim': 3,  # the x-y coordinates of the pendulum's free end and its angular velocity.
        'action_dim': 1,  # the torque applied to free end of the pendulum
        'if_discrete': False  # continuous action space, symbols → direction, value → force
    }
    get_gym_env_args(env=PendulumEnv(), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `erl_config.py Arguments()` for hyperparameter explanation
    args.break_step = int(1e5)  # break training if 'total_step > break_step'
    args.net_dims = [64, 32]  # the middle layer dimension of MultiLayer Perceptron
    args.gamma = 0.97  # discount factor of future rewards

    args.gpu_id = gpu_id  # the ID of single GPU, -1 means CPU
    train_agent(args)
    if input("| Press 'y' to load actor.pth and render:") == 'y':
        actor_name = sorted([s for s in os.listdir(args.cwd) if s[-4:] == '.pth'])[-1]
        actor_path = f"{args.cwd}/{actor_name}"
        valid_agent(env_class, env_args, args.net_dims, agent_class, actor_path)


if __name__ == "__main__":
    GPU_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    train_ddpg_for_pendulum(gpu_id=GPU_ID)
