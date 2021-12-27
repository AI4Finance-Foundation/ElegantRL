import gym
import numpy as np
import numpy.random as rd
import torch

from elegantrl.train.config import Arguments
from elegantrl.train.run_parallel import train_and_evaluate_mp
from elegantrl.train.run_tutorial import train_and_evaluate
from elegantrl.envs.Gym import GymEnv
from elegantrl.agents.AgentPPO import AgentPPO, AgentModPPO

"""delete"""

"""demo: train on standard gym env"""


def demo_continuous_action():
    args = Arguments(agent=AgentPPO(), env=GymEnv(env_name='BipedalWalker-v3'))

    args.net_dim = 2 ** 8
    args.batch_size = args.net_dim * 2
    args.target_step = args.max_step * 4
    args.reward_scale = 2 ** -2

    args.learner_gpus = (0,)
    train_and_evaluate(args)


"""demo: train on custom env"""

if __name__ == '__main__':
    check__go_after_env()
    check__go_after_vec_env()
    exit()

    # import sys
    # sys.argv.extend('0 1'.split(' '))
    GPU_ID = 0  # int(sys.argv[1])
    # demo_continuous_action()
    # demo_custom_env1()
    # demo_custom_env2()
    demo_custom_env3()
