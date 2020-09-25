from AgentRun import *
from AgentNet import *
from AgentZoo import *

"""
auto TTUR
try MSE in AMAX1
"""


def run_continuous_action(gpu_id=None):
    # import AgentZoo as Zoo
    args = Arguments(rl_agent=AgentInterGAE, gpu_id=gpu_id)
    args.show_gap = 2 ** 9
    args.eval_times2 = 2 ** 5
    args.if_stop = False  # todo

    args.net_dim = 2 ** 8
    args.max_memo = 2 ** 12
    args.batch_size = 2 ** 9
    args.repeat_times = 2 ** 4

    # args.env_name = "Pendulum-v0"  # It is easy to reach target score -200.0 (-100 is harder)
    # args.max_total_step = int(1e5 * 4)
    # args.reward_scale = 2 ** -2
    # args.init_for_training()
    # train_agent(**vars(args))
    # # exit()
    #
    # args.env_name = "LunarLanderContinuous-v2"
    # args.max_total_step = int(1e5 * 16)
    # args.init_for_training()
    # train_agent(**vars(args))
    # # exit()
    #
    # args.env_name = "BipedalWalker-v3"
    # args.max_total_step = int(3e6 * 4)
    # args.init_for_training()
    # train_agent(**vars(args))
    # # exit()
    #
    # import pybullet_envs  # for python-bullet-gym
    # dir(pybullet_envs)
    # args.env_name = "AntBulletEnv-v0"
    # args.max_total_step = int(1e6 * 8)
    # args.net_dim = 2 ** 9
    # # exit()
    #
    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)
    args.env_name = "MinitaurBulletEnv-v0"
    args.max_total_step = int(2e7 * 8)
    args.reward_scale = 2 ** 4
    args.net_dim = 2 ** 9
    args.init_for_training()
    train_agent(**vars(args))
    # # exit()

    # args.env_name = "BipedalWalkerHardcore-v3"  # 2020-08-24 plan
    # args.max_total_step = int(4e6 * 8)
    # args.reward_scale = 2 ** 0
    # args.net_dim = 2 ** 9
    # args.init_for_training()
    # train_agent(**vars(args))
    # exit()


run_continuous_action()
