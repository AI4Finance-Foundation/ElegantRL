from AgentRun import *
from AgentNet import *
from AgentZoo import *

"""
        if loss_c > 4:  # todo loaded ISAC 
"""


def run_continuous_action(gpu_id=None):
    # import AgentZoo as Zoo
    args = Arguments(rl_agent=AgentInterSAC, gpu_id=gpu_id)
    args.show_gap = 2 ** 9
    args.eval_times1 = 2 ** 4
    args.eval_times2 = 2 ** 6
    args.if_stop = False

    # args.env_name = "Pendulum-v0"  # It is easy to reach target score -200.0 (-100 is harder)
    # args.max_total_step = int(1e4 * 4)
    # args.reward_scale = 2 ** -2
    # args.init_for_training()
    # build_for_mp(args)  # train_offline_policy(**vars(args))
    # exit()
    #
    args.env_name = "LunarLanderContinuous-v2"
    args.max_total_step = int(1e5 * 4)
    args.init_for_training()
    build_for_mp(args)  # train_agent(**vars(args))
    # exit()

    # args.env_name = "BipedalWalker-v3"
    # args.random_seed = 1945
    # args.max_total_step = int(2e5 * 4)
    # args.init_for_training()
    # # build_for_mp(args)
    # train_agent(**vars(args))
    # exit()
    #
    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)
    args.env_name = "AntBulletEnv-v0"
    args.max_total_step = int(5e5 * 4)
    args.max_epoch = 2 ** 13
    args.max_memo = 2 ** 20
    args.max_step = 2 ** 10
    args.batch_size = 2 ** 9
    args.reward_scale = 2 ** -2
    args.eval_times2 = 2 ** 3  # for Recorder
    args.show_gap = 2 ** 8  # for Recorder
    args.init_for_training()
    # build_for_mp(args)
    train_agent(**vars(args))

    # import pybullet_envs  # for python-bullet-gym
    # dir(pybullet_envs)
    # args.env_name = "MinitaurBulletEnv-v0"
    # args.max_total_step = int(1e6 * 2)
    # args.max_epoch = 2 ** 13
    # args.max_memo = 2 ** 20  # todo
    # args.max_step = 2 ** 11  # todo 10
    # args.net_dim = 2 ** 8
    # args.batch_size = 2 ** 8
    # args.reward_scale = 2 ** 4
    # args.eva_size = 2 ** 5  # for Recorder
    # args.show_gap = 2 ** 9  # for Recorder
    # args.init_for_training(cpu_threads=4)
    # build_for_mp(args)  # train_agent(**vars(args))


run_continuous_action()
