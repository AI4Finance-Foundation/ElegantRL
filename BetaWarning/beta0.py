from AgentRun import *
from AgentNet import *
from AgentZoo import *

"""
beta3 OtherPPO CarRacing


InterOffPPO BiWalker
ceta4 all_old_value_std = 1.0
beta1 ceta4 Inter
beta2 beta1 InterLamb
ceta1 beta2 InterLamb a__log_std

OffPPO
ceta3 Minitaur
ceta2 BiWalker
"""


def run_continuous_action(gpu_id=None):
    rl_agent = AgentInterSAC1101
    args = Arguments(rl_agent, gpu_id)
    args.if_break_early = False
    args.if_remove_history = True

    args.random_seed += 123

    # args.env_name = "LunarLanderContinuous-v2"
    # args.break_step = int(5e4 * 16)  # (2e4) 5e4
    # args.reward_scale = 2 ** -3  # (-800) -200 ~ 200 (302)
    # args.init_for_training()
    # train_agent_mp(args)  # train_agent(**vars(args))
    # # exit()
    #
    # args.env_name = "BipedalWalker-v3"
    # args.break_step = int(2e5 * 8)  # (1e5) 2e5
    # args.reward_scale = 2 ** -1  # (-200) -140 ~ 300 (341)
    # args.init_for_training()
    # train_agent_mp(args)  # train_agent(**vars(args))
    # exit()
    #
    # import pybullet_envs  # for python-bullet-gym
    # dir(pybullet_envs)
    # args.env_name = "AntBulletEnv-v0"
    # args.break_step = int(1e6 * 8)  # (8e5) 10e5
    # args.reward_scale = 2 ** -3  # (-50) 0 ~ 2500 (3340)
    # args.batch_size = 2 ** 8
    # args.max_memo = 2 ** 20
    # args.eva_size = 2 ** 3  # for Recorder
    # args.show_gap = 2 ** 8  # for Recorder
    # args.init_for_training()
    # train_agent_mp(args)  # train_agent(**vars(args))
    # exit()
    #
    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)
    args.env_name = "MinitaurBulletEnv-v0"
    args.break_step = int(4e6 * 4)  # (2e6) 4e6
    args.reward_scale = 2 ** 5  # (-2) 0 ~ 16 (20)
    args.batch_size = (2 ** 8)
    args.net_dim = int(2 ** 8)
    args.max_step = 2 ** 11  # todo
    args.max_memo = 2 ** 20
    args.eval_times2 = 3  # for Recorder
    args.eval_times2 = 9  # for Recorder
    args.show_gap = 2 ** 9  # for Recorder
    args.init_for_training()
    train_agent_mp(args)  # train_agent(**vars(args))
    exit()
    # args.env_name = "BipedalWalkerHardcore-v3"  # 2020-08-24 plan
    # args.reward_scale = 2 ** 0  # (-200) -150 ~ 300 (334)
    # args.break_step = int(4e6 * 8)  # (2e6) 4e6
    # args.net_dim = int(2 ** 8)  # int(2 ** 8.5) #
    # args.max_memo = int(2 ** 21)
    # args.batch_size = int(2 ** 8)
    # args.eval_times2 = 2 ** 5  # for Recorder
    # args.show_gap = 2 ** 8  # for Recorder
    # args.init_for_training()
    # train_agent_mp(args)  # train_offline_policy(**vars(args))
    # exit()


def run_continuous_action_off_ppo(gpu_id=None):
    args = Arguments()
    args.rl_agent = AgentOffPPO
    args.if_break_early = False
    args.if_remove_history = True

    args.random_seed += 123

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)
    args.env_name = "MinitaurBulletEnv-v0"  # PPO is the best, I don't know why.
    args.break_step = int(5e5 * 8)  # (PPO 3e5) 5e5
    args.reward_scale = 2 ** 4  # (-2) 0 ~ 16 (PPO 34)
    args.net_dim = 2 ** 8
    args.max_memo = 2 ** 11  # todo
    args.batch_size = 2 ** 9
    args.repeat_times = 2 ** 4
    args.init_for_training()
    train_agent_mp(args)  # train_agent(**vars(args))
    exit()


run_continuous_action_off_ppo()
