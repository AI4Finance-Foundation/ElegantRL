from AgentRun import *
from AgentZoo import *
from AgentNet import *

"""
ModPPO use_dn=True
beta0 BW 256
beta1 BW 128
ceta0 Mini 256
ceta1 Mini 128

"""


def run__on_policy():
    args = Arguments(rl_agent=None, env_name=None, gpu_id=None)
    args.rl_agent = AgentModPPO
    args.random_seed += 12
    args.if_break_early = False

    args.net_dim = 2 ** 7  # todo
    args.max_memo = 2 ** 12
    args.batch_size = 2 ** 9
    args.repeat_times = 2 ** 4
    args.reward_scale = 2 ** 0  # unimportant hyper-parameter in PPO which do normalization on Q value
    args.gamma = 0.99  # important hyper-parameter, related to episode steps

    # args.env_name = "BipedalWalker-v3"
    # args.break_step = int(8e5 * 8)  # (6e5) 8e5 (6e6), UsedTimes: (800s) 1500s (8000s)
    # args.reward_scale = 2 ** 0  # (-150) -90 ~ 300 (324)
    # args.gamma = 0.95  # important hyper-parameter, related to episode steps
    # args.init_for_training()
    # train_agent_mp(args)  # train_agent(args)
    # exit()

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)
    args.env_name = "MinitaurBulletEnv-v0"  # PPO is the best, I don't know why.
    args.break_step = int(5e5 * 8)  # (PPO 3e5) 5e5
    args.reward_scale = 2 ** 4  # (-2) 0 ~ 16 (PPO 34)
    args.gamma = 0.95  # important hyper-parameter, related to episode steps
    args.net_dim = 2 ** 8
    args.max_memo = 2 ** 11
    args.batch_size = 2 ** 9
    args.repeat_times = 2 ** 4
    args.init_for_training()
    train_agent_mp(args)
    exit()


run__on_policy()
