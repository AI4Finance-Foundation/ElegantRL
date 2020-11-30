from AgentRun import *
from AgentNet import *
from AgentZoo import *


def test__train_agent():
    args = Arguments()
    args.rl_agent = AgentInterPPO

    args.net_dim = 2 ** 8
    args.max_memo = 2 ** 11  # 12
    args.batch_size = 2 ** 9
    args.repeat_times = 2 ** 3  # 4

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)
    args.env_name = "ReacherBulletEnv-v0"
    args.break_step = int(5e6 * 8)  # (4e4) 5e4
    args.reward_scale = 2 ** 0  # (-37) 0 ~ 18 (29) # todo wait update
    args.net_dim = 2 ** 8
    args.max_memo = 2 ** 11
    args.batch_size = 2 ** 9
    args.repeat_times = 2 ** 3
    args.init_for_training()
    train_agent(**vars(args))
    # train_agent_mp(args)
    exit()

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
    train_agent(**vars(args))
    exit()


test__train_agent()
