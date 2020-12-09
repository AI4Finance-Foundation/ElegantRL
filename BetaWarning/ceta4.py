from AgentRun import *
from AgentZoo import *
from AgentNet import *

"""
ModPPO
ceta0 Minitaur
ceta4 BW

PPO
ceta2 Minitaur
ceta3 BW


"""


def run__on_policy():
    import AgentZoo as Zoo
    args = Arguments(rl_agent=None, env_name=None, gpu_id=None)
    args.rl_agent = [
        Zoo.AgentPPO,  # 2018. PPO2 + GAE, slow but quite stable, especially in high-dim
        Zoo.AgentModPPO,  # 2018+ Reliable Lambda
        Zoo.AgentInterPPO,  # 2019. Integrated Network, useful in pixel-level task (state2D)
    ][1]

    args.net_dim = 2 ** 8
    args.max_memo = 2 ** 12
    args.batch_size = 2 ** 9
    args.repeat_times = 2 ** 4
    args.reward_scale = 2 ** 0  # unimportant hyper-parameter in PPO which do normalization on Q value
    args.gamma = 0.99  # important hyper-parameter, related to episode steps

    # args.env_name = "Pendulum-v0"  # It is easy to reach target score -200.0 (-100 is harder)
    # args.break_step = int(8e4 * 8)  # 5e5 means the average total training step of ModPPO to reach target_reward
    # args.reward_scale = 2 ** 0  # (-1800) -1000 ~ -200 (-50), UsedTime:  (100s) 200s
    # args.gamma = 0.9  # important hyper-parameter, related to episode steps
    # args.init_for_training()
    # train_agent_mp(args)  # train_agent(args)
    # exit()
    #
    # args.env_name = "LunarLanderContinuous-v2"
    # args.break_step = int(3e5 * 8)  # (2e5) 3e5 , used time: (400s) 600s
    # args.reward_scale = 2 ** 0  # (-800) -200 ~ 200 (301)
    # args.gamma = 0.99  # important hyper-parameter, related to episode steps
    # args.init_for_training()
    # train_agent_mp(args)  # train_agent(args)
    # # exit()
    #
    args.env_name = "BipedalWalker-v3"
    args.break_step = int(8e5 * 8)  # (4e5) 8e5 (4e6), UsedTimes: (600s) 1500s (8000s)
    args.reward_scale = 2 ** 0  # (-150) -90 ~ 300 (325)
    args.gamma = 0.95  # important hyper-parameter, related to episode steps
    args.init_for_training()
    train_agent_mp(args)  # train_agent(args)
    exit()
    #
    # import pybullet_envs  # for python-bullet-gym
    # dir(pybullet_envs)
    # args.env_name = "ReacherBulletEnv-v0"
    # args.break_step = int(2e6 * 8)  # (1e6) 2e6 (4e6), UsedTimes: 2000s (6000s)
    # args.reward_scale = 2 ** 0  # (-15) 0 ~ 18 (25)
    # args.gamma = 0.95  # important hyper-parameter, related to episode steps
    # args.init_for_training()
    # train_agent_mp(args)  # train_agent(args)
    # exit()
    #
    # import pybullet_envs  # for python-bullet-gym
    # dir(pybullet_envs)
    # args.env_name = "AntBulletEnv-v0"
    # args.break_step = int(5e6 * 8)  # (1e6) 5e6 UsedTime: 25697s
    # args.reward_scale = 2 ** -3  #
    # args.gamma = 0.99  # important hyper-parameter, related to episode steps
    # args.net_dim = 2 ** 9
    # args.init_for_training()
    # train_agent_mp(args)
    # exit()

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)
    args.env_name = "MinitaurBulletEnv-v0"  # PPO is the best, I don't know why.
    args.break_step = int(1e6 * 8)  # (4e5) 1e6 (8e6)
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
