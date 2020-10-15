from AgentRun import *
from AgentNet import *
from AgentZoo import *

"""     Minitaur
beta14  ModSAC (best, 1.17e+06     15.00 |   15.33      1.44 |   10.81     21371  ########)
InterSAC (0    1.25e+06     13.05 |   12.69      2.41 |    9.54    -66.44      0.06
ModSAC   (2    3.98e+06     11.07 |    2.74      3.53 |    2.99    -16.21      0.02

beta0   clip_grad_norm_(self.act.parameters(), 1.0)
beta10  clip_grad_norm_(self.act.parameters(), 0.5)
"""


def run_continuous_action(gpu_id=None):
    rl_agent = AgentMixSAC
    args = Arguments(rl_agent, gpu_id)
    args.if_break_early = False
    args.if_remove_history = True

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)
    args.env_name = "AntBulletEnv-v0"
    args.break_step = int(1e6 * 8)  # (8e5) 10e5
    args.reward_scale = 2 ** -3  # (-50) 0 ~ 2500 (3340)
    args.max_step = 2 ** 11  # todo
    args.batch_size = 2 ** 8
    args.max_memo = 2 ** 20
    args.eva_size = 2 ** 3  # for Recorder
    args.show_gap = 2 ** 8  # for Recorder
    args.init_for_training(8)
    train_agent_mp(args)  # train_agent(**vars(args))
    exit()

    args.env_name = "BipedalWalkerHardcore-v3"  # 2020-08-24 plan
    args.reward_scale = 2 ** 0  # (-200) -150 ~ 300 (335)
    args.break_step = int(4e6 * 8)  # (2e6) 4e6
    args.net_dim = int(2 ** 8)  # int(2 ** 8.5) #
    args.max_step = 2 ** 11  # todo
    args.max_memo = int(2 ** 21)
    args.batch_size = int(2 ** 8)
    args.eval_times2 = 2 ** 5  # for Recorder
    args.show_gap = 2 ** 8  # for Recorder
    args.init_for_training()
    train_agent_mp(args)  # train_offline_policy(**vars(args))
    exit()


run_continuous_action()
