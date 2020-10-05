from AgentRun import *
from AgentNet import *
from AgentZoo import *

"""     Minitaur
beta14  ModSAC (best, 1.17e+06     15.00 |   15.33      1.44 |   10.81     21371  ########)
InterSAC (1    4.00e+06     13.17 |   11.26      1.22 |    8.79    -33.22      0.06)
ModSAC   (2    3.98e+06     11.07 |    2.74      3.53 |    2.99    -16.21      0.02

beta1   ModSAC      args.reward_scale = 2 ** 4 (if_norm)
beta14  InterSAC    args.reward_scale = 2 ** 4 (if_norm)
beta10  PPO
beta0   PPO
beta11  GAE
beta12  InterSAC    anchor
"""


def run_continuous_action(gpu_id=None):
    rl_agent = AgentPPO
    args = Arguments(rl_agent, gpu_id)
    args.if_break_early = False
    args.if_remove_history = True

    args.net_dim = 2 ** 8
    args.max_memo = 2 ** 12
    args.batch_size = 2 ** 9
    args.repeat_times = 2 ** 4

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)
    args.env_name = "MinitaurBulletEnv-v0"
    args.break_step = int(1e6 * 8)
    args.reward_scale = 2 ** 4
    args.net_dim = 2 ** 9
    args.init_for_training()
    train_agent(**vars(args))
    exit()




run_continuous_action()
