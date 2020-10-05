from AgentRun import *
from AgentNet import *
from AgentZoo import *

"""     Minitaur
beta14  ModSAC (best, 1.17e+06     15.00 |   15.33      1.44 |   10.81     21371  ########)
beta0   InterSAC
        self.target_entropy = np.log(action_dim + 1) * 0.02  # todo
beta1   InterSAC
beta2   ModSAC
        self.target_entropy = np.log(action_dim + 1) * 0.02  # todo
"""


def run_continuous_action(gpu_id=None):
    rl_agent = AgentModSAC
    args = Arguments(rl_agent, gpu_id)
    args.if_break_early = False
    args.if_remove_history = True

    # import pybullet_envs  # for python-bullet-gym
    # dir(pybullet_envs)
    # args.env_name = "AntBulletEnv-v0"
    # args.break_step = int(5e6 * 4)
    # args.reward_scale = 2 ** -3
    # args.batch_size = 2 ** 8
    # args.max_memo = 2 ** 20
    # args.eva_size = 2 ** 3  # for Recorder
    # args.show_gap = 2 ** 8  # for Recorder
    # args.init_for_training()
    # train_agent_mp(args)  # train_offline_policy(**vars(args))
    # exit()

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)
    args.env_name = "MinitaurBulletEnv-v0"
    args.break_step = int(1e6 * 4)
    args.reward_scale = 2 ** 4
    args.batch_size = 2 ** 8
    args.max_memo = 2 ** 21
    args.net_dim = 2 ** 8
    args.eval_times1 = 2 ** 2  # for Recorder
    args.eval_times2 = 2 ** 4  # for Recorder
    args.show_gap = 2 ** 9  # for Recorder
    args.init_for_training(cpu_threads=8)
    train_agent_mp(args)  # train_agent(**vars(args))
    exit()


run_continuous_action()
