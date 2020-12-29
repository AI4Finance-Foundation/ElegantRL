import numpy as np
import numpy.random as rd
# import matplotlib.pyplot as plt

from AgentRun import *


def train__rl():
    from beta0 import BeamFormerEnv1524
    env = BeamFormerEnv1524()
    env.r_offset = 1.3  # todo

    from AgentZoo import AgentModSAC
    # AgentModSAC().alpha_log = -5.0

    args = Arguments(rl_agent=AgentModSAC, env=env)
    args.rollout_workers_num = 2  # todo
    args.eval_times1 = 1
    args.eval_times2 = 2
    args.random_seed += 1943

    args.break_step = int(6e4 * 8)  # UsedTime: 900s (reach target_reward 200)
    args.net_dim = 2 ** 7
    args.init_for_training()
    train_agent_mp(args)  # train_agent(args)


if __name__ == '__main__':
    # run__plan()
    # read_print_terminal_data()
    # read_print_terminal_data1()
    train__rl()
