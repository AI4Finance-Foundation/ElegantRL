import numpy as np
import numpy.random as rd
# import matplotlib.pyplot as plt
from beta0 import *


def train__rl():
    antennas_num = 8
    user_num = 4
    noise_y_std = 1.0
    # noise_h_std = 1.0
    max_power = 1.0

    # snr_db_l = np.linspace(-10, 20, 7)  # 13
    # print(snr_db_l.round(3))
    # power__l = 10 ** (snr_db_l / 10) * (noise_y_std ** 2)
    # print(power__l.round(3))

    # from beta0 import BeamFormerEnv
    for noise_h_std in (0.0, 0.5):  # todo
        print(f'\n| noise_h_std:{noise_h_std:.2f} \n')
        env = BeamFormerEnvNoisy(antennas_num, user_num, max_power, noise_h_std, noise_y_std)

        from AgentZoo import AgentModSAC
        # AgentModSAC().alpha_log = -5.0

        args = Arguments(rl_agent=AgentModSAC, env=env)
        args.rollout_workers_num = 4
        args.eval_times1 = 1
        args.eval_times2 = 2

        args.break_step = int(6e4 * 8)  # UsedTime: 900s (reach target_reward 200)
        args.net_dim = 2 ** 7
        args.init_for_training()
        train_agent_mp(args)  # train_agent(args)


if __name__ == '__main__':
    # run__plan()
    # read_print_terminal_data()
    # read_print_terminal_data1()
    train__rl()
