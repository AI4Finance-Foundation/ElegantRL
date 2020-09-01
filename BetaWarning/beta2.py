from AgentPixel import *

"""
GAE
2    1.87e+05    215.43 |   37.06      2.67 |  189.52     -0.07      0.06  # expR > 100
2    3.67e+05    307.46 |  221.76     16.09 |  516.27     -0.10      0.16  # evaR > 300
"""


def run__car_racing(gpu_id=None):
    print('pixel-level state')

    """run online policy"""
    args = Arguments(rl_agent=AgentGAE, gpu_id=gpu_id)
    args.env_name = "CarRacing-v0"
    args.random_seed = 1943
    args.max_total_step = int(2e6 * 1)
    args.max_memo = 2 ** 11
    args.batch_size = 2 ** 9  # todo beta2 and (1, 96, 96)
    args.repeat_times = 2 ** 4
    args.net_dim = 2 ** 7
    args.gamma = 0.99
    args.random_seed = 1942
    args.max_total_step = int(1e6 * 4)
    args.max_step = int(1000)
    args.eva_size = 3
    args.reward_scale = 2 ** -1
    args.show_gap = 2 ** 8  # for Recorder
    args.init_for_training()
    train_agent(**vars(args))
    # build_for_mp(args)


run__car_racing()
