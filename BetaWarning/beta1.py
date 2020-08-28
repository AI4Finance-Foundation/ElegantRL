from AgentPixel import *


def run__car_racing(gpu_id=None):
    print('pixel-level state')

    """run online policy"""
    args = Arguments(rl_agent=AgentGAE, gpu_id=gpu_id)
    args.env_name = "CarRacing-v0"
    args.max_total_step = int(2e6 * 1)
    args.max_memo = 2 ** 10  # 11
    args.batch_size = 2 ** 7  # 8
    args.repeat_times = 2 ** 4
    args.net_dim = 2 ** 8
    args.gamma = 0.99
    args.random_seed = 1942
    args.max_total_step = int(1e6 * 4)
    args.max_step = int(2000)
    args.eva_size = 3
    args.reward_scale = 2 ** -1
    args.show_gap = 2 ** 8  # for Recorder
    args.init_for_training()
    train_agent(**vars(args))


run__car_racing()
