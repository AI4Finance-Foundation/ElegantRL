from AgentRun import *
from AgentNet import *
from AgentZoo import *



def train__car_racing(gpu_id=None, random_seed=0):
    print('pixel-level state')
    rl_agent = (AgentModPPO, AgentInterPPO)[1]  # choose DRl algorithm.
    args = Arguments(rl_agent=rl_agent, gpu_id=gpu_id)
    args.if_break_early = True
    args.eval_times2 = 2
    args.eval_times2 = 3

    args.env_name = "CarRacing-v0"
    args.random_seed = 1943 + random_seed
    args.break_step = int(5e5 * 4)  # (2e5) 5e5, used time 25000s
    args.reward_scale = 2 ** -2  # (-1) 80 ~ 900 (1001)
    args.max_memo = 2 ** 11
    args.batch_size = 2 ** 7
    args.repeat_times = 2 ** 4
    args.net_dim = 2 ** 7
    args.max_step = 2 ** 10
    args.show_gap = 2 ** 8  # for Recorder
    args.init_for_training()
    train_agent_mp(args)  # train_agent(**vars(args))


if __name__ == '__main__':
    # test_conv2d()
    # test_car_racing()
    train__car_racing(random_seed=321)
