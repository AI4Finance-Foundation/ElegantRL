from AgentRun import *
from AgentZoo import *
from AgentNet import *

"""
    args.net_dim = 2 ** 7 # todo beta0

"""


def run__mp(gpu_id=None):
    import multiprocessing as mp
    q_i_buf = mp.Queue(maxsize=8)  # buffer I
    q_o_buf = mp.Queue(maxsize=8)  # buffer O
    q_i_eva = mp.Queue(maxsize=8)  # evaluate I
    q_o_eva = mp.Queue(maxsize=8)  # evaluate O

    def build_for_mp():
        process = [mp.Process(target=mp__update_params, args=(args, q_i_buf, q_o_buf, q_i_eva, q_o_eva)),
                   mp.Process(target=mp__update_buffer, args=(args, q_i_buf, q_o_buf,)),
                   mp.Process(target=mp_evaluate_agent, args=(args, q_i_eva, q_o_eva)), ]
        [p.start() for p in process]
        [p.join() for p in process]
        print('\n')

    # import AgentZoo as Zoo
    # args = Arguments(rl_agent=Zoo.AgentInterSAC, gpu_id=gpu_id)
    args = Arguments(rl_agent=AgentInterSAC, gpu_id=gpu_id)
    args.is_remove = False
    args.random_seed = 19432

    # args.env_name = "LunarLanderContinuous-v2"
    # args.max_total_step = int(1e5 * 8)
    # args.init_for_training()
    # train_offline_policy(**vars(args))  # build_for_mp()
    #
    # args.env_name = "BipedalWalker-v3"
    # args.max_total_step = int(2e5 * 8)
    # args.init_for_training()
    # build_for_mp()

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)
    args.env_name = "AntBulletEnv-v0"
    args.max_total_step = int(5e5 * 8)
    args.max_epoch = 2 ** 13
    args.max_memo = 2 ** 20
    args.max_step = 2 ** 10
    args.net_dim = 2 ** 7  # todo beta0
    args.batch_size = 2 ** 9
    args.reward_scale = 2 ** -2
    args.eva_size = 2 ** 5  # for Recorder
    args.show_gap = 2 ** 8  # for Recorder
    args.init_for_training()
    build_for_mp()

    # import pybullet_envs  # for python-bullet-gym
    # dir(pybullet_envs)
    # args.env_name = "MinitaurBulletEnv-v0" # todo bad 3.0 score
    # args.max_total_step = int(2e6 * 8)
    # args.max_epoch = 2 ** 13
    # args.max_memo = 2 ** 21
    # args.max_step = 2 ** 10
    # args.batch_size = 2 ** 8
    # args.reward_scale = 2 ** 4
    # args.eva_size = 2 ** 3  # for Recorder
    # args.show_gap = 2 ** 8  # for Recorder
    # args.init_for_training()
    # train_offline_policy(**vars(args))  # build_for_mp()


if __name__ == '__main__':
    run__mp()
