from AgentRun import *


def run__sac(gpu_id=0, cwd='AC_SAC'):
    from AgentZoo import AgentSAC
    args = Arguments(AgentSAC)
    args.gpu_id = gpu_id

    # args.env_name = "LunarLanderContinuous-v2"
    # args.cwd = './{}/LL_{}'.format(cwd, gpu_id)
    # args.init_for_training()
    # while not train_agent(**vars(args)):
    #     args.random_seed += 42

    # args.env_name = "BipedalWalker-v3"
    # args.cwd = './{}/BW_{}'.format(cwd, gpu_id)
    # args.init_for_training()
    # while not train_agent_sac(**vars(args)):
    #     args.random_seed += 42

    import pybullet_envs  # for python-bullet-gym
    args.env_name = "MinitaurBulletEnv-v0"
    args.cwd = './{}/Minitaur_{}'.format(cwd, args.gpu_id)
    args.max_epoch = 2 ** 13
    args.max_memo = 2 ** 20
    args.net_dim = 2 ** 9
    args.max_step = 2 ** 12
    args.batch_size = 2 ** 8
    args.reward_scale = 2 ** 3
    args.is_remove = True

    args.eva_size = 2 ** 5  # for Recorder
    args.show_gap = 2 ** 8  # for Recorder

    args.init_for_training()
    while not train_agent(**vars(args)):
        args.random_seed += 42


if __name__ == '__main__':
    run__multi_process(run__sac, gpu_tuple=(2, 3), cwd='AC_SAC')
