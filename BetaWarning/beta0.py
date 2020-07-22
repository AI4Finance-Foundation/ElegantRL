from AgentRun import *


def run__mp(gpu_id=None, cwd='MP__IntelAC'):
    import AgentZoo as Zoo
    args = Arguments()
    args.class_agent = Zoo.AgentInterAC
    args.gpu_id = gpu_id if gpu_id is not None else sys.argv[-1][-4]
    assert args.class_agent in {
        Zoo.AgentDDPG, Zoo.AgentTD3, Zoo.ActorSAC, Zoo.AgentDeepSAC,
        Zoo.AgentBasicAC, Zoo.AgentSNAC, Zoo.AgentInterAC, Zoo.AgentInterSAC,
    }

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
        [p.close() for p in process]

    args.env_name = "LunarLanderContinuous-v2"
    args.cwd = f'./{cwd}/{args.env_name}_{gpu_id}'
    build_for_mp()

    args.env_name = "BipedalWalker-v3"
    args.cwd = f'./{cwd}/{args.env_name}_{gpu_id}'
    build_for_mp()

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)
    args.env_name = "AntBulletEnv-v0"
    args.cwd = f'./{cwd}/{args.env_name}_{gpu_id}'
    args.max_epoch = 2 ** 13
    args.max_memo = 2 ** 20
    args.max_step = 2 ** 10
    args.net_dim = 2 ** 8
    args.batch_size = 2 ** 9
    args.reward_scale = 2 ** -2
    args.is_remove = True
    args.eva_size = 2 ** 5  # for Recorder
    args.show_gap = 2 ** 8  # for Recorder
    build_for_mp()

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)
    args.env_name = "MinitaurBulletEnv-v0"
    args.cwd = f'./{cwd}/{args.env_name}_{gpu_id}'
    args.max_epoch = 2 ** 13
    args.max_memo = 2 ** 20
    args.net_dim = 2 ** 8
    args.max_step = 2 ** 10
    args.batch_size = 2 ** 9
    args.reward_scale = 2 ** 3
    args.is_remove = True
    args.eva_size = 2 ** 5  # for Recorder
    args.show_gap = 2 ** 8  # for Recorder
    build_for_mp()


run__mp()
