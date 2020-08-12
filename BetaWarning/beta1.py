from AgentRun import *

# from AgentZoo import *
# from AgentNet import *

"""
beta2    np.log(1.0 / action_dim)
beta0    args.batch_size = 2 ** 6 
"""


def run__mp(gpu_id=None, cwd='MP__InterSAC'):
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
        # [p.close() for p in process]
        [p.terminate() for p in process]  # use p.terminate() instead of p.close()
        time.sleep(8)

    import AgentZoo as Zoo
    class_agent = Zoo.AgentInterSAC
    # class_agent = AgentInterSAC

    # args = ArgumentsBeta(class_agent, gpu_id, cwd, env_name="LunarLanderContinuous-v2")
    # build_for_mp()

    # args = ArgumentsBeta(class_agent, gpu_id, cwd, env_name="BipedalWalker-v3")
    # build_for_mp()

    # import pybullet_envs  # for python-bullet-gym
    # dir(pybullet_envs)
    # args = ArgumentsBeta(class_agent, gpu_id, cwd, env_name="AntBulletEnv-v0")
    # args.max_epoch = 2 ** 13
    # args.max_memo = 2 ** 20
    # args.max_step = 2 ** 10
    # args.net_dim = 2 ** 8
    # args.batch_size = 2 ** 9
    # args.reward_scale = 2 ** -2
    # args.eva_size = 2 ** 5  # for Recorder
    # args.show_gap = 2 ** 8  # for Recorder
    # build_for_mp()

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)
    args = ArgumentsBeta(class_agent, gpu_id, cwd, env_name="MinitaurBulletEnv-v0")
    args.max_epoch = 2 ** 13
    args.batch_size = 2 ** 7
    args.max_memo = 2 ** 21
    args.net_dim = 2 ** 8
    args.max_step = 2 ** 11
    args.reward_scale = 2 ** 5
    args.is_remove = True
    args.eva_size = 2 ** 5  # for Recorder
    args.show_gap = 2 ** 8  # for Recorder
    build_for_mp()

    # import pybullet_envs  # for python-bullet-gym
    # dir(pybullet_envs)
    # args = ArgumentsBeta(class_agent, gpu_id, cwd, env_name="BipedalWalkerHardcore-v3")
    # args.max_epoch = 2 ** 13
    # args.max_memo = 2 ** 21
    # args.net_dim = 2 ** 8
    # args.max_step = 2 ** 12
    # args.is_remove = True
    # args.eva_size = 2 ** 5  # for Recorder
    # args.show_gap = 2 ** 8  # for Recorder
    # build_for_mp()


if __name__ == '__main__':
    run__mp()
