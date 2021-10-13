from elegantrl.demo import *

"""delete 2021-10-12"""


def demo_continuous_action_off_policy():  # [ElegantRL.2021.10.12]
    args = Arguments()
    args.agent = AgentModSAC()  # AgentSAC AgentTD3 AgentDDPG
    args.learner_gpus = (GPU_ID,)  # todo check
    args.eval_gpu_id = (GPU_ID,)  # todo check

    env_name = ENV_NAME  # todo check
    if env_name in {'Pendulum-v1', 'Pendulum-v0'}:
        "TotalStep: 2e5, TargetReward: -200, UsedTime: 200s, ModSAC"
        # One way to build env
        # args.env = build_env(env=env_name)
        # Pendulum-v1  gym.__version__ == 0.21.0
        # Pendulum-v0  gym.__version__ == 0.17.0

        # Another way to build env
        args.env = env_name  # 'Pendulum-v1' or 'Pendulum-v0'
        args.env_num = 1
        args.max_step = 200
        args.state_dim = 3
        args.action_dim = 1
        args.if_discrete = False
        args.target_return = -200

        args.gamma = 0.97
        args.net_dim = 2 ** 7
        args.worker_num = 2
        args.reward_scale = 2 ** -2
        args.target_step = 200 * 4  # max_step = 200
    if env_name in {'LunarLanderContinuous-v2', 'LunarLanderContinuous-v1'}:
        "TotalStep: 4e5, TargetReward: 200, UsedTime:  900s, TD3"
        "TotalStep: 5e5, TargetReward: 200, UsedTime: 1500s, ModSAC"
        args.env = build_env(env=env_name)
        args.eval_times1 = 2 ** 4
        args.eval_times2 = 2 ** 6

        args.target_step = args.env.max_step
    if env_name in {'BipedalWalker-v3', 'BipedalWalker-v2'}:
        "TotalStep: 08e5, TargetReward: 300, UsedTime: 1800s TD3"
        "TotalStep: 11e5, TargetReward: 329, UsedTime: 6000s TD3"
        "TotalStep:  4e5, TargetReward: 300, UsedTime: 2000s ModSAC"
        "TotalStep:  8e5, TargetReward: 330, UsedTime: 5000s ModSAC"
        args.env = build_env(env=env_name)
        args.eval_times1 = 2 ** 3
        args.eval_times2 = 2 ** 5

        args.target_step = args.env.max_step
        args.gamma = 0.98
    if env_name in {'BipedalWalkerHardcore-v3', 'BipedalWalkerHardcore-v2'}:
        "TotalStep: 10e5, TargetReward:   0, UsedTime: 10ks ModSAC"
        "TotalStep: 25e5, TargetReward: 150, UsedTime: 20ks ModSAC"
        "TotalStep: 35e5, TargetReward: 295, UsedTime: 40ks ModSAC"
        "TotalStep: 40e5, TargetReward: 300, UsedTime: 50ks ModSAC"
        args.env = build_env(env=env_name)
        args.target_step = args.env.max_step
        args.gamma = 0.98
        args.net_dim = 2 ** 8
        args.batch_size = args.net_dim * 2
        args.learning_rate = 2 ** -15
        args.repeat_times = 1.5

        args.max_memo = 2 ** 22
        args.break_step = int(80e6)

        args.eval_gap = 2 ** 9
        args.eval_times1 = 2 ** 2
        args.eval_times2 = 2 ** 5

        args.worker_num = 4
        args.target_step = args.env.max_step * 1

    # args.learner_gpus = (0, )  # single GPU
    # args.learner_gpus = (0, 1)  # multiple GPUs
    # train_and_evaluate(args)  # single process
    train_and_evaluate_mp(args)  # multiple process


"""
IP 194
GPU 1 if_train_pendulum
GPU 2 if_train_lunar_lander                 40e4, 200, 2686s
GPU 3 if_train_bipedal_walker               49e4, 322, 2977s

GPU 3 BipedalWalkerHardcore WORKER_NUM=4    
GPU 4 BipedalWalkerHardcore WORKER_NUM=2     8e5,  13  16e5, 136  23e5, 219  38e5, 302, 99ks
GPU 5 BipedalWalkerHardcore WORKER_NUM=4    14e5,  15  18e5, 117  28e5, 212  45e5, 306, 67ks
"""

if __name__ == '__main__':
    # sys.argv.extend(['WORK GPU ENV', '2', '3', '0'])

    GPU_ID = eval(sys.argv[-2])
    ENV_NAME = ['Pendulum-v1',
                'LunarLanderContinuous-v2',
                'BipedalWalker-v3',
                'BipedalWalkerHardcore-v3',
                ][eval(sys.argv[-1])]
    print(f"| GPU_ID       {GPU_ID}")
    print(f"| ENV_NAME     {ENV_NAME}")

    demo_continuous_action_off_policy()
