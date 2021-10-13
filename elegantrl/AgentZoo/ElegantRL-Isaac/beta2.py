from elegantrl.demo import *

"""run 2021-10-13"""


def demo_continuous_action_on_policy():  # [ElegantRL.2021.10.13]
    args = Arguments()
    args.agent = AgentPPO()
    args.learner_gpus = (GPU_ID,)  # todo check
    args.eval_gpu_id = (GPU_ID,)  # todo check

    env_name = ENV_NAME  # todo check
    if env_name in {'Pendulum-v1', 'Pendulum-v0'}:
        """
        Step: 45e4, Reward: -138, UsedTime: 373s PPO
        Step: 40e4, Reward: -200, UsedTime: 400s PPO
        Step: 46e4, Reward: -213, UsedTime: 300s PPO
        """
        # One way to build env
        args.env = build_env(env=env_name)

        # Another way to build env
        # args.env = env_name  # 'Pendulum-v1' or 'Pendulum-v0'
        # args.env_num = 1
        # args.max_step = 200
        # args.state_dim = 3
        # args.action_dim = 1
        # args.if_discrete = False
        # args.target_return = -200

        args.gamma = 0.97
        args.net_dim = 2 ** 8
        args.worker_num = 2
        args.reward_scale = 2 ** -2
        args.target_step = 200 * 16  # max_step = 200
    if env_name in {'LunarLanderContinuous-v2', 'LunarLanderContinuous-v1'}:
        """
        Step: 80e4, Reward: 246, UsedTime: 3000s PPO
        """
        args.env = build_env(env=env_name)
        args.eval_times1 = 2 ** 4
        args.eval_times2 = 2 ** 6

        args.target_step = args.env.max_step * 8
    if env_name in {'BipedalWalker-v3', 'BipedalWalker-v2'}:
        """
        Step: 57e5, Reward: 295, UsedTime: 17ks PPO
        Step: 70e5, Reward: 300, UsedTime: 21ks PPO
        """
        args.env = build_env(env=env_name)
        args.eval_times1 = 2 ** 3
        args.eval_times2 = 2 ** 5

        args.gamma = 0.98
        args.target_step = args.env.max_step * 16
    if env_name in {'BipedalWalkerHardcore-v3', 'BipedalWalkerHardcore-v2'}:
        """
        Step: 57e5, Reward: 295, UsedTime: 17ks PPO
        Step: 70e5, Reward: 300, UsedTime: 21ks PPO
        """
        args.env = build_env(env=env_name)

        args.gamma = 0.98
        args.net_dim = 2 ** 8
        args.max_memo = 2 ** 22
        args.batch_size = args.net_dim * 4
        args.repeat_times = 2 ** 4
        args.learning_rate = 2 ** -16

        args.eval_gap = 2 ** 8
        args.eval_times1 = 2 ** 2
        args.eval_times2 = 2 ** 5
        # args.break_step = int(80e5)

        args.worker_num = WORKER_NUM
        args.target_step = args.env.max_step * 16

    # args.learner_gpus = (0, )  # single GPU
    # args.learner_gpus = (0, 1)  # multiple GPUs
    # train_and_evaluate(args)  # single process
    train_and_evaluate_mp(args)  # multiple process


"""
IP83
GPU 0 BipedalWalkerHardcore WORKER_NUM=4 GAE CriTarget   6e7,  15  14e7, 115  21e7,
GPU 4 BipedalWalkerHardcore WORKER_NUM=4                 6e7,  30  21e7, 156  26e7,

IP194
GPU 2 BipedalWalkerHardcore WORKER_NUM=4 GAE             7e7,  15  18e7, 101  23e7, 149
GPU 3 BipedalWalkerHardcore WORKER_NUM=4 max_step * 16                 
GPU 4 BipedalWalkerHardcore WORKER_NUM=8 max_step * 16                 
"""

if __name__ == '__main__':
    # sys.argv.extend(['WORK GPU ENV', '2', '3', '0'])
    # sys.argv.extend(['GPU ENV', '3', '0'])

    WORKER_NUM = eval(sys.argv[-2])
    GPU_ID = eval(sys.argv[-1])
    ENV_NAME = ['Pendulum-v1',
                'LunarLanderContinuous-v2',
                'BipedalWalker-v3',
                'BipedalWalkerHardcore-v3',
                ][3]
    print(f"| WORKER_NUM   {WORKER_NUM}")
    print(f"| GPU_ID       {GPU_ID}")

    demo_continuous_action_on_policy()
