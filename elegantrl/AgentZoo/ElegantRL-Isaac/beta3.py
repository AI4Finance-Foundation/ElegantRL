from elegantrl.demo import *

GPU_ID = 3
"""
IP 194
GPU 1 if_train_pendulum
GPU 2 if_train_lunar_lander
GPU 3 if_train_bipedal_walker
"""


def demo_continuous_action_off_policy():  # [ElegantRL.2021.10.10]
    args = Arguments()
    args.agent = AgentModSAC()  # AgentSAC AgentTD3 AgentDDPG

    if_train_pendulum = 1
    if if_train_pendulum:
        "TotalStep: 2e5, TargetReward: -200, UsedTime: 200s"
        # One way to build env
        # args.env = build_env(env='Pendulum-v1')  # gym.__version__ == 0.21.0
        # args.env = build_env(env='Pendulum-v0')  # gym.__version__ == 0.17.0

        # Another way to build env
        args.env = 'Pendulum-v1'  # or 'Pendulum-v0'
        args.env_num = 1
        args.max_step = 200
        args.state_dim = 3
        args.action_dim = 1
        args.if_discrete = False
        args.target_return = -200

        args.gamma = 0.97
        args.worker_num = 2
        args.reward_scale = 2 ** -2
        args.target_step = 200 * 2  # max_step = 200

        args.learner_gpus = (GPU_ID, )  # todo check

        # train_and_evaluate(args)  # single process
        train_and_evaluate_mp(args)  # multiple process

    if_train_lunar_lander = 0
    if if_train_lunar_lander:
        "TotalStep: 4e5, TargetReward: 200, UsedTime:  900s, TD3"
        "TotalStep: 5e5, TargetReward: 200, UsedTime: 1500s, ModSAC"
        args.env = build_env(env='LunarLanderContinuous-v2')
        args.target_step = args.env.max_step
        args.reward_scale = 2 ** 0

        args.eval_times1 = 2 ** 4
        args.eval_times2 = 2 ** 6  # use CPU to draw learning curve

    if_train_bipedal_walker = 0
    if if_train_bipedal_walker:
        "TotalStep: 08e5, TargetReward: 300, UsedTime: 1800s TD3"
        "TotalStep: 11e5, TargetReward: 329, UsedTime: 6000s TD3"
        "TotalStep:  4e5, TargetReward: 300, UsedTime: 2000s ModSAC"
        "TotalStep:  8e5, TargetReward: 330, UsedTime: 5000s ModSAC"
        args.env = build_env(env='BipedalWalker-v3')
        args.target_step = args.env.max_step
        args.gamma = 0.98

        args.eval_times1 = 2 ** 3
        args.eval_times2 = 2 ** 5

    if_train_bipedal_walker_hard_core = 0
    if if_train_bipedal_walker_hard_core:
        "TotalStep: 10e5, TargetReward:   0, UsedTime: 10ks ModSAC"
        "TotalStep: 25e5, TargetReward: 150, UsedTime: 20ks ModSAC"
        "TotalStep: 35e5, TargetReward: 295, UsedTime: 40ks ModSAC"
        "TotalStep: 40e5, TargetReward: 300, UsedTime: 50ks ModSAC"
        args.env = build_env(env='BipedalWalkerHardcore-v3')
        args.target_step = args.env.max_step
        args.gamma = 0.98
        args.net_dim = 2 ** 8
        args.batch_size = args.net_dim * 2
        args.learning_rate = 2 ** -15
        args.repeat_times = 1.5

        args.max_memo = 2 ** 22
        args.break_step = 2 ** 24

        args.eval_gap = 2 ** 8
        args.eval_times1 = 2 ** 2
        args.eval_times2 = 2 ** 5

        args.target_step = args.env.max_step * 1

    # args.init_before_training()  # necessary!
    #
    # train_and_evaluate(args)  # single process
    # args.worker_num = 4
    # args.visible_gpu = sys.argv[-1]
    # train_and_evaluate_mp(args)  # multiple process
    # args.worker_num = 6
    # args.visible_gpu = '0,1'
    # train_and_evaluate_mp(args)  # multiple GPU


if __name__ == '__main__':
    demo_continuous_action_off_policy()
