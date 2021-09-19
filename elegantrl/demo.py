import sys
from elegantrl.agent import *
from elegantrl.env import PreprocessEnv, build_env
from elegantrl.run import Arguments, train_and_evaluate, train_and_evaluate_mp

"""[ElegantRL.2021.09.18](https://github.com/AI4Finance-LLC/ElegantRL)"""


def demo_continuous_action_off_policy():  # 2021-09-07
    args = Arguments(if_on_policy=False)
    args.agent = AgentModSAC()  # AgentSAC AgentTD3 AgentDDPG

    if_train_pendulum = 1
    if if_train_pendulum:
        "TotalStep: 2e5, TargetReward: -200, UsedTime: 200s"
        import gym
        args.env = PreprocessEnv(env=gym.make('Pendulum-v0'))
        args.env.target_return = -200  # set target_reward manually for env 'Pendulum-v0'
        # args.env = PreprocessEnv(env='Pendulum-v0')  # It is Ok.
        # args.env = build_env(env='Pendulum-v0')  # It is Ok.
        args.reward_scale = 2 ** -2
        args.gamma = 0.97

        args.worker_num = 2
        args.target_step = args.env.max_step * 2

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

    # train_and_evaluate(args)  # single process
    args.worker_num = 4
    args.visible_gpu = sys.argv[-1]
    train_and_evaluate_mp(args)  # multiple process
    # args.worker_num = 6
    # args.visible_gpu = '0,1'
    # train_and_evaluate_mp(args)  # multiple GPU


def demo_continuous_action_on_policy():
    args = Arguments(if_on_policy=True)  # hyper-parameters of on-policy is different from off-policy
    args.agent = AgentPPO()
    args.agent.cri_target = True
    args.visible_gpu = '0'  # sys.argv[-1]

    if_train_pendulum = 1
    if if_train_pendulum:
        "TotalStep: 4e5, TargetReward: -200, UsedTime: 400s"
        import gym
        args.env = PreprocessEnv(env=gym.make('Pendulum-v0'))
        args.env.target_return = -200  # set target_reward manually for env 'Pendulum-v0'
        # args.env = PreprocessEnv(env='Pendulum-v0')  # It is Ok.
        # args.env = build_env(env='Pendulum-v0')  # It is Ok.
        args.reward_scale = 2 ** -2  # RewardRange: -1800 < -200 < -50 < 0

        args.gamma = 0.97
        args.net_dim = 2 ** 7
        args.batch_size = args.net_dim * 2

        args.target_step = args.env.max_step * 8

    if_train_lunar_lander = 0
    if if_train_lunar_lander:
        "TotalStep: 4e5, TargetReward: 200, UsedTime: 2000s, TD3"
        args.env = build_env(env='LunarLanderContinuous-v2')
        args.gamma = 0.99
        args.break_step = int(4e6)

        args.target_step = args.env.max_step * 8

    if_train_bipedal_walker = 0
    if if_train_bipedal_walker:
        "TotalStep: 8e5, TargetReward: 300, UsedTime: 1800s"
        args.env = build_env(env='BipedalWalker-v3')

        args.gamma = 0.98
        args.if_per_or_gae = True
        args.break_step = int(8e6)

        args.target_step = args.env.max_step * 16

    # train_and_evaluate(args)
    args.worker_num = 4
    train_and_evaluate_mp(args)


def demo_discrete_action_off_policy():
    args = Arguments(if_on_policy=False)
    args.agent = AgentD3QN()  # AgentD3QN AgentDuelDQN AgentDoubleDQN AgentDQN
    args.visible_gpu = '3,1'

    if_train_cart_pole = 0
    if if_train_cart_pole:
        "TotalStep: 5e4, TargetReward: 200, UsedTime: 60s, D3QN"
        args.env = build_env('CartPole-v0', if_print=True)
        args.reward_scale = 2 ** -1
        args.target_step = args.env.max_step * 8

    if_train_lunar_lander = 1
    if if_train_lunar_lander:
        "TotalStep: 6e5, TargetReturn: 200, UsedTime: 1500s, LunarLander-v2, DQN"
        """
        0  4.14e+05  202.48 |  202.48   67.0    602   189 |    0.18   0.40  12.58    | UsedTime: 2054 | D3QN
        0  6.08e+05  240.13 |  240.13   14.4    384    65 |    0.39   0.14  11.12    | UsedTime: 2488 | D3QN
        0  2.75e+05  218.51 |  218.51   19.6    442    46 |    0.11   0.50  12.92    | UsedTime: 1991 | D3QN
        0  6.08e+05  240.13 |  240.13   14.4    384    65 |    0.39   0.14  11.12    | UsedTime: 2488 | D3QN 2GPU
        """
        args.env = build_env(env='LunarLander-v2', if_print=True)
        args.target_step = args.env.max_step
        args.max_memo = 2 ** 19
        args.repeat_times = 2 ** 1

    # train_and_evaluate(args)
    args.worker_num = 4
    args.learning_rate = 2 ** -16
    args.target_step = args.env.max_step // 2
    train_and_evaluate_mp(args)


def demo_discrete_action_on_policy():
    args = Arguments(if_on_policy=True)  # hyper-parameters of on-policy is different from off-policy
    args.agent = AgentDiscretePPO()
    args.visible_gpu = '0'

    if_train_cart_pole = 1
    if if_train_cart_pole:
        "TotalStep: 5e4, TargetReward: 200, UsedTime: 60s"
        args.env = build_env('CartPole-v0')
        args.reward_scale = 2 ** -1
        args.target_step = args.env.max_step * 8

    if_train_lunar_lander = 0
    if if_train_lunar_lander:
        "TotalStep: 6e5, TargetReturn: 200, UsedTime: 1500s, LunarLander-v2, PPO"
        args.env = build_env(env='LunarLander-v2')
        args.repeat_times = 2 ** 5
        args.if_per_or_gae = True

    train_and_evaluate(args)


def demo_pybullet_off_policy():
    args = Arguments(if_on_policy=False)  # hyper-parameters of on-policy is different from off-policy
    args.agent = AgentModSAC()
    args.visible_gpu = sys.argv[-1]  # '0'
    args.random_seed += 1943161

    if_train_ant = 0
    if if_train_ant:
        args.env = build_env(env='AntBulletEnv-v0', if_print=True)
        args.agent.if_use_act_target = False

        args.break_step = int(8e7)

        args.reward_scale = 2 ** -3
        args.net_dim = 2 ** 9
        args.batch_size = args.net_dim * 2
        args.target_step = args.env.max_step * 2

        args.break_step = int(2e7)
        args.if_allow_break = False

        args.eval_gap = 2 ** 8
        args.eval_times1 = 2 ** 1
        args.eval_times2 = 2 ** 4

    if_train_humanoid = 0
    if if_train_humanoid:
        args.env = build_env(env='HumanoidBulletEnv-v0', if_print=True)

        args.agent.lambda_entropy = 0.02
        args.agent.lambda_gae_adv = 0.97
        args.learning_rate = 2 ** -14
        args.if_per_or_gae = True
        args.break_step = int(8e7)

        args.reward_scale = 2 ** -1
        args.repeat_times = 2 ** 3
        args.net_dim = 2 ** 9
        args.batch_size = args.net_dim * 2 ** 3
        args.target_step = args.env.max_step * 4

        args.break_step = int(8e7)
        args.if_allow_break = False

    if_train_reacher = 1
    if if_train_reacher:
        args.env = build_env(env='ReacherBulletEnv-v0', if_print=True)

        args.explore_rate = 0.9
        args.learning_rate = 2 ** -15

        args.gamma = 0.99
        args.reward_scale = 2 ** 2
        args.break_step = int(4e7)

        args.net_dim = 2 ** 8
        args.batch_size = args.net_dim * 2
        args.repeat_times = 2 ** 0

        args.target_step = args.env.max_step * 4

        args.eval_gap = 2 ** 8
        args.eval_times1 = 2 ** 3
        args.eval_times2 = 2 ** 5

    # train_and_evaluate(args)
    args.worker_num = 4
    train_and_evaluate_mp(args)


def demo_pybullet_on_policy():
    args = Arguments(if_on_policy=True)  # hyper-parameters of on-policy is different from off-policy
    args.agent = AgentPPO()
    args.visible_gpu = '0'
    args.random_seed += 1943

    if_train_ant = 1
    if if_train_ant:
        args.env = build_env(env='AntBulletEnv-v0', if_print=True)
        """
        0  1.98e+07 3322.16 | 3322.16   48.7    999     0 |    0.78   0.48  -0.01  -0.80 | UsedTime: 12380 PPO
        0  1.99e+07 3104.05 | 3071.44   14.5    999     0 |    0.74   0.47   0.01  -0.79 | UsedTime: 12976
        0  1.98e+07 3246.79 | 3245.98   25.3    999     0 |    0.75   0.48  -0.02  -0.81 | UsedTime: 13170
        0  1.97e+07 3345.48 | 3345.48   29.0    999     0 |    0.80   0.49  -0.01  -0.81 | UsedTime: 8169  PPO 2GPU
        0  1.98e+07 3028.69 | 3004.67   10.3    999     0 |    0.72   0.48   0.05  -0.82 | UsedTime: 8734  PPO 2GPU
        """

        args.agent.lambda_entropy = 0.05
        args.agent.lambda_gae_adv = 0.97
        args.learning_rate = 2 ** -15
        args.if_per_or_gae = True
        args.break_step = int(8e7)

        args.reward_scale = 2 ** -2  # (-50) 0 ~ 2500 (3340)
        args.repeat_times = 2 ** 3
        args.net_dim = 2 ** 9
        args.batch_size = args.net_dim * 2 ** 3
        args.target_step = args.env.max_step * 2

        args.break_step = int(2e7)
        args.if_allow_break = False

    if_train_humanoid = 0
    if if_train_humanoid:
        args.env = build_env(env='HumanoidBulletEnv-v0', if_print=True)
        """
        0  2.00e+07 2049.87 | 1905.57  686.5    883   308 |    0.93   0.42  -0.02  -1.14 | UsedTime: 15292
        0  3.99e+07 2977.80 | 2611.64  979.6    879   317 |    1.29   0.46  -0.01  -1.16 | UsedTime: 19685
        0  7.99e+07 3047.88 | 3041.95   41.1    999     0 |    1.37   0.46  -0.04  -1.15 | UsedTime: 38693
        """

        args.agent.lambda_entropy = 0.02
        args.agent.lambda_gae_adv = 0.97
        args.learning_rate = 2 ** -14
        args.if_per_or_gae = True
        args.break_step = int(8e7)

        args.reward_scale = 2 ** -1
        args.repeat_times = 2 ** 3
        args.net_dim = 2 ** 9
        args.batch_size = args.net_dim * 2 ** 3
        args.target_step = args.env.max_step * 4

        args.break_step = int(8e7)
        args.if_allow_break = False

    if_train_reacher = 0
    if if_train_reacher:
        args.env = build_env(env='ReacherBulletEnv-v0', if_print=True)
        "TotalStep: 1e5, TargetReturn: 18, UsedTime:  3ks, PPO eval_times < 4"
        "TotalStep: 1e6, TargetReturn: 18, UsedTime: 30ks, PPO eval_times < 4"
        '''The probability of the following results is only 25%.
        0  5.00e+05    3.23 |    3.23   12.6    149     0 |   -0.03   0.64  -0.03  -0.51
        0  3.55e+06    7.69 |    7.69   10.3    149     0 |   -0.19   0.56  -0.04  -0.59
        0  5.07e+06    9.72 |    7.89    7.6    149     0 |    0.27   0.24   0.02  -0.71
        0  6.85e+06   11.89 |    6.52   12.3    149     0 |    0.22   0.18  -0.06  -0.85
        0  7.87e+06   18.59 |   18.59    9.4    149     0 |    0.39   0.18  -0.01  -0.94

        0  1.01e+06   -2.19 |   -7.30   10.9    149     0 |   -0.05   0.70   0.03  -0.52
        0  4.05e+06    9.29 |   -1.86   15.0    149     0 |    0.08   0.28  -0.05  -0.65
        0  4.82e+06   11.12 |   11.12   12.4    149     0 |    0.13   0.26  -0.07  -0.71
        0  6.07e+06   15.66 |   15.66   11.1    149     0 |    0.16   0.14   0.00  -0.81
        0  9.46e+06   18.58 |   18.58    8.2    149     0 |    0.19   0.10  -0.06  -1.09

        0  2.20e+06    3.63 |    3.26    7.3    149     0 |   -0.05   0.43  -0.01  -0.55
        0  4.19e+06    5.24 |    4.60    9.2    149     0 |    0.04   0.24   0.00  -0.66
        0  6.16e+06    5.24 |    4.80    9.2    149     0 |    0.03   0.15  -0.00  -0.81
        0  7.40e+06   12.99 |   12.99   13.2    149     0 |    0.07   0.19  -0.03  -0.91
        0  1.01e+07   18.09 |   18.09    7.6    149     0 |    0.18   0.16  -0.00  -1.09

        0  1.06e+06    3.25 |    3.25    7.6    149     0 |   -0.21   0.72  -0.05  -0.51 
        0  2.13e+06    3.56 |    0.94    6.1    149     0 |    0.08   0.54   0.02  -0.56
        0  5.85e+06   11.61 |   11.61   11.0    149     0 |    0.13   0.22  -0.05  -0.78
        0  9.04e+06   14.07 |   13.57   10.5    149     0 |    0.27   0.17   0.01  -1.05
        0  1.01e+07   16.16 |   16.16   10.8    149     0 |    0.29   0.19  -0.08  -1.13
        0  1.14e+07   21.33 |   21.33    7.8    149     0 |    0.21   0.24  -0.06  -1.21

        0  1.02e+07    4.06 |   -3.27   11.1    149     0 |   -0.01   0.34  -0.03  -0.88
        0  2.00e+07    9.23 |   -1.57    7.6    149     0 |    0.06   0.12  -0.08  -1.26
        0  3.00e+07   11.78 |   11.78    7.6    149     0 |    0.05   0.08  -0.04  -1.40
        0  4.01e+07   13.20 |   12.35    7.8    149     0 |    0.14   0.08   0.01  -1.42
        0  5.02e+07   14.13 |   11.53    6.5    149     0 |    0.10   0.03   0.03  -1.42
        0  6.00e+07   15.75 |    6.33    6.1    149     0 |    0.18   0.13  -0.03  -1.43
        0  7.29e+07   20.71 |   20.71    8.1    149     0 |    0.16   0.03  -0.00  -1.41
        '''

        args.explore_rate = 0.9
        args.learning_rate = 2 ** -16
        args.agent.if_use_cri_target = True

        args.gamma = 0.99
        args.agent.lambda_gae_adv = 0.97
        args.agent.ratio_clip = 0.5
        args.reward_scale = 2 ** 1
        args.if_per_or_gae = True
        args.break_step = int(8e7)

        args.net_dim = 2 ** 8
        args.batch_size = args.net_dim * 4
        args.repeat_times = 2 ** 4

        args.target_step = args.env.max_step * 4

        args.eval_gap = 2 ** 8
        args.eval_times1 = 2 ** 3
        args.eval_times2 = 2 ** 5

    # train_and_evaluate(args)
    args.worker_num = 4
    train_and_evaluate_mp(args)


def demo_pixel_level_task():  # 2021-09-07
    args = Arguments(if_on_policy=True)  # hyper-parameters of on-policy is different from off-policy
    args.agent = AgentPPO()
    args.agent.cri_target = True
    args.visible_gpu = sys.argv[-1]

    if_train_car_racing = 1
    if if_train_car_racing:
        "TotalStep: 12e5, TargetReward: 300, UsedTime: 10ks PPO"
        "TotalStep: 20e5, TargetReward: 700, UsedTime: 25ks PPO"
        "TotalStep: 40e5, TargetReward: 800, UsedTime: 50ks PPO"
        from elegantrl.env import build_env
        env_name = 'CarRacingFix'
        args.env = build_env(env=env_name)  # register this environment in `env.py build_env()`
        args.agent.explore_rate = 0.75
        args.agent.ratio_clip = 0.5

        args.gamma = 0.98
        args.net_dim = 2 ** 8
        args.repeat_times = 2 ** 4
        args.learning_rate = 2 ** -17
        args.soft_update_tau = 2 ** -11
        args.batch_size = args.net_dim * 4
        args.if_per_or_gae = True
        args.agent.lambda_gae_adv = 0.96

        args.eval_gap = 2 ** 9
        args.eval_times1 = 2 ** 2
        args.eval_times1 = 2 ** 4
        args.if_allow_break = False
        args.break_step = int(2 ** 22)

        # args.target_step = args.env.max_step * 2
        # train_and_evaluate(args)

        args.worker_num = 6
        args.target_step = args.env.max_step * 2
        train_and_evaluate_mp(args)


if __name__ == '__main__':
    # demo_continuous_action_off_policy()
    # demo_continuous_action_on_policy()
    # demo_discrete_action_off_policy()
    # demo_discrete_action_on_policy()
    # demo_pybullet_off_policy()
    # demo_pybullet_on_policy()
    # demo_pixel_level_task()
    pass
