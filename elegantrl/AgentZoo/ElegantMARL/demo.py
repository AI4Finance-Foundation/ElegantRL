import sys
from elegantrl.AgentZoo.ElegantMARL.agent import *
from elegantrl.AgentZoo.ElegantMARL.env import PreprocessEnv, build_env
from elegantrl.AgentZoo.ElegantMARL.run import Arguments, train_and_evaluate, train_and_evaluate_mp

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
    args.random_seed += 19431

    if_train_ant = 1
    if if_train_ant:
        args.env = build_env(env='AntBulletEnv-v0', if_print=True)
        """
        0  4.29e+06 2446.47 |  431.34   82.1    999     0 |    0.08   1.65-275.32   0.26 | UsedTime:   14393 |
        0  1.41e+07 3499.37 | 3317.42    5.9    999     0 |    0.24   0.06 -49.94   0.03 | UsedTime:   70020 |
        0  3.54e+06 2875.30 |  888.67    4.7    999     0 |    0.19   0.11 -69.10   0.05 | UsedTime:   54701 |
        0  2.00e+07 2960.38 |  698.58   42.5    999     0 |    0.08   0.05 -39.44   0.03 | UsedTime:   53545 |
        """
        args.agent.if_use_act_target = False

        args.break_step = int(8e7)

        args.reward_scale = 2 ** -2
        args.net_dim = 2 ** 9
        args.batch_size = args.net_dim * 2
        args.max_memo = 2 ** 22
        args.repeat_times = 2 ** 1
        args.target_step = args.env.max_step * 2

        args.break_step = int(2e7)
        args.if_allow_break = False

        args.eval_gap = 2 ** 8
        args.eval_times1 = 2 ** 1
        args.eval_times2 = 2 ** 4

    if_train_humanoid = 0
    if if_train_humanoid:
        args.env = build_env(env='HumanoidBulletEnv-v0', if_print=True)
        """
        0  1.50e+07 2571.46 |   53.63   66.8    128    58 |    0.04   0.96-153.29   0.06 | UsedTime:    74470 |
        0  1.51e+07 2822.93 |   -1.51   27.1     99    36 |    0.03   0.58 -96.48   0.04 | UsedTime:    74480 |
        0  1.09e+06   66.96 |   58.69    8.2     58    12 |    0.22   0.28 -22.92   0.00
        0  3.01e+06  129.69 |  101.39   40.6     96    33 |    0.14   0.28 -20.16   0.03
        0  5.02e+06  263.13 |  208.69  122.6    195    59 |    0.11   0.29 -32.71   0.03
        0  6.03e+06  791.89 |  527.79  282.7    360   144 |    0.21   0.26 -36.51   0.03
        0  8.00e+06 2432.21 |   35.78   49.3    113    54 |   -0.08   1.30-168.28   0.05
        0  8.13e+06 2432.21 |  907.28  644.9    606   374 |    0.11   0.72-134.01   0.05
        0  8.29e+06 2432.21 | 2341.30   39.4    999     0 |    0.41   0.41 -96.96   0.03
        0  1.09e+07 2936.10 | 2936.10   24.8    999     0 |    0.60   0.13 -68.74   0.02
        0  2.83e+07 2968.08 | 2737.18   15.9    999     0 |    0.57   0.21 -81.07   0.03 | UsedTime:    74512 |
        """
        args.break_step = int(8e7)

        args.reward_scale = 2 ** -2
        args.net_dim = 2 ** 9
        args.batch_size = args.net_dim * 2
        args.target_step = args.env.max_step * 2

        args.break_step = int(8e7)
        args.if_allow_break = False

    if_train_reacher = 0
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

    if_train_minitaur = 0
    if if_train_minitaur:
        args.env = build_env(env='MinitaurBulletEnv-v0', if_print=True)
        """
        0  1.00e+06    0.46 |    0.24    0.0     98    37 |    0.06   0.06  -7.64   0.02
        0  1.26e+06    1.36 |    1.36    0.7    731   398 |    0.10   0.08 -10.40   0.02
        0  1.30e+06    3.18 |    3.18    0.8    999     0 |    0.13   0.08 -10.99   0.02
        0  2.00e+06    3.18 |    0.04    0.0     28     0 |    0.13   0.09 -16.02   0.02
        0  4.04e+06    7.11 |    6.68    0.6    999     0 |    0.17   0.08 -19.67   0.02
        0  5.72e+06    9.79 |    9.28    0.1    999     0 |    0.22   0.03 -23.89   0.01
        0  6.01e+06   10.69 |   10.09    0.8    999     0 |    0.22   0.03 -24.98   0.01
        """

        args.learning_rate = 2 ** -16
        args.break_step = int(8e7)

        args.reward_scale = 2 ** 5  # (-2) 0 ~ 16 (20)
        args.net_dim = 2 ** 9
        args.batch_size = args.net_dim * 2
        args.target_step = args.env.max_step * 2

        args.break_step = int(4e7)
        args.if_allow_break = False

        args.eval_gap = 2 ** 8
        args.eval_times1 = 2 ** 2
        args.eval_times2 = 2 ** 4

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

    if_train_minitaur = 0
    if if_train_minitaur:
        args.env = build_env(env='MinitaurBulletEnv-v0', if_print=True)
        """
        0  5.91e+05   10.59 |   10.59    3.9    727   282 |    0.27   0.69  -0.03  -0.52
        0  1.15e+06   14.91 |   12.48    2.2    860   158 |    0.40   0.65  -0.02  -0.55
        0  2.27e+06   25.38 |   22.54    4.7    968    54 |    0.75   0.61  -0.06  -0.60
        0  4.13e+06   29.05 |   28.33    1.0    999     0 |    0.89   0.51  -0.07  -0.65
        0  8.07e+06   32.66 |   32.17    0.9    999     0 |    0.97   0.45  -0.06  -0.73
        0  1.10e+07   32.66 |   32.33    1.3    999     0 |    0.94   0.40  -0.07  -0.80 | UsedTime:   20208 |

        0  5.91e+05    5.48 |    5.48    1.5    781   219 |    0.24   0.66  -0.04  -0.52
        0  1.01e+06   12.35 |    9.77    2.9    754   253 |    0.34   0.74  -0.05  -0.54
        0  2.10e+06   12.35 |   12.21    4.8    588   285 |    0.60   0.65  -0.01  -0.58
        0  4.09e+06   28.31 |   22.88   12.6    776   385 |    0.88   0.51  -0.03  -0.66
        0  8.03e+06   30.96 |   28.32    6.8    905   163 |    0.93   0.52  -0.05  -0.76
        0  1.09e+07   32.07 |   31.29    0.9    999     0 |    0.95   0.47  -0.07  -0.82 | UsedTime:   20238 |
        """

        args.agent.lambda_entropy = 0.05
        args.agent.lambda_gae_adv = 0.97
        args.learning_rate = 2 ** -15
        args.if_per_or_gae = True
        args.break_step = int(8e7)

        args.reward_scale = 2 ** 5  # (-2) 0 ~ 16 (20)
        args.repeat_times = 2 ** 4
        args.net_dim = 2 ** 9
        args.batch_size = args.net_dim * 4
        args.target_step = args.env.max_step * 2

        args.break_step = int(4e7)
        args.if_allow_break = False

        args.eval_gap = 2 ** 8
        args.eval_times1 = 2 ** 2
        args.eval_times2 = 2 ** 4

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
