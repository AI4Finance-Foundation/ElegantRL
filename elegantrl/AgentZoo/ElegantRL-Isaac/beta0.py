from elegantrl.demo import *


def demo_pybullet_on_policy():
    args = Arguments(if_on_policy=True)  # hyper-parameters of on-policy is different from off-policy
    args.agent = AgentPPO()
    args.learner_gpus = (1, )
    args.random_seed += 1943

    if_train_ant = 1
    if if_train_ant:
        # args.env = build_env(env='AntBulletEnv-v0', if_print=True)
        args.env = 'AntBulletEnv-v0'
        args.max_step = 1000
        args.state_dim = 28
        args.action_dim = 8
        args.if_discrete = False
        args.target_return = 2500
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
        args.target_step = 1000 * 2

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

    args.init_before_training()

    # train_and_evaluate(args)
    args.worker_num = 4
    train_and_evaluate_mp(args)


if __name__ == '__main__':
    demo_pybullet_on_policy()
