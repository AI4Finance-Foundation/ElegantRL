from beta0 import *
from elegantrl.demo import *


def demo_pybullet_off_policy():
    args = Arguments(if_on_policy=False)  # hyper-parameters of on-policy is different from off-policy
    args.agent = AgentModSAC()
    args.visible_gpu = '0'
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
    args.visible_gpu = '2'  # todo
    args.random_seed += 1943

    if_train_ant = 0
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

    if_train_isaac_ant = 1
    if if_train_isaac_ant:
        args.env = build_env(env='IsaacGymAnt', if_print=True)

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

    train_and_evaluate(args)
    # args.worker_num = 1
    # train_and_evaluate_mp(args)


if __name__ == '__main__':
    demo_pybullet_on_policy()
