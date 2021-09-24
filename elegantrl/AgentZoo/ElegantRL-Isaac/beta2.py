from beta0 import *
from elegantrl.demo import *


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
