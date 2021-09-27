from envs.IsaacGym import *
from elegantrl.demo import *


def demo_isaac_on_policy():
    args = Arguments(if_on_policy=True)  # hyper-parameters of on-policy is different from off-policy
    args.agent = AgentPPO()
    args.agent.if_use_cri_target = True  # todo
    args.random_seed += 1943

    if_train_ant = 1
    if if_train_ant:
        # env = build_env('IsaacOneEnvAnt', if_print=True, device_id=0, env_num=1)
        args.eval_env = 'IsaacOneEnvAnt'
        args.eval_gpu_id = 7

        # env = build_env('IsaacVecEnvAnt', if_print=True, device_id=0, env_num=2)
        args.env = 'IsaacVecEnvAnt'
        args.env_num = 1024
        args.max_step = 1000
        args.state_dim = 60
        args.action_dim = 8
        args.if_discrete = False
        args.target_return = 4000

        args.agent.lambda_entropy = 0.05
        args.agent.lambda_gae_adv = 0.97
        args.learning_rate = 2 ** -14
        args.if_per_or_gae = True
        args.break_step = int(8e7)

        args.reward_scale = 2 ** -2  # (-50) 0 ~ 2500 (3340)
        args.repeat_times = 2 ** 4
        args.net_dim = 2 ** 9
        args.batch_size = args.net_dim * 2 ** 3
        args.target_step = 2 ** 10  # todo

        args.break_step = int(2e8)
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

    args.init_before_training()

    # train_and_evaluate(args)
    args.learner_gpus = (2,)
    args.workers_gpus = args.learner_gpus
    args.worker_num = 1
    train_and_evaluate_mp(args)


if __name__ == '__main__':
    demo_isaac_on_policy()
