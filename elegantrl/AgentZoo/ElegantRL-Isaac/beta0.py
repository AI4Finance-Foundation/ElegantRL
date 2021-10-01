from envs.IsaacGym import *
from elegantrl.demo import *

"""
Humanoid 
GPU 4 
GPU 5 if_use_cri_target = True
"""


def demo_isaac_on_policy():
    args = Arguments(if_on_policy=True)  # hyper-parameters of on-policy is different from off-policy
    args.agent = AgentPPO()
    args.random_seed += 1943

    if_train_ant = 0
    if if_train_ant:
        # env = build_env('IsaacOneEnvAnt', if_print=True, device_id=0, env_num=1)
        args.eval_env = 'IsaacOneEnvAnt'
        args.eval_gpu_id = 7

        # env = build_env('IsaacVecEnvAnt', if_print=True, device_id=0, env_num=2)
        args.env = f'IsaacVecEnvAnt'
        args.env_num = 1024
        args.max_step = 1000
        args.state_dim = 60
        args.action_dim = 8
        args.if_discrete = False
        args.target_return = 4000

        args.agent.lambda_entropy = 0.05
        args.agent.lambda_gae_adv = 0.97
        args.learning_rate = 2 ** -15
        args.if_per_or_gae = True
        args.break_step = int(8e7)

        args.reward_scale = 2 ** -2  # (-50) 0 ~ 2500 (3340)
        args.repeat_times = 2 ** 3
        args.net_dim = 2 ** 9
        args.batch_size = args.net_dim * 2 ** 3
        args.target_step = 2 ** 10

        args.break_step = int(2e7)
        args.if_allow_break = False

    if_train_humanoid = 1
    if if_train_humanoid:
        # env = build_env('IsaacOneEnvHumanoid', if_print=True, device_id=0, env_num=1)
        args.eval_env = 'IsaacOneEnvHumanoid'
        args.eval_gpu_id = 7

        # env = build_env('IsaacVecEnvHumanoid', if_print=True, device_id=0, env_num=2)
        args.env = f'IsaacVecEnvHumanoid'
        args.env_num = 1024
        args.max_step = 1000
        args.state_dim = 108
        args.action_dim = 21
        args.if_discrete = False
        args.target_return = 7000

        args.agent.lambda_entropy = 0.02
        args.agent.lambda_gae_adv = 0.97
        args.learning_rate = 2 ** -14
        args.if_per_or_gae = True
        args.break_step = int(8e7)

        args.reward_scale = 2 ** -1
        args.repeat_times = 2 ** 3
        args.net_dim = 2 ** 9
        args.batch_size = args.net_dim * 2 ** 4
        args.target_step = 2 ** 10

        args.break_step = int(2e8)
        args.if_allow_break = False

    args.init_before_training()

    # train_and_evaluate(args)
    args.learner_gpus = (4, )
    args.workers_gpus = args.learner_gpus
    args.worker_num = 1
    train_and_evaluate_mp(args)


if __name__ == '__main__':
    demo_isaac_on_policy()
