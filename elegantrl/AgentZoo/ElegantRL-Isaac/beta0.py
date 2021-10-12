from envs.IsaacGym import *
from elegantrl.demo import *

"""
Ant
501  GPU 0, args.max_step * 1, learning_rate = 2 ** -15, S 23e7, R 7111, S 97e7, R 12k
398  GPU 1, args.max_step * 2, learning_rate = 2 ** -15, S 22e7, R 5412, S 91e7, R 10k

Humanoid 
639  GPU 3, env_num = 4096, args.max_step * 2,           S 86e7, R 5795
887  GPU 4, env_num = 4096, args.max_step * 1,           S 90e7, R 6900 

net_dim * 2 ** 5, repeat_times = 2 ** 5
802  GPU 6, env_num = 2048, args.max_step * 1,           S 71e7, R 7800
806  GPU 7, env_num = 4096, args.max_step * 1,           S 72e7, R 6250, T 130ks
# 583  GPU 5, 
"""


def demo_isaac_on_policy():
    args = Arguments(if_on_policy=True)  # hyper-parameters of on-policy is different from off-policy
    args.agent = AgentPPO()
    args.random_seed += 1943
    gpu_id = 6  # todo

    if_train_ant = 1
    if if_train_ant:
        # env = build_env('IsaacOneEnvAnt', if_print=True, device_id=0, env_num=1)
        args.eval_env = 'IsaacOneEnvAnt'
        args.eval_gpu_id = 6

        # env = build_env('IsaacVecEnvAnt', if_print=True, device_id=0, env_num=2)
        args.env = f'IsaacVecEnvAnt'
        args.env_num = 4096
        args.max_step = 1000
        args.state_dim = 60
        args.action_dim = 8
        args.if_discrete = False
        args.target_return = 8000
        args.if_per_or_gae = True
        args.learning_rate = 2 ** -14  # todo

        args.agent.lambda_entropy = 0.05
        args.agent.lambda_gae_adv = 0.97
        args.agent.if_use_cri_target = True

        args.net_dim = int(2 ** 8 * 1.5)
        args.batch_size = args.net_dim * 2 ** 4
        args.target_step = args.max_step * 1
        args.repeat_times = 2 ** 4
        args.reward_scale = 2 ** -2  # (-50) 0 ~ 2500 (3340)

        args.break_step = int(8e14)
        args.if_allow_break = False

    if_train_humanoid = 1
    if if_train_humanoid:
        # env = build_env('IsaacOneEnvHumanoid', if_print=True, device_id=0, env_num=1)
        args.eval_env = 'IsaacOneEnvHumanoid'
        args.eval_gpu_id = gpu_id

        # env = build_env('IsaacVecEnvHumanoid', if_print=True, device_id=0, env_num=2)
        args.env = f'IsaacVecEnvHumanoid'
        args.env_num = 2048  # todo
        args.max_step = 1000
        args.state_dim = 108
        args.action_dim = 21
        args.if_discrete = False
        args.target_return = 7000

        args.agent.lambda_entropy = 0.05
        args.agent.lambda_gae_adv = 0.97
        args.agent.if_use_cri_target = True

        args.net_dim = int(2 ** 8 * 1.5)
        args.batch_size = args.net_dim * 2 ** 5  # todo
        args.target_step = args.max_step * 1  # todo
        args.repeat_times = 2 ** 5  # todo
        args.reward_scale = 2 ** -2  # (-50) 0 ~ 2500 (3340)
        args.if_per_or_gae = True
        args.learning_rate = 2 ** -15  # todo

        args.break_step = int(8e14)
        args.if_allow_break = False

    args.learner_gpus = (gpu_id,)
    args.workers_gpus = args.learner_gpus
    args.worker_num = 1

    args.init_before_training()
    train_and_evaluate_mp(args)  # train_and_evaluate(args)


"""
Ant
501  GPU 0, S 18e7, R 8350, T 35sk
398  GPU 2, S  2e8, R 9196, T 35ks, if_use_cri_target = True

Humanoid 
639  GPU 3, S  2e8, R 5892, T 31ks
887  GPU 4, S  8e7, R  711, T 11ks
583  GPU 0, S  2e8, R 3787, T 30ks, batch_size = args.net_dim * 2 ** 5
802  GPU 5, S  2e8, R 4295, T 33ks, batch_size = args.net_dim * 2 ** 4
806  GPU 6, S  2e8, R 3725, T 33ks, batch_size = args.net_dim * 2 ** 4, if_use_cri_target = True
"""

if __name__ == '__main__':
    demo_isaac_on_policy()
