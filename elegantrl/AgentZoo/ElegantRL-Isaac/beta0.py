from envs.IsaacGym import *
from elegantrl.demo import *

"""2021-10-14
IDEA ENV GPU 0 2 beta0.py
IDEA ENV GPU 1 3 beta1.py
"""


def demo_isaac_on_policy():
    env_name = ['IsaacVecEnvAnt', 'IsaacVecEnvHumanoid'][ENV_ID]
    args = Arguments(env=env_name, agent=AgentPPO())
    args.eval_gpu_id = GPU_ID
    args.learner_gpus = (GPU_ID,)
    args.workers_gpus = args.learner_gpus

    if env_name in {'IsaacVecEnvAnt', 'IsaacOneEnvAnt'}:
        '''
        Step  21e7, Reward  8350, UsedTime  35ks
        Step 484e7, Reward 16206, UsedTime 960ks  PPO, if_use_cri_target = False
        Step  20e7, Reward  9196, UsedTime  35ks
        Step 471e7, Reward 15021, UsedTime 960ks  PPO, if_use_cri_target = True
        '''
        args.eval_env = 'IsaacOneEnvAnt'
        args.env = f'IsaacVecEnvAnt'
        args.env_num = 4096
        args.max_step = 1000
        args.state_dim = 60
        args.action_dim = 8
        args.if_discrete = False
        args.target_return = 8000

        args.agent.lambda_entropy = 0.05
        args.agent.lambda_gae_adv = 0.97
        args.agent.if_use_cri_target = False

        args.if_per_or_gae = True
        args.learning_rate = 2 ** -14

        args.net_dim = int(2 ** 8 * 1.5)
        args.batch_size = args.net_dim * 2 ** 4
        args.target_step = args.max_step * 1
        args.repeat_times = 2 ** 4
        args.reward_scale = 2 ** -2  # (-50) 0 ~ 2500 (3340)

        args.break_step = int(8e14)
        args.if_allow_break = False
        args.eval_times1 = 2 ** 1
        args.eval_times1 = 2 ** 4
        args.eval_gap = 2 ** 9

    if env_name in {'IsaacVecEnvHumanoid', 'IsaacOneEnvHumanoid'}:
        '''
        Step 126e7, Reward  8021
        Step 216e7, Reward  9517
        Step 283e7, Reward  9998
        Step 438e7, Reward 10749, UsedTime 960ks  PPO
        Step 215e7, Reward  9794, UsedTime 465ks  PPO
        Step   1e7, Reward   117
        Step  16e7, Reward   538
        Step  21e7, Reward  3044
        Step  38e7, Reward  5015
        Step  65e7, Reward  6010
        Step  72e7, Reward  6257, UsedTime 129ks  PPO, if_use_cri_target = True
        Step  77e7, Reward  5399, UsedTime 143ks  PPO
        Step  86e7, Reward  5822, UsedTime 157ks  PPO
        Step  86e7, Reward  5822, UsedTime 157ks  PPO
        '''
        args.eval_env = 'IsaacOneEnvHumanoid'
        args.env = f'IsaacVecEnvHumanoid'
        args.env_num = 2048
        args.max_step = 1000
        args.state_dim = 108
        args.action_dim = 21
        args.if_discrete = False
        args.target_return = 7000

        args.agent.lambda_entropy = 0.05
        args.agent.lambda_gae_adv = 0.97
        args.agent.if_use_cri_target = True

        args.net_dim = int(2 ** 8 * 1.5)
        args.batch_size = args.net_dim * 2 ** 5
        args.target_step = args.max_step * 1
        args.repeat_times = 2 ** 5
        args.reward_scale = 2 ** -2  # (-50) 0 ~ 2500 (3340)
        args.if_per_or_gae = True
        args.learning_rate = 2 ** -15

        args.break_step = int(8e14)
        args.if_allow_break = False
        args.eval_times1 = 2 ** 1
        args.eval_times1 = 2 ** 4
        args.eval_gap = 2 ** 9

    args.worker_num = 1
    args.workers_gpus = args.learner_gpus
    train_and_evaluate_mp(args)  # train_and_evaluate(args)


if __name__ == '__main__':
    # import sys  # todo
    ENV_ID = 0  # eval(sys.argv[-2])
    GPU_ID = 2  # eval(sys.argv[-1])

    demo_isaac_on_policy()
