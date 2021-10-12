from elegantrl.demo import *

"""
IP 83
GPU 1 LL RPT=1
GPU 2 LL RPT=2

IP 111 DiscretePPO
GPU 0 LL  max_step * 2 GAE
GPU 1 LL  max_step * 2
GPU 2 LL  max_step * 1 GAE
GPU 3 LL  max_step * 1
"""


def demo_discrete_action_off_policy():  # [ElegantRL.2021.10.10]
    args = Arguments()
    args.agent = AgentD3QN()  # AgentD3QN AgentDuelDQN AgentDoubleDQN AgentDQN

    env_name = ['CartPole-v0',
                'LunarLander-v2',
                'SlimeVolley-v0', ][ENV_ID]

    if env_name in {'CartPole-v0', }:
        "Step: 1e5, Reward: 200, UsedTime: 40s, AgentD3QN"
        args.env = build_env(env=env_name)
        args.target_return = 195

        args.eval_gap = 2 ** 5

        args.reward_scale = 2 ** -1
        args.target_step = args.env.max_step * 4
    if env_name in {'LunarLander-v2', }:
        "Step: 2e5, Reward: -200, UsedTime: 200s ModSAC"
        args.env = build_env(env=env_name)

        args.max_memo = 2 ** 19
        args.reward_scale = 2 ** -1
        args.repeat_times = 1
        args.target_step = args.env.max_step

    args.learner_gpus = (GPU_ID,)  # todo
    args.eval_gpu_id = (GPU_ID,)
    # args.learner_gpus = (0, )  # single GPU
    # args.learner_gpus = (0, 1)  # multiple GPUs
    # train_and_evaluate(args)  # single process
    train_and_evaluate_mp(args)  # multiple process


def demo_discrete_action_on_policy():  # [ElegantRL.2021.10.12]
    args = Arguments()
    args.agent = AgentDiscretePPO()

    env_name = ['CartPole-v0',
                'LunarLander-v2',
                'SlimeVolley-v0', ][ENV_ID]

    if env_name in {'CartPole-v0', }:
        "Step: 1e5, Reward: 200, UsedTime: 40s, DiscretePPO"
        args.env = build_env(env=env_name)
        args.target_return = 195

        args.eval_gap = 2 ** 5
        args.reward_scale = 2 ** -1
        args.target_step = args.env.max_step * 8
    if env_name in {'LunarLander-v2', }:
        "Step: 2e5, Reward: -200, UsedTime: 200s, DiscretePPO"
        args.env = build_env(env=env_name)

        args.if_per_or_gae = False  # todo
        args.reward_scale = 2 ** -1
        args.repeat_times = 2 ** 5
        args.target_step = args.env.max_step * TGS

    args.learner_gpus = (GPU_ID,)  # todo
    args.eval_gpu_id = (GPU_ID,)
    # args.learner_gpus = (0, )  # single GPU
    # args.learner_gpus = (0, 1)  # multiple GPUs
    # train_and_evaluate(args)  # single process
    train_and_evaluate_mp(args)  # multiple process


if __name__ == '__main__':
    # sys.argv.extend(['1', '1'])  # todo
    TGS = eval(sys.argv[-3])
    GPU_ID = eval(sys.argv[-2])
    ENV_ID = eval(sys.argv[-1])

    demo_discrete_action_on_policy()
