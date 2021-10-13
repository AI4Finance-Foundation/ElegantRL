from elegantrl.demo import *


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
        "Step: 12e5, Reward: 207, UsedTime: 19ks, DiscretePPO"
        args.env = build_env(env=env_name)

        args.reward_scale = 2 ** -1
        args.repeat_times = 2 ** 4
        args.if_per_or_gae = True

        args.worker_num = 4
        args.target_step = args.env.max_step * TGS

    args.learner_gpus = (GPU_ID,)  # todo
    args.eval_gpu_id = (GPU_ID,)
    # args.learner_gpus = (0, )  # single GPU
    # args.learner_gpus = (0, 1)  # multiple GPUs
    # train_and_evaluate(args)  # single process
    train_and_evaluate_mp(args)  # multiple process


"""
IP 83

IP 111 DiscretePPO
GPU 2 LL TGS=2
GPU 3 LL TGS=4
"""

if __name__ == '__main__':
    # sys.argv.extend(['1', '1'])  # todo
    TGS = eval(sys.argv[-3])
    GPU_ID = eval(sys.argv[-2])
    ENV_ID = eval(sys.argv[-1])

    demo_discrete_action_on_policy()
