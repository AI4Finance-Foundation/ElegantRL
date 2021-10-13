from elegantrl.demo import *


def demo_discrete_action_on_policy():  # [ElegantRL.2021.10.12]
    args = Arguments()
    args.agent = AgentDiscretePPO()

    env_name = ['CartPole-v0',
                'LunarLander-v2',
                'SlimeVolley-v0', ][ENV]

    if env_name in {'CartPole-v0', }:
        "Step: 1e5, Reward: 200, UsedTime: 40s, DiscretePPO"
        args.env = build_env(env=env_name)
        args.target_return = 195

        args.eval_gap = 2 ** 5
        args.reward_scale = 2 ** -1
        args.target_step = args.env.max_step * 8
    if env_name in {'LunarLander-v2', }:
        '''
        Step 70e5, Reward 110, UsedTime 9961s  DiscretePPO, repeat_times = 2 ** 4
        Step 10e5, Reward 218, UsedTime 1336s  DiscretePPO, repeat_times = 2 ** 5
        '''
        args.env = build_env(env=env_name)

        args.reward_scale = 2 ** -1
        args.repeat_times = 2 ** 5

        args.worker_num = 2
        args.target_step = args.env.max_step * 4

        args.eval_gap = 2 ** 8
        args.random_seed = 1943

    args.learner_gpus = (GPU,)  # todo
    args.eval_gpu_id = GPU
    # args.learner_gpus = (0, )  # single GPU
    # args.learner_gpus = (0, 1)  # multiple GPUs
    # train_and_evaluate(args)  # single process
    train_and_evaluate_mp(args)  # multiple process


"""
IP 83 DiscretePPO GAE=False
GPU 2 LL worker_num = 2, max_step * 4, repeat_times = 2 ** 4
GPU 3 LL worker_num = 2, max_step * 4, repeat_times = 2 ** 5  5e5, 253, 605, again

IP 111 DiscretePPO 
GPU 2 LL worker_num = 2, max_step * 4, repeat_times = 2 ** 4 GAE=True
GPU 3 LL worker_num = 2, max_step * 4, repeat_times = 2 ** 4 cri_target=True
"""

if __name__ == '__main__':
    # sys.argv.extend(['1', '1'])  # todo
    GPU = eval(sys.argv[-1])
    ENV = 1  # eval(sys.argv[-1])

    demo_discrete_action_on_policy()
