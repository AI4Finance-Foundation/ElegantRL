from elegantrl.demo import *


def demo_continuous_action_off_policy():  # [ElegantRL.2021.11.11]
    env_name = ['Pendulum-v1', 'LunarLanderContinuous-v2',
                'BipedalWalker-v3', 'BipedalWalkerHardcore-v3'][ENV_ID]
    agent_class = [AgentModSAC, AgentSAC,
                   AgentTD3, AgentDDPG][DRL_ID]
    args = Arguments(env=build_env(env_name), agent=agent_class())

    if env_name in {'Pendulum-v1', 'Pendulum-v0'}:
        """EpisodeReturn: (-1800) -1000 ~ -200 (-50)
        Step 2e5,  Reward -200,  UsedTime 200s ModSAC
        """
        # args = Arguments(env=build_env(env_name), agent=agent_class())  # One way to build env
        # args = Arguments(env=env_name, agent=agent_class())  # Another way to build env
        # args.env_num = 1
        # args.max_step = 200
        # args.state_dim = 3
        # args.action_dim = 1
        # args.if_discrete = False
        # args.target_return = -200

        args.gamma = 0.97
        args.net_dim = 2 ** 7
        args.worker_num = 2
        args.reward_scale = 2 ** -2
        args.target_step = 200 * 4  # max_step = 200
    if env_name in {'LunarLanderContinuous-v2', 'LunarLanderContinuous-v1'}:
        """EpisodeReturn: (-800) -200 ~ 200 (302)
        Step 40e4,  Reward 200,  UsedTime  900s, TD3
        Step 27e4,  Reward 200,  UsedTime 1600s, ModSAC
        Step 38e4,  Reward 200,  UsedTime 2700s, ModSAC
        """
        args.reward_scale = 2 ** -1
        args.eval_times1 = 2 ** 4
        args.eval_times2 = 2 ** 6

        args.target_step = args.env.max_step
    if env_name in {'BipedalWalker-v3', 'BipedalWalker-v2'}:
        """EpisodeReturn: (-200) -140 ~ 300 (341)
        Step 08e5,  Reward 300,  UsedTime 1800s TD3
        Step 11e5,  Reward 329,  UsedTime 6000s TD3
        Step  5e5,  Reward 300,  UsedTime 3500s ModSAC
        Step  8e5,  Reward 330,  UsedTime 5000s ModSAC
        """
        args.eval_times1 = 2 ** 3
        args.eval_times2 = 2 ** 5

        args.gamma = 0.98
        args.target_step = args.env.max_step
    if env_name in {'BipedalWalkerHardcore-v3', 'BipedalWalkerHardcore-v2'}:
        '''EpisodeReturn: (-200) -150 ~ 300 (334)
        TotalStep (2e6) 4e6

        Step 12e5,  Reward  20
        Step 18e5,  Reward 135
        Step 25e5,  Reward 202
        Step 43e5,  Reward 309, UsedTime 68ks,  ModSAC, worker_num=4

        Step 14e5,  Reward  15
        Step 18e5,  Reward 117
        Step 28e5,  Reward 212
        Step 45e5,  Reward 306,  UsedTime 67ks,  ModSAC, worker_num=4

        Step  8e5,  Reward  13
        Step 16e5,  Reward 136
        Step 23e5,  Reward 219
        Step 38e5,  Reward 302
        UsedTime 99ks  ModSAC, worker_num=2
        '''
        args.gamma = 0.98
        args.net_dim = 2 ** 8
        args.max_memo = 2 ** 22
        args.break_step = int(80e6)
        args.batch_size = args.net_dim * 2
        args.repeat_times = 1.5
        args.learning_rate = 2 ** -15

        args.eval_gap = 2 ** 9
        args.eval_times1 = 2 ** 2
        args.eval_times2 = 2 ** 5

        args.worker_num = 4
        args.target_step = args.env.max_step * 1

    args.learner_gpus = (GPU_ID,)  # single GPU
    # args.learner_gpus = (0, 1)  # multiple GPUs
    if_use_single_process = 0
    if if_use_single_process:
        train_and_evaluate(args)  # single process
    else:
        train_and_evaluate_mp(args)  # multiple process


"""
111 GPU 1 LL  reward_scale=2**-1
111 GPU 2 LL  reward_scale=2**-1
111 GPU 3 LL  reward_scale=2**0
111 GPU 0 LL  reward_scale=2**0
83  GPU 0 BW
"""

if __name__ == '__main__':
    sys.argv.extend('2 2 0'.split(' '))
    GPU_ID = eval(sys.argv[1])
    ENV_ID = eval(sys.argv[2])
    DRL_ID = eval(sys.argv[3])
    demo_continuous_action_off_policy()
