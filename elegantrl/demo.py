import sys
from elegantrl.agent import *
from elegantrl.env import PreprocessEnv, build_env
from elegantrl.run import Arguments, train_and_evaluate, train_and_evaluate_mp

"""[ElegantRL.2021.09.01](https://github.com/AI4Finance-LLC/ElegantRL)"""

'''train'''


def demo_continuous_action_off_policy():  # 2021-09-07
    args = Arguments(if_on_policy=False)
    args.agent = AgentModSAC()  # AgentSAC AgentTD3 AgentDDPG

    if_train_pendulum = 1
    if if_train_pendulum:
        "TotalStep: 2e5, TargetReward: -200, UsedTime: 200s"
        import gym
        args.env = PreprocessEnv(env=gym.make('Pendulum-v0'))
        args.env.target_return = -200  # set target_reward manually for env 'Pendulum-v0'
        # args.env = PreprocessEnv(env='Pendulum-v0')  # It is Ok.
        # args.env = build_env(env='Pendulum-v0')  # It is Ok.
        args.reward_scale = 2 ** -2
        args.gamma = 0.97

        args.worker_num = 2
        args.target_step = args.env.max_step * 2

    if_train_lunar_lander = 0
    if if_train_lunar_lander:
        "TotalStep: 4e5, TargetReward: 200, UsedTime:  900s, TD3"
        "TotalStep: 5e5, TargetReward: 200, UsedTime: 1500s, ModSAC"
        args.env = build_env(env='LunarLanderContinuous-v2')
        args.target_step = args.env.max_step
        args.reward_scale = 2 ** 0

        args.eval_times1 = 2 ** 4
        args.eval_times2 = 2 ** 6  # use CPU to draw learning curve

    if_train_bipedal_walker = 0
    if if_train_bipedal_walker:
        "TotalStep: 08e5, TargetReward: 300, UsedTime: 1800s TD3"
        "TotalStep: 11e5, TargetReward: 329, UsedTime: 6000s TD3"
        "TotalStep:  4e5, TargetReward: 300, UsedTime: 2000s ModSAC"
        "TotalStep:  8e5, TargetReward: 330, UsedTime: 5000s ModSAC"
        args.env = build_env(env='BipedalWalker-v3')
        args.target_step = args.env.max_step
        args.gamma = 0.98

        args.eval_times1 = 2 ** 3
        args.eval_times2 = 2 ** 5

    if_train_bipedal_walker_hard_core = 0
    if if_train_bipedal_walker_hard_core:
        "TotalStep: 10e5, TargetReward:   0, UsedTime: 10ks ModSAC"
        "TotalStep: 25e5, TargetReward: 150, UsedTime: 20ks ModSAC"
        "TotalStep: 35e5, TargetReward: 295, UsedTime: 40ks ModSAC"
        "TotalStep: 40e5, TargetReward: 300, UsedTime: 50ks ModSAC"
        args.env = build_env(env='BipedalWalkerHardcore-v3')
        args.target_step = args.env.max_step
        args.gamma = 0.98
        args.net_dim = 2 ** 8
        args.batch_size = args.net_dim * 2
        args.learning_rate = 2 ** -15
        args.repeat_times = 1.5

        args.max_memo = 2 ** 22
        args.break_step = 2 ** 24

        args.eval_gap = 2 ** 8
        args.eval_times1 = 2 ** 2
        args.eval_times2 = 2 ** 5

        args.target_step = args.env.max_step * 1

    # train_and_evaluate(args)  # single process
    args.worker_num = 4
    args.visible_gpu = sys.argv[-1]
    train_and_evaluate_mp(args)  # multiple process
    # args.worker_num = 4
    # args.visible_gpu = '0,1'
    # train_and_evaluate_mp(args)  # multiple GPU


def demo_continuous_action_on_policy():
    args = Arguments(if_on_policy=True)  # hyper-parameters of on-policy is different from off-policy
    args.agent = AgentPPO()
    args.agent.cri_target = True
    args.visible_gpu = sys.argv[-1]
    args.random_seed += 1943

    if_train_pendulum = 1
    if if_train_pendulum:
        "TotalStep: 4e5, TargetReward: -200, UsedTime: 400s"
        import gym
        args.env = PreprocessEnv(env=gym.make('Pendulum-v0'))
        args.env.target_return = -200  # set target_reward manually for env 'Pendulum-v0'
        # args.env = PreprocessEnv(env='Pendulum-v0')  # It is Ok.
        # args.env = build_env(env='Pendulum-v0')  # It is Ok.
        args.reward_scale = 2 ** -2  # RewardRange: -1800 < -200 < -50 < 0

        args.gamma = 0.97
        args.net_dim = 2 ** 7
        args.batch_size = args.net_dim * 2

        args.target_step = args.env.max_step * 8

    if_train_lunar_lander = 0
    if if_train_lunar_lander:
        "TotalStep: 4e5, TargetReward: 200, UsedTime: 2000s, TD3"
        args.env = build_env(env='LunarLanderContinuous-v2')
        args.gamma = 0.99
        args.break_step = int(4e6)

        args.target_step = args.env.max_step * 8

    if_train_bipedal_walker = 0
    if if_train_bipedal_walker:
        "TotalStep: 8e5, TargetReward: 300, UsedTime: 1800s"
        args.env = build_env(env='BipedalWalker-v3')

        args.gamma = 0.98
        args.if_per_or_gae = True
        args.break_step = int(8e6)

        args.target_step = args.env.max_step * 16

    # train_and_evaluate(args)
    args.worker_num = 4
    train_and_evaluate_mp(args)


def demo_discrete_action_off_policy():
    args = Arguments(if_on_policy=False)
    args.agent = AgentDoubleDQN()  # AgentDQN()
    args.visible_gpu = '0'

    if_train_cart_pole = 0
    if if_train_cart_pole:
        "TotalStep: 5e4, TargetReward: 200, UsedTime: 60s"
        args.env = build_env('CartPole-v0')
        args.reward_scale = 2 ** -1
        args.target_step = args.env.max_step * 8

    if_train_lunar_lander = 1
    if if_train_lunar_lander:
        "TotalStep: 6e5, TargetReturn: 200, UsedTime: 1500s, LunarLander-v2, DQN"
        args.env = build_env(env='LunarLander-v2')
        args.repeat_times = 2 ** 5
        args.if_per_or_gae = True

    train_and_evaluate(args)


def demo_discrete_action_on_policy():
    args = Arguments(if_on_policy=True)  # hyper-parameters of on-policy is different from off-policy
    args.agent = AgentDiscretePPO()
    args.visible_gpu = '0'

    if_train_cart_pole = 1
    if if_train_cart_pole:
        "TotalStep: 5e4, TargetReward: 200, UsedTime: 60s"
        args.env = build_env('CartPole-v0')
        args.reward_scale = 2 ** -1
        args.target_step = args.env.max_step * 8

    if_train_lunar_lander = 0
    if if_train_lunar_lander:
        "TotalStep: 6e5, TargetReturn: 200, UsedTime: 1500s, LunarLander-v2, PPO"
        args.env = build_env(env='LunarLander-v2')
        args.repeat_times = 2 ** 5
        args.if_per_or_gae = True

    train_and_evaluate(args)


def demo_pixel_level_task():  # 2021-09-07
    args = Arguments(if_on_policy=True)  # hyper-parameters of on-policy is different from off-policy
    args.agent = AgentPPO()
    args.agent.cri_target = True
    args.visible_gpu = sys.argv[-1]

    if_train_car_racing = 1
    if if_train_car_racing:
        "TotalStep: 12e5, TargetReward: 300, UsedTime: 10ks PPO"
        "TotalStep: 20e5, TargetReward: 700, UsedTime: 25ks PPO"
        "TotalStep: 40e5, TargetReward: 800, UsedTime: 50ks PPO"
        from elegantrl.env import build_env
        env_name = 'CarRacingFix'
        args.env = build_env(env=env_name)  # register this environment in `env.py build_env()`
        args.agent.explore_rate = 0.75
        args.agent.ratio_clip = 0.5

        args.gamma = 0.98
        args.net_dim = 2 ** 8
        args.repeat_times = 2 ** 4
        args.learning_rate = 2 ** -17
        args.soft_update_tau = 2 ** -11
        args.batch_size = args.net_dim * 4
        args.if_per_or_gae = True
        args.agent.lambda_gae_adv = 0.96

        args.eval_gap = 2 ** 9
        args.eval_times1 = 2 ** 2
        args.eval_times1 = 2 ** 4
        args.if_allow_break = False
        args.break_step = int(2 ** 22)

        # args.target_step = args.env.max_step * 2
        # train_and_evaluate(args)

        args.worker_num = 6
        args.target_step = args.env.max_step * 2
        train_and_evaluate_mp(args)


'''test'''

if __name__ == '__main__':
    # demo_continuous_action_off_policy()
    demo_continuous_action_on_policy()
    # demo_discrete_action_off_policy()
    # demo_discrete_action_on_policy()
    # demo_pixel_level_task()
