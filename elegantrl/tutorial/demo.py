from elegantrl.tutorial.run import *

gym.logger.set_level(40)  # Block warning


def demo_continuous_action_off_policy():
    args = Arguments(if_on_policy=False)
    args.agent = AgentModSAC()  # AgentModSAC AgentSAC AgentTD3 AgentDDPG
    args.visible_gpu = '0'

    if_train_pendulum = 1
    if if_train_pendulum:
        "TotalStep: 2e5, TargetReward: -200, UsedTime: 200s"
        args.env = PreprocessEnv(env=gym.make('Pendulum-v0'))  # env='Pendulum-v0' is OK.
        args.env.target_return = -200  # set target_reward manually for env 'Pendulum-v0'
        args.reward_scale = 2 ** -2  # RewardRange: -1800 < -200 < -50 < 0
        args.gamma = 0.97
        args.target_step = args.env.max_step * 8

    if_train_lunar_lander = 0
    if if_train_lunar_lander:
        "TotalStep: 4e5, TargetReward: 200, UsedTime: 900s"
        args.env = PreprocessEnv(env=gym.make('LunarLanderContinuous-v2'))
        args.target_step = args.env.max_step * 4
        args.gamma = 0.98

    if_train_bipedal_walker = 0
    if if_train_bipedal_walker:
        "TotalStep: 8e5, TargetReward: 300, UsedTime: 1800s"
        args.env = PreprocessEnv(env=gym.make('BipedalWalker-v3'))
        args.gamma = 0.98

    args.eval_times = 2 ** 5  # evaluate times of the average episode return
    train_and_evaluate(args)


def demo_continuous_action_on_policy():
    args = Arguments(if_on_policy=True)  # hyper-parameters of on-policy is different from off-policy
    args.agent = AgentPPO()
    args.visible_gpu = '0'

    if_train_pendulum = 1
    if if_train_pendulum:
        "TotalStep: 4e5, TargetReward: -200, UsedTime: 400s"
        args.env = PreprocessEnv(env=gym.make('Pendulum-v0'))  # env='Pendulum-v0' is OK.
        args.env.target_return = -200  # set target_reward manually for env 'Pendulum-v0'
        args.reward_scale = 2 ** -3  # RewardRange: -1800 < -200 < -50 < 0
        args.gamma = 0.97
        args.net_dim = 2 ** 7
        args.batch_size = args.net_dim * 2
        args.target_step = args.env.max_step * 8

    if_train_lunar_lander = 0
    if if_train_lunar_lander:
        "TotalStep: 4e5, TargetReward: 200, UsedTime: 900s"
        args.env = PreprocessEnv(env=gym.make('LunarLanderContinuous-v2'))
        args.target_step = args.env.max_step * 4
        args.gamma = 0.98
        args.if_per_or_gae = True

    if_train_bipedal_walker = 0
    if if_train_bipedal_walker:
        "TotalStep: 8e5, TargetReward: 300, UsedTime: 1800s"
        args.env = PreprocessEnv(env=gym.make('BipedalWalker-v3'))
        args.gamma = 0.98
        args.if_per_or_gae = True
        args.agent.cri_target = True

    args.eval_times = 2 ** 5  # evaluate times of the average episode return
    train_and_evaluate(args)


def demo_discrete_action_off_policy():
    args = Arguments(if_on_policy=False)
    args.agent = AgentDoubleDQN()  # AgentDoubleDQN AgentDQN
    args.visible_gpu = '0'

    if_train_cart_pole = 1
    if if_train_cart_pole:
        "TotalStep: 5e4, TargetReward: 200, UsedTime: 60s"
        args.env = PreprocessEnv(env='CartPole-v0')
        args.reward_scale = 2 ** -1
        args.target_step = args.env.max_step * 8

    if_train_lunar_lander = 0
    if if_train_lunar_lander:
        "TotalStep: 6e5, TargetReturn: 200, UsedTime: 1500s, LunarLander-v2, DQN"
        args.env = PreprocessEnv(env=gym.make('LunarLander-v2'))
        args.repeat_times = 2 ** 5
        args.if_per_or_gae = True

    args.eval_times = 2 ** 5  # evaluate times of the average episode return
    train_and_evaluate(args)


def demo_discrete_action_on_policy():
    args = Arguments(if_on_policy=True)  # hyper-parameters of on-policy is different from off-policy
    args.agent = AgentDiscretePPO()
    args.visible_gpu = '0'

    if_train_cart_pole = 1
    if if_train_cart_pole:
        "TotalStep: 5e4, TargetReward: 200, UsedTime: 60s"
        args.env = PreprocessEnv(env='CartPole-v0')
        args.reward_scale = 2 ** -1
        args.target_step = args.env.max_step * 8

    if_train_lunar_lander = 0
    if if_train_lunar_lander:
        "TotalStep: 6e5, TargetReturn: 200, UsedTime: 1500s, LunarLander-v2, PPO"
        args.env = PreprocessEnv(env=gym.make('LunarLander-v2'))
        args.repeat_times = 2 ** 5
        args.if_per_or_gae = True

    args.eval_times = 2 ** 5  # evaluate times of the average episode return
    train_and_evaluate(args)


if __name__ == '__main__':
    # demo_continuous_action_off_policy()
    demo_continuous_action_on_policy()
    # demo_discrete_action_off_policy()
    # demo_discrete_action_on_policy()
