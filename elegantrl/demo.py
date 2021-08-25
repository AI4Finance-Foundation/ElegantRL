import sys
import gym  # not necessary

from elegantrl.agent import *
from elegantrl.env import PreprocessEnv
from elegantrl.run import Arguments, train_and_evaluate, train_and_evaluate_mp

gym.logger.set_level(40)  # Block warning


def demo_continuous_action_off_policy():
    args = Arguments(if_on_policy=False)
    args.agent = AgentModSAC()  # AgentSAC AgentTD3 AgentDDPG
    args.agent.if_use_act_target = True
    args.agent.if_use_cri_target = True
    args.visible_gpu = sys.argv[-1]

    if_train_pendulum = 0
    if if_train_pendulum:
        "TotalStep: 2e5, TargetReward: -200, UsedTime: 200s"
        args.env = PreprocessEnv(env=gym.make('Pendulum-v0'))  # env='Pendulum-v0' is OK.
        args.env.target_return = -200  # set target_reward manually for env 'Pendulum-v0'
        args.reward_scale = 2 ** -2
        args.gamma = 0.97

        # train_and_evaluate(args)
        args.env_num = 2
        args.worker_num = 2
        args.target_step = args.env.max_step * 4 // (args.env_num * args.worker_num)
        train_and_evaluate_mp(args)

    if_train_lunar_lander = 1
    if if_train_lunar_lander:
        "TotalStep: 4e5, TargetReward: 200, UsedTime: 900s"
        args.env = PreprocessEnv(env=gym.make('LunarLanderContinuous-v2'))
        args.gamma = 0.99
        args.break_step = int(4e6)

        # train_and_evaluate(args)
        args.env_num = 2
        args.worker_num = 4
        args.target_step = args.env.max_step * 2 // (args.env_num * args.worker_num)
        train_and_evaluate_mp(args)

    if_train_bipedal_walker = 1
    if if_train_bipedal_walker:
        "TotalStep: 08e5, TargetReward: 300, UsedTime: 1800s TD3"
        "TotalStep: 11e5, TargetReward: 329, UsedTime: 3000s TD3"
        args.env = PreprocessEnv(env=gym.make('BipedalWalker-v3'))
        args.gamma = 0.98
        args.break_step = int(4e6)
        args.max_memo = 2 ** 20

        train_and_evaluate(args)
        # args.env_num = 2
        # args.worker_num = 4
        # args.target_step = args.env.max_step * 2 // (args.env_num * args.worker_num)
        # train_and_evaluate_mp(args)


def demo_continuous_action_on_policy():
    args = Arguments(if_on_policy=True)  # hyper-parameters of on-policy is different from off-policy
    args.agent = AgentPPO()
    args.agent.cri_target = True
    args.visible_gpu = sys.argv[-1]
    args.random_seed += 1943

    if_train_pendulum = 0
    if if_train_pendulum:
        "TotalStep: 4e5, TargetReward: -200, UsedTime: 400s"
        env = PreprocessEnv(env=gym.make('Pendulum-v0'))
        env.target_return = -200

        args.env_eval = env
        args.env = env
        args.env.env_num = 2

        args.agent.cri_target = False
        args.reward_scale = 2 ** -2  # RewardRange: -1800 < -200 < -50 < 0
        args.gamma = 0.97
        args.net_dim = 2 ** 7
        args.batch_size = args.net_dim * 2
        args.target_step = args.env_eval.max_step * 2

        train_and_evaluate(args)
        # args.worker_num = 2
        # train_and_evaluate_mp(args)

    if_train_lunar_lander = 0
    if if_train_lunar_lander:
        "TotalStep: 4e5, TargetReward: 200, UsedTime: 2000s, TD3"
        args.env = PreprocessEnv(env=gym.make('LunarLanderContinuous-v2'))
        args.gamma = 0.99
        args.break_step = int(4e6)

        # train_and_evaluate(args)
        args.env_num = 2
        args.worker_num = 4
        args.target_step = args.env.max_step * 2 // (args.env_num * args.worker_num)
        train_and_evaluate_mp(args)

    if_train_bipedal_walker = 1
    if if_train_bipedal_walker:
        "TotalStep: 8e5, TargetReward: 300, UsedTime: 1800s"
        args.env_eval = PreprocessEnv(env=gym.make('BipedalWalker-v3'))
        args.env = PreprocessEnv(env=gym.make('BipedalWalker-v3'), if_print=False)
        args.env.env_num = 1
        args.agent.cri_target = False

        args.gamma = 0.98
        args.if_per_or_gae = True
        args.break_step = int(8e6)

        # train_and_evaluate(args)
        args.env_num = 2
        args.worker_num = 4
        args.target_step = args.env.max_step * 16 // (args.env_num * args.worker_num)
        train_and_evaluate_mp(args)


def demo_discrete_action_off_policy():
    args = Arguments(if_on_policy=False)
    args.agent = AgentDoubleDQN()  # AgentDQN()
    args.visible_gpu = '0'

    if_train_cart_pole = 0
    if if_train_cart_pole:
        "TotalStep: 5e4, TargetReward: 200, UsedTime: 60s"
        args.env = PreprocessEnv(env='CartPole-v0')
        args.reward_scale = 2 ** -1
        args.target_step = args.env.max_step * 8

    if_train_lunar_lander = 1
    if if_train_lunar_lander:
        "TotalStep: 6e5, TargetReturn: 200, UsedTime: 1500s, LunarLander-v2, DQN"
        args.env = PreprocessEnv(env=gym.make('LunarLander-v2'))
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
        args.env = PreprocessEnv(env='CartPole-v0')
        args.reward_scale = 2 ** -1
        args.target_step = args.env.max_step * 8

    if_train_lunar_lander = 0
    if if_train_lunar_lander:
        "TotalStep: 6e5, TargetReturn: 200, UsedTime: 1500s, LunarLander-v2, PPO"
        args.env = PreprocessEnv(env=gym.make('LunarLander-v2'))
        args.repeat_times = 2 ** 5
        args.if_per_or_gae = True

    train_and_evaluate(args)


if __name__ == '__main__':
    # demo_continuous_action_off_policy()
    demo_continuous_action_on_policy()
    # demo_discrete_action_off_policy()
    # demo_discrete_action_on_policy()
