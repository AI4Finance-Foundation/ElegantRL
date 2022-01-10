from elegantrl.run import *

'''custom env'''


class PendulumEnv(gym.Wrapper):  # [ElegantRL.2021.11.11]
    def __init__(self, gym_env_id='Pendulum-v1', target_return=-200):
        # Pendulum-v0 gym.__version__ == 0.17.0
        # Pendulum-v1 gym.__version__ == 0.21.0
        super(PendulumEnv, self).__init__(env=gym.make(gym_env_id))

        # from elegantrl.envs.Gym import get_gym_env_info
        # get_gym_env_info(env, if_print=True)  # use this function to print the env information
        self.env_num = 1  # the env number of VectorEnv is greater than 1
        self.env_name = gym_env_id  # the name of this env.
        self.max_step = 200  # the max step of each episode
        self.state_dim = 3  # feature number of state
        self.action_dim = 1  # feature number of action
        self.if_discrete = False  # discrete action or continuous action
        self.target_return = target_return  # episode return is between (-1600, 0)

    def reset(self):
        return self.env.reset().astype(np.float32)

    def step(self, action: np.ndarray):
        # PendulumEnv set its action space as (-2, +2). It is bad.  # https://github.com/openai/gym/wiki/Pendulum-v0
        # I suggest to set action space as (-1, +1) when you design your own env.
        state, reward, done, info_dict = self.env.step(action * 2)  # state, reward, done, info_dict
        return state.astype(np.float32), reward, done, info_dict


'''demo'''


def demo_off_policy():
    gpu_id = 0
    args = None

    if_train_pendulum = 0
    if if_train_pendulum:
        "TotalStep: 1e5, TargetReward: -200, UsedTime: 600s"
        args = Arguments(AgentModSAC, env=PendulumEnv('Pendulum-v0'))
        args.learner_gpus = gpu_id
        args.reward_scale = 2 ** -1  # RewardRange: -1800 < -200 < -50 < 0
        args.gamma = 0.97
        args.target_step = args.max_step * 2
        args.eval_times = 2 ** 3

    if_train_lunar_lander = 1
    if if_train_lunar_lander:
        "TotalStep: 4e5, TargetReward: 200, UsedTime: 900s"
        # env = gym.make('LunarLanderContinuous-v2')
        # get_gym_env_args(env=env, if_print=True)
        env_func = gym.make
        env_args = {'env_num': 1,
                    'env_name': 'LunarLanderContinuous-v2',
                    'max_step': 1000,
                    'state_dim': 8,
                    'action_dim': 2,
                    'if_discrete': False,
                    'target_return': 200,

                    'id': 'LunarLanderContinuous-v2'}
        args = Arguments(AgentModSAC, env_func=env_func, env_args=env_args)

        args.target_step = args.max_step
        args.gamma = 0.99
        args.eval_times = 2 ** 5

    if_train_bipedal_walker = 0
    if if_train_bipedal_walker:
        "TotalStep: 8e5, TargetReward: 300, UsedTime: 1800s"
        env_func = gym.make
        env_args = {'env_num': 1,
                    'env_name': 'LunarLanderContinuous-v2',
                    'max_step': 1000,
                    'state_dim': 8,
                    'action_dim': 2,
                    'if_discrete': False,
                    'target_return': 200,

                    'id': 'LunarLanderContinuous-v2'}
        args = Arguments(AgentModSAC, env_func=env_func, env_args=env_args)
        args.target_step = args.max_step
        args.gamma = 0.98
        args.eval_times = 2 ** 4

    if_check = 0
    if if_check:
        train_and_evaluate(args)
    else:
        train_and_evaluate_mp(args)


def demo_on_policy():
    gpu_id = 1
    args = None

    if_train_pendulum = 0
    if if_train_pendulum:
        "TotalStep: 1e5, TargetReward: -200, UsedTime: 600s"
        args = Arguments(AgentPPO, env=PendulumEnv('Pendulum-v0'))
        args.learner_gpus = gpu_id
        args.reward_scale = 2 ** -1  # RewardRange: -1800 < -200 < -50 < 0
        args.gamma = 0.97
        args.target_step = args.max_step * 16
        args.eval_times = 2 ** 3

    if_train_lunar_lander = 1
    if if_train_lunar_lander:
        "TotalStep: 4e5, TargetReward: 200, UsedTime: 900s"
        # env = gym.make('LunarLanderContinuous-v2')
        # get_gym_env_args(env=env, if_print=True)
        env_func = gym.make
        env_args = {'env_num': 1,
                    'env_name': 'LunarLanderContinuous-v2',
                    'max_step': 1000,
                    'state_dim': 8,
                    'action_dim': 2,
                    'if_discrete': False,
                    'target_return': 200,

                    'id': 'LunarLanderContinuous-v2'}
        args = Arguments(AgentPPO, env_func=env_func, env_args=env_args)

        args.target_step = args.max_step * 4
        args.gamma = 0.99
        args.eval_times = 2 ** 5

    if_train_bipedal_walker = 0
    if if_train_bipedal_walker:
        "TotalStep: 8e5, TargetReward: 300, UsedTime: 1800s"
        env_func = gym.make
        env_args = {'env_num': 1,
                    'env_name': 'LunarLanderContinuous-v2',
                    'max_step': 1000,
                    'state_dim': 8,
                    'action_dim': 2,
                    'if_discrete': False,
                    'target_return': 200,

                    'id': 'LunarLanderContinuous-v2'}
        args = Arguments(AgentPPO, env_func=env_func, env_args=env_args)
        args.target_step = args.max_step * 2
        args.gamma = 0.98
        args.eval_times = 2 ** 4

    if_check = 0
    if if_check:
        train_and_evaluate(args)
    else:
        train_and_evaluate_mp(args)


if __name__ == '__main__':
    demo_off_policy()
    # demo_on_policy()
