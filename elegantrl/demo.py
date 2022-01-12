import sys

from elegantrl.run import *

'''custom env'''


class PendulumEnv(gym.Wrapper):  # [ElegantRL.2021.11.11]
    def __init__(self, gym_env_id='Pendulum-v1', target_return=-200):
        # Pendulum-v0 gym.__version__ == 0.17.0
        # Pendulum-v1 gym.__version__ == 0.21.0
        gym.logger.set_level(40)  # Block warning
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
    env_name = ['Pendulum-v0',
                'Pendulum-v1',
                'LunarLanderContinuous-v2',
                'BipedalWalker-v3'][ENV_ID]
    gpu_id = GPU_ID  # >=0 means GPU ID, -1 means CPU

    if env_name in {'Pendulum-v0', 'Pendulum-v1'}:
        env = PendulumEnv(env_name, target_return=-500)
        "TotalStep: 1e5, TargetReward: -200, UsedTime: 600s"
        args = Arguments(AgentModSAC, env)
        args.reward_scale = 2 ** -1  # RewardRange: -1800 < -200 < -50 < 0
        args.gamma = 0.97
        args.target_step = args.max_step * 2
        args.eval_times = 2 ** 3
    elif env_name == 'LunarLanderContinuous-v2':
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
    elif env_name == 'BipedalWalker-v3':
        "TotalStep: 8e5, TargetReward: 300, UsedTime: 1800s"
        env_func = gym.make
        env_args = {'env_num': 1,
                    'env_name': 'BipedalWalker-v3',
                    'max_step': 1600,
                    'state_dim': 24,
                    'action_dim': 4,
                    'if_discrete': False,
                    'target_return': 300,

                    'id': 'BipedalWalker-v3', }
        args = Arguments(AgentModSAC, env_func=env_func, env_args=env_args)
        args.target_step = args.max_step
        args.gamma = 0.98
        args.eval_times = 2 ** 4
    else:
        raise ValueError('env_name:', env_name)

    args.learner_gpus = gpu_id

    if_check = 0
    if if_check:
        train_and_evaluate(args)
    else:
        train_and_evaluate_mp(args)


def demo_on_policy():
    env_name = ['Pendulum-v0',
                'Pendulum-v1',
                'LunarLanderContinuous-v2',
                'BipedalWalker-v3'][ENV_ID]
    gpu_id = GPU_ID  # >=0 means GPU ID, -1 means CPU

    if env_name in {'Pendulum-v0', 'Pendulum-v1'}:
        env = PendulumEnv(env_name, target_return=-500)
        "TotalStep: 1e5, TargetReward: -200, UsedTime: 600s"
        args = Arguments(AgentPPO, env)
        args.reward_scale = 2 ** -1  # RewardRange: -1800 < -200 < -50 < 0
        args.gamma = 0.97
        args.target_step = args.max_step * 8
        args.eval_times = 2 ** 3
    elif env_name == 'LunarLanderContinuous-v2':
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

        args.target_step = args.max_step * 2
        args.gamma = 0.99
        args.eval_times = 2 ** 5
    elif env_name == 'BipedalWalker-v3':
        "TotalStep: 8e5, TargetReward: 300, UsedTime: 1800s"
        env_func = gym.make
        env_args = {'env_num': 1,
                    'env_name': 'BipedalWalker-v3',
                    'max_step': 1600,
                    'state_dim': 24,
                    'action_dim': 4,
                    'if_discrete': False,
                    'target_return': 300,

                    'id': 'BipedalWalker-v3', }
        args = Arguments(AgentPPO, env_func=env_func, env_args=env_args)
        args.target_step = args.max_step * 4
        args.gamma = 0.98
        args.eval_times = 2 ** 4
    else:
        raise ValueError('env_name:', env_name)

    args.learner_gpus = gpu_id

    if_check = 1
    if if_check:
        train_and_evaluate(args)
    else:
        train_and_evaluate_mp(args)


if __name__ == '__main__':
    GPU_ID = 0  # int(sys.argv[1])
    ENV_ID = 3  # int(sys.argv[2])

    demo_off_policy()
    # demo_on_policy()
