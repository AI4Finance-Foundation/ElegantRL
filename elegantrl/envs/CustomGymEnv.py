import gym
import numpy as np

'''custom env'''


class PendulumEnv(gym.Wrapper):  # [ElegantRL.2022.04.04]
    def __init__(self, gym_env_id='Pendulum-v1', target_return=-200):
        if gym.__version__ < '0.18.0':
            gym_env_id = 'Pendulum-v0'
        elif gym.__version__ >= '0.20.0':
            gym_env_id = 'Pendulum-v1'
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

        print(f'\n| {self.__class__.__name__}: Pendulum Env set its action space as (-2, +2).'
              f'\n| And we scale the action, and set the action space as (-1, +1).'
              f'\n| So do not use your policy network on raw env directly.')

    def reset(self):
        return self.env.reset().astype(np.float32)

    def step(self, action: np.ndarray):
        # PendulumEnv set its action space as (-2, +2). It is bad.  # https://github.com/openai/gym/wiki/Pendulum-v0
        # I suggest to set action space as (-1, +1) when you design your own env.
        state, reward, done, info_dict = self.env.step(action * 2)  # state, reward, done, info_dict
        return state.astype(np.float32), reward, done, info_dict


class GymNormaEnv(gym.Wrapper):  # [ElegantRL.2022.04.04] # todo plan
    def __init__(self, gym_env_id='BipdealWalker-v3', target_return=-200):
        gym.logger.set_level(40)  # Block warning
        super(GymNormaEnv, self).__init__(env=gym.make(gym_env_id))

        # from elegantrl.envs.Gym import get_gym_env_info
        # get_gym_env_info(env, if_print=True)  # use this function to print the env information
        self.env_num = 1  # the env number of VectorEnv is greater than 1
        self.env_name = gym_env_id  # the name of this env.
        self.max_step = 200  # the max step of each episode
        self.state_dim = 3  # feature number of state
        self.action_dim = 1  # feature number of action
        self.if_discrete = False  # discrete action or continuous action
        self.target_return = target_return  # episode return is between (-1600, 0)

        if gym_env_id.find('BipdealWalker') >= 0:
            state_avg = np.array([1.0143e-05, -3.1358e-07, 1.3466e-05, -1.2471e-06, -2.8488e-05,
                                  -1.3139e-06, 3.1929e-05, -6.0103e-06, 6.6120e-05, 7.8306e-05,
                                  1.6425e-07, 5.2854e-06, -2.2617e-06, 7.5263e-05, 3.9760e-05,
                                  4.0206e-05, 4.1609e-05, 4.4142e-05, 4.8155e-05, 5.4313e-05,
                                  6.3924e-05, 7.9849e-05, 1.0633e-04, 1.3038e-04])
            state_std = np.array([2.3532e-05, 4.6646e-06, 8.3879e-06, 6.6188e-06, 4.0461e-05, 6.8184e-05,
                                  4.2910e-05, 9.2036e-05, 6.3987e-05, 4.0925e-05, 6.0233e-05, 3.7091e-05,
                                  7.2178e-05, 6.2552e-05, 5.1780e-06, 5.2399e-06, 5.4365e-06, 5.7947e-06,
                                  6.3670e-06, 7.2504e-06, 8.6473e-06, 1.1004e-05, 1.0739e-05,
                                  1.4903e-06])

    def reset(self):
        return self.env.reset().astype(np.float32)

    def step(self, action: np.ndarray):
        # PendulumEnv set its action space as (-2, +2). It is bad.  # https://github.com/openai/gym/wiki/Pendulum-v0
        # I suggest to set action space as (-1, +1) when you design your own env.
        state, reward, done, info_dict = self.env.step(action * 2)  # state, reward, done, info_dict
        return state.astype(np.float32), reward, done, info_dict


class HumanoidEnv(gym.Wrapper):  # [ElegantRL.2021.11.11]
    def __init__(self, gym_env_id='Humanoid-v3', target_return=3000):
        gym.logger.set_level(40)  # Block warning
        super(HumanoidEnv, self).__init__(env=gym.make(gym_env_id))

        # from elegantrl.envs.Gym import get_gym_env_info
        # get_gym_env_info(env, if_print=True)  # use this function to print the env information
        self.env_num = 1  # the env number of VectorEnv is greater than 1
        self.env_name = gym_env_id  # the name of this env.
        self.max_step = 1000  # the max step of each episode
        self.state_dim = 376  # feature number of state
        self.action_dim = 17  # feature number of action
        self.if_discrete = False  # discrete action or continuous action
        self.target_return = target_return  # episode return is between (-1600, 0)

        print(f'\n| {self.self.__class__.__name__}: MuJoCo Humanoid Env set its action space as (-0.4, +0.4).'
              f'\n| And we scale the action, and set the action space as (-1, +1).'
              f'\n| So do not use your policy network on raw env directly.')

    def reset(self):
        return self.env.reset()

    def step(self, action: np.ndarray):
        # MuJoCo Humanoid Env set its action space as (-0.4, +0.4). It is bad.
        # I suggest to set action space as (-1, +1) when you design your own env.
        # action_space.high = 0.4
        # action_space.low = -0.4
        state, reward, done, info_dict = self.env.step(action * 2.5)  # state, reward, done, info_dict
        return state.astype(np.float32), reward, done, info_dict
