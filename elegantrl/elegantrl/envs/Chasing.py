import gym
import torch
import numpy as np
import numpy.random as rd


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
        return self.env.reset()

    def step(self, action: np.ndarray):
        # PendulumEnv set its action space as (-2, +2). It is bad.  # https://github.com/openai/gym/wiki/Pendulum-v0
        # I suggest to set action space as (-1, +1) when you design your own env.
        return self.env.step(action * 2)  # state, reward, done, info_dict


class ChasingEnv:
    def __init__(self, dim=2):
        self.dim = dim
        self.init_distance = 8.

        # reset
        self.p0 = None  # position of point 0
        self.v0 = None  # velocity of point 0
        self.p1 = None  # position of point 1
        self.v1 = None  # velocity of point 1

        self.distance = None  # distance between point0 and point1
        self.cur_step = None  # current step number

        '''env info'''
        self.env_name = 'GoAfterEnv'
        self.state_dim = self.dim * 4
        self.action_dim = self.dim
        self.max_step = 2 ** 10
        self.if_discrete = False
        self.target_return = 6.3

    def reset(self):
        self.p0 = rd.normal(0, 1, size=self.dim)
        self.v0 = np.zeros(self.dim)

        self.p1 = rd.normal(-self.init_distance, 1, size=self.dim)
        self.v1 = np.zeros(self.dim)

        self.distance = ((self.p0 - self.p1) ** 2).sum() ** 0.5
        self.cur_step = 0

        return self.get_state()

    def step(self, action):
        action_l2 = (action ** 2).sum() ** 0.5
        action_l2 = max(action_l2, 1.)
        action = action / action_l2

        self.v0 *= 0.50
        self.v0 += rd.rand(self.dim)
        self.p0 += self.v0 * 0.01

        self.v1 *= 0.75
        self.v1 += action
        self.p1 += self.v1 * 0.01

        '''next_state'''
        next_state = self.get_state()

        '''reward'''
        distance = ((self.p0 - self.p1) ** 2).sum() ** 0.5
        reward = self.distance - distance - action_l2 * 0.02
        self.distance = distance

        '''done'''
        self.cur_step += 1

        done = False
        if distance < self.dim:
            done = True
            reward += self.init_distance
        if self.cur_step == self.max_step:
            done = True
        return next_state, reward, done, None

    def get_state(self):
        return np.hstack((self.p0, self.v0, self.p1, self.v1))

    @staticmethod
    def get_action(state):
        states_reshape = state.reshape((4, -1))
        p0 = states_reshape[0]
        p1 = states_reshape[2]
        action = p0 - p1
        return action


class ChasingVecEnv:
    def __init__(self, dim=2, env_num=32, device_id=0):
        self.dim = dim
        self.init_distance = 8.0

        # reset
        self.p0s = None  # position
        self.v0s = None  # velocity
        self.p1s = None
        self.v1s = None

        self.distances = None  # a list of distance between point0 and point1
        self.cur_steps = None  # a list of current step number
        # env.step() is a function, so I can't name it `steps`

        '''env info'''
        self.env_name = 'ChasingVecEnv'
        self.state_dim = self.dim * 4
        self.action_dim = self.dim
        self.max_step = 2 ** 10
        self.if_discrete = False
        self.target_return = 6.3

        self.env_num = env_num
        self.device = torch.device(f"cuda:{device_id}")

    def reset(self):
        self.p0s = torch.zeros((self.env_num, self.dim), dtype=torch.float32, device=self.device)
        self.v0s = torch.zeros((self.env_num, self.dim), dtype=torch.float32, device=self.device)
        self.p1s = torch.zeros((self.env_num, self.dim), dtype=torch.float32, device=self.device)
        self.v1s = torch.zeros((self.env_num, self.dim), dtype=torch.float32, device=self.device)

        self.cur_steps = np.zeros(self.env_num, dtype=np.int)

        for env_i in range(self.env_num):
            self.reset_env_i(env_i)

        self.distances = ((self.p0s - self.p1s) ** 2).sum(dim=1) ** 0.5

        return self.get_state()

    def reset_env_i(self, i):
        self.p0s[i] = torch.normal(0, 1, size=(self.dim,))
        self.v0s[i] = torch.zeros((self.dim,))
        self.p1s[i] = torch.normal(-self.init_distance, 1, size=(self.dim,))
        self.v1s[i] = torch.zeros((self.dim,))

        self.cur_steps[i] = 0

    def step(self, actions):
        """
        :param actions: [tensor] actions.shape == (env_num, action_dim)
        :return: next_states [tensor] next_states.shape == (env_num, state_dim)
        :return: rewards [tensor] rewards == (env_num, )
        :return: dones [tensor] dones == (env_num, ), done = 1. if done else 0.
        :return: None [None or dict]
        """
        action0s = torch.rand(size=(self.env_num, self.dim), dtype=torch.float32, device=self.device)
        action0s_l2 = (action0s ** 2).sum(dim=1, keepdim=True) ** 0.5
        action0s = action0s / action0s_l2.clamp_min(1.0)

        self.v0s *= 0.50
        self.v0s += action0s
        self.p0s += self.v0s * 0.01

        actions_l2 = (actions ** 2).sum(dim=1, keepdim=True) ** 0.5
        actions = actions / actions_l2.clamp_min(1.0)

        self.v1s *= 0.75
        self.v1s += actions
        self.p1s += self.v1s * 0.01

        '''reward'''
        distances = ((self.p0s - self.p1s) ** 2).sum(dim=1) ** 0.5
        rewards = self.distances - distances - actions_l2.squeeze(1) * 0.02
        self.distances = distances

        '''done'''
        self.cur_steps += 1  # array
        dones = torch.zeros(self.env_num, dtype=torch.float32, device=self.device)
        for env_i in range(self.env_num):
            done = 0
            if distances[env_i] < self.dim:
                done = 1
                rewards[env_i] += self.init_distance
            elif self.cur_steps[env_i] == self.max_step:
                done = 1

            if done:
                self.reset_env_i(env_i)
            dones[env_i] = done

        '''next_state'''
        next_states = self.get_state()
        return next_states, rewards, dones, None

    def get_state(self):
        return torch.cat((self.p0s, self.v0s, self.p1s, self.v1s), dim=1)

    @staticmethod
    def get_action(states):
        states_reshape = states.reshape((states.shape[0], 4, -1))
        p0s = states_reshape[:, 0]
        p1s = states_reshape[:, 2]
        actions = p0s - p1s
        return actions


def check_chasing_env():
    env = ChasingEnv()

    reward_sum = 0.0  # episode return
    reward_sum_list = list()

    state = env.reset()
    for _ in range(env.max_step * 4):
        action = env.get_action(state)
        state, reward, done, _ = env.step(action)

        reward_sum += reward
        if done:
            print(f"{env.distance:8.4f}    {action.round(2)}")
            reward_sum_list.append(reward_sum)
            reward_sum = 0.0
            state = env.reset()

    print('len: ', len(reward_sum_list))
    print('mean:', np.mean(reward_sum_list))
    print('std: ', np.std(reward_sum_list))


def check_chasing_vec_env():
    env = ChasingVecEnv(dim=2, env_num=2, device_id=0)

    reward_sums = [0.0, ] * env.env_num  # episode returns
    reward_sums_list = [[], ] * env.env_num

    states = env.reset()
    for _ in range(env.max_step * 4):
        actions = env.get_action(states)
        states, rewards, dones, _ = env.step(actions)

        for env_i in range(env.env_num):
            reward_sums[env_i] += rewards[env_i].item()

            if dones[env_i]:
                print(f"{env.distances[env_i].item():8.4f}    {actions[env_i].detach().cpu().numpy().round(2)}")
                reward_sums_list[env_i].append(reward_sums[env_i])
                reward_sums[env_i] = 0.0

    reward_sums_list = np.array(reward_sums_list)
    print('shape:', reward_sums_list.shape)
    print('mean: ', np.mean(reward_sums_list, axis=1))
    print('std:  ', np.std(reward_sums_list, axis=1))


if __name__ == '__main__':
    check_chasing_env()
    check_chasing_vec_env()
