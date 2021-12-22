import torch
import numpy as np


class ChasingVecEnv:
    def __init__(self, dim=2, env_num=4096, device_id=0):
        self.dim = dim
        self.init_distance = 8.0

        # reset
        self.p0s = None  # position
        self.v0s = None  # velocity
        self.p1s = None
        self.v1s = None

        self.distances = None
        self.steps = None

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

        self.steps = np.zeros(self.env_num, dtype=np.int)

        for env_i in range(self.env_num):
            self.reset_env_i(env_i)

        self.distances = ((self.p0s - self.p1s) ** 2).sum(dim=1) ** 0.5

        return self.get_state()

    def reset_env_i(self, i):
        self.p0s[i] = torch.normal(0, 1, size=(self.dim,))
        self.v0s[i] = torch.zeros((self.dim,))
        self.p1s[i] = torch.normal(-self.init_distance, 1, size=(self.dim,))
        self.v1s[i] = torch.zeros((self.dim,))

        self.steps[i] = 0

    def step(self, action1s):
        action0s = torch.rand(size=(self.env_num, self.dim), dtype=torch.float32, device=self.device)
        action0s_l2 = (action0s ** 2).sum(dim=1, keepdim=True) ** 0.5
        action0s = action0s / action0s_l2.clamp_min(1.0)

        self.v0s *= 0.50
        self.v0s += action0s
        self.p0s += self.v0s * 0.01

        action1s_l2 = (action1s ** 2).sum(dim=1, keepdim=True) ** 0.5
        action1s = action1s / action1s_l2.clamp_min(1.0)

        self.v1s *= 0.75
        self.v1s += action1s
        self.p1s += self.v1s * 0.01

        '''reward'''
        distances = ((self.p0s - self.p1s) ** 2).sum(dim=1) ** 0.5
        rewards = self.distances - distances - action1s_l2.squeeze(1) * 0.02
        self.distances = distances

        '''done'''
        self.steps += 1  # array
        masks = torch.zeros(self.env_num, dtype=torch.float32, device=self.device)
        for env_i in range(self.env_num):
            done = 0
            if distances[env_i] < 1:
                done = 1
                rewards[env_i] += self.init_distance
            elif self.steps[env_i] == self.max_step:
                done = 1

            if done:
                self.reset_env_i(env_i)
            masks[env_i] = done

        '''next_state'''
        next_states = self.get_state()
        return next_states, rewards, masks, None

    def get_state(self):
        return torch.cat((self.p0s, self.v0s, self.p1s, self.v1s), dim=1)


def vec_policy(states):
    states_reshape = states.reshape((states.shape[0], 4, -1))
    p0s = states_reshape[:, 0]
    p1s = states_reshape[:, 2]
    actions = p0s - p1s
    return actions


def check_env():
    env = ChasingVecEnv(dim=2, env_num=4096, device_id=0)

    rewards = [0.0, ] * env.env_num  # episode returns
    rewards_list = [[], ] * env.env_num

    states = env.reset()
    for _ in range(env.max_step * 4):
        actions = vec_policy(states)
        states, rewards, masks, _ = env.step(actions)

        for env_i in range(env.env_num):
            rewards[env_i] += rewards[env_i].item()

            if masks[env_i]:
                print(f"{env.distances[env_i].item():8.4f}    {actions[env_i].detach().cpu().numpy().round(2)}")
                rewards_list[env_i].append(rewards[env_i])
                rewards[env_i] = 0.0

    rewards_list = np.array(rewards_list)
    print('shape:', rewards_list.shape)
    print('mean: ', np.mean(rewards_list, axis=1))
    print('std:  ', np.std(rewards_list, axis=1))


if __name__ == '__main__':
    check_env()
    exit()
