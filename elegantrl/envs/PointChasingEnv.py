from typing import Tuple

import numpy as np
import numpy.random as rd
import torch as th

ARY = np.ndarray
TEN = th.Tensor


class PointChasingEnv:
    def __init__(self, dim=2):
        self.dim = dim
        self.init_distance = 8.0

        # reset
        self.p0 = None  # position of point 0
        self.v0 = None  # velocity of point 0
        self.p1 = None  # position of point 1
        self.v1 = None  # velocity of point 1

        self.distance = None  # distance between point0 and point1
        self.cur_step = None  # current step number

        """env info"""
        self.env_name = "PointChasingEnv"
        self.state_dim = self.dim * 4
        self.action_dim = self.dim
        self.max_step = 2 ** 10
        self.if_discrete = False

    def reset(self, **_kwargs) -> Tuple[ARY, dict]:
        self.p0 = rd.normal(0, 1, size=self.dim)
        self.v0 = np.zeros(self.dim)

        self.p1 = rd.normal(-self.init_distance, 1, size=self.dim)
        self.v1 = np.zeros(self.dim)

        self.distance = ((self.p0 - self.p1) ** 2).sum() ** 0.5
        self.cur_step = 0

        state = self.get_state()
        return state, dict()

    def step(self, action: ARY) -> Tuple[ARY, ARY, bool, bool, dict]:
        action_l2 = (action ** 2).sum() ** 0.5
        action_l2 = max(action_l2, 1.0)
        action = action / action_l2

        self.v1 *= 0.75
        self.v1 += action
        self.p1 += self.v1 * 0.01

        self.v0 *= 0.50
        self.v0 += rd.rand(self.dim)
        self.p0 += self.v0 * 0.01

        """next_state"""
        next_state = self.get_state()

        """reward"""
        distance = ((self.p0 - self.p1) ** 2).sum() ** 0.5
        reward = self.distance - distance - action_l2 * 0.02
        self.distance = distance

        """done"""
        self.cur_step += 1

        terminal = (distance < self.dim) or (self.cur_step == self.max_step)
        truncate = False
        return next_state, reward, terminal, truncate, dict()

    def get_state(self) -> ARY:
        return np.hstack((self.p0, self.v0, self.p1, self.v1))

    @staticmethod
    def get_action(state: ARY) -> ARY:
        states_reshape = state.reshape((4, -1))
        p0 = states_reshape[0]
        p1 = states_reshape[2]
        return p0 - p1


class PointChasingVecEnv:
    def __init__(self, dim=2, env_num=32, sim_gpu_id=0):
        self.dim = dim
        self.init_distance = 8.0

        # reset
        self.p0s = None  # position
        self.v0s = None  # velocity
        self.p1s = None
        self.v1s = None

        self.distances = None  # a tensor of distance between point0 and point1
        self.cur_steps = None  # a tensor of current step number
        # env.step() is a function, so I can't name it `steps`

        """env info"""
        self.env_name = "PointChasingVecEnv"
        self.state_dim = self.dim * 4
        self.action_dim = self.dim
        self.max_step = 2 ** 10
        self.if_discrete = False

        self.num_envs = env_num
        self.device = th.device("cpu" if sim_gpu_id == -1 else f"cuda:{sim_gpu_id}")

    def reset(self, **_kwargs) -> Tuple[TEN, dict]:
        self.p0s = th.zeros((self.num_envs, self.dim), dtype=th.float32, device=self.device)
        self.v0s = th.zeros((self.num_envs, self.dim), dtype=th.float32, device=self.device)
        self.p1s = th.zeros((self.num_envs, self.dim), dtype=th.float32, device=self.device)
        self.v1s = th.zeros((self.num_envs, self.dim), dtype=th.float32, device=self.device)

        self.cur_steps = th.zeros(self.num_envs, dtype=th.float32, device=self.device)

        for env_i in range(self.num_envs):
            self.reset_env_i(env_i)

        self.distances = ((self.p0s - self.p1s) ** 2).sum(dim=1) ** 0.5

        state = self.get_state()
        return state, dict()

    def reset_env_i(self, i: int):
        self.p0s[i] = th.normal(0, 1, size=(self.dim,))
        self.v0s[i] = th.zeros((self.dim,))
        self.p1s[i] = th.normal(-self.init_distance, 1, size=(self.dim,))
        self.v1s[i] = th.zeros((self.dim,))

        self.cur_steps[i] = 0

    def step(self, actions: TEN) -> Tuple[TEN, TEN, TEN, TEN, dict]:
        """
        :param actions: [tensor] actions.shape == (num_envs, action_dim)
        :return: next_states [tensor] next_states.shape == (num_envs, state_dim)
        :return: rewards [tensor] rewards == (num_envs, )
        :return: terminal [tensor] terminal == (num_envs, ), done = 1. if done else 0.
        :return: None [None or dict]
        """
        # assert actions.get_device() == self.device.index
        actions_l2 = (actions ** 2).sum(dim=1, keepdim=True) ** 0.5
        actions_l2 = actions_l2.clamp_min(1.0)
        actions = actions / actions_l2

        self.v1s *= 0.75
        self.v1s += actions
        self.p1s += self.v1s * 0.01

        self.v0s *= 0.50
        self.v0s += th.rand(
            size=(self.num_envs, self.dim), dtype=th.float32, device=self.device
        )
        self.p0s += self.v0s * 0.01

        """reward"""
        distances = ((self.p0s - self.p1s) ** 2).sum(dim=1) ** 0.5
        rewards = self.distances - distances - actions_l2.squeeze(1) * 0.02
        self.distances = distances

        """done"""
        self.cur_steps += 1  # array
        terminal = (distances < self.dim) | (self.cur_steps == self.max_step)
        for env_i in range(self.num_envs):
            if terminal[env_i]:
                self.reset_env_i(env_i)
        terminal = terminal.type(th.float32)
        truncate = th.zeros(size=(self.num_envs,), dtype=th.bool, device=self.device)

        """next_state"""
        next_states = self.get_state()
        return next_states, rewards, terminal, truncate, dict()

    def get_state(self) -> TEN:
        return th.cat((self.p0s, self.v0s, self.p1s, self.v1s), dim=1)

    @staticmethod
    def get_action(states: TEN) -> TEN:
        states_reshape = states.reshape((states.shape[0], 4, -1))
        p0s = states_reshape[:, 0]
        p1s = states_reshape[:, 2]
        return p0s - p1s


class PointChasingDiscreteEnv(PointChasingEnv):
    def __init__(self, dim=2):
        PointChasingEnv.__init__(self, dim)
        self.env_name = "PointChasingDiscreteEnv"
        self.action_dim = 3 ** self.dim
        self.if_discrete = True

    def step(self, action: ARY) -> Tuple[ARY, ARY, bool, bool, dict]:
        action_ary = np.zeros(self.dim, dtype=np.float32)  # continuous_action
        for dim in range(self.dim):
            idx = (action // (3 ** dim)) % 3
            action_ary[dim] = idx - 1  # map `idx` to `value` using {0: -1, 1: 0, 2: +1}
        return PointChasingEnv.step(self, action_ary)

    def get_action(self, state: ARY) -> int:
        action_ary = PointChasingEnv.get_action(state)
        action_idx = 0
        for dim in range(self.dim):
            action_value = action_ary[dim]
            if action_value < -0.5:
                action_idx += dim ** 3 * 0
            elif action_value < +0.5:
                action_idx += dim ** 3 * 1
            else:
                action_idx += dim ** 3 * 2
        return action_idx


def check_chasing_env():
    env = PointChasingEnv()

    reward_sum = 0.0  # episode return
    reward_sum_list = []

    state = env.reset()
    for _ in range(env.max_step * 4):
        action = env.get_action(state)
        state, reward, terminal, truncate, _ = env.step(action)

        reward_sum += reward
        if terminal or truncate:
            print(f"{env.distance:8.4f}    {action.round(2)}")
            reward_sum_list.append(reward_sum)
            reward_sum = 0.0
            state = env.reset()

    print("len: ", len(reward_sum_list))
    print("mean:", np.mean(reward_sum_list))
    print("std: ", np.std(reward_sum_list))


def check_chasing_vec_env():
    env = PointChasingVecEnv(dim=2, env_num=2, sim_gpu_id=0)

    reward_sums = [
                      0.0,
                  ] * env.num_envs  # episode returns
    reward_sums_list = [
                           [],
                       ] * env.num_envs

    states = env.reset()
    for _ in range(env.max_step * 4):
        actions = env.get_action(states)
        states, rewards, terminal, truncate, _ = env.step(actions)

        dones = th.logical_or(terminal, truncate)
        for env_i in range(env.num_envs):
            reward_sums[env_i] += rewards[env_i].item()

            if dones[env_i]:
                print(f"{env.distances[env_i].item():8.4f}    {actions[env_i].detach().cpu().numpy().round(2)}")
                reward_sums_list[env_i].append(reward_sums[env_i])
                reward_sums[env_i] = 0.0

    reward_sums_list = np.array(reward_sums_list)
    print("shape:", reward_sums_list.shape)
    print("mean: ", np.mean(reward_sums_list, axis=1))
    print("std:  ", np.std(reward_sums_list, axis=1))


if __name__ == "__main__":
    check_chasing_env()
    check_chasing_vec_env()
