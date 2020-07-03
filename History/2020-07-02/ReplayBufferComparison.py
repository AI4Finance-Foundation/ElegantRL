import os
import sys
from time import time as timer

import gym
import torch
import numpy as np
import numpy.random as rd

'''
2020-0505 ZenJiaHao Github: YonV1943
Compare the running speed of different ReplayBuffer(Memory) implement.

ReplayBuffer    UsedTime(s) Storage(memories)
MemoryList:     24          list()
MemoryTuple:    20          collections.namedtuple
MemoryArray:    13          numpy.array
MemoryTensor:   13          torch.tensor (GPU/CPU)
'''


class BufferList:
    def __init__(self, memo_max_len):
        self.memories = list()

        self.max_len = memo_max_len
        self.now_len = len(self.memories)

    def add_memo(self, memory_tuple):
        self.memories.append(memory_tuple)

    def init_after_add_memo(self):
        del_len = len(self.memories) - self.max_len
        if del_len > 0:
            del self.memories[:del_len]
            # print('Length of Deleted Memories:', del_len)

        self.now_len = len(self.memories)

    def random_sample(self, batch_size, device):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # indices = rd.choice(self.memo_len, batch_size, replace=False)  # why perform worse?
        # indices = rd.choice(self.memo_len, batch_size, replace=True)  # why perform better?
        # same as:
        indices = rd.randint(self.now_len, size=batch_size)

        '''convert list into array'''
        arrays = [list()
                  for _ in range(5)]  # len(self.memories[0]) == 5
        for index in indices:
            items = self.memories[index]
            for item, array in zip(items, arrays):
                array.append(item)

        '''convert array into torch.tensor'''
        tensors = [torch.tensor(np.array(ary), dtype=torch.float32, device=device)
                   for ary in arrays]
        return tensors


class BufferTuple:
    def __init__(self, memo_max_len):
        self.memories = list()

        self.max_len = memo_max_len
        self.now_len = None  # init in init_after_add_memo()

        from collections import namedtuple
        self.transition = namedtuple(
            'Transition', ('reward', 'mask', 'state', 'action', 'next_state',)
        )

    def add_memo(self, args):
        self.memories.append(self.transition(*args))

    def init_after_add_memo(self):
        del_len = len(self.memories) - self.max_len
        if del_len > 0:
            del self.memories[:del_len]
            # print('Length of Deleted Memories:', del_len)

        self.now_len = len(self.memories)

    def random_sample(self, batch_size, device):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # indices = rd.choice(self.memo_len, batch_size, replace=False)  # why perform worse?
        # indices = rd.choice(self.memo_len, batch_size, replace=True)  # why perform better?
        # same as:
        indices = rd.randint(self.now_len, size=batch_size)

        '''convert tuple into array'''
        arrays = self.transition(*zip(*[self.memories[i] for i in indices]))

        '''convert array into torch.tensor'''
        tensors = [torch.tensor(np.array(ary), dtype=torch.float32, device=device)
                   for ary in arrays]
        return tensors


class BufferArray:  # 2020-05-20
    def __init__(self, memo_max_len, state_dim, action_dim, ):
        memo_dim = 1 + 1 + state_dim + action_dim + state_dim
        self.memories = np.empty((memo_max_len, memo_dim), dtype=np.float32)

        self.next_idx = 0
        self.is_full = False
        self.max_len = memo_max_len
        self.now_len = self.max_len if self.is_full else self.next_idx

        self.state_idx = 1 + 1 + state_dim  # reward_dim==1, done_dim==1
        self.action_idx = self.state_idx + action_dim

    def add_memo(self, memo_tuple):
        self.memories[self.next_idx] = np.hstack(memo_tuple)
        self.next_idx = self.next_idx + 1
        if self.next_idx >= self.max_len:
            self.is_full = True
            self.next_idx = 0

    def init_after_add_memo(self):
        self.now_len = self.max_len if self.is_full else self.next_idx

    def random_sample(self, batch_size, device):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # indices = rd.choice(self.memo_len, batch_size, replace=False)  # why perform worse?
        # indices = rd.choice(self.memo_len, batch_size, replace=True)  # why perform better?
        # same as:
        indices = rd.randint(self.now_len, size=batch_size)

        memory = self.memories[indices]
        memory = torch.tensor(memory, device=device)

        '''convert array into torch.tensor'''
        tensors = (
            memory[:, 0:1],  # rewards
            memory[:, 1:2],  # masks, mark == (1-float(done)) * gamma
            memory[:, 2:self.state_idx],  # states
            memory[:, self.state_idx:self.action_idx],  # actions
            memory[:, self.action_idx:],  # next_states
        )
        return tensors


def uniform_exploration(env, max_step, max_action, gamma, reward_scale, memo, action_dim):
    state = env.reset()

    rewards = list()
    reward_sum = 0.0
    steps = list()
    step = 0

    global_step = 0
    while global_step < max_step:
        # action = np.tanh(rd.normal(0, 0.5, size=action_dim))  # zero-mean gauss exploration
        action = rd.uniform(-1.0, +1.0, size=action_dim)  # uniform exploration

        next_state, reward, done, _ = env.step(action * max_action)
        reward_sum += reward
        step += 1

        adjust_reward = reward * reward_scale
        mask = 0.0 if done else gamma
        memo.add_memo((adjust_reward, mask, state, action, next_state))

        state = next_state
        if done:
            rewards.append(reward_sum)
            steps.append(step)
            global_step += step

            state = env.reset()  # reset the environment
            reward_sum = 0.0
            step = 1

    memo.init_after_add_memo()
    return rewards, steps


def run_compare_speed_of_replay_buffer():
    from AgentRun import get_env_info

    os.environ['CUDA_VISIBLE_DEVICES'] = '3'  # sys.argv[-1][-4]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 2 ** 8
    max_step = 2 ** 10
    gamma = 0.99
    reward_scale = 1
    memo_max_len = 2 ** 13

    start_time = timer()
    for env_name in ("LunarLanderContinuous-v2", "BipedalWalker-v3"):

        env = gym.make(env_name)
        state_dim, action_dim, max_action, target_reward = get_env_info(env)

        # memo = MemoryList(memo_max_len)
        # memo = MemoryTuple(memo_max_len)
        memo = BufferArray(memo_max_len, state_dim, action_dim)

        uniform_exploration(env, max_step, max_action, gamma, reward_scale, memo, action_dim)
        for i in range(8):
            uniform_exploration(env, max_step, max_action, gamma, reward_scale, memo, action_dim)

            for _ in range(max_step):
                batches = memo.random_sample(batch_size, device)

                for batch in batches:
                    assert torch.is_tensor(batch)

    print("Used Time: {:.1f}".format(timer() - start_time))


if __name__ == '__main__':
    run_compare_speed_of_replay_buffer()
