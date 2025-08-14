import json
import math
import os
import random
import threading
from collections import namedtuple

import numpy as np
import torch

from rlsolver.methods.eco_s2v.config import *

Transition = namedtuple(
    'Transition', ('state', 'action', 'reward', 'state_next', 'done')
)


class TestMetric(Enum):
    CUMULATIVE_REWARD = 1
    BEST_ENERGY = 2
    ENERGY_ERROR = 3
    MAX_CUT = 4
    FINAL_CUT = 5


def set_global_seed(seed, env):
    torch.manual_seed(seed)
    env.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class ReplayBuffer:
    def __init__(self, capacity):
        self._capacity = capacity
        self._memory = {}
        self._position = 0

        self.next_batch_process = None
        self.next_batch_size = None
        self.next_batch_device = None
        self.next_batch = None

    def add(self, *args):
        """
        Saves a transition.
        """
        if self.next_batch_process is not None:
            # Don't add to the buffer when sampling from it.
            self.next_batch_process.join()
        if ALG == Alg.eeco:
            for i in range(len(args[0])):
                self._memory[self._position] = Transition(args[0][i], args[1][i], args[2][i], args[3][i], args[4][i], )
                self._position = (self._position + 1) % self._capacity
        else:
            self._memory[self._position] = Transition(*args)
            self._position = (self._position + 1) % self._capacity

    def _prepare_sample(self, batch_size, device=None):
        self.next_batch_size = batch_size
        self.next_batch_device = device

        batch = random.sample(list(self._memory.values()), batch_size)

        self.next_batch = [torch.stack(tensors).to(device) for tensors in zip(*batch)]
        self.next_batch_ready = True

    def launch_sample(self, *args):
        self.next_batch_process = threading.Thread(target=self._prepare_sample, args=args)
        self.next_batch_process.start()

    def sample(self, batch_size, device=None):
        """
        Samples a batch of Transitions, with the tensors already stacked
        and transfered to the specified device.
        Return a list of tensors in the order specified in Transition.
        """
        if self.next_batch_process is not None:
            self.next_batch_process.join()
        else:
            self.launch_sample(batch_size, device)
            self.sample(batch_size, device)

        if self.next_batch_size == batch_size and self.next_batch_device == device:
            next_batch = self.next_batch
            self.launch_sample(batch_size, device)
            return next_batch
        else:
            self.launch_sample(batch_size, device)
            self.sample(batch_size, device)

    def __len__(self):
        return len(self._memory)


class eeco_ReplayBuffer:
    def __init__(self, capacity, sampling_patten=None, device=None, n_matrix=None, matrix_index_cycle=None):
        self._capacity = capacity
        self.device = device
        # 调试
        self._memory = {'state': torch.zeros((self._capacity, 7, NUM_TRAIN_NODES), dtype=torch.float16, device=device),
                        'action': torch.zeros((self._capacity,), dtype=torch.long, device=device),
                        'reward': torch.zeros((self._capacity,), dtype=torch.float16, device=device),
                        'state_next': torch.zeros((self._capacity, 7, NUM_TRAIN_NODES), dtype=torch.float16, device=device),
                        'done': torch.zeros((self._capacity,), dtype=torch.bool, device=device, ),
                        'matrix_index': torch.zeros((self._capacity,), dtype=torch.long, device=device),
                        'score': torch.zeros((self._capacity,), dtype=torch.float, device=device)}

        self._position = 0
        self.matrix_index_cycle = matrix_index_cycle
        self.sampling_patten = sampling_patten
        self.matrix = torch.zeros((n_matrix, NUM_TRAIN_NODES, NUM_TRAIN_NODES), dtype=torch.float, device=device)

    def add(self, state, action, reward, state_next, done, score):
        batch_size = state.shape[0]
        # indices = [(self._position + i) % self._capacity for i in range(batch_size)]
        indices = (self._position + torch.arange(batch_size, device=self.device)) % self._capacity

        self._memory['state'][indices] = state[:, :7, :].to(self.device)
        self._memory['action'][indices] = action.to(self.device)
        self._memory['reward'][indices] = reward.to(self.device)
        self._memory['state_next'][indices] = state_next[:, :7, :].to(self.device)
        self._memory['done'][indices] = done.to(self.device)
        self._memory['score'][indices] = score.to(self.device)
        self._memory['matrix_index'][indices] = self.matrix_indices
        self._position = (self._position + batch_size) % self._capacity

    def record_matrix(self, matrix):
        start_index = next(self.matrix_index_cycle)
        self.matrix_indices = start_index + torch.arange(matrix.shape[0], device=self.device)
        self.matrix[self.matrix_indices] = matrix.to(self.device)

    # 有cat版本
    # def sample(self, batch_size,biased=None):
    #     if biased is not None:
    #         indices = torch.randint(0,biased,(batch_size,),dtype=torch.long,device=self.device)
    #     else:
    #         indices = torch.randint(0,self._capacity,(batch_size,),dtype=torch.long,device=self.device)
    #     traj = []
    #     matrix_index = self._memory['matrix_index'][indices]
    #     matrix = self.matrix[matrix_index]
    #     for key in list(self._memory.keys())[:-2]:
    #         if key == 'state' or key == 'state_next':
    #             traj.append(torch.cat((self._memory[key][indices],matrix), dim=-2).to(TRAIN_DEVICE))
    #         else:
    #             traj.append(self._memory[key][indices].to(TRAIN_DEVICE))
    #     return traj
    # 去掉cat
    def sample(self, batch_size, biased=None):
        if biased is not None:
            indices = torch.randint(0, biased, (batch_size,), dtype=torch.long, device=self.device)
        else:
            indices = torch.randint(0, self._capacity, (batch_size,), dtype=torch.long, device=self.device)
        traj = []
        matrix_index = self._memory['matrix_index'][indices]
        matrix = self.matrix[matrix_index]
        for key in list(self._memory.keys())[:-2]:
            if key == 'state' or key == 'state_next':
                traj.append(torch.cat((self._memory[key][indices], matrix), dim=-2).to(TRAIN_DEVICE))
            else:
                traj.append(self._memory[key][indices].to(TRAIN_DEVICE))
        return traj

    # def sample(self, batch_size,biased=None):
    #     if self.sampling_patten == "best_score":
    #         indices = torch.argsort(self._memory['score'],descending=True)[:batch_size]
    #     elif self.sampling_patten == "best_reward":
    #         indices = torch.argsort(self._memory['reward'],descending=True)[:batch_size]
    #     else:
    #         if biased is not None:
    #             indices = torch.randint(0,biased,(batch_size,),dtype=torch.long,device=TRAIN_DEVICE)
    #         else:
    #             indices = torch.randint(0,self._capacity,(batch_size,),dtype=torch.long,device=TRAIN_DEVICE)
    #     return [self._memory[key][indices] for key in list(self._memory.keys())[:-1]]


class PrioritisedReplayBuffer:

    def __init__(self, capacity=10000, alpha=0.7, beta0=0.5):

        # The capacity of the replay buffer.
        self._capacity = capacity

        # A binary (max-)heap of the buffer contents, sorted by the td_error <--> priority.
        self.priority_heap = {}  # heap_position : [buffer_position, td_error, transition]

        # Maps a buffer position (when the transition was added) to the position of the
        # transition in the priority_heap.
        self.buffer2heap = {}  # buffer_position : heap_position

        # The current position in the replay buffer.  Starts at 1 for ease of binary-heap calcs.
        self._buffer_position = 1

        # Flag for when the replay buffer reaches max capicity.
        self.full = False

        self.alpha = alpha
        self.beta = beta0
        self.beta_step = 0

        self.partitions = []
        self.probabilities = []
        self.__partitions_fixed = False

    def __get_max_td_err(self):
        try:
            return self.priority_heap[0][1]
        except KeyError:
            # Nothing exists in the priority heap yet!
            return 1

    def add(self, *args):
        """
        Add the transition described by *args : (state, action, reward, state_next, done), to the
        memory.
        """
        # By default a new transition has equal highest priority in the heap.
        trans = [self._buffer_position, self.__get_max_td_err(), Transition(*args)]
        try:
            # Find the heap position of the transition to be replaced
            heap_pos = self.buffer2heap[self._buffer_position]
            self.full = True  # We found a transition in this buffer slot --> the memory is at capacity.
        except KeyError:
            # No transition in the buffer slot, therefore we will be adding one fresh.
            heap_pos = self._buffer_position

        # Update the heap, associated data stuctures and re-sort.
        self.__update_heap(heap_pos, trans)
        self.up_heap(heap_pos)
        if self.full:
            self.down_heap(heap_pos)

        # Iterate to the next buffer position.
        self._buffer_position = (self._buffer_position % self._capacity) + 1

    def __update_heap(self, heap_pos, val):
        """
        heapList[heap_pos] <-- val = [buffer_position, td_error, transition]
        """
        self.priority_heap[heap_pos] = val
        self.buffer2heap[val[0]] = heap_pos

    def up_heap(self, i):
        """
        Iteratively swap heap items with their parents until they are in the correct order.
        """
        if i >= 2:
            i_parent = i // 2
            if self.priority_heap[i_parent][1] < self.priority_heap[i][1]:
                tmp = self.priority_heap[i]
                self.__update_heap(i, self.priority_heap[i_parent])
                self.__update_heap(i_parent, tmp)
                self.up_heap(i_parent)

    def down_heap(self, i):
        """
        Iteratively swap heap items with their children until they are in the correct order.
        """
        i_largest = i
        left = 2 * i
        right = 2 * i + 1

        size = self._capacity if self.full else len(self)

        if left < size and self.priority_heap[left][1] > self.priority_heap[i_largest][1]:
            i_largest = left
        if right < size and self.priority_heap[right][1] > self.priority_heap[i_largest][1]:
            i_largest = right

        if i_largest != i:
            tmp = self.priority_heap[i]
            self.__update_heap(i, self.priority_heap[i_largest])
            self.__update_heap(i_largest, tmp)
            self.down_heap(i_largest)

    def rebalance(self):
        """
        rebalance priority_heap
        """
        sort_array = sorted(self.priority_heap.values(), key=lambda x: x[1], reverse=True)
        # reconstruct priority_queue
        self.priority_heap.clear()
        self.buffer2heap.clear()

        count = 1
        while count <= self._capacity:
            self.__update_heap(count, sort_array[count - 1])
            count += 1

        # sort the heap
        for i in range(self._capacity // 2, 1, -1):
            self.down_heap(i)

    def update_partitions(self, num_partitions):

        # P(t_i) = p_i^alpha / Sum_k(p_k^alpha), where the priority p_i = 1 / rank_i.
        priorities = [math.pow(rank, -self.alpha) for rank in range(1, len(self.priority_heap) + 1)]
        priorities_sum = sum(priorities)
        probabilities = dict(
            [(rank0index + 1, priority / priorities_sum) for rank0index, priority in enumerate(priorities)])

        partitions = [1]
        partition_num = 1

        cum_probabilty = 0
        next_boundary = partition_num / num_partitions
        rank = 1
        while partition_num < num_partitions:
            cum_probabilty += probabilities[rank]
            rank += 1
            if cum_probabilty >= next_boundary:
                partitions.append(rank)
                partition_num += 1
                next_boundary = partition_num / num_partitions
        partitions.append(len(self.priority_heap))

        partitions = [(a, b) for a, b in zip(partitions, partitions[1:])]

        return partitions, probabilities

    def update_priorities(self, buffer_positions, td_error):
        for buf_id, td_err in zip(buffer_positions, td_error):
            heap_id = self.buffer2heap[buf_id]
            [id, _, trans] = self.priority_heap[heap_id]
            self.priority_heap[heap_id] = [id, td_err, trans]
            self.down_heap(heap_id)
            self.up_heap(heap_id)

    def sample(self, batch_size, device=None):

        # print("\nStarting sample():...")
        # t = time.time()

        if batch_size != len(self.partitions) or not self.__partitions_fixed:
            # t1 = time.time()
            self.partitions, self.probabilities = self.update_partitions(batch_size)
            if self.full:
                # Once the buffer is full, the partitions no longer need to be updated
                # (as they depend only on the number of stored transitions and alpha).
                self.__partitions_fixed = True
            # print("\tupdate_partitions in :", time.time()-t1)

        self.beta = min(self.beta + self.beta_step, 1)

        # t1 = time.time()

        batch_ranks = [np.random.randint(low, high) for low, high in self.partitions]
        batch_buffer_positions, batch_td_errors, batch_transitions = zip(
            *[self.priority_heap[rank] for rank in batch_ranks])
        batch = [torch.stack(tensors).to(device) for tensors in zip(*batch_transitions)]

        # print("\tbatch sampled in :", time.time() - t1)
        # t1 = time.time()

        N = self._capacity if self.full else len(self)
        # Note this is a column vector to match the dimensions of weights and td_target in dqn.train_step(...)
        sample_probs = torch.FloatTensor([[self.probabilities[rank]] for rank in batch_ranks])
        weights = (N * sample_probs).pow(-self.beta)
        weights /= weights.max()

        # print("\tweights calculated in :", time.time() - t1)
        # print("...finished in :", time.time() - t)

        return batch, weights.to(device), batch_buffer_positions

    def configure_beta_anneal_time(self, beta_max_at_samples):
        self.beta_step = (1 - self.beta) / beta_max_at_samples

    def __len__(self):
        return len(self.priority_heap)


class Logger:
    def __init__(self, save_path, args, n_sims):
        self._memory = {}
        self._saves = 0
        self._maxsize = NB_STEPS
        self._dumps = 0
        self.save_path = save_path
        self.result = {}
        self.result['args'] = str(args['args'])
        self.result['n_sims'] = n_sims
        self.result['alg'] = ALG.value

    # def add_scalar(self, name, data, timestep):
    #     """
    #     Saves a scalar
    #     """
    #     if isinstance(data, torch.Tensor):
    #         data = data.item()
    #
    #     self._memory.setdefault(name, []).append([data, timestep])
    #
    #     self._saves += 1
    #     if self._saves == self._maxsize - 1:
    #         # with open('log_data_' + str((self._dumps + 1) * self._maxsize) + '.pkl', 'wb') as output:
    #         #     pickle.dump(self._memory, output, pickle.HIGHEST_PROTOCOL)
    #         self._dumps += 1
    #         self._saves = 0
    #         self._memory = {}

    def add_scalar(self, name_str, key, value):
        """
        Saves a scalar
        """
        if isinstance(key, torch.Tensor):
            key = key.item()
        if isinstance(value, torch.Tensor):
            value = value.item()
        if name_str not in self._memory:
            self._memory[name_str] = {}
        if key not in self._memory[name_str]:
            self._memory[name_str][key] = {}
        self._memory[name_str][key] = value

    # def save(self):
    #     self.result = {}
    #     # 保存所有内存中的数据到txt文件
    #     if not os.path.exists(os.path.dirname(self.save_path)):
    #         os.makedirs(os.path.dirname(self.save_path))
    #     keys = ["step_vs_num_samples_per_second", "step_vs_obj", "time_vs_obj", "step_vs_loss", "time_vs_loss"]
    #     with open(self.save_path, 'w') as output:
    #         for key, values in self._memory.items():
    #             if key in keys:
    #                 tmp_dict = {}
    #                 for value in values:
    #                     if type(value[1]) == torch.Tensor:
    #                         value[1] = value[1].tolist()
    #                     tmp_dict[f'{value[0]}'] = value[1]
    #                 self.result[key] = tmp_dict
    #         # for key, value in self.result.items():
    #         #     tmp_dict = {}
    #         #     tmp_dict[key] = value
    #         #     json.dump(tmp_dict, output, ensure_ascii=True, indent=4)
    #         json.dump(self.result, output, ensure_ascii=True, indent=4)
    #         print(f"result saved to {self.save_path}")

    def save(self):
        if not os.path.exists(os.path.dirname(self.save_path)):
            os.makedirs(os.path.dirname(self.save_path))
        with open(self.save_path, 'w') as output:
            json.dump(self._memory, output, ensure_ascii=True, indent=4)
            print(f"result saved to {self.save_path}")
