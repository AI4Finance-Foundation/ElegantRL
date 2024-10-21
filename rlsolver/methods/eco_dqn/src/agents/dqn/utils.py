import math
import pickle
import random
import threading
from collections import namedtuple
from enum import Enum

import numpy as np
import torch

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

        self.next_batch_process=None
        self.next_batch_size=None
        self.next_batch_device=None
        self.next_batch = None

    def add(self, *args):
        """
        Saves a transition.
        """
        if self.next_batch_process is not None:
            # Don't add to the buffer when sampling from it.
            self.next_batch_process.join()
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

        if self.next_batch_size==batch_size and self.next_batch_device==device:
            next_batch = self.next_batch
            self.launch_sample(batch_size, device)
            return next_batch
        else:
            self.launch_sample(batch_size, device)
            self.sample(batch_size, device)

    def __len__(self):
        return len(self._memory)


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
        for buf_id, td_err in  zip(buffer_positions, td_error):
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
        batch_buffer_positions, batch_td_errors, batch_transitions = zip(*[self.priority_heap[rank] for rank in batch_ranks])
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
    def __init__(self):
        self._memory = {}
        self._saves = 0
        self._maxsize = 1000000
        self._dumps = 0

    def add_scalar(self, name, data, timestep):
        """
        Saves a scalar
        """
        if isinstance(data, torch.Tensor):
            data = data.item()

        self._memory.setdefault(name, []).append([data, timestep])

        self._saves += 1
        if self._saves == self._maxsize - 1:
            with open('log_data_' + str((self._dumps + 1) * self._maxsize) + '.pkl', 'wb') as output:
                pickle.dump(self._memory, output, pickle.HIGHEST_PROTOCOL)
            self._dumps += 1
            self._saves = 0
            self._memory = {}

    def save(self):
        with open('log_data.pkl', 'wb') as output:
            pickle.dump(self._memory, output, pickle.HIGHEST_PROTOCOL)
