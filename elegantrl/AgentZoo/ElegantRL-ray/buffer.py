import numpy as np
import numpy.random as rd
import torch

"""
Modify [ElegantRL](https://github.com/AI4Finance-Foundation/ElegantRL)
by https://github.com/GyChou
"""


class ReplayBuffer:
    def __init__(self, max_len, state_dim, action_dim, reward_dim, if_off_policy, if_per, if_gpu):
        """Experience Replay Buffer

        save environment transition in a continuous RAM for high performance training
        we save trajectory in order and save state and other (action, reward, mask, ...) separately.

        `int max_len` the maximum capacity of ReplayBuffer. First In First Out
        `int state_dim` the dimension of state
        `int action_dim` the dimension of action (action_dim==1 for discrete action)
        `bool if_off_policy` on-policy or off-policy
        `bool if_gpu` create buffer space on CPU RAM or GPU
        `bool if_per` Prioritized Experience Replay for sparse reward
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.now_len = 0
        self.next_idx = 0
        self.if_full = False
        self.action_dim = action_dim
        self.if_off_policy = if_off_policy
        self.if_per = if_per
        self.if_gpu = if_gpu
        if if_per:
            self.tree = BinarySearchTree(max_len)
        if self.if_gpu:
            self.buf_state = torch.empty((max_len, state_dim), dtype=torch.float32, device=self.device)
            self.buf_action = torch.empty((max_len, action_dim), dtype=torch.float32, device=self.device)
            self.buf_reward = torch.empty((max_len, reward_dim), dtype=torch.float32, device=self.device)
            self.buf_gamma = torch.empty((max_len, reward_dim), dtype=torch.float32, device=self.device)
        else:
            self.buf_state = np.empty((max_len, state_dim), dtype=np.float32)
            self.buf_action = np.empty((max_len, action_dim), dtype=np.float32)
            self.buf_reward = np.empty((max_len, reward_dim), dtype=np.float32)
            self.buf_gamma = np.empty((max_len, reward_dim), dtype=np.float32)

    def append_buffer(self, state, action, reward, gamma):  # CPU array to CPU array
        if self.if_gpu:
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
            reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
            gamma = torch.as_tensor(gamma, dtype=torch.float32, device=self.device)
        self.buf_state[self.next_idx] = state
        self.buf_action[self.next_idx] = action
        self.buf_reward[self.next_idx] = reward
        self.buf_gamma[self.next_idx] = gamma

        if self.if_per:
            self.tree.update_id(self.next_idx)

        self.next_idx += 1
        if self.next_idx >= self.max_len:
            self.if_full = True
            self.next_idx = 0

    def extend_buffer(self, state, action, reward, gamma):  # CPU array to CPU array
        if self.if_gpu:
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
            reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
            gamma = torch.as_tensor(gamma, dtype=torch.float32, device=self.device)

        size = len(state)
        next_idx = self.next_idx + size

        if self.if_per:
            for data_id in (np.arange(self.next_idx, next_idx) % self.max_len):
                self.tree.update_ids(data_id)

        if next_idx > self.max_len:
            if next_idx > self.max_len:
                self.buf_state[self.next_idx:self.max_len] = state[:self.max_len - self.next_idx]
                self.buf_action[self.next_idx:self.max_len] = action[:self.max_len - self.next_idx]
                self.buf_reward[self.next_idx:self.max_len] = reward[:self.max_len - self.next_idx]
                self.buf_gamma[self.next_idx:self.max_len] = gamma[:self.max_len - self.next_idx]
            self.if_full = True
            next_idx = next_idx - self.max_len

            self.buf_state[0:next_idx] = state[-next_idx:]
            self.buf_action[0:next_idx] = action[-next_idx:]
            self.buf_reward[0:next_idx] = reward[-next_idx:]
            self.buf_gamma[0:next_idx] = gamma[-next_idx:]
        else:
            self.buf_state[self.next_idx:next_idx] = state
            self.buf_action[self.next_idx:next_idx] = action
            self.buf_reward[self.next_idx:next_idx] = reward
            self.buf_gamma[self.next_idx:next_idx] = gamma
        self.next_idx = next_idx

    def sample_batch(self, batch_size) -> tuple:
        """randomly sample a batch of data for training

        :int batch_size: the number of data in a batch for Stochastic Gradient Descent
        :return torch.Tensor reward: reward.shape==(now_len, 1)
        :return torch.Tensor mask:   mask.shape  ==(now_len, 1), mask = 0.0 if done else gamma
        :return torch.Tensor action: action.shape==(now_len, action_dim)
        :return torch.Tensor state:  state.shape ==(now_len, state_dim)
        :return torch.Tensor state:  state.shape ==(now_len, state_dim), next state
        """
        if self.if_per:
            beg = -self.max_len
            end = (self.now_len - self.max_len) if (self.now_len < self.max_len) else None

            indices, is_weights = self.tree.get_indices_is_weights(batch_size, beg, end)

            return (self.buf_reward[indices],
                    self.buf_gamma[indices],
                    self.buf_action[indices],
                    self.buf_state[indices],
                    self.buf_state[indices + 1],
                    torch.as_tensor(is_weights, dtype=torch.float32, device=self.device))
        else:
            indices = torch.randint(self.now_len - 1, size=(batch_size,), device=self.device) if self.if_gpu \
                else rd.randint(self.now_len - 1, size=batch_size)
            return (self.buf_reward[indices],
                    self.buf_gamma[indices],
                    self.buf_action[indices],
                    self.buf_state[indices],
                    self.buf_state[indices + 1])

    def sample_all(self) -> tuple:
        """sample all the data in ReplayBuffer (for on-policy)

        :return torch.Tensor reward: reward.shape==(now_len, 1)
        :return torch.Tensor mask:   mask.shape  ==(now_len, 1), mask = 0.0 if done else gamma
        :return torch.Tensor action: action.shape==(now_len, action_dim)
        :return torch.Tensor noise:  noise.shape ==(now_len, action_dim)
        :return torch.Tensor state:  state.shape ==(now_len, state_dim)
        """
        return (torch.as_tensor(self.buf_reward[:self.now_len], device=self.device),
                torch.as_tensor(self.buf_gamma[:self.now_len], device=self.device),
                torch.as_tensor(self.buf_action[:self.now_len], device=self.device),
                torch.as_tensor(self.buf_state[:self.now_len], device=self.device))

    def update_now_len_before_sample(self):
        """update the a pointer `now_len`, which is the current data number of ReplayBuffer
        """
        self.now_len = self.max_len if self.if_full else self.next_idx

    def empty_buffer_before_explore(self):
        """we empty the buffer by set now_len=0. On-policy need to empty buffer before exploration
        """
        self.next_idx = 0
        self.now_len = 0
        self.if_full = False


class ReplayBufferMP:
    def __init__(self, max_len, state_dim, action_dim, reward_dim, if_off_policy, if_per, rollout_num):
        """Experience Replay Buffer for Multiple Processing

        `int rollout_num` the rollout workers number
        """
        self.now_len = 0
        self.max_len = max_len
        self.rollout_num = rollout_num
        self.if_off_policy=if_off_policy
        _max_len = max_len // rollout_num
        if_gpu = True
        if not self.if_off_policy:
            if_gpu = False
            if_per = False
        self.buffers = [ReplayBuffer(_max_len, state_dim, action_dim, reward_dim, if_off_policy, if_per, if_gpu=if_gpu)
                        for _ in range(rollout_num)]

    def extend_buffer(self, state, action, reward, gamma, i):
        self.buffers[i].extend_buffer(state, action, reward, gamma, )

    def sample_batch(self, batch_size) -> list:
        bs = batch_size // self.rollout_num
        list_items = []
        list_items.append(self.buffers[0].sample_batch(bs + (batch_size % self.rollout_num)))
        for i in range(1, self.rollout_num):
            list_items.append(self.buffers[i].sample_batch(bs))

        return [torch.cat([item[i] for item in list_items], dim=0)
                for i in range(len(list_items[0]))]

    def sample_all(self) -> list:
        l__r_m_a_s = [self.buffers[i].sample_all()
                      for i in range(self.rollout_num)]
        # list of reward, mask, action, state
        return [torch.cat([item[i] for item in l__r_m_a_s], dim=0)
                for i in range(len(l__r_m_a_s[0]))]

    def update_now_len_before_sample(self):
        self.now_len = 0
        for buffer in self.buffers:
            buffer.update_now_len_before_sample()
            self.now_len += buffer.now_len

    def empty_buffer_before_explore(self):
        for buffer in self.buffers:
            buffer.empty_buffer_before_explore()

    def td_error_update(self, td_error):
        td_errors = td_error.view(self.rollout_num, -1, 1)
        for i in range(self.rollout_num):
            self.buffers[i].tree.td_error_update(td_errors[i])


class BinarySearchTree:
    """Binary Search Tree for PER

    Contributor: Github GyChou, Github mississippiu
    Reference: https://github.com/kaixindelele/DRLib/tree/main/algos/pytorch/td3_sp
    Reference: https://github.com/jaromiru/AI-blog/blob/master/SumTree.py
    """

    def __init__(self, memo_len):
        self.memo_len = memo_len  # replay buffer len
        self.prob_ary = np.zeros((memo_len - 1) + memo_len)  # parent_nodes_num + leaf_nodes_num
        self.max_len = len(self.prob_ary)
        self.now_len = self.memo_len - 1  # pointer
        self.indices = None
        self.depth = int(np.log2(self.max_len))

        # PER.  Prioritized Experience Replay. Section 4
        # alpha, beta = 0.7, 0.5 for rank-based variant
        # alpha, beta = 0.6, 0.4 for proportional variant
        self.per_alpha = 0.6  # alpha = (Uniform:0, Greedy:1)
        self.per_beta = 0.4  # beta = (PER:0, NotPER:1)

    def update_id(self, data_id, prob=10):  # 10 is max_prob
        tree_id = data_id + self.memo_len - 1
        if self.now_len == tree_id:
            self.now_len += 1

        delta = prob - self.prob_ary[tree_id]
        self.prob_ary[tree_id] = prob

        while tree_id != 0:  # propagate the change through tree
            tree_id = (tree_id - 1) // 2  # faster than the recursive loop
            self.prob_ary[tree_id] += delta

    def update_ids(self, data_ids, prob=10):  # 10 is max_prob
        ids = data_ids + self.memo_len - 1
        self.now_len += (ids >= self.now_len).sum()

        upper_step = self.depth - 1
        self.prob_ary[ids] = prob  # here, ids means the indices of given children (maybe the right ones or left ones)
        p_ids = (ids - 1) // 2

        while upper_step:  # propagate the change through tree
            ids = p_ids * 2 + 1  # in this while loop, ids means the indices of the left children
            self.prob_ary[p_ids] = self.prob_ary[ids] + self.prob_ary[ids + 1]
            p_ids = (p_ids - 1) // 2
            upper_step -= 1

        self.prob_ary[0] = self.prob_ary[1] + self.prob_ary[2]
        # because we take depth-1 upper steps, ps_tree[0] need to be updated alone

    def get_leaf_id(self, v):
        """Tree structure and array storage:

        Tree index:
              0       -> storing priority sum
            |  |
          1     2
         | |   | |
        3  4  5  6    -> storing priority for transitions
        Array type for storing: [0, 1, 2, 3, 4, 5, 6]
        """
        parent_idx = 0
        while True:
            l_idx = 2 * parent_idx + 1  # the leaf's left node
            r_idx = l_idx + 1  # the leaf's right node
            if l_idx >= (len(self.prob_ary)):  # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.prob_ary[l_idx]:
                    parent_idx = l_idx
                else:
                    v -= self.prob_ary[l_idx]
                    parent_idx = r_idx
        return min(leaf_idx, self.now_len - 2)  # leaf_idx

    def get_indices_is_weights(self, batch_size, beg, end):
        self.per_beta = min(1., self.per_beta + 0.001)

        # get random values for searching indices with proportional prioritization
        values = (rd.rand(batch_size) + np.arange(batch_size)) * (self.prob_ary[0] / batch_size)

        # get proportional prioritization
        leaf_ids = np.array([self.get_leaf_id(v) for v in values])
        self.indices = leaf_ids - (self.memo_len - 1)

        prob_ary = self.prob_ary[leaf_ids] / self.prob_ary[beg:end].min()
        is_weights = np.power(prob_ary, -self.per_beta)  # important sampling weights
        return self.indices, is_weights

    def td_error_update(self, td_error):  # td_error = (q-q).detach_().abs()
        prob = td_error.squeeze().clamp(1e-6, 10).pow(self.per_alpha)
        prob = prob.cpu().numpy()
        self.update_ids(self.indices, prob)
