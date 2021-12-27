import os
import torch
import numpy as np
import numpy.random as rd
from types import SimpleNamespace


class ReplayBuffer:
    def __init__(self, max_len, state_dim, action_dim, if_use_per, gpu_id=0, state_type=torch.float32):
        """Experience Replay Buffer

        save environment transition in a continuous RAM for high performance training
        we save trajectory in order and save state and other (action, reward, mask, ...) separately.

        `int max_len` the maximum capacity of ReplayBuffer. First In First Out
        `int state_dim` the dimension of state
        `int action_dim` the dimension of action (action_dim==1 for discrete action)
        `bool if_on_policy` on-policy or off-policy
        `bool if_gpu` create buffer space on CPU RAM or GPU
        `bool if_per` Prioritized Experience Replay for sparse reward
        """
        self.now_len = 0
        self.next_id = 0
        self.if_full = False
        self.max_len = max_len
        self.data_type = torch.float32
        self.action_dim = action_dim
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        self.per_tree = BinarySearchTree(max_len) if if_use_per else None

        other_dim = 1 + 1 + self.action_dim
        self.buf_other = torch.empty((max_len, other_dim), dtype=torch.float32, device=self.device)

        buf_state_shape = (max_len, state_dim) if isinstance(state_dim, int) else (max_len, *state_dim)
        self.buf_state = torch.empty(buf_state_shape, dtype=state_type, device=self.device)

    def append_buffer(self, state, other):  # CPU array to CPU array
        self.buf_state[self.next_id] = state
        self.buf_other[self.next_id] = other

        if self.per_tree:
            self.per_tree.update_id(self.next_id)

        self.next_id += 1
        if self.next_id >= self.max_len:
            self.if_full = True
            self.next_id = 0

    def extend_buffer(self, state, other):
        size = len(other)
        next_idx = self.next_id + size

        if self.per_tree:
            self.per_tree.update_ids(data_ids=np.arange(self.next_id, next_idx) % self.max_len)

        if next_idx > self.max_len:
            self.buf_state[self.next_id:self.max_len] = state[:self.max_len - self.next_id]
            self.buf_other[self.next_id:self.max_len] = other[:self.max_len - self.next_id]
            self.if_full = True

            next_idx = next_idx - self.max_len
            self.buf_state[0:next_idx] = state[-next_idx:]
            self.buf_other[0:next_idx] = other[-next_idx:]
        else:
            self.buf_state[self.next_id:next_idx] = state
            self.buf_other[self.next_id:next_idx] = other
        self.next_id = next_idx

    def sample_batch(self, batch_size) -> tuple:
        """randomly sample a batch of data for training

        :int batch_size: the number of data in a batch for Stochastic Gradient Descent
        :return torch.Tensor reward: reward.shape==(now_len, 1)
        :return torch.Tensor mask:   mask.shape  ==(now_len, 1), mask = 0.0 if done else gamma
        :return torch.Tensor action: action.shape==(now_len, action_dim)
        :return torch.Tensor state:  state.shape ==(now_len, state_dim)
        :return torch.Tensor state:  state.shape ==(now_len, state_dim), next state
        """
        if self.per_tree:
            beg = -self.max_len
            end = (self.now_len - self.max_len) if (self.now_len < self.max_len) else None

            indices, is_weights = self.per_tree.get_indices_is_weights(batch_size, beg, end)
            r_m_a = self.buf_other[indices]
            return (r_m_a[:, 0:1].type(torch.float32),  # reward
                    r_m_a[:, 1:2].type(torch.float32),  # mask
                    r_m_a[:, 2:].type(torch.float32),  # action
                    self.buf_state[indices].type(torch.float32),  # state
                    self.buf_state[indices + 1].type(torch.float32),  # next state
                    torch.as_tensor(is_weights, dtype=torch.float32, device=self.device))  # important sampling weights
        else:
            indices = rd.randint(self.now_len - 1, size=batch_size)
            r_m_a = self.buf_other[indices]
            return (r_m_a[:, 0:1],  # reward
                    r_m_a[:, 1:2],  # mask
                    r_m_a[:, 2:],  # action
                    self.buf_state[indices],
                    self.buf_state[indices + 1])

    def sample_batch_one_step(self, batch_size) -> tuple:
        if self.per_tree:
            beg = -self.max_len
            end = (self.now_len - self.max_len) if (self.now_len < self.max_len) else None

            indices, is_weights = self.per_tree.get_indices_is_weights(batch_size, beg, end)
            r_m_a = self.buf_other[indices]
            return (r_m_a[:, 0:1].type(torch.float32),  # reward
                    r_m_a[:, 2:].type(torch.float32),  # action
                    self.buf_state[indices].type(torch.float32),  # state
                    torch.as_tensor(is_weights, dtype=torch.float32, device=self.device))  # important sampling weights
        else:
            indices = rd.randint(self.now_len - 1, size=batch_size)
            r_m_a = self.buf_other[indices]
            return (r_m_a[:, 0:1],  # reward
                    r_m_a[:, 2:],  # action
                    self.buf_state[indices],)

    def update_now_len(self):
        """update the a pointer `now_len`, which is the current data number of ReplayBuffer
        """
        self.now_len = self.max_len if self.if_full else self.next_id

    def print_state_norm(self, neg_avg=None, div_std=None):  # non-essential
        """print the state norm information: state_avg, state_std

        We don't suggest to use running stat state.
        We directly do normalization on state using the historical avg and std
        eg. `state = (state + self.neg_state_avg) * self.div_state_std` in `PreprocessEnv.step_norm()`
        neg_avg = -states.mean()
        div_std = 1/(states.std()+1e-5) or 6/(states.max()-states.min())

        :array neg_avg: neg_avg.shape=(state_dim)
        :array div_std: div_std.shape=(state_dim)
        """
        max_sample_size = 2 ** 14

        '''check if pass'''
        state_shape = self.buf_state.shape
        if len(state_shape) > 2 or state_shape[1] > 64:
            print(f"| print_state_norm(): state_dim: {state_shape} is too large to print its norm. ")
            return None

        '''sample state'''
        indices = np.arange(self.now_len)
        rd.shuffle(indices)
        indices = indices[:max_sample_size]  # len(indices) = min(self.now_len, max_sample_size)

        batch_state = self.buf_state[indices]

        '''compute state norm'''
        if isinstance(batch_state, torch.Tensor):
            batch_state = batch_state.cpu().data.numpy()
        assert isinstance(batch_state, np.ndarray)

        if batch_state.shape[1] > 64:
            print(f"| _print_norm(): state_dim: {batch_state.shape[1]:.0f} is too large to print its norm. ")
            return None

        if np.isnan(batch_state).any():  # 2020-12-12
            batch_state = np.nan_to_num(batch_state)  # nan to 0

        ary_avg = batch_state.mean(axis=0)
        ary_std = batch_state.std(axis=0)
        fix_std = ((np.max(batch_state, axis=0) - np.min(batch_state, axis=0)) / 6 + ary_std) / 2

        if neg_avg is not None:  # norm transfer
            ary_avg = ary_avg - neg_avg / div_std
            ary_std = fix_std / div_std

        print(f"print_state_norm: state_avg, state_std (fixed)")
        print(f"avg = np.{repr(ary_avg).replace('=float32', '=np.float32')}")
        print(f"std = np.{repr(ary_std).replace('=float32', '=np.float32')}")

    def td_error_update(self, td_error):
        self.per_tree.td_error_update(td_error)

    def save_or_load_history(self, cwd, if_save, buffer_id=0):  # [ElegantRL.2021.11.11]
        save_path = f"{cwd}/buffer_{buffer_id}.npz"
        if_load = None

        if if_save:
            self.update_now_len()
            state_dim = self.buf_state.shape[1]
            other_dim = self.buf_other.shape[1]

            buf_state_data_type = np.float16 \
                if self.buf_state.dtype in {np.float, np.float64, np.float32} \
                else np.uint8

            buf_state = np.empty((self.now_len, state_dim), dtype=buf_state_data_type)
            buf_other = np.empty((self.now_len, other_dim), dtype=np.float16)

            temp_len = self.now_len - self.next_id
            buf_state[0:temp_len] = self.buf_state[self.next_id:self.now_len].cpu().numpy()
            buf_other[0:temp_len] = self.buf_other[self.next_id:self.now_len].cpu().numpy()

            buf_state[temp_len:] = self.buf_state[:self.next_id].detach().cpu().numpy()
            buf_other[temp_len:] = self.buf_other[:self.next_id].detach().cpu().numpy()

            np.savez_compressed(save_path, buf_state=buf_state, buf_other=buf_other)
            print(f"| ReplayBuffer save in: {save_path}")
        elif os.path.isfile(save_path):
            buf_dict = np.load(save_path)
            buf_state = buf_dict['buf_state']
            buf_other = buf_dict['buf_other']

            bs = 512
            for i in range(0, buf_state.shape[0], bs):
                tmp_state = torch.as_tensor(buf_state[i:i + bs], dtype=torch.float32, device=self.device)
                tmp_other = torch.as_tensor(buf_other[i:i + bs], dtype=torch.float32, device=self.device)
                self.extend_buffer(tmp_state, tmp_other)

            self.update_now_len()
            print(f"| ReplayBuffer load: {save_path}")
            if_load = True
        else:
            # print(f"| ReplayBuffer FileNotFound: {save_path}")
            if_load = False
        return if_load


class ReplayBufferMP:
    def __init__(self, state_dim, action_dim, max_len, if_use_per, buffer_num, gpu_id):
        """Experience Replay Buffer for Multiple Processing

        `int max_len` the max_len of ReplayBuffer, not the total len of ReplayBufferMP
        `int worker_num` the rollout workers number
        """
        self.now_len = 0
        self.max_len = max_len
        self.worker_num = buffer_num

        buf_max_len = max_len // buffer_num
        self.buffers = [ReplayBuffer(max_len=buf_max_len, state_dim=state_dim, action_dim=action_dim,
                                     if_use_per=if_use_per, gpu_id=gpu_id)
                        for _ in range(buffer_num)]

    def sample_batch(self, batch_size) -> list:
        bs = batch_size // self.worker_num
        list_items = [self.buffers[i].sample_batch(bs)
                      for i in range(self.worker_num)]
        # list_items of reward, mask, action, state, next_state
        # list_items of reward, mask, action, state, next_state, is_weights (PER)

        list_items = list(map(list, zip(*list_items)))  # 2D-list transpose
        return [torch.cat(item, dim=0) for item in list_items]

    def sample_batch_one_step(self, batch_size) -> list:
        bs = batch_size // self.worker_num
        list_items = [self.buffers[i].sample_batch_one_step(bs)
                      for i in range(self.worker_num)]
        # list_items of reward, mask, action, state, next_state
        # list_items of reward, mask, action, state, next_state, is_weights (PER)

        list_items = list(map(list, zip(*list_items)))  # 2D-list transpose
        return [torch.cat(item, dim=0) for item in list_items]

    def update_now_len(self):
        self.now_len = 0
        for buffer in self.buffers:
            buffer.update_now_len()
            self.now_len += buffer.now_len

    def print_state_norm(self, neg_avg=None, div_std=None):  # non-essential
        # for buffer in self.l_buffer:
        self.buffers[0].print_state_norm(neg_avg, div_std)

    def td_error_update(self, td_error):
        td_errors = td_error.view(self.worker_num, -1, 1)
        for i in range(self.worker_num):
            self.buffers[i].per_tree.td_error_update(td_errors[i])

    def save_or_load_history(self, cwd, if_save):
        for i in range(self.worker_num):
            self.buffers[i].save_or_load_history(cwd, if_save, buffer_id=i)


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


'''MARL'''


class EpisodeBatch:
    def __init__(self,
                 scheme,
                 groups,
                 batch_size,
                 max_seq_length,
                 data=None,
                 preprocess=None,
                 device="cpu"):
        self.scheme = scheme.copy()
        self.groups = groups
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.preprocess = {} if preprocess is None else preprocess
        self.device = device

        if data is not None:
            self.data = data
        else:
            self.data = SimpleNamespace()
            self.data.transition_data = {}
            self.data.episode_data = {}
            self._setup_data(self.scheme, self.groups, batch_size, max_seq_length, self.preprocess)

    def _setup_data(self, scheme, groups, batch_size, max_seq_length, preprocess):
        if preprocess is not None:
            for k in preprocess:
                assert k in scheme
                new_k = preprocess[k][0]
                transforms = preprocess[k][1]

                vshape = self.scheme[k]["vshape"]
                dtype = self.scheme[k]["dtype"]
                for transform in transforms:
                    vshape, dtype = transform.infer_output_info(vshape, dtype)

                self.scheme[new_k] = {
                    "vshape": vshape,
                    "dtype": dtype
                }
                if "group" in self.scheme[k]:
                    self.scheme[new_k]["group"] = self.scheme[k]["group"]
                if "episode_const" in self.scheme[k]:
                    self.scheme[new_k]["episode_const"] = self.scheme[k]["episode_const"]

        assert "filled" not in scheme, '"filled" is a reserved key for masking.'
        scheme.update({
            "filled": {"vshape": (1,), "dtype": torch.long},
        })

        for field_key, field_info in scheme.items():
            assert "vshape" in field_info, "Scheme must define vshape for {}".format(field_key)
            vshape = field_info["vshape"]
            episode_const = field_info.get("episode_const", False)
            group = field_info.get("group", None)
            dtype = field_info.get("dtype", torch.float32)

            if isinstance(vshape, int):
                vshape = (vshape,)

            if group:
                assert group in groups, "Group {} must have its number of members defined in _groups_".format(group)
                shape = (groups[group], *vshape)
            else:
                shape = vshape

            if episode_const:
                self.data.episode_data[field_key] = torch.zeros((batch_size, *shape), dtype=dtype, device=self.device)
            else:
                self.data.transition_data[field_key] = torch.zeros((batch_size, max_seq_length, *shape), dtype=dtype,
                                                                   device=self.device)

    def extend(self, scheme, groups=None):
        self._setup_data(scheme, self.groups if groups is None else groups, self.batch_size, self.max_seq_length)

    def to(self, device):
        for k, v in self.data.transition_data.items():
            self.data.transition_data[k] = v.to(device)
        for k, v in self.data.episode_data.items():
            self.data.episode_data[k] = v.to(device)
        self.device = device

    def update(self, data, bs=slice(None), ts=slice(None), mark_filled=True):
        slices = self._parse_slices((bs, ts))
        for k, v in data.items():
            if k in self.data.transition_data:
                target = self.data.transition_data
                if mark_filled:
                    target["filled"][slices] = 1
                    mark_filled = False
                _slices = slices
            elif k in self.data.episode_data:
                target = self.data.episode_data
                _slices = slices[0]
            else:
                raise KeyError("{} not found in transition or episode data".format(k))

            dtype = self.scheme[k].get("dtype", torch.float32)
            v = torch.tensor(v, dtype=dtype, device=self.device)
            self._check_safe_view(v, target[k][_slices])
            target[k][_slices] = v.view_as(target[k][_slices])

            if k in self.preprocess:
                new_k = self.preprocess[k][0]
                v = target[k][_slices]
                for transform in self.preprocess[k][1]:
                    v = transform.transform(v)
                target[new_k][_slices] = v.view_as(target[new_k][_slices])

    def _check_safe_view(self, v, dest):
        idx = len(v.shape) - 1
        for s in dest.shape[::-1]:
            if v.shape[idx] != s:
                if s != 1:
                    raise ValueError("Unsafe reshape of {} to {}".format(v.shape, dest.shape))
            else:
                idx -= 1

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in self.data.episode_data:
                return self.data.episode_data[item]
            elif item in self.data.transition_data:
                return self.data.transition_data[item]
            else:
                raise ValueError
        elif isinstance(item, tuple) and all([isinstance(it, str) for it in item]):
            new_data = self._new_data_sn()
            for key in item:
                if key in self.data.transition_data:
                    new_data.transition_data[key] = self.data.transition_data[key]
                elif key in self.data.episode_data:
                    new_data.episode_data[key] = self.data.episode_data[key]
                else:
                    raise KeyError("Unrecognised key {}".format(key))

            # Update the scheme to only have the requested keys
            new_scheme = {key: self.scheme[key] for key in item}
            new_groups = {self.scheme[key]["group"]: self.groups[self.scheme[key]["group"]]
                          for key in item if "group" in self.scheme[key]}
            ret = EpisodeBatch(new_scheme, new_groups, self.batch_size, self.max_seq_length, data=new_data,
                               device=self.device)
            return ret
        else:
            item = self._parse_slices(item)
            new_data = self._new_data_sn()
            for k, v in self.data.transition_data.items():
                new_data.transition_data[k] = v[item]
            for k, v in self.data.episode_data.items():
                new_data.episode_data[k] = v[item[0]]

            ret_bs = self._get_num_items(item[0], self.batch_size)
            ret_max_t = self._get_num_items(item[1], self.max_seq_length)

            ret = EpisodeBatch(self.scheme, self.groups, ret_bs, ret_max_t, data=new_data, device=self.device)
            return ret

    def _get_num_items(self, indexing_item, max_size):
        if isinstance(indexing_item, list) or isinstance(indexing_item, np.ndarray):
            return len(indexing_item)
        elif isinstance(indexing_item, slice):
            _range = indexing_item.indices(max_size)
            return 1 + (_range[1] - _range[0] - 1) // _range[2]

    def _new_data_sn(self):
        new_data = SimpleNamespace()
        new_data.transition_data = {}
        new_data.episode_data = {}
        return new_data

    def _parse_slices(self, items):
        parsed = []
        # Only batch slice given, add full time slice
        if (isinstance(items, slice)  # slice a:b
                or isinstance(items, int)  # int i
                or (isinstance(items, (list, np.ndarray, torch.LongTensor, torch.cuda.LongTensor)))  # [a,b,c]
        ):
            items = (items, slice(None))

        # Need the time indexing to be contiguous
        if isinstance(items[1], list):
            raise IndexError("Indexing across Time must be contiguous")

        for item in items:
            # TODO: stronger checks to ensure only supported options get through
            if isinstance(item, int):
                # Convert single indices to slices
                parsed.append(slice(item, item + 1))
            else:
                # Leave slices and lists as is
                parsed.append(item)
        return parsed

    def max_t_filled(self):
        return torch.sum(self.data.transition_data["filled"], 1).max(0)[0]

    def __repr__(self):
        return "EpisodeBatch. Batch Size:{} Max_seq_len:{} Keys:{} Groups:{}".format(self.batch_size,
                                                                                     self.max_seq_length,
                                                                                     self.scheme.keys(),
                                                                                     self.groups.keys())


class ReplayBufferMARL(EpisodeBatch):
    def __init__(self, scheme, groups, buffer_size, max_seq_length, preprocess=None, device="cpu"):
        super().__init__(scheme, groups, buffer_size, max_seq_length, preprocess=preprocess, device=device)
        self.buffer_size = buffer_size  # same as self.batch_size but more explicit
        self.buffer_index = 0
        self.episodes_in_buffer = 0

    def insert_episode_batch(self, ep_batch):
        if self.buffer_index + ep_batch.batch_size <= self.buffer_size:
            self.update(ep_batch.data.transition_data,
                        slice(self.buffer_index, self.buffer_index + ep_batch.batch_size),
                        slice(0, ep_batch.max_seq_length),
                        mark_filled=False)
            self.update(ep_batch.data.episode_data,
                        slice(self.buffer_index, self.buffer_index + ep_batch.batch_size))
            self.buffer_index = (self.buffer_index + ep_batch.batch_size)
            self.episodes_in_buffer = max(self.episodes_in_buffer, self.buffer_index)
            self.buffer_index = self.buffer_index % self.buffer_size
            assert self.buffer_index < self.buffer_size
        else:
            buffer_left = self.buffer_size - self.buffer_index
            self.insert_episode_batch(ep_batch[0:buffer_left, :])
            self.insert_episode_batch(ep_batch[buffer_left:, :])

    def can_sample(self, batch_size):
        return self.episodes_in_buffer >= batch_size

    def sample(self, batch_size):
        assert self.can_sample(batch_size)
        if self.episodes_in_buffer == batch_size:
            return self[:batch_size]
        else:
            # Uniform sampling only atm
            ep_ids = np.random.choice(self.episodes_in_buffer, batch_size, replace=False)
            return self[ep_ids]

    def uni_sample(self, batch_size):
        return self.sample(batch_size)

    def sample_latest(self, batch_size):
        assert self.can_sample(batch_size)
        if self.buffer_index - batch_size < 0:
            # Uniform sampling
            return self.uni_sample(batch_size)
        else:
            # Return the latest
            return self[self.buffer_index - batch_size: self.buffer_index]

    def __repr__(self):
        return "ReplayBuffer. {}/{} episodes. Keys:{} Groups:{}".format(self.episodes_in_buffer,
                                                                        self.buffer_size,
                                                                        self.scheme.keys(),
                                                                        self.groups.keys())
