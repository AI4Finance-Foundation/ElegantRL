import os
import math
import torch
from typing import Tuple
from torch import Tensor

from elegantrl.train.config import Config


class ReplayBuffer:  # for off-policy
    def __init__(self,
                 max_size: int,
                 state_dim: int,
                 action_dim: int,
                 gpu_id: int = 0,
                 num_seqs: int = 1,
                 if_use_per: bool = False,
                 args: Config = Config()):
        self.p = 0  # pointer
        self.if_full = False
        self.cur_size = 0
        self.add_size = 0
        self.add_item = None
        self.max_size = max_size
        self.num_seqs = num_seqs
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        """The struction of ReplayBuffer (for example, num_seqs = num_workers * num_envs == 2*4 = 8
        ReplayBuffer:
        worker0 for env0:   sequence of sub_env0.0  self.states  = Tensor[s, s, ..., s, ..., s]     
                                                    self.actions = Tensor[a, a, ..., a, ..., a]   
                                                    self.rewards = Tensor[r, r, ..., r, ..., r]   
                                                    self.undones = Tensor[d, d, ..., d, ..., d]
                                                                          <-----max_size----->
                                                                          <-cur_size->
                                                                                     ↑ pointer
                            sequence of sub_env0.1  s, s, ..., s    a, a, ..., a    r, r, ..., r    d, d, ..., d
                            sequence of sub_env0.2  s, s, ..., s    a, a, ..., a    r, r, ..., r    d, d, ..., d
                            sequence of sub_env0.3  s, s, ..., s    a, a, ..., a    r, r, ..., r    d, d, ..., d
        worker1 for env1:   sequence of sub_env1.0  s, s, ..., s    a, a, ..., a    r, r, ..., r    d, d, ..., d
                            sequence of sub_env1.1  s, s, ..., s    a, a, ..., a    r, r, ..., r    d, d, ..., d
                            sequence of sub_env1.2  s, s, ..., s    a, a, ..., a    r, r, ..., r    d, d, ..., d
                            sequence of sub_env1.3  s, s, ..., s    a, a, ..., a    r, r, ..., r    d, d, ..., d
        
        D: done=True
        d: done=False
        sequence of transition: s-a-r-d, s-a-r-d, s-a-r-D  s-a-r-d, s-a-r-d, s-a-r-d, s-a-r-d, s-a-r-D  s-a-r-d, ...
                                <------trajectory------->  <----------trajectory--------------------->  <-----------
        """
        self.states = torch.empty((max_size, num_seqs, state_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.empty((max_size, num_seqs, action_dim), dtype=torch.float32, device=self.device)
        self.rewards = torch.empty((max_size, num_seqs), dtype=torch.float32, device=self.device)
        self.undones = torch.empty((max_size, num_seqs), dtype=torch.float32, device=self.device)

        self.if_use_per = if_use_per
        if if_use_per:
            self.sum_trees = [SumTree(buf_len=max_size) for _ in range(num_seqs)]
            self.per_alpha = getattr(args, 'per_alpha', 0.6)  # alpha = (Uniform:0, Greedy:1)
            self.per_beta = getattr(args, 'per_beta', 0.4)  # alpha = (Uniform:0, Greedy:1)
            """PER.  Prioritized Experience Replay. Section 4
            alpha, beta = 0.7, 0.5 for rank-based variant
            alpha, beta = 0.6, 0.4 for proportional variant
            """
        else:
            self.sum_trees = None
            self.per_alpha = None
            self.per_beta = None

    def update(self, items: Tuple[Tensor, ...]):
        self.add_item = items
        states, actions, rewards, undones = items
        # assert states.shape[1:] == (env_num, state_dim)
        # assert actions.shape[1:] == (env_num, action_dim)
        # assert rewards.shape[1:] == (env_num,)
        # assert undones.shape[1:] == (env_num,)
        self.add_size = rewards.shape[0]

        p = self.p + self.add_size  # pointer
        if p > self.max_size:
            self.if_full = True
            p0 = self.p
            p1 = self.max_size
            p2 = self.max_size - self.p
            p = p - self.max_size

            self.states[p0:p1], self.states[0:p] = states[:p2], states[-p:]
            self.actions[p0:p1], self.actions[0:p] = actions[:p2], actions[-p:]
            self.rewards[p0:p1], self.rewards[0:p] = rewards[:p2], rewards[-p:]
            self.undones[p0:p1], self.undones[0:p] = undones[:p2], undones[-p:]
        else:
            self.states[self.p:p] = states
            self.actions[self.p:p] = actions
            self.rewards[self.p:p] = rewards
            self.undones[self.p:p] = undones

        if self.if_use_per:
            '''data_ids for single env'''
            data_ids = torch.arange(self.p, p, dtype=torch.long, device=self.device)
            if p > self.max_size:
                data_ids = torch.fmod(data_ids, self.max_size)

            '''apply data_ids for vectorized env'''
            for sum_tree in self.sum_trees:
                sum_tree.update_ids(data_ids=data_ids.cpu(), prob=10.)

        self.p = p
        self.cur_size = self.max_size if self.if_full else self.p

    def sample(self, batch_size: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        sample_len = self.cur_size - 1

        ids = torch.randint(sample_len * self.num_seqs, size=(batch_size,), requires_grad=False)
        ids0 = torch.fmod(ids, sample_len)  # ids % sample_len
        ids1 = torch.div(ids, sample_len, rounding_mode='floor')  # ids // sample_len

        return (self.states[ids0, ids1],
                self.actions[ids0, ids1],
                self.rewards[ids0, ids1],
                self.undones[ids0, ids1],
                self.states[ids0 + 1, ids1],)  # next_state

    def sample_for_per(self, batch_size: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        beg = -self.max_size
        end = (self.cur_size - self.max_size) if (self.cur_size < self.max_size) else -1

        '''get is_indices, is_weights'''
        is_indices: list = []
        is_weights: list = []

        assert batch_size % self.num_seqs == 0
        sub_batch_size = batch_size // self.num_seqs
        for env_i in range(self.num_seqs):
            sum_tree = self.sum_trees[env_i]
            _is_indices, _is_weights = sum_tree.important_sampling(batch_size, beg, end, self.per_beta)
            is_indices.append(_is_indices + sub_batch_size * env_i)
            is_weights.append(_is_weights)

        is_indices: Tensor = torch.hstack(is_indices).to(self.device)
        is_weights: Tensor = torch.hstack(is_weights).to(self.device)

        ids0 = torch.fmod(is_indices, self.cur_size)  # is_indices % sample_len
        ids1 = torch.div(is_indices, self.cur_size, rounding_mode='floor')  # is_indices // sample_len
        return (
            self.states[ids0, ids1],
            self.actions[ids0, ids1],
            self.rewards[ids0, ids1],
            self.undones[ids0, ids1],
            self.states[ids0 + 1, ids1],  # next_state
            is_weights,  # important sampling weights
            is_indices,  # important sampling indices
        )

    def td_error_update_for_per(self, is_indices: Tensor, td_error: Tensor):  # td_error = (q-q).detach_().abs()
        prob = td_error.clamp(1e-8, 10).pow(self.per_alpha)

        # self.sum_tree.update_ids(is_indices.cpu(), prob.cpu())
        batch_size = td_error.shape[0]
        sub_batch_size = batch_size // self.num_seqs
        for env_i in range(self.num_seqs):
            sum_tree = self.sum_trees[env_i]
            slice_i = env_i * sub_batch_size
            slice_j = slice_i + sub_batch_size

            sum_tree.update_ids(is_indices[slice_i:slice_j].cpu(), prob[slice_i:slice_j].cpu())

    def save_or_load_history(self, cwd: str, if_save: bool):
        item_names = (
            (self.states, "states"),
            (self.actions, "actions"),
            (self.rewards, "rewards"),
            (self.undones, "undones"),
        )

        if if_save:
            for item, name in item_names:
                if self.cur_size == self.p:
                    buf_item = item[:self.cur_size]
                else:
                    buf_item = torch.vstack((item[self.p:self.cur_size], item[0:self.p]))
                file_path = f"{cwd}/replay_buffer_{name}.pth"
                print(f"| buffer.save_or_load_history(): Save {file_path}")
                torch.save(buf_item, file_path)

        elif all([os.path.isfile(f"{cwd}/replay_buffer_{name}.pth") for item, name in item_names]):
            max_sizes = []
            for item, name in item_names:
                file_path = f"{cwd}/replay_buffer_{name}.pth"
                print(f"| buffer.save_or_load_history(): Load {file_path}")
                buf_item = torch.load(file_path)

                max_size = buf_item.shape[0]
                item[:max_size] = buf_item
                max_sizes.append(max_size)
            assert all([max_size == max_sizes[0] for max_size in max_sizes])
            self.cur_size = self.p = max_sizes[0]
            self.if_full = self.cur_size == self.max_size


class SumTree:
    """ BinarySearchTree for PER (SumTree)
    Contributor: Github GyChou, Github mississippiu
    Reference: https://github.com/kaixindelele/DRLib/tree/main/algos/pytorch/td3_sp
    Reference: https://github.com/jaromiru/AI-blog/blob/master/SumTree.py
    """

    def __init__(self, buf_len: int):
        self.buf_len = buf_len  # replay buffer len
        self.max_len = (buf_len - 1) + buf_len  # parent_nodes_num + leaf_nodes_num
        self.depth = math.ceil(math.log2(self.max_len))

        self.tree = torch.zeros(self.max_len, dtype=torch.float32)

    def update_id(self, data_id: int, prob=10):  # 10 is max_prob
        tree_id = data_id + self.buf_len - 1

        delta = prob - self.tree[tree_id]
        self.tree[tree_id] = prob

        for depth in range(self.depth - 2):  # propagate the change through tree
            tree_id = (tree_id - 1) // 2  # faster than the recursive loop
            self.tree[tree_id] += delta

    def update_ids(self, data_ids: Tensor, prob: Tensor = 10.):  # 10 is max_prob
        l_ids = data_ids + self.buf_len - 1

        self.tree[l_ids] = prob
        for depth in range(self.depth - 2):  # propagate the change through tree
            p_ids = ((l_ids - 1) // 2).unique()  # parent indices
            l_ids = p_ids * 2 + 1  # left children indices
            r_ids = l_ids + 1  # right children indices
            self.tree[p_ids] = self.tree[l_ids] + self.tree[r_ids]

            l_ids = p_ids

    def get_leaf_id_and_value(self, v) -> Tuple[int, float]:
        """Tree structure and array storage:
        Tree index:
              0       -> storing priority sum
            |  |
          1     2
         | |   | |
        3  4  5  6    -> storing priority for transitions
        Array type for storing: [0, 1, 2, 3, 4, 5, 6]
        """
        p_id = 0  # the leaf's parent node

        for depth in range(self.depth - 2):  # propagate the change through tree
            l_id = min(2 * p_id + 1, self.max_len - 1)  # the leaf's left node
            r_id = l_id + 1  # the leaf's right node
            if v <= self.tree[l_id]:
                p_id = l_id
            else:
                v -= self.tree[l_id]
                p_id = r_id
        return p_id, self.tree[p_id]  # leaf_id and leaf_value

    def important_sampling(self, batch_size: int, beg: int, end: int, per_beta: float) -> Tuple[Tensor, Tensor]:
        # get random values for searching indices with proportional prioritization
        values = (torch.arange(batch_size) + torch.rand(batch_size)) * (self.tree[0] / batch_size)

        # get proportional prioritization
        leaf_ids, leaf_values = list(zip(*[self.get_leaf_id_and_value(v) for v in values]))
        leaf_ids = torch.tensor(leaf_ids, dtype=torch.long)
        leaf_values = torch.tensor(leaf_values, dtype=torch.float32)

        indices = leaf_ids - (self.buf_len - 1)
        assert 0 <= indices.min()
        assert indices.max() < self.buf_len

        prob_ary = leaf_values / self.tree[beg:end].min()
        weights = torch.pow(prob_ary, -per_beta)
        return indices, weights
