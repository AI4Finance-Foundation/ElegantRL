import math
import os
from typing import Tuple

import torch as th

from .config import Config

TEN = th.Tensor


class ReplayBuffer:  # for off-policy
    def __init__(self,
                 max_size: int,
                 state_dim: int,
                 action_dim: int,
                 gpu_id: int = 0,
                 num_seqs: int = 1,
                 if_use_per: bool = False,
                 if_discrete: bool = False,
                 args: Config = Config()):
        self.p = 0  # pointer
        self.if_full = False
        self.cur_size = 0
        self.add_size = 0
        self.max_size = max_size
        self.num_seqs = num_seqs
        self.device = th.device(f"cuda:{gpu_id}" if (th.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        """The struction of ReplayBuffer (for example, num_seqs = num_workers * num_envs == 2*4 = 8
        ReplayBuffer:
        worker0 for env0:   sequence of sub_env0.0  self.states  = TEN[s, s, ..., s, ..., s]     
                                                    self.actions = TEN[a, a, ..., a, ..., a]   
                                                    self.rewards = TEN[r, r, ..., r, ..., r]   
                                                    self.undones = TEN[d, d, ..., d, ..., d]
                                                    self.unmasks = TEN[m, m, ..., m, ..., m]
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
        assert (action_dim < 256) or (not if_discrete)  # if_discrete==True, then action_dim < 256
        self.states = th.empty((max_size, num_seqs, state_dim), dtype=th.float32, device=self.device)
        self.actions = th.empty((max_size, num_seqs, action_dim), dtype=th.float32, device=self.device) \
            if not if_discrete else th.empty((max_size, num_seqs), dtype=th.uint8, device=self.device)
        self.rewards = th.empty((max_size, num_seqs), dtype=th.float32, device=self.device)
        self.undones = th.empty((max_size, num_seqs), dtype=th.float32, device=self.device)
        self.unmasks = th.empty((max_size, num_seqs), dtype=th.float32, device=self.device)

        self.cum_rewards = th.empty_like(self.rewards)
        self.ids0 = th.tensor((), dtype=th.long, device=self.device)
        self.ids1 = th.tensor((), dtype=th.long, device=self.device)

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

    def update(self, items: Tuple[TEN, ...]):
        states, actions, rewards, undones, unmasks = items
        # assert states.shape[1:] == (num_envs, state_dim)
        # assert actions.shape[1:] == (num_envs, action_dim if if_discrete else 1)
        # assert rewards.shape[1:] == (num_envs,)
        # assert undones.shape[1:] == (num_envs,)
        # assert unmasks.shape[1:] == (num_envs,)
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
            self.unmasks[p0:p1], self.unmasks[0:p] = unmasks[:p2], unmasks[-p:]
        else:
            self.states[self.p:p] = states
            self.actions[self.p:p] = actions
            self.rewards[self.p:p] = rewards
            self.undones[self.p:p] = undones
            self.unmasks[self.p:p] = unmasks

        if self.if_use_per:
            '''data_ids for single env'''
            data_ids = th.arange(self.p, p, dtype=th.long, device=self.device)
            if p > self.max_size:
                data_ids = th.fmod(data_ids, self.max_size)

            '''apply data_ids for vectorized env'''
            for sum_tree in self.sum_trees:
                sum_tree.update_ids(data_ids=data_ids.cpu(), prob=10.)

        self.p = p
        self.cur_size = self.max_size if self.if_full else self.p

    def sample(self, batch_size: int) -> Tuple[TEN, TEN, TEN, TEN, TEN, TEN]:
        sample_len = self.cur_size - 1

        ids = th.randint(sample_len * self.num_seqs, size=(batch_size,), requires_grad=False, device=self.device)
        self.ids0 = ids0 = th.fmod(ids, sample_len)  # ids % sample_len
        self.ids1 = ids1 = th.div(ids, sample_len, rounding_mode='floor')  # ids // sample_len

        return (
            self.states[ids0, ids1],
            self.actions[ids0, ids1],
            self.rewards[ids0, ids1],
            self.undones[ids0, ids1],
            self.unmasks[ids0, ids1],
            self.states[ids0 + 1, ids1],  # next_state
        )

    def sample_for_per(self, batch_size: int) -> Tuple[TEN, TEN, TEN, TEN, TEN, TEN, TEN, TEN]:
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

        is_indices: TEN = th.hstack(is_indices).to(self.device)
        is_weights: TEN = th.hstack(is_weights).to(self.device)

        self.ids0 = ids0 = th.fmod(is_indices, self.cur_size)  # is_indices % sample_len
        self.ids1 = ids1 = th.div(is_indices, self.cur_size, rounding_mode='floor')  # is_indices // sample_len
        return (
            self.states[ids0, ids1],
            self.actions[ids0, ids1],
            self.rewards[ids0, ids1],
            self.undones[ids0, ids1],
            self.unmasks[ids0, ids1],
            self.states[ids0 + 1, ids1],  # next_state
            is_weights,  # important sampling weights
            is_indices,  # important sampling indices
        )

    def td_error_update_for_per(self, is_indices: TEN, td_error: TEN):  # td_error = (q-q).detach_().abs()
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
                    buf_item = th.vstack((item[self.p:self.cur_size], item[0:self.p]))
                file_path = f"{cwd}/replay_buffer_{name}.pth"
                print(f"| buffer.save_or_load_history(): Save {file_path}", flush=True)
                th.save(buf_item, file_path)

        elif all([os.path.isfile(f"{cwd}/replay_buffer_{name}.pth") for item, name in item_names]):
            max_sizes = []
            for item, name in item_names:
                file_path = f"{cwd}/replay_buffer_{name}.pth"
                print(f"| buffer.save_or_load_history(): Load {file_path}", flush=True)
                buf_item = th.load(file_path)

                max_size = buf_item.shape[0]
                item[:max_size] = buf_item
                max_sizes.append(max_size)
            assert all([max_size == max_sizes[0] for max_size in max_sizes])
            self.cur_size = self.p = max_sizes[0]
            self.if_full = self.cur_size == self.max_size

    def update_cum_rewards(self, get_cumulative_rewards):
        if self.p >= self.add_size:
            p1 = self.p
            p0 = self.p - self.add_size

        else:
            p1 = self.max_size
            p0 = p1 - self.add_size
        cum_rewards = get_cumulative_rewards(rewards=self.rewards[p0:p1, :],
                                             undones=self.undones[p0:p1, :])
        self.cum_rewards[p0:p1, :] = cum_rewards


class SumTree:
    """ BinarySearchTree for PER (SumTree)
    Contributor: GitHub GyChou, GitHub MissIsSipPiu
    Reference: https://github.com/kaixindelele/DRLib/tree/main/algos/pytorch/td3_sp
    Reference: https://github.com/jaromiru/AI-blog/blob/master/SumTree.py
    """

    def __init__(self, buf_len: int):
        self.buf_len = buf_len  # replay buffer len
        self.max_len = (buf_len - 1) + buf_len  # parent_nodes_num + leaf_nodes_num
        self.depth = math.ceil(math.log2(self.max_len))

        self.tree = th.zeros(self.max_len, dtype=th.float32)

    def update_id(self, data_id: int, prob=10):  # 10 is max_prob
        tree_id = data_id + self.buf_len - 1

        delta = prob - self.tree[tree_id]
        self.tree[tree_id] = prob

        for depth in range(self.depth - 2):  # propagate the change through tree
            tree_id = (tree_id - 1) // 2  # faster than the recursive loop
            self.tree[tree_id] += delta

    def update_ids(self, data_ids: TEN, prob: TEN = 10.):  # 10 is max_prob
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
        ARY type for storing: [0, 1, 2, 3, 4, 5, 6]
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
        return p_id, float(self.tree[p_id])  # leaf_id and leaf_value

    def important_sampling(self, batch_size: int, beg: int, end: int, per_beta: float) -> Tuple[TEN, TEN]:
        # get random values for searching indices with proportional prioritization
        values = (th.arange(batch_size) + th.rand(batch_size)) * (self.tree[0] / batch_size)

        # get proportional prioritization
        leaf_ids, leaf_values = list(zip(*[self.get_leaf_id_and_value(v) for v in values]))
        leaf_ids = th.tensor(leaf_ids, dtype=th.long)
        leaf_values = th.tensor(leaf_values, dtype=th.float32)

        indices = leaf_ids - (self.buf_len - 1)
        assert 0 <= indices.min()
        assert indices.max() < self.buf_len

        prob_ary = leaf_values / self.tree[beg:end].min()
        weights = th.pow(prob_ary, -per_beta)
        return indices, weights
