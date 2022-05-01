import os

import numpy as np
import numpy.random as rd
import torch


class ReplayBuffer:  # for off-policy
    def __init__(self, max_capacity, state_dim, action_dim, gpu_id=0):
        self.prev_p = 0  # previous pointer
        self.next_p = 0  # next pointer
        self.if_full = False
        self.cur_capacity = 0  # current capacity
        self.max_capacity = max_capacity
        self.add_capacity = 0  # update in self.update_buffer

        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        self.buf_action = torch.empty((max_capacity, action_dim), dtype=torch.float32, device=self.device)
        self.buf_reward = torch.empty((max_capacity, 1), dtype=torch.float32, device=self.device)
        self.buf_mask = torch.empty((max_capacity, 1), dtype=torch.float32, device=self.device)

        buf_state_size = (max_capacity, state_dim) if isinstance(state_dim, int) else (max_capacity, *state_dim)
        self.buf_state = torch.empty(buf_state_size, dtype=torch.float32, device=self.device)

    def update_buffer(self, traj_list):
        traj_items = list(map(list, zip(*traj_list)))

        states, rewards, masks, actions = [torch.cat(item, dim=0) for item in traj_items]
        self.add_capacity = rewards.shape[0]
        p = self.next_p + self.add_capacity  # pointer

        if p > self.max_capacity:
            self.buf_state[self.next_p:self.max_capacity] = states[:self.max_capacity - self.next_p]
            self.buf_reward[self.next_p:self.max_capacity] = rewards[:self.max_capacity - self.next_p]
            self.buf_mask[self.next_p:self.max_capacity] = masks[:self.max_capacity - self.next_p]
            self.buf_action[self.next_p:self.max_capacity] = actions[:self.max_capacity - self.next_p]
            self.if_full = True

            p = p - self.max_capacity
            self.buf_state[0:p] = states[-p:]
            self.buf_reward[0:p] = rewards[-p:]
            self.buf_mask[0:p] = masks[-p:]
            self.buf_action[0:p] = actions[-p:]
        else:
            self.buf_state[self.next_p:p] = states
            self.buf_reward[self.next_p:p] = rewards
            self.buf_mask[self.next_p:p] = masks
            self.buf_action[self.next_p:p] = actions
        self.next_p = p

        self.cur_capacity = self.max_capacity if self.if_full else self.next_p

        steps = rewards.shape[0]
        r_exp = rewards.mean().item()
        return steps, r_exp

    def sample_batch(self, batch_size) -> tuple:
        indices = torch.randint(self.cur_capacity - 1, size=(batch_size,), device=self.device)

        i1 = self.next_p
        i0 = self.next_p - self.add_capacity
        num_new_indices = 1
        new_indices = torch.randint(i0, i1, size=(num_new_indices,)) % (self.max_capacity - 1)
        indices[0:num_new_indices] = new_indices  # todo
        return (
            self.buf_reward[indices],
            self.buf_mask[indices],
            self.buf_action[indices],
            self.buf_state[indices],
            self.buf_state[indices + 1]  # next state
        )

    def save_or_load_history(self, cwd, if_save):
        obj_names = (
            (self.buf_reward, "reward"),
            (self.buf_mask, "mask"),
            (self.buf_action, "action"),
            (self.buf_state, "state"),
        )

        if if_save:
            print(f"| {self.__class__.__name__}: Saving in cwd {cwd}")
            for obj, name in obj_names:
                if self.cur_capacity == self.next_p:
                    buf_tensor = obj[:self.cur_capacity]
                else:
                    buf_tensor = torch.vstack((obj[self.next_p:self.cur_capacity], obj[0:self.next_p]))

                torch.save(buf_tensor, f"{cwd}/replay_buffer_{name}.pt")

            print(f"| {self.__class__.__name__}: Saved in cwd {cwd}")

        elif os.path.isfile(f"{cwd}/replay_buffer_state.pt"):
            print(f"| {self.__class__.__name__}: Loading from cwd {cwd}")
            buf_capacity = 0
            for obj, name in obj_names:
                buf_tensor = torch.load(f"{cwd}/replay_buffer_{name}.pt")
                buf_capacity = buf_tensor.shape[0]

                obj[:buf_capacity] = buf_tensor
            self.cur_capacity = buf_capacity

            print(f"| {self.__class__.__name__}: Loaded from cwd {cwd}")

    def get_state_norm(self, cwd='.', neg_state_avg=0, div_state_std=1):
        state_avg, state_std = get_state_avg_std(
            buf_state=self.buf_state, batch_size=2 ** 10,
            neg_state_avg=neg_state_avg, div_state_std=div_state_std,
        )

        torch.save(state_avg, f"{cwd}/state_norm_avg.pt")
        print(f"| {self.__class__.__name__}: state_avg = {state_avg}")
        torch.save(state_std, f"{cwd}/state_norm_std.pt")
        print(f"| {self.__class__.__name__}: state_std = {state_std}")


class ReplayBufferList(list):  # for on-policy
    def __init__(self):
        list.__init__(self)  # (buf_state, buf_reward, buf_mask, buf_action, buf_noise) = self[:]

    def update_buffer(self, traj_list):
        cur_items = list(map(list, zip(*traj_list)))
        self[:] = [torch.cat(item, dim=0) for item in cur_items]

        steps = self[1].shape[0]
        r_exp = self[1].mean().item()
        return steps, r_exp

    def get_state_norm(self, cwd='.', neg_state_avg=0, div_state_std=1):
        state_avg, state_std = get_state_avg_std(
            buf_state=self[0], batch_size=2 ** 10,
            neg_state_avg=neg_state_avg, div_state_std=div_state_std,
        )

        torch.save(state_avg, f"{cwd}/state_norm_avg.pt")
        print(f"| {self.__class__.__name__}: state_avg = {state_avg}")
        torch.save(state_std, f"{cwd}/state_norm_std.pt")
        print(f"| {self.__class__.__name__}: state_std = {state_std}")


def get_state_avg_std(buf_state, batch_size=2 ** 10, neg_state_avg=0, div_state_std=1):
    state_len = buf_state.shape[0]
    state_avg = torch.zeros_like(buf_state[0])
    state_std = torch.zeros_like(buf_state[0])

    from tqdm import trange
    for i in trange(0, state_len, batch_size):
        state_part = buf_state[i:i + batch_size]
        state_avg += state_part.mean(axis=0)
        state_std += state_part.std(axis=0)

    num = max(1, state_len // batch_size)
    state_avg /= num
    state_std /= num

    state_avg = state_avg / div_state_std - neg_state_avg
    state_std = state_std / div_state_std - neg_state_avg
    return state_avg.cpu(), state_std.cpu()


class BinarySearchTree:
    """Binary Search Tree for PER

    Contributor: Github GyChou, Github mississippiu
    Reference: https://github.com/kaixindelele/DRLib/tree/main/algos/pytorch/td3_sp
    Reference: https://github.com/jaromiru/AI-blog/blob/master/SumTree.py
    """

    def __init__(self, memo_len):
        self.memo_len = memo_len  # replay buffer len
        self.prob_ary = np.zeros(
            (memo_len - 1) + memo_len
        )  # parent_nodes_num + leaf_nodes_num
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
        self.prob_ary[
            ids
        ] = prob  # here, ids means the indices of given children (maybe the right ones or left ones)
        p_ids = (ids - 1) // 2

        while upper_step:  # propagate the change through tree
            ids = (
                    p_ids * 2 + 1
            )  # in this while loop, ids means the indices of the left children
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
        self.per_beta = min(1.0, self.per_beta + 0.001)

        # get random values for searching indices with proportional prioritization
        values = (rd.rand(batch_size) + np.arange(batch_size)) * (
                self.prob_ary[0] / batch_size
        )

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
