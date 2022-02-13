import os
import torch
import numpy as np
import numpy.random as rd


class ReplayBuffer:  # for off-policy
    def __init__(self, max_len, state_dim, action_dim, gpu_id=0):
        self.now_len = 0
        self.next_idx = 0
        self.prev_idx = 0
        self.if_full = False
        self.max_len = max_len
        self.action_dim = action_dim
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        other_dim = 1 + 1 + self.action_dim  # reward_dim + mask_dim + action_dim
        self.buf_other = torch.empty((max_len, other_dim), dtype=torch.float32, device=self.device)

        buf_state_size = (max_len, state_dim) if isinstance(state_dim, int) else (max_len, *state_dim)
        self.buf_state = torch.empty(buf_state_size, dtype=torch.float32, device=self.device)

    def extend_buffer(self, state, other):  # CPU array to CPU array
        size = len(other)
        next_idx = self.next_idx + size

        if next_idx > self.max_len:
            self.buf_state[self.next_idx:self.max_len] = state[:self.max_len - self.next_idx]
            self.buf_other[self.next_idx:self.max_len] = other[:self.max_len - self.next_idx]
            self.if_full = True

            next_idx = next_idx - self.max_len
            self.buf_state[0:next_idx] = state[-next_idx:]
            self.buf_other[0:next_idx] = other[-next_idx:]
        else:
            self.buf_state[self.next_idx:next_idx] = state
            self.buf_other[self.next_idx:next_idx] = other
        self.next_idx = next_idx

    def update_buffer(self, traj_lists):
        steps = 0
        r_exp = 0.
        for traj_list in traj_lists:
            self.extend_buffer(state=traj_list[0], other=torch.hstack(traj_list[1:]))

            steps += traj_list[1].shape[0]
            r_exp += traj_list[1].mean().item()
        return steps, r_exp / len(traj_lists)

    def sample_batch(self, batch_size) -> tuple:
        indices = rd.randint(self.now_len - 1, size=batch_size)
        # r_m_a = self.buf_other[indices]
        # return (r_m_a[:, 0:1],
        #         r_m_a[:, 1:2],
        #         r_m_a[:, 2:],
        #         self.buf_state[indices],
        #         self.buf_state[indices + 1])
        return (self.buf_other[indices, 0:1],
                self.buf_other[indices, 1:2],
                self.buf_other[indices, 2:],
                self.buf_state[indices],
                self.buf_state[indices + 1])

    def sample_batch_r_m_a_s(self):
        if self.prev_idx <= self.next_idx:
            r = self.buf_other[self.prev_idx:self.next_idx, 0:1]
            m = self.buf_other[self.prev_idx:self.next_idx, 1:2]
            a = self.buf_other[self.prev_idx:self.next_idx, 2:]
            s = self.buf_state[self.prev_idx:self.next_idx]
        else:
            r = torch.vstack((self.buf_other[self.prev_idx:, 0:1],
                              self.buf_other[:self.next_idx, 0:1]))
            m = torch.vstack((self.buf_other[self.prev_idx:, 1:2],
                              self.buf_other[:self.next_idx, 1:2]))
            a = torch.vstack((self.buf_other[self.prev_idx:, 2:],
                              self.buf_other[:self.next_idx, 2:]))
            s = torch.vstack((self.buf_state[self.prev_idx:],
                              self.buf_state[:self.next_idx],))
        self.prev_idx = self.next_idx
        return r, m, a, s  # reward, mask, action, state

    def update_now_len(self):
        self.now_len = self.max_len if self.if_full else self.next_idx

    def save_or_load_history(self, cwd, if_save, buffer_id=0):
        save_path = f"{cwd}/replay_{buffer_id}.npz"

        if if_save:
            self.update_now_len()
            state_dim = self.buf_state.shape[1]
            other_dim = self.buf_other.shape[1]
            buf_state = np.empty((self.max_len, state_dim), dtype=np.float16)  # sometimes np.uint8
            buf_other = np.empty((self.max_len, other_dim), dtype=np.float16)

            temp_len = self.max_len - self.now_len
            buf_state[0:temp_len] = self.buf_state[self.now_len:self.max_len].detach().cpu().numpy()
            buf_other[0:temp_len] = self.buf_other[self.now_len:self.max_len].detach().cpu().numpy()

            buf_state[temp_len:] = self.buf_state[:self.now_len].detach().cpu().numpy()
            buf_other[temp_len:] = self.buf_other[:self.now_len].detach().cpu().numpy()

            np.savez_compressed(save_path, buf_state=buf_state, buf_other=buf_other)
            print(f"| ReplayBuffer save in: {save_path}")
        elif os.path.isfile(save_path):
            buf_dict = np.load(save_path)
            buf_state = buf_dict['buf_state']
            buf_other = buf_dict['buf_other']

            buf_state = torch.as_tensor(buf_state, dtype=torch.float32, device=self.device)
            buf_other = torch.as_tensor(buf_other, dtype=torch.float32, device=self.device)
            self.extend_buffer(buf_state, buf_other)
            self.update_now_len()
            print(f"| ReplayBuffer load: {save_path}")


class ReplayBufferList(list):  # for on-policy
    def __init__(self):
        list.__init__(self)

    def update_buffer(self, traj_list):
        cur_items = list(map(list, zip(*traj_list)))
        self[:] = [torch.cat(item, dim=0) for item in cur_items]

        steps = self[1].shape[0]
        r_exp = self[1].mean().item()
        return steps, r_exp
