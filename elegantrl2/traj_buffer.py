import os
import torch as th

from .traj_config import Config


TEN = th.Tensor



class TrajBuffer:  # for off-policy
    def __init__(self,
                 max_size: int,
                 state_dim: int,
                 action_dim: int,
                 gpu_id: int = 0,
                 num_seqs: int = 1,
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

        action_save_dim = 1 if if_discrete else action_dim
        self.observ_i, self.observ_j = (0, state_dim)
        self.reward_i = state_dim
        self.undone_i = state_dim + 1
        self.unmask_i = state_dim + 2
        self.action_i, self.action_j = state_dim + 3, state_dim + 3 + action_save_dim

        self.buffer_dim = state_dim + 3 + action_save_dim
        self.seqs = th.empty((max_size, num_seqs, self.buffer_dim), dtype=th.float32, device=self.device)

        self.ids0 = th.tensor((), dtype=th.long, device=self.device)
        self.ids1 = th.tensor((), dtype=th.long, device=self.device)

    def update(self, items: tuple[TEN, ...]):
        observ, action, reward, undone, unmask = items
        # assert observ.shape[1:] == (num_envs, state_dim)
        # assert action.shape[1:] == (num_envs, action_dim if if_discrete else 1)
        # assert reward.shape[1:] == (num_envs,)
        # assert undone.shape[1:] == (num_envs,)
        # assert unmask.shape[1:] == (num_envs,)
        self.add_size = reward.shape[0]
        seqs = th.empty((self.add_size, self.num_seqs, self.buffer_dim), dtype=th.float32, device=self.device)
        seqs[:, :, self.observ_i:self.observ_j] = observ
        seqs[:, :, self.action_i:self.action_j] = action
        seqs[:, :, self.reward_i] = reward
        seqs[:, :, self.undone_i] = undone
        seqs[:, :, self.unmask_i] = unmask

        p = self.p + self.add_size  # pointer
        if p > self.max_size:
            self.if_full = True
            p0 = self.p
            p1 = self.max_size
            p2 = self.max_size - self.p
            p = p - self.max_size

            self.seqs[p0:p1], self.seqs[0:p] = seqs[:p2], seqs[-p:]
        else:
            self.seqs[self.p:p] = seqs

        self.p = p
        self.cur_size = self.max_size if self.if_full else self.p

    def sample_seqs(self, batch_size: int, seq_len: int) -> tuple[TEN, TEN, TEN, TEN, TEN, TEN]:  # TODO
        seq_len = min((self.cur_size - 1) // 2, seq_len)
        sample_len = self.cur_size - 1 - seq_len
        assert sample_len > 0

        # ids = th.randint(sample_len * self.num_seqs, size=(batch_size,), requires_grad=False, device=self.device)
        # self.ids0 = ids0 = th.fmod(ids, sample_len)  # ids % sample_len
        # self.ids1 = ids1 = th.div(ids, sample_len, rounding_mode='floor')  # ids // sample_len
        #
        # return (
        #     self.states[ids0, ids1],
        #     self.actions[ids0, ids1],
        #     self.rewards[ids0, ids1],
        #     self.undones[ids0, ids1],
        #     self.unmasks[ids0, ids1],
        #     self.states[ids0 + 1, ids1],  # next_state
        # )

    def save_or_load_history(self, cwd: str, if_save: bool):
        file_path = f"{cwd}/replay_buffer.pth"

        if if_save:
            if self.cur_size == self.p:
                seqs = self.seqs[:self.cur_size]
            else:
                seqs = th.vstack((self.seqs[self.p:self.cur_size], self.seqs[0:self.p]))
            print(f"| buffer.save_or_load_history(): Save {file_path}", flush=True)
            th.save(seqs, file_path)

        elif os.path.isfile(file_path):
            print(f"| buffer.save_or_load_history(): Load {file_path}", flush=True)
            seqs = th.load(file_path)

            max_size = seqs.shape[0]
            self.seqs[:max_size] = seqs
            self.cur_size = self.p = max_size
            self.if_full = self.cur_size == self.max_size
