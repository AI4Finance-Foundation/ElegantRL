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
                 if_logprob: bool = False,
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
                            sequence of sub_env0.3  s, s, ..., s    a, a, ..., a    r, r, ..., r    d, d, ..., d
        worker1 for env1:   sequence of sub_env1.0  s, s, ..., s    a, a, ..., a    r, r, ..., r    d, d, ..., d
                            sequence of sub_env1.1  s, s, ..., s    a, a, ..., a    r, r, ..., r    d, d, ..., d

        D: done=True
        d: done=False
        sequence of transition: s-a-r-d, s-a-r-d, s-a-r-D  s-a-r-d, s-a-r-d, s-a-r-d, s-a-r-d, s-a-r-D  s-a-r-d, ...
                                <------trajectory------->  <----------trajectory--------------------->  <-----------
        """
        assert (action_dim < 256) or (not if_discrete)  # if_discrete==True, then action_dim < 256

        self.observ_i = 0
        self.observ_j = self.reward_i = self.observ_i + state_dim
        self.reward_j = self.undone_i = self.observ_j + 1
        self.undone_j = self.unmask_i = self.reward_j + 1
        self.unmask_j = self.action_i = self.undone_j + 1
        self.action_j = self.logprob_i = self.unmask_j + (1 if if_discrete else action_dim)
        self.logprob_j = self.action_j + (action_dim if if_logprob else 0)

        self.buffer_dim = self.logprob_j
        self.seqs = th.empty((max_size, num_seqs, self.buffer_dim), dtype=th.float32, device=self.device)

        self.if_logprob = if_logprob
        self.t_int = th.int32
        self.ids0 = th.tensor((), dtype=self.t_int, device=self.device)
        self.ids1 = th.tensor((), dtype=self.t_int, device=self.device)

    def update_seqs(self, seqs: TEN):
        # observ, reward, undone, unmask, action = items
        # assert observ.shape[1:] == (num_envs, state_dim)
        # assert reward.shape[1:] == (num_envs,)
        # assert undone.shape[1:] == (num_envs,)
        # assert unmask.shape[1:] == (num_envs,)
        # assert action.shape[1:] == (num_envs, action_dim if if_discrete else 1)

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

    def sample_seqs(self, batch_size: int, seq_len: int) -> TEN:  # TODO
        seq_len = min((self.cur_size - 1) // 2, seq_len)
        sample_len = self.cur_size - 1 - seq_len
        assert sample_len > 0

        ids = th.randint(sample_len * self.num_seqs, size=(batch_size,), dtype=self.t_int, device=self.device)
        self.ids0 = ids0 = th.fmod(ids, sample_len)[:, None].repeat(1, seq_len)  # ids % sample_len
        self.ids1 = ids1 = (th.div(ids, sample_len, rounding_mode='floor')[:, None] +
                            th.arange(seq_len, dtype=self.t_int, device=self.device)[None, :])  # ids // sample_len
        return self.seqs[ids0, ids1]


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
