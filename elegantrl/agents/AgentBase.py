import os
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from elegantrl.train.replay_buffer import ReplayBuffer

'''[ElegantRL.2022.05.05](github.com/AI4Fiance-Foundation/ElegantRL)'''

Tensor = torch.Tensor


class AgentBase:  # [ElegantRL.2022.05.05]
    def __init__(self, net_dim: int, state_dim: int, action_dim: int, gpu_id=0, args=None):
        """initialize
        replace by different DRL algorithms

        :param net_dim: the dimension of networks (the width of neural networks)
        :param state_dim: the dimension of state (the number of state vector)
        :param action_dim: the dimension of action (the number of discrete action)
        :param gpu_id: the gpu_id of the training device. Use CPU when cuda is not available.
        :param args: the arguments for agent training. `args = Arguments()`
        """

        self.gamma = getattr(args, 'gamma', 0.99)
        self.env_num = getattr(args, 'env_num', 1)
        self.num_layer = getattr(args, 'num_layer', 3)
        self.batch_size = getattr(args, 'batch_size', 128)
        self.action_dim = getattr(args, 'action_dim', 3)
        self.repeat_times = getattr(args, 'repeat_times', 1.)
        self.reward_scale = getattr(args, 'reward_scale', 1.)
        self.clip_grad_norm = getattr(args, 'clip_grad_norm', 3.0)
        self.learning_rate = getattr(args, 'learning_rate', 2 ** -15)  # 2**-15 ~= 3e-5
        self.soft_update_tau = getattr(args, 'soft_update_tau', 2 ** -8)

        self.if_use_per = getattr(args, 'if_use_per', None)
        self.if_act_target = getattr(args, 'if_act_target', None)
        self.if_cri_target = getattr(args, 'if_cri_target', None)
        self.if_use_old_traj = getattr(args, 'if_use_old_traj', False)
        self.if_off_policy = getattr(args, 'if_off_policy', None)

        self.states = None  # assert self.states == (env_num, state_dim)
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        self.traj_list = [[torch.tensor((), dtype=torch.float32, device=self.device)
                           for _ in range(4 if self.if_off_policy else 5)]
                          for _ in range(self.env_num)]  # for `self.explore_vec_env()`

        act_class = getattr(self, "act_class", None)
        cri_class = getattr(self, "cri_class", None)
        self.act = act_class(net_dim, self.num_layer, state_dim, action_dim).to(self.device)
        self.cri = cri_class(net_dim, self.num_layer, state_dim, action_dim).to(self.device) \
            if cri_class else self.act

        from copy import deepcopy
        self.act_target = deepcopy(self.act) if self.if_act_target else self.act
        self.cri_target = deepcopy(self.cri) if self.if_cri_target else self.cri

        self.act_optimizer = torch.optim.Adam(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), self.learning_rate) \
            if cri_class else self.act_optimizer

        from types import MethodType  # built-in package of Python3
        self.act_optimizer.parameters = MethodType(get_optim_param, self.act_optimizer)
        self.cri_optimizer.parameters = MethodType(get_optim_param, self.cri_optimizer)

        """attribute"""
        if self.env_num == 1:
            self.explore_env = self.explore_one_env
        else:
            self.explore_env = self.explore_vec_env

        if getattr(args, 'if_use_per', False):
            self.criterion = torch.nn.SmoothL1Loss(reduction="none")
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.criterion = torch.nn.SmoothL1Loss(reduction="mean")
            self.get_obj_critic = self.get_obj_critic_raw

        '''h_term'''
        self.lambda_h_term = getattr(args, 'lambda_h_term', 2 ** -3)
        self.h_term_drop_rate = getattr(args, 'h_term_drop_rate', 2 ** -3)  # drop the data in H-term ReplayBuffer
        self.h_term_sample_rate = getattr(args, 'h_term_sample_rate', 2 ** -4)  # sample the data in H-term ReplayBuffer
        self.h_term_buffer = list()
        self.ten_state = None
        self.ten_action = None
        self.ten_r_sum = None
        self.ten_r_norm = None

    def explore_one_env(self, env, target_step: int) -> list:
        """
        Collect trajectories through the actor-environment interaction for a **single** environment instance.

        :param env: RL training environment. env.reset() env.step(). It should be a vector env.
        :param target_step: explored target_step number of step in env
        :return: `[traj, ]`
        `traj = [(state, reward, mask, action, noise), ...]` for on-policy
        `traj = [(state, reward, mask, action), ...]` for off-policy
        """
        traj_list = []
        last_done = [0, ]
        state = self.states[0]

        i = 0
        done = False
        while i < target_step or not done:
            ten_s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            ten_a = self.act.get_action(ten_s.to(self.device)).detach().cpu()  # different
            next_s, reward, done, _ = env.step(ten_a[0].numpy())  # different

            traj_list.append((ten_s, reward, done, ten_a))  # different

            i += 1
            state = env.reset() if done else next_s

        self.states[0] = state
        last_done[0] = i
        return self.convert_trajectory(traj_list, last_done)  # traj_list

    def explore_vec_env(self, env, target_step: int) -> list:
        """
        Collect trajectories through the actor-environment interaction for a **vectorized** environment instance.

        :param env: RL training environment. env.reset() env.step(). It should be a vector env.
        :param target_step: explored target_step number of step in env
        :return: `[traj, ...]`
        `traj = [(state, reward, mask, action, noise), ...]` for on-policy
        `traj = [(state, reward, mask, action), ...]` for off-policy
        """
        traj_list = []
        last_done = torch.zeros(self.env_num, dtype=torch.int, device=self.device)
        ten_s = self.states

        i = 0
        ten_dones = torch.zeros(self.env_num, dtype=torch.int, device=self.device)
        while i < target_step or not any(ten_dones):
            ten_a = self.act.get_action(ten_s).detach()  # different
            ten_s_next, ten_rewards, ten_dones, _ = env.step(ten_a)  # different

            traj_list.append((ten_s.clone(), ten_rewards.clone(), ten_dones.clone(), ten_a))  # different

            i += 1
            last_done[torch.where(ten_dones)[0]] = i  # behind `step_i+=1`
            ten_s = ten_s_next

        self.states = ten_s
        return self.convert_trajectory(traj_list, last_done)  # traj_list

    def get_obj_critic_raw(self, buffer: ReplayBuffer, batch_size: int) -> (Tensor, Tensor):
        """
        Calculate the loss of networks with **uniform sampling**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and states.
        """
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_a = self.act_target(next_s)
            critic_targets: torch.Tensor = self.cri_target(next_s, next_a)
            (next_q, min_indices) = torch.min(critic_targets, dim=1, keepdim=True)
            q_label = reward + mask * next_q

        q = self.cri(state, action)
        obj_critic = self.criterion(q, q_label)

        return obj_critic, state

    def get_obj_critic_per(self, buffer: ReplayBuffer, batch_size: int) -> (Tensor, Tensor):
        """
        Calculate the loss of the network with **Prioritized Experience Replay (PER)**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and states.
        """
        with torch.no_grad():
            reward, mask, action, state, next_s, is_weights = buffer.sample_batch(batch_size)
            next_a = self.act_target(next_s)
            critic_targets: torch.Tensor = self.cri_target(next_s, next_a)
            # taking a minimum while preserving the dimension for possible twin critics
            (next_q, min_indices) = torch.min(critic_targets, dim=1, keepdim=True)
            q_label = reward + mask * next_q

        q = self.cri(state, action)
        td_error = self.criterion(q, q_label)
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, state

    def optimizer_update(self, optimizer, objective):  # [ElegantRL 2021.11.11]
        """minimize the optimization objective via update the network parameters

        :param optimizer: `optimizer = torch.optim.SGD(net.parameters(), learning_rate)`
        :param objective: `objective = net(...)` the optimization objective, sometimes is a loss function.
        """
        optimizer.zero_grad()
        objective.backward()
        clip_grad_norm_(parameters=optimizer.param_groups[0]["params"], max_norm=self.clip_grad_norm)
        optimizer.step()

    def optim_update_amp(self, optimizer, objective):  # automatic mixed precision
        """minimize the optimization objective via update the network parameters

        amp: Automatic Mixed Precision

        :param optimizer: `optimizer = torch.optim.SGD(net.parameters(), learning_rate)`
        :param objective: `objective = net(...)` the optimization objective, sometimes is a loss function.
        """
        amp_scale = torch.cuda.amp.GradScaler()  # write in __init__()

        optimizer.zero_grad()
        amp_scale.scale(objective).backward()  # loss.backward()
        amp_scale.unscale_(optimizer)  # amp

        # from torch.nn.utils import clip_grad_norm_
        clip_grad_norm_(parameters=optimizer.param_groups[0]["params"], max_norm=self.clip_grad_norm)
        amp_scale.step(optimizer)  # optimizer.step()
        amp_scale.update()  # optimizer.step()

    @staticmethod
    def soft_update(target_net, current_net, tau: float):
        """soft update target network via current network

        :param target_net: update target network via current network to make training more stable.
        :param current_net: current network update via an optimizer
        :param tau: tau of soft target update: `target_net = target_net * (1-tau) + current_net * tau`
        """
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

    def save_or_load_agent(self, cwd: str, if_save: bool):
        """save or load training files for Agent

        :param cwd: Current Working Directory. ElegantRL save training files in CWD.
        :param if_save: True: save files. False: load files.
        """

        def load_torch_file(model, path: str):
            state_dict = torch.load(path, map_location=lambda storage, loc: storage)
            model.load_state_dict(state_dict)

        name_obj_list = [
            ("actor", self.act),
            ("act_target", self.act_target),
            ("act_optimizer", self.act_optimizer),
            ("critic", self.cri),
            ("cri_target", self.cri_target),
            ("cri_optimizer", self.cri_optimizer),
        ]
        name_obj_list = [(name, obj) for name, obj in name_obj_list if obj is not None]

        if if_save:
            for name, obj in name_obj_list:
                save_path = f"{cwd}/{name}.pth"
                torch.save(obj.state_dict(), save_path)
        else:
            for name, obj in name_obj_list:
                save_path = f"{cwd}/{name}.pth"
                load_torch_file(obj, save_path) if os.path.isfile(save_path) else None

    def convert_trajectory(self, traj_list, last_done: [list or Tensor]) -> list:  # [ElegantRL.2022.02.02]
        # assert len(buf_items) == step_i
        # assert len(buf_items[0]) in {4, 5}
        # assert len(buf_items[0][0]) == self.env_num
        traj_list = list(map(list, zip(*traj_list)))  # state, reward, done, action, noise
        # assert len(buf_items) == {4, 5}
        # assert len(buf_items[0]) == step
        # assert len(buf_items[0][0]) == self.env_num

        '''stack items'''
        traj_list[0] = torch.stack(traj_list[0])
        traj_list[3:] = [torch.stack(item) for item in traj_list[3:]]

        if len(traj_list[3].shape) == 2:
            traj_list[3] = traj_list[3].unsqueeze(2)

        if self.env_num > 1:
            traj_list[1] = (torch.stack(traj_list[1]) * self.reward_scale).unsqueeze(2)
            traj_list[2] = ((1 - torch.stack(traj_list[2])) * self.gamma).unsqueeze(2)
        else:
            traj_list[1] = (torch.tensor(traj_list[1], dtype=torch.float32) * self.reward_scale
                            ).unsqueeze(1).unsqueeze(2)
            traj_list[2] = ((1 - torch.tensor(traj_list[2], dtype=torch.float32)) * self.gamma
                            ).unsqueeze(1).unsqueeze(2)
        # assert all([buf_item.shape[:2] == (step, self.env_num) for buf_item in buf_items])

        '''splice items'''
        for j in range(len(traj_list)):
            cur_item = list()
            buf_item = traj_list[j]

            for env_i in range(self.env_num):
                last_step = last_done[env_i]

                pre_item = self.traj_list[env_i][j]
                if len(pre_item):
                    cur_item.append(pre_item)

                cur_item.append(buf_item[:last_step, env_i])

                if self.if_use_old_traj:
                    self.traj_list[env_i][j] = buf_item[last_step:, env_i]

            traj_list[j] = torch.vstack(cur_item)

        # on-policy:  buf_item = [states, rewards, dones, actions, noises]
        # off-policy: buf_item = [states, rewards, dones, actions]
        # buf_items = [buf_item, ...]
        return traj_list

    def get_buf_h_term(self, buf_state: Tensor, buf_action: Tensor, buf_r_sum: Tensor):
        buf_r_norm = buf_r_sum - buf_r_sum.mean()
        buf_r_diff = torch.where(buf_r_norm[:-1] * buf_r_norm[1:] <= 0)[0].detach().cpu().numpy() + 1
        buf_r_diff = list(buf_r_diff) + [buf_r_norm.shape[0], ]

        step_i = 0
        min_len = 8
        positive_list = list()
        for step_j in buf_r_diff:
            if buf_r_norm[step_i] > 0 and step_i + min_len < step_j:
                positive_list.append((step_i, step_j))
            step_i = step_j

        for step_i, step_j in positive_list:
            index = np.arange(step_i, step_j)

            ten_state = buf_state[index]
            ten_action = buf_action[index]
            ten_r_sum = buf_r_sum[index]

            q_avg = ten_r_sum.mean().item()
            q_min = ten_r_sum.min().item()
            q_max = ten_r_sum.max().item()

            self.h_term_buffer.append((ten_state, ten_action, ten_r_sum, q_avg, q_min, q_max))

        # q_arg_sort = np.argsort([item[3] for item in self.h_term_buffer])
        # h_term_throw = max(0, int(len(self.h_term_buffer) * self.h_term_drop_rate) - 1)
        # self.h_term_buffer = [self.h_term_buffer[i] for i in q_arg_sort[h_term_throw:]]
        # q_arg_sort = np.argsort([item[3] for item in self.h_term_buffer])
        h_term_throw = max(0, int(len(self.h_term_buffer) * self.h_term_drop_rate) - 1)
        del self.h_term_buffer[:h_term_throw]  # todo

        q_min = np.min(np.array([item[4] for item in self.h_term_buffer]))
        q_max = np.max(np.array([item[5] for item in self.h_term_buffer]))

        self.ten_r_sum = torch.hstack([item[2] for item in self.h_term_buffer])  # ten_r_sum.shape == (-1, )
        self.ten_r_norm = (self.ten_r_sum - q_min) / (q_max - q_min)
        self.ten_state = torch.vstack([item[0] for item in self.h_term_buffer])  # ten_state.shape == (-1, state_dim)
        self.ten_action = torch.vstack([item[1] for item in self.h_term_buffer])  # ten_action.shape == (-1, action_dim)

    def get_obj_h_term(self) -> Tensor:
        if self.ten_state is None:
            return torch.zeros(1, dtype=torch.float32, device=self.device)
        ten_size = self.ten_state.shape[0]
        if ten_size < 1024:
            return torch.zeros(1, dtype=torch.float32, device=self.device)

        '''rd sample'''
        indices = torch.randint(ten_size, size=(int(ten_size * self.h_term_sample_rate),),
                                requires_grad=False, device=self.device)
        ten_state = self.ten_state[indices]
        ten_action = self.ten_action[indices]
        ten_r_norm = self.ten_r_norm[indices]

        '''hamilton'''
        ten_logprob = self.act.get_logprob(ten_state, ten_action)
        ten_hamilton = ten_logprob.exp().prod(dim=1)
        return -(ten_hamilton * ten_r_norm).mean() * self.lambda_h_term

    def get_r_sum_h_term(self, buf_reward: Tensor, buf_mask: Tensor) -> Tensor:
        buf_len = buf_reward.shape[0]
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # reward sum

        pre_r_sum = 0
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
        return buf_r_sum


def get_optim_param(optim) -> list:
    # optim = torch.optim.Adam(network_param, learning_rate)
    params_list = []
    for params_dict in optim.state_dict()["state"].values():
        params_list.extend([t for t in params_dict.values() if isinstance(t, torch.Tensor)])
    return params_list
