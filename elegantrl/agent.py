import os
import torch

import numpy as np
import numpy.random as rd

from copy import deepcopy
from elegantrl.net import Actor, ActorSAC, ActorFixSAC, CriticTwin, CriticREDq
from elegantrl.net import ActorPPO, ActorDiscretePPO, CriticPPO
from elegantrl.net import QNet, QNetDuel, QNetTwin, QNetTwinDuel


class AgentBase:
    def __init__(self, net_dim: int, state_dim: int, action_dim: int, gpu_id=0, args=None):
        self.gamma = getattr(args, 'gamma', 0.99)
        self.env_num = getattr(args, 'env_num', 1)
        self.batch_size = getattr(args, 'batch_size', 128)
        self.repeat_times = getattr(args, 'repeat_times', 1.)
        self.reward_scale = getattr(args, 'reward_scale', 1.)
        self.lambda_gae_adv = getattr(args, 'lambda_entropy', 0.98)
        self.if_use_old_traj = getattr(args, 'if_use_old_traj', False)
        self.soft_update_tau = getattr(args, 'soft_update_tau', 2 ** -8)

        if_act_target = getattr(args, 'if_act_target', False)
        if_cri_target = getattr(args, 'if_cri_target', False)
        if_off_policy = getattr(args, 'if_off_policy', True)
        learning_rate = getattr(args, 'learning_rate', 2 ** -12)

        self.states = None
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        self.traj_list = [[list() for _ in range(4 if if_off_policy else 5)]
                          for _ in range(self.env_num)]  # for `self.explore_vec_env()`

        act_class = getattr(self, 'act_class', None)
        cri_class = getattr(self, 'cri_class', None)
        self.act = act_class(net_dim, state_dim, action_dim).to(self.device)
        self.cri = cri_class(net_dim, state_dim, action_dim).to(self.device) if cri_class else self.act
        self.act_target = deepcopy(self.act) if if_act_target else self.act
        self.cri_target = deepcopy(self.cri) if if_cri_target else self.cri

        self.act_optimizer = torch.optim.Adam(self.act.parameters(), learning_rate)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), learning_rate) if cri_class else self.act_optimizer

        '''function'''
        self.criterion = torch.nn.SmoothL1Loss()

        if self.env_num == 1:
            self.explore_env = self.explore_one_env
        else:
            self.explore_env = self.explore_vec_env

        if getattr(args, 'if_use_per', False):  # PER (Prioritized Experience Replay) for sparse reward
            self.criterion = torch.nn.SmoothL1Loss(reduction='none')
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.criterion = torch.nn.SmoothL1Loss(reduction='mean')
            self.get_obj_critic = self.get_obj_critic_raw

    def explore_one_env(self, env, target_step) -> list:
        traj_list = list()
        last_done = [0, ]
        state = self.states[0]

        step_i = 0
        done = False
        while step_i < target_step or not done:
            ten_s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            ten_a = self.act.get_action(ten_s.to(self.device)).detach().cpu()  # different
            next_s, reward, done, _ = env.step(ten_a[0].numpy())  # different

            traj_list.append((ten_s, reward, done, ten_a))  # different

            step_i += 1
            state = env.reset() if done else next_s

        self.states[0] = state
        last_done[0] = step_i
        return self.convert_trajectory(traj_list, last_done)  # traj_list

    def explore_vec_env(self, env, target_step) -> list:
        traj_list = list()
        last_done = torch.zeros(self.env_num, dtype=torch.int, device=self.device)
        ten_s = self.states

        step_i = 0
        ten_dones = torch.zeros(self.env_num, dtype=torch.int, device=self.device)
        while step_i < target_step or not any(ten_dones):
            ten_a = self.act.get_action(ten_s).detach()  # different
            ten_s_next, ten_rewards, ten_dones, _ = env.step(ten_a)  # different

            traj_list.append((ten_s.clone(), ten_rewards.clone(), ten_dones.clone(), ten_a))  # different

            step_i += 1
            last_done[torch.where(ten_dones)[0]] = step_i  # behind `step_i+=1`
            ten_s = ten_s_next

        self.states = ten_s
        return self.convert_trajectory(traj_list, last_done)  # traj_list

    def convert_trajectory(self, buf_items, last_done):  # [ElegantRL.2022.01.01]
        # assert len(buf_items) == step_i
        # assert len(buf_items[0]) in {4, 5}
        # assert len(buf_items[0][0]) == self.env_num
        buf_items = list(map(list, zip(*buf_items)))  # state, reward, done, action, noise
        # assert len(buf_items) == {4, 5}
        # assert len(buf_items[0]) == step
        # assert len(buf_items[0][0]) == self.env_num

        '''stack items'''
        buf_items[0] = torch.stack(buf_items[0])
        buf_items[3:] = [torch.stack(item) for item in buf_items[3:]]

        if len(buf_items[3].shape) == 2:
            buf_items[3] = buf_items[3].unsqueeze(2)

        if self.env_num > 1:
            buf_items[1] = (torch.stack(buf_items[1]) * self.reward_scale).unsqueeze(2)
            buf_items[2] = ((1 - torch.stack(buf_items[2])) * self.gamma).unsqueeze(2)
        else:
            buf_items[1] = (torch.tensor(buf_items[1], dtype=torch.float32) * self.reward_scale
                            ).unsqueeze(1).unsqueeze(2)
            buf_items[2] = ((1 - torch.tensor(buf_items[2], dtype=torch.float32)) * self.gamma
                            ).unsqueeze(1).unsqueeze(2)
        # assert all([buf_item.shape[:2] == (step, self.env_num) for buf_item in buf_items])

        '''splice items'''
        for j in range(len(buf_items)):
            cur_item = list()
            buf_item = buf_items[j]

            for env_i in range(self.env_num):
                last_step = last_done[env_i]

                pre_item = self.traj_list[env_i][j]
                if len(pre_item):
                    cur_item.append(pre_item)

                cur_item.append(buf_item[:last_step, env_i])

                if self.if_use_old_traj:
                    self.traj_list[env_i][j] = buf_item[last_step:, env_i]

            buf_items[j] = torch.vstack(cur_item)

        # on-policy:  buf_item = [states, rewards, dones, actions, noises]
        # off-policy: buf_item = [states, rewards, dones, actions]
        # buf_items = [buf_item, ...]
        return buf_items

    def get_obj_critic_raw(self, buffer, batch_size):
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

    def get_obj_critic_per(self, buffer, batch_size):
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

    @staticmethod
    def optimizer_update(optimizer, objective):
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    @staticmethod
    def soft_update(target_net, current_net, tau):
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

    def save_or_load_agent(self, cwd, if_save):
        def load_torch_file(model_or_optim, _path):
            state_dict = torch.load(_path, map_location=lambda storage, loc: storage)
            model_or_optim.load_state_dict(state_dict)

        name_obj_list = [('actor', self.act), ('act_target', self.act_target), ('act_optim', self.act_optimizer),
                         ('critic', self.cri), ('cri_target', self.cri_target), ('cri_optim', self.cri_optimizer), ]
        name_obj_list = [(name, obj) for name, obj in name_obj_list if obj is not None]
        if if_save:
            for name, obj in name_obj_list:
                save_path = f"{cwd}/{name}.pth"
                torch.save(obj.state_dict(), save_path)
        else:
            for name, obj in name_obj_list:
                save_path = f"{cwd}/{name}.pth"
                load_torch_file(obj, save_path) if os.path.isfile(save_path) else None


'''DQN'''


class AgentDQN(AgentBase):
    def __init__(self, net_dim, state_dim, action_dim, gpu_id=0, args=None):
        self.if_off_policy = True
        self.act_class = getattr(self, 'act_class', QNet)
        self.cri_class = None  # = act_class
        AgentBase.__init__(self, net_dim, state_dim, action_dim, gpu_id, args)
        self.act.explore_rate = getattr(args, 'explore_rate', 0.125)
        # the probability of choosing action randomly in epsilon-greedy

    def explore_one_env(self, env, target_step) -> list:
        traj_list = list()
        last_done = [0, ]
        state = self.states[0]

        step_i = 0
        done = False
        while step_i < target_step or not done:
            ten_s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            ten_a = self.act.get_action(ten_s.to(self.device)).detach().cpu()
            next_s, reward, done, _ = env.step(ten_a[0, 0].numpy())  # different

            traj_list.append((ten_s, reward, done, ten_a))  # different

            step_i += 1
            state = env.reset() if done else next_s

        self.states[0] = state
        last_done[0] = step_i
        return self.convert_trajectory(traj_list, last_done)  # traj_list

    def explore_vec_env(self, env, target_step) -> list:
        traj_list = list()
        last_done = torch.zeros(self.env_num, dtype=torch.int, device=self.device)
        ten_s = self.states

        step_i = 0
        ten_dones = torch.zeros(self.env_num, dtype=torch.int, device=self.device)
        while step_i < target_step or not any(ten_dones):
            ten_a = self.act.get_action(ten_s).detach()
            ten_s_next, ten_rewards, ten_dones, _ = env.step(ten_a)  # different

            traj_list.append((ten_s.clone(), ten_rewards.clone(), ten_dones.clone(), ten_a))  # different

            step_i += 1
            last_done[torch.where(ten_dones)[0]] = step_i  # behind `step_i+=1`
            ten_s = ten_s_next

        self.states = ten_s
        return self.convert_trajectory(traj_list, last_done)  # traj_list

    def update_net(self, buffer) -> tuple:
        buffer.update_now_len()
        obj_critic = q_value = None
        for _ in range(int(1 + buffer.now_len * self.repeat_times / self.batch_size)):
            obj_critic, q_value = self.get_obj_critic(buffer, self.batch_size)
            self.optimizer_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
        return obj_critic.item(), q_value.mean().item()

    def get_obj_critic_raw(self, buffer, batch_size):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_q = self.cri_target(next_s).max(dim=1, keepdim=True)[0]
            q_label = reward + mask * next_q

        q_value = self.cri(state).gather(1, action.long())
        obj_critic = self.criterion(q_value, q_label)
        return obj_critic, q_value

    def get_obj_critic_per(self, buffer, batch_size):
        with torch.no_grad():
            reward, mask, action, state, next_s, is_weights = buffer.sample_batch(batch_size)
            next_q = self.cri_target(next_s).max(dim=1, keepdim=True)[0]
            q_label = reward + mask * next_q

        q_value = self.cri(state).gather(1, action.long())
        td_error = self.criterion(q_value, q_label)  # or td_error = (q_value - q_label).abs()
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, q_value


class AgentDuelingDQN(AgentDQN):
    def __init__(self, net_dim, state_dim, action_dim, gpu_id=0, args=None):
        self.act_class = getattr(self, 'act_class', QNetDuel)
        super().__init__(net_dim, state_dim, action_dim, gpu_id, args)


class AgentDoubleDQN(AgentDQN):
    def __init__(self, net_dim, state_dim, action_dim, gpu_id=0, args=None):
        self.act_class = getattr(self, 'act_class', QNetTwin)
        super().__init__(net_dim, state_dim, action_dim, gpu_id, args)

    def get_obj_critic_raw(self, buffer, batch_size):
        """
        Calculate the loss of the network and predict Q values with **uniform sampling**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and Q values.
        """
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s)).max(dim=1, keepdim=True)[0]
            q_label = reward + mask * next_q

        q1, q2 = [qs.gather(1, action.long()) for qs in self.act.get_q1_q2(state)]
        obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)
        return obj_critic, q1

    def get_obj_critic_per(self, buffer, batch_size):
        """
        Calculate the loss of the network and predict Q values with **Prioritized Experience Replay (PER)**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and Q values.
        """
        with torch.no_grad():
            reward, mask, action, state, next_s, is_weights = buffer.sample_batch(batch_size)
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s)).max(dim=1, keepdim=True)[0]
            q_label = reward + mask * next_q

        q1, q2 = [qs.gather(1, action.long()) for qs in self.act.get_q1_q2(state)]
        td_error = self.criterion(q1, q_label) + self.criterion(q2, q_label)
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, q1


class AgentD3QN(AgentDoubleDQN):  # D3QN: DuelingDoubleDQN
    def __init__(self, net_dim, state_dim, action_dim, gpu_id=0, args=None):
        self.act_class = getattr(self, 'act_class', QNetTwinDuel)
        super().__init__(net_dim, state_dim, action_dim, gpu_id, args)


'''off-policy'''


class AgentDDPG(AgentBase):
    def __init__(self, net_dim, state_dim, action_dim, gpu_id=0, args=None):
        self.if_off_policy = True
        self.act_class = getattr(self, 'act_class', ActorSAC)
        self.cri_class = getattr(self, 'cri_class', CriticTwin)
        super().__init__(net_dim, state_dim, action_dim, gpu_id, args)
        self.act.explore_noise = getattr(args, 'explore_noise', 0.1)  # set for `get_action()`

    def update_net(self, buffer):
        buffer.update_now_len()
        obj_critic = obj_actor = None
        for _ in range(int(1 + buffer.now_len * self.repeat_times / self.batch_size)):
            obj_critic, state = self.get_obj_critic(buffer, self.batch_size)
            self.optimizer_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

            action_pg = self.act(state)  # policy gradient
            obj_actor = -self.cri(state, action_pg).mean()
            self.optimizer_update(self.act_optimizer, obj_actor)
            self.soft_update(self.act_target, self.act, self.soft_update_tau)
        return obj_critic.item(), -obj_actor.item()


class AgentTD3(AgentDDPG):
    def __init__(self, net_dim, state_dim, action_dim, gpu_id=0, args=None):
        self.act_class = getattr(self, 'act_class', Actor)
        self.cri_class = getattr(self, 'cri_class', CriticTwin)
        super().__init__(net_dim, state_dim, action_dim, gpu_id, args)
        self.policy_noise = getattr(args, 'policy_noise', 0.15)  # standard deviation of policy noise
        self.update_freq = getattr(args, 'update_freq', 2)  # delay update frequency

    def update_net(self, buffer) -> tuple:
        buffer.update_now_len()
        obj_critic = obj_actor = None
        for update_c in range(int(1 + buffer.now_len * self.repeat_times / self.batch_size)):
            obj_critic, state = self.get_obj_critic(buffer, self.batch_size)
            self.optimizer_update(self.cri_optimizer, obj_critic)

            action_pg = self.act(state)  # policy gradient
            obj_actor = -self.cri_target(state, action_pg).mean()  # use cri_target instead of cri for stable training
            self.optimizer_update(self.act_optimizer, obj_actor)
            if update_c % self.update_freq == 0:  # delay update
                self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
                self.soft_update(self.act_target, self.act, self.soft_update_tau)
        return obj_critic.item() / 2, -obj_actor.item()

    def get_obj_critic_raw(self, buffer, batch_size):
        """
        Calculate the loss of networks with **uniform sampling**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and states.
        """
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_a = self.act_target.get_action_noise(next_s, self.policy_noise)  # policy noise
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))  # twin critics
            q_label = reward + mask * next_q
        q1, q2 = self.cri.get_q1_q2(state, action)
        obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)  # twin critics
        return obj_critic, state

    def get_obj_critic_per(self, buffer, batch_size):
        """
        Calculate the loss of the network with **Prioritized Experience Replay (PER)**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and states.
        """
        with torch.no_grad():
            reward, mask, action, state, next_s, is_weights = buffer.sample_batch(batch_size)
            next_a = self.act_target.get_action_noise(next_s, self.policy_noise)  # policy noise
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))  # twin critics
            q_label = reward + mask * next_q

        q1, q2 = self.cri.get_q1_q2(state, action)
        td_error = self.criterion(q1, q_label) + self.criterion(q2, q_label)
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, state


class AgentSAC(AgentBase):
    def __init__(self, net_dim, state_dim, action_dim, gpu_id=0, args=None):
        self.if_off_policy = True
        self.act_class = getattr(self, 'act_class', ActorSAC)
        self.cri_class = getattr(self, 'cri_class', CriticTwin)
        super().__init__(net_dim, state_dim, action_dim, gpu_id, args)

        self.alpha_log = torch.tensor((-np.log(action_dim) * np.e,), dtype=torch.float32,
                                      requires_grad=True, device=self.device)  # trainable parameter
        self.alpha_optim = torch.optim.Adam((self.alpha_log,), lr=args.learning_rate)
        self.target_entropy = np.log(action_dim)

    def update_net(self, buffer):
        buffer.update_now_len()

        obj_critic = obj_actor = None
        for _ in range(int(1 + buffer.now_len * self.repeat_times / self.batch_size)):
            '''objective of critic (loss function of critic)'''
            obj_critic, state = self.get_obj_critic(buffer, self.batch_size)
            self.optimizer_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

            '''objective of alpha (temperature parameter automatic adjustment)'''
            a_noise_pg, log_prob = self.act.get_action_logprob(state)  # policy gradient
            obj_alpha = (self.alpha_log * (log_prob - self.target_entropy).detach()).mean()
            self.optimizer_update(self.alpha_optim, obj_alpha)

            '''objective of actor'''
            alpha = self.alpha_log.exp().detach()
            with torch.no_grad():
                self.alpha_log[:] = self.alpha_log.clamp(-20, 2)

            q_value_pg = self.cri(state, a_noise_pg)
            obj_actor = -(q_value_pg + log_prob * alpha).mean()
            self.optimizer_update(self.act_optimizer, obj_actor)
            # self.soft_update(self.act_target, self.act, self.soft_update_tau) # SAC don't use act_target network

        return obj_critic.item(), -obj_actor.item(), self.alpha_log.exp().detach().item()

    def get_obj_critic_raw(self, buffer, batch_size):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)

            next_a, next_log_prob = self.act_target.get_action_logprob(next_s)  # stochastic policy
            next_q = self.cri_target.get_q_min(next_s, next_a)

            alpha = self.alpha_log.exp().detach()
            q_label = reward + mask * (next_q + next_log_prob * alpha)
        q1, q2 = self.cri.get_q1_q2(state, action)
        obj_critic = (self.criterion(q1, q_label) + self.criterion(q2, q_label)) / 2
        return obj_critic, state

    def get_obj_critic_per(self, buffer, batch_size):
        with torch.no_grad():
            reward, mask, action, state, next_s, is_weights = buffer.sample_batch(batch_size)

            next_a, next_log_prob = self.act_target.get_action_logprob(next_s)  # stochastic policy
            next_q = self.cri_target.get_q_min(next_s, next_a)

            alpha = self.alpha_log.exp().detach()
            q_label = reward + mask * (next_q + next_log_prob * alpha)
        q1, q2 = self.cri.get_q1_q2(state, action)
        td_error = (self.criterion(q1, q_label) + self.criterion(q2, q_label)) / 2.
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, state


class AgentModSAC(AgentSAC):  # Modified SAC using reliable_lambda and TTUR (Two Time-scale Update Rule)
    def __init__(self, net_dim, state_dim, action_dim, gpu_id=0, args=None):
        self.act_class = getattr(self, 'act_class', ActorFixSAC)
        self.cri_class = getattr(self, 'cri_class', CriticTwin)
        super().__init__(net_dim, state_dim, action_dim, gpu_id, args)
        self.obj_c = (-np.log(0.5)) ** 0.5  # for reliable_lambda

        self.lambda_a_log_std = getattr(args, 'lambda_a_log_std', 2 ** -4)

    def update_net(self, buffer):
        buffer.update_now_len()

        with torch.no_grad():  # H term
            # buf_state = buffer.sample_batch_r_m_a_s()[3]
            if buffer.prev_idx <= buffer.next_idx:
                buf_state = buffer.buf_state[buffer.prev_idx:buffer.next_idx]
            else:
                buf_state = torch.vstack((buffer.buf_state[buffer.prev_idx:],
                                          buffer.buf_state[:buffer.next_idx],))
            buffer.prev_idx = buffer.next_idx

            avg_a_log_std = self.act.get_a_log_std(buf_state).mean(dim=0, keepdim=True)
            avg_a_log_std = avg_a_log_std * torch.ones((self.batch_size, 1), device=self.device)
            del buf_state

        alpha = self.alpha_log.exp().detach()
        update_a = 0
        obj_actor = torch.zeros(1)
        for update_c in range(1, int(2 + buffer.now_len * self.repeat_times / self.batch_size)):
            '''objective of critic (loss function of critic)'''
            obj_critic, state = self.get_obj_critic(buffer, self.batch_size)
            self.optimizer_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
            self.obj_c = 0.995 * self.obj_c + 0.005 * obj_critic.item()  # for reliable_lambda

            a_noise_pg, logprob = self.act.get_action_logprob(state)  # policy gradient
            '''objective of alpha (temperature parameter automatic adjustment)'''
            obj_alpha = (self.alpha_log * (logprob - self.target_entropy).detach()).mean()
            self.optimizer_update(self.alpha_optim, obj_alpha)
            with torch.no_grad():
                self.alpha_log[:] = self.alpha_log.clamp(-16, 2)
            alpha = self.alpha_log.exp().detach()

            '''objective of actor using reliable_lambda and TTUR (Two Time-scales Update Rule)'''
            reliable_lambda = np.exp(-self.obj_c ** 2)  # for reliable_lambda
            if_update_a = update_a / update_c < 1 / (2 - reliable_lambda)
            if if_update_a:  # auto TTUR
                update_a += 1

                obj_a_std = self.criterion(self.act.get_a_log_std(state), avg_a_log_std) * self.lambda_a_log_std

                q_value_pg = self.cri(state, a_noise_pg)
                obj_actor = -(q_value_pg + logprob * alpha).mean() + obj_a_std

                self.optimizer_update(self.act_optimizer, obj_actor)
                self.soft_update(self.act_target, self.act, self.soft_update_tau)
        return self.obj_c, -obj_actor.item(), alpha.item()


class AgentREDqSAC(AgentSAC):  # Modified SAC using reliable_lambda and TTUR (Two Time-scale Update Rule)
    def __init__(self, net_dim, state_dim, action_dim, gpu_id=0, args=None):
        self.act_class = getattr(self, 'act_class', ActorFixSAC)
        self.cri_class = getattr(self, 'cri_class', CriticREDq)
        super().__init__(net_dim, state_dim, action_dim, gpu_id, args)
        self.obj_c = (-np.log(0.5)) ** 0.5  # for reliable_lambda

    def get_obj_critic_raw(self, buffer, batch_size):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)

            next_a, next_log_prob = self.act_target.get_action_logprob(next_s)  # stochastic policy
            next_q = self.cri_target.get_q_min(next_s, next_a)

            alpha = self.alpha_log.exp().detach()
            q_label = reward + mask * (next_q + next_log_prob * alpha)
        qs = self.cri.get_q_values(state, action)
        obj_critic = self.criterion(qs, q_label * torch.ones_like(qs))
        return obj_critic, state

    def get_obj_critic_per(self, buffer, batch_size):
        with torch.no_grad():
            reward, mask, action, state, next_s, is_weights = buffer.sample_batch(batch_size)

            next_a, next_log_prob = self.act_target.get_action_logprob(next_s)  # stochastic policy
            next_q = self.cri_target.get_q_min(next_s, next_a)

            alpha = self.alpha_log.exp().detach()
            q_label = reward + mask * (next_q + next_log_prob * alpha)
        qs = self.cri.get_q_values(state, action)
        td_error = self.criterion(qs, q_label * torch.ones_like(qs)).mean(dim=1)
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, state


'''on-policy'''


class AgentPPO(AgentBase):
    def __init__(self, net_dim: int, state_dim: int, action_dim: int, gpu_id=0, args=None):
        self.if_off_policy = False
        self.act_class = getattr(self, 'act_class', ActorPPO)
        self.cri_class = getattr(self, 'cri_class', CriticPPO)
        self.if_cri_target = getattr(args, 'if_cri_target', False)
        AgentBase.__init__(self, net_dim, state_dim, action_dim, gpu_id, args)

        self.ratio_clip = getattr(args, 'ratio_clip', 0.25)  # could be 0.00 ~ 0.50 `ratio.clamp(1 - clip, 1 + clip)`
        self.lambda_entropy = getattr(args, 'lambda_entropy', 0.02)  # could be 0.00~0.10
        self.lambda_gae_adv = getattr(args, 'lambda_entropy', 0.98)  # could be 0.95~0.99, GAE (ICLR.2016.)

        if getattr(args, 'if_use_gae', False):  # GAE (Generalized Advantage Estimation) for sparse reward
            self.get_reward_sum = self.get_reward_sum_gae
        else:
            self.get_reward_sum = self.get_reward_sum_raw

    def explore_one_env(self, env, target_step) -> list:
        traj_list = list()
        last_done = [0, ]
        state = self.states[0]

        step_i = 0
        done = False
        get_action = self.act.get_action
        get_a_to_e = self.act.get_a_to_e
        while step_i < target_step or not done:
            ten_s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            ten_a, ten_n = [ten.cpu() for ten in get_action(ten_s.to(self.device))]  # different
            next_s, reward, done, _ = env.step(get_a_to_e(ten_a)[0].numpy())

            traj_list.append((ten_s, reward, done, ten_a, ten_n))  # different

            step_i += 1
            state = env.reset() if done else next_s
        self.states[0] = state
        last_done[0] = step_i
        return self.convert_trajectory(traj_list, last_done)  # traj_list

    def explore_vec_env(self, env, target_step) -> list:
        traj_list = list()
        last_done = torch.zeros(self.env_num, dtype=torch.int, device=self.device)
        ten_s = self.states

        step_i = 0
        ten_dones = torch.zeros(self.env_num, dtype=torch.int, device=self.device)
        get_action = self.act.get_action
        get_a_to_e = self.act.get_a_to_e
        while step_i < target_step or not any(ten_dones):
            ten_a, ten_n = get_action(ten_s)  # different
            ten_s_next, ten_rewards, ten_dones, _ = env.step(get_a_to_e(ten_a))

            traj_list.append((ten_s.clone(), ten_rewards.clone(), ten_dones.clone(), ten_a, ten_n))  # different

            step_i += 1
            last_done[torch.where(ten_dones)[0]] = step_i  # behind `step_i+=1`
            ten_s = ten_s_next

        self.states = ten_s
        return self.convert_trajectory(traj_list, last_done)  # traj_list

    def update_net(self, buffer):
        with torch.no_grad():
            buf_state, buf_reward, buf_mask, buf_action, buf_noise = [ten.to(self.device) for ten in buffer]
            buf_len = buf_state.shape[0]

            '''get buf_r_sum, buf_logprob'''
            bs = 2 ** 10  # set a smaller 'BatchSize' when out of GPU memory.
            buf_value = [self.cri_target(buf_state[i:i + bs]) for i in range(0, buf_len, bs)]
            buf_value = torch.cat(buf_value, dim=0)
            buf_logprob = self.act.get_old_logprob(buf_action, buf_noise)

            buf_r_sum, buf_adv_v = self.get_reward_sum(buf_len, buf_reward, buf_mask, buf_value)  # detach()
            buf_adv_v = (buf_adv_v - buf_adv_v.mean()) / (buf_adv_v.std() + 1e-5)
            # buf_adv_v: buffer data of adv_v value
            del buf_noise

        '''update network'''
        obj_critic = None
        obj_actor = None
        assert buf_len >= self.batch_size
        for _ in range(int(1 + buf_len * self.repeat_times / self.batch_size)):
            indices = torch.randint(buf_len, size=(self.batch_size,), requires_grad=False, device=self.device)

            state = buf_state[indices]
            r_sum = buf_r_sum[indices]
            adv_v = buf_adv_v[indices]
            action = buf_action[indices]
            logprob = buf_logprob[indices]

            '''PPO: Surrogate objective of Trust Region'''
            new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)  # it is obj_actor
            ratio = (new_logprob - logprob.detach()).exp()
            surrogate1 = adv_v * ratio
            surrogate2 = adv_v * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy
            self.optimizer_update(self.act_optimizer, obj_actor)

            value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, r_sum)
            self.optimizer_update(self.cri_optimizer, obj_critic)
            if self.if_cri_target:
                self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

        a_std_log = getattr(self.act, 'a_std_log', torch.zeros(1)).mean()
        return obj_critic.item(), -obj_actor.item(), a_std_log.item()  # logging_tuple

    def get_reward_sum_raw(self, buf_len, buf_reward, buf_mask, buf_value) -> (torch.Tensor, torch.Tensor):
        """
        Calculate the **reward-to-go** and **advantage estimation**.

        :param buf_len: the length of the ``ReplayBuffer``.
        :param buf_reward: a list of rewards for the state-action pairs.
        :param buf_mask: a list of masks computed by the product of done signal and discount factor.
        :param buf_value: a list of state values estimated by the ``Critic`` network.
        :return: the reward-to-go and advantage estimation.
        """
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # reward sum

        pre_r_sum = 0
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
        buf_adv_v = buf_r_sum - buf_value[:, 0]
        return buf_r_sum, buf_adv_v

    def get_reward_sum_gae(self, buf_len, ten_reward, ten_mask, ten_value) -> (torch.Tensor, torch.Tensor):
        """
        Calculate the **reward-to-go** and **advantage estimation** using GAE.

        :param buf_len: the length of the ``ReplayBuffer``.
        :param ten_reward: a list of rewards for the state-action pairs.
        :param ten_mask: a list of masks computed by the product of done signal and discount factor.
        :param ten_value: a list of state values estimated by the ``Critic`` network.
        :return: the reward-to-go and advantage estimation.
        """
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # old policy value
        buf_adv_v = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # advantage value

        pre_r_sum = 0
        pre_adv_v = 0  # advantage value of previous step
        for i in range(buf_len - 1, -1, -1):  # Notice: mask = (1-done) * gamma
            buf_r_sum[i] = ten_reward[i] + ten_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]

            buf_adv_v[i] = ten_reward[i] + ten_mask[i] * pre_adv_v - ten_value[i]
            pre_adv_v = ten_value[i] + buf_adv_v[i] * self.lambda_gae_adv
            # ten_mask[i] * pre_adv_v == (1-done) * gamma * pre_adv_v
        return buf_r_sum, buf_adv_v



class AgentDiscretePPO(AgentPPO):
    def __init__(self, net_dim: int, state_dim: int, action_dim: int, gpu_id=0, args=None):
        self.act_class = getattr(self, 'act_class', ActorDiscretePPO)
        self.cri_class = getattr(self, 'cri_class', CriticPPO)
        super().__init__(net_dim, state_dim, action_dim, gpu_id, args)
