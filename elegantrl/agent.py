import os
import numpy.random as rd
from copy import deepcopy
from elegantrl.net import *


class AgentBase:
    def __init__(self, net_dim: int, state_dim: int, action_dim: int,
                 act_class=None, cri_class=None, gpu_id=0, args=None):
        self.gamma = args.gamma
        self.reward_scale = args.reward_scale
        self.if_act_target = args.if_act_target
        self.if_cri_target = args.if_cri_target
        self.env_num = getattr(args, 'env_num', 1)
        self.explore_noise = getattr(args, 'explore_noise', 0.1)
        self.if_use_old_traj = getattr(args, 'if_use_old_traj', False)

        self.states = None
        self.action_dim = action_dim
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        self.traj_list = [[list() for _ in range(4 if args.if_off_policy else 5)]
                          for _ in range(self.env_num)]  # set for `self.explore_vec_env()`

        self.act = act_class(net_dim, state_dim, action_dim).to(self.device)
        self.cri = cri_class(net_dim, state_dim, action_dim).to(self.device) if cri_class else self.act
        self.act_target = deepcopy(self.act) if self.if_act_target else self.act
        self.cri_target = deepcopy(self.cri) if self.if_cri_target else self.cri

        self.act_optim = torch.optim.Adam(self.act.parameters(), args.learning_rate)
        self.cri_optim = torch.optim.Adam(self.cri.parameters(), args.learning_rate) if cri_class else self.act_optim

        '''function'''
        self.criterion = torch.nn.SmoothL1Loss()
        self.explore_env = None  # one env or vec env
        self.get_obj_critic = None  # for off-policy (TD-error)
        self.get_reward_sum = None  # for on-policy (V-trace)

    @staticmethod
    def optim_update(optimizer, objective):
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

        name_obj_list = [('actor', self.act), ('act_target', self.act_target), ('act_optim', self.act_optim),
                         ('critic', self.cri), ('cri_target', self.cri_target), ('cri_optim', self.cri_optim), ]
        name_obj_list = [(name, obj) for name, obj in name_obj_list if obj is not None]
        if if_save:
            for name, obj in name_obj_list:
                save_path = f"{cwd}/{name}.pth"
                torch.save(obj.state_dict(), save_path)
        else:
            for name, obj in name_obj_list:
                save_path = f"{cwd}/{name}.pth"
                load_torch_file(obj, save_path) if os.path.isfile(save_path) else None

    def convert_trajectory(self, buf_items, last_done):  # [ElegantRL.2022.01.01]
        # assert len(buf_items) == step_i
        # assert len(buf_items[0]) in {4, 5}
        # assert len(buf_items[0][0]) == self.env_num
        buf_items = list(map(list, zip(*buf_items)))  # s_r_d_a_n: state, reward, done, action, noise
        # assert len(buf_items) == {4, 5}
        # assert len(buf_items[0]) == step
        # assert len(buf_items[0][0]) == self.env_num

        '''stack items'''
        buf_items[0] = torch.stack(buf_items[0])
        buf_items[3:] = [torch.stack(item) for item in buf_items[3:]]
        if self.env_num > 1:
            buf_items[1] = (torch.stack(buf_items[1]) * self.reward_scale).unsqueeze(2)
            buf_items[2] = ((1 - torch.stack(buf_items[2])) * self.gamma).unsqueeze(2)
        else:
            buf_items[1] = (torch.tensor(buf_items[1], dtype=torch.float32) * self.reward_scale).unsqueeze(1).unsqueeze(
                2)
            buf_items[2] = ((1 - torch.tensor(buf_items[2], dtype=torch.float32)) * self.gamma).unsqueeze(1).unsqueeze(
                2)
        # assert all([buf_item.shape[:2] == (step, self.env_num) for buf_item in buf_items])

        '''splice items'''
        out_item = list()
        for j in range(len(buf_items)):
            cur_item = list()
            buf_item = buf_items.pop(0)

            for env_i in range(self.env_num):
                last_step = last_done[env_i]

                pre_item = self.traj_list[env_i][j]
                if len(pre_item):
                    cur_item.append(pre_item)

                cur_item.append(buf_item[:last_step, env_i])

                if self.if_use_old_traj:
                    self.traj_list[env_i][j] = buf_item[last_step:, env_i]

            out_item.append(torch.vstack(cur_item))

        # on-policy:  out_item = [states, rewards, dones, actions, noises]
        # off-policy: out_item = [states, rewards, dones, actions]
        return out_item


'''DQN'''


class AgentDQN(AgentBase):
    def __init__(self, net_dim, state_dim, action_dim, gpu_id=0, args=None):
        act_class = QNet
        cri_class = None  # = act_class
        AgentBase.__init__(self, net_dim, state_dim, action_dim, act_class, cri_class, gpu_id, args)

        self.explore_rate = 0.25  # the probability of choosing action randomly in epsilon-greedy
        self.if_use_cri_target = True
        self.ClassCri = QNet

        if self.env_num == 1:
            self.explore_env = self.explore_one_env
        else:
            self.explore_env = self.explore_vec_env

    def explore_one_env(self, env, target_step) -> list:
        traj_list = list()
        last_done = 0
        state = self.states[0]

        '''get traj_list and last_done'''
        step_i = 0
        done = False
        while step_i < target_step or not done:
            ten_s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            # ten_a = self.act.get_action(ten_s.to(self.device), self.explore_noise).detach().cpu()  # different
            if rd.rand() > self.explore_rate:  # epsilon-greedy
                ten_a = self.act(ten_s).detach().argmax(dim=0, keepdim=True)
            else:
                ten_a = torch.randint(self.action_dim, size=(1, 1))  # choosing action randomly
            next_s, reward, done, _ = env.step(ten_a[0].numpy())  # different

            traj_list.append((ten_s, reward, done, ten_a))  # different

            step_i += 1
            if done:
                state = env.reset()
                last_done = step_i  # behind `step_i += 1`
            else:
                state = next_s

        last_done = (last_done,)
        self.states[0] = state

        out_items = self.convert_trajectory(traj_list, last_done)
        return [out_items, ]  # plan to be elegant

    def explore_vec_env(self, env, target_step) -> list:
        traj_list = list()
        last_done = torch.zeros(self.env_num, dtype=torch.int, device=self.device)

        '''get traj_list and last_done'''
        ten_s = self.states
        step_i = 0
        ten_dones = torch.zeros(self.env_num, dtype=torch.int, device=self.device)
        while step_i < target_step or not any(ten_dones):
            # ten_a = self.act.get_action(ten_s, self.explore_noise).detach()  # different
            if rd.rand() > self.explore_rate:  # epsilon-greedy
                ten_a = self.act(ten_s).detach().argmax(dim=1)
            else:
                ten_a = torch.randint(self.action_dim, size=(self.env_num, 1))  # choosing action randomly
            ten_s_next, ten_rewards, ten_dones, _ = env.step(ten_a)  # different

            traj_list.append((ten_s.clone(), ten_rewards.clone(), ten_dones.clone(), ten_a))  # different

            ten_s = ten_s_next

            step_i += 1
            last_done[torch.where(ten_dones)[0]] = step_i  # behind `step_i+=1`
        # assert len(traj_list) == step_i
        # assert len(traj_list[0]) == 4  # different
        # assert len(traj_list[0][0]) == self.env_num
        self.states = ten_s

        out_items = self.convert_trajectory(traj_list, last_done)
        return [out_items, ]  # plan to be elegant

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> tuple:
        buffer.update_now_len()
        obj_critic = q_value = None
        for _ in range(int(buffer.now_len / batch_size * repeat_times)):
            obj_critic, q_value = self.get_obj_critic(buffer, batch_size)
            self.optim_update(self.cri_optim, obj_critic)
            self.soft_update(self.cri_target, self.cri, soft_update_tau)
        return obj_critic.item(), q_value.mean().item()

    def get_obj_critic(self, buffer, batch_size) -> (torch.Tensor, torch.Tensor):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_q = self.cri_target(next_s).max(dim=1, keepdim=True)[0]
            q_label = reward + mask * next_q

        q_value = self.cri(state).gather(1, action.long())
        obj_critic = self.criterion(q_value, q_label)
        return obj_critic, q_value


class AgentDoubleDQN(AgentDQN):
    def __init__(self, net_dim, state_dim, action_dim, gpu_id=0, args=None):
        act_class = QNetTwin
        cri_class = None  # = act_class
        AgentBase.__init__(self, net_dim, state_dim, action_dim, act_class, cri_class, gpu_id, args)
        self.softMax = torch.nn.Softmax(dim=1)  # todo plan to

    def select_action(self, state) -> int:  # for discrete action space
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
        actions = self.act(states)
        if rd.rand() < self.explore_rate:  # epsilon-greedy
            a_prob = self.softMax(actions)[0].detach().cpu().numpy()
            a_int = rd.choice(self.action_dim, p=a_prob)  # choose action according to Q value
        else:
            action = actions[0]
            a_int = action.argmax(dim=0).detach().cpu().numpy()
        return a_int

    def get_obj_critic(self, buffer, batch_size) -> (torch.Tensor, torch.Tensor):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s)).max(dim=1, keepdim=True)[0]
            q_label = reward + mask * next_q

        q1, q2 = [qs.gather(1, action.long()) for qs in self.act.get_q1_q2(state)]
        obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)
        return obj_critic, q1


'''off-policy'''


class AgentDDPG(AgentBase):
    def __init__(self, net_dim, state_dim, action_dim, gpu_id=0, args=None):
        act_class = Actor
        cri_class = Critic
        self.if_act_target = True
        self.if_cri_target = True
        AgentBase.__init__(self, net_dim, state_dim, action_dim, act_class, cri_class, gpu_id, args)
        self.explore_noise = 0.1  # explore noise of action

        if self.env_num == 1:
            self.explore_env = self.explore_one_env
        else:
            self.explore_env = self.explore_vec_env

        if getattr(args, 'if_use_per', False):
            self.criterion = torch.nn.MSELoss(reduction='none')
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.criterion = torch.nn.MSELoss(reduction='mean')
            self.get_obj_critic = self.get_obj_critic_raw

    def explore_one_env(self, env, target_step) -> list:
        traj_list = list()
        last_done = 0
        state = self.states[0]

        '''get traj_list and last_done'''
        step_i = 0
        done = False
        while step_i < target_step or not done:
            ten_s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            ten_a = self.act.get_action(ten_s.to(self.device), self.explore_noise).detach().cpu()  # different
            next_s, reward, done, _ = env.step(ten_a[0].numpy())  # different

            traj_list.append((ten_s, reward, done, ten_a))  # different

            step_i += 1
            if done:
                state = env.reset()
                last_done = step_i  # behind `step_i += 1`
            else:
                state = next_s
        last_done = (last_done,)
        # assert len(traj_list) == step_i
        # assert len(traj_list[0]) == 4  # different
        # assert len(traj_list[0][0]) == self.env_num
        self.states[0] = state

        out_items = self.convert_trajectory(traj_list, last_done)
        return [out_items, ]  # plan to be elegant

    def explore_vec_env(self, env, target_step) -> list:
        traj_list = list()
        last_done = torch.zeros(self.env_num, dtype=torch.int, device=self.device)

        '''get traj_list and last_done'''
        ten_s = self.states
        step_i = 0
        ten_dones = torch.zeros(self.env_num, dtype=torch.int, device=self.device)
        while step_i < target_step or not any(ten_dones):
            ten_a = self.act.get_action(ten_s, self.explore_noise).detach()  # different
            ten_s_next, ten_rewards, ten_dones, _ = env.step(ten_a)  # different

            traj_list.append((ten_s.clone(), ten_rewards.clone(), ten_dones.clone(), ten_a))  # different

            ten_s = ten_s_next

            step_i += 1
            last_done[torch.where(ten_dones)[0]] = step_i  # behind `step_i+=1`
        # assert len(traj_list) == step_i
        # assert len(traj_list[0]) == 4  # different
        # assert len(traj_list[0][0]) == self.env_num
        self.states = ten_s

        out_items = self.convert_trajectory(traj_list, last_done)
        return [out_items, ]  # plan to be elegant

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> (float, float):
        buffer.update_now_len()
        obj_critic = obj_actor = None
        for _ in range(int(buffer.now_len / batch_size * repeat_times)):
            obj_critic, state = self.get_obj_critic(buffer, batch_size)
            self.optim_update(self.cri_optim, obj_critic)
            self.soft_update(self.cri_target, self.cri, soft_update_tau)

            action_pg = self.act(state)  # policy gradient
            obj_actor = -self.cri(state, action_pg).mean()
            self.optim_update(self.act_optim, obj_actor)
            self.soft_update(self.act_target, self.act, soft_update_tau)
        return obj_actor.item(), obj_critic.item()

    def get_obj_critic_raw(self, buffer, batch_size):
        """
        Calculate the loss of networks with **uniform sampling**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and states.
        """
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_q = self.cri_target(next_s, self.act_target(next_s))
            q_label = reward + mask * next_q
        q_value = self.cri(state, action)
        obj_critic = self.criterion(q_value, q_label)
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
            next_q = self.cri_target(next_s, self.act_target(next_s))
            q_label = reward + mask * next_q

        q_value = self.cri(state, action)
        td_error = self.criterion(q_value, q_label)  # or td_error = (q_value - q_label).abs()
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, state


class AgentTD3(AgentDDPG):
    def __init__(self, net_dim, state_dim, action_dim, gpu_id=0, args=None):
        act_class = Actor
        cri_class = CriticTwin
        self.if_act_target = True
        self.if_cri_target = True
        AgentBase.__init__(self, net_dim, state_dim, action_dim, act_class, cri_class, gpu_id, args)
        self.explore_noise = 0.1  # standard deviation of exploration noise
        self.policy_noise = 0.2  # standard deviation of policy noise
        self.update_freq = 2  # delay update frequency

        if self.env_num == 1:
            self.explore_env = self.explore_one_env
        else:
            self.explore_env = self.explore_vec_env

        if getattr(args, 'if_use_per', False):
            self.criterion = torch.nn.MSELoss(reduction='none')
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.criterion = torch.nn.MSELoss(reduction='mean')
            self.get_obj_critic = self.get_obj_critic_raw

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> tuple:
        buffer.update_now_len()
        obj_critic = obj_actor = None
        for update_c in range(int(buffer.now_len / batch_size * repeat_times)):
            obj_critic, state = self.get_obj_critic(buffer, batch_size)
            self.optim_update(self.cri_optim, obj_critic)

            action_pg = self.act(state)  # policy gradient
            obj_actor = -self.cri_target(state, action_pg).mean()  # use cri_target instead of cri for stable training
            self.optim_update(self.act_optim, obj_actor)
            if update_c % self.update_freq == 0:  # delay update
                self.soft_update(self.cri_target, self.cri, soft_update_tau)
                self.soft_update(self.act_target, self.act, soft_update_tau)
        return obj_critic.item() / 2, obj_actor.item()

    def get_obj_critic_raw(self, buffer, batch_size):
        """
        Calculate the loss of networks with **uniform sampling**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and states.
        """
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_a = self.act_target.get_action(next_s, self.policy_noise)  # policy noise
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
            next_a = self.act_target.get_action(next_s, self.policy_noise)  # policy noise
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))  # twin critics
            q_label = reward + mask * next_q

        q1, q2 = self.cri.get_q1_q2(state, action)
        td_error = self.criterion(q1, q_label) + self.criterion(q2, q_label)
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, state


class AgentSAC(AgentBase):
    def __init__(self, net_dim, state_dim, action_dim, gpu_id=0, args=None):
        act_class = ActorSAC
        cri_class = CriticTwin
        self.if_act_target = False
        self.if_cri_target = True
        AgentBase.__init__(self, net_dim, state_dim, action_dim, act_class, cri_class, gpu_id, args)

        self.alpha_log = torch.tensor((-np.log(action_dim) * np.e,), dtype=torch.float32,
                                      requires_grad=True, device=self.device)  # trainable parameter
        self.alpha_optim = torch.optim.Adam((self.alpha_log,), lr=args.learning_rate)
        self.target_entropy = np.log(action_dim)

        if self.env_num == 1:
            self.explore_env = self.explore_one_env
        else:
            self.explore_env = self.explore_vec_env

        if getattr(args, 'if_use_per', False):
            self.criterion = torch.nn.MSELoss(reduction='none')
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.criterion = torch.nn.MSELoss(reduction='mean')
            self.get_obj_critic = self.get_obj_critic_raw

    def explore_one_env(self, env, target_step) -> list:
        traj_list = list()
        last_done = 0
        state = self.states[0]

        '''get traj_list and last_done'''
        step_i = 0
        done = False
        while step_i < target_step or not done:
            ten_s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            ten_a = self.act.get_action(ten_s.to(self.device), self.explore_noise).detach().cpu()  # different
            next_s, reward, done, _ = env.step(ten_a[0].numpy())  # different

            traj_list.append((ten_s, reward, done, ten_a))  # different

            step_i += 1
            if done:
                state = env.reset()
                last_done = step_i  # behind `step_i += 1`
            else:
                state = next_s
        last_done = (last_done,)
        # assert len(traj_list) == step_i
        # assert len(traj_list[0]) == 4  # different
        # assert len(traj_list[0][0]) == self.env_num
        self.states[0] = state

        out_items = self.convert_trajectory(traj_list, last_done)
        return [out_items, ]  # plan to be elegant

    def explore_vec_env(self, env, target_step) -> list:
        traj_list = list()
        last_done = torch.zeros(self.env_num, dtype=torch.int, device=self.device)

        '''get traj_list and last_done'''
        ten_s = self.states
        step_i = 0
        ten_dones = torch.zeros(self.env_num, dtype=torch.int, device=self.device)
        while step_i < target_step or not any(ten_dones):
            ten_a = self.act.get_action(ten_s, self.explore_noise).detach()  # different
            ten_s_next, ten_rewards, ten_dones, _ = env.step(ten_a)  # different

            traj_list.append((ten_s.clone(), ten_rewards.clone(), ten_dones.clone(), ten_a))  # different

            ten_s = ten_s_next

            step_i += 1
            last_done[torch.where(ten_dones)[0]] = step_i  # behind `step_i+=1`
        # assert len(traj_list) == step_i
        # assert len(traj_list[0]) == 4  # different
        # assert len(traj_list[0][0]) == self.env_num
        self.states = ten_s

        out_items = self.convert_trajectory(traj_list, last_done)
        return [out_items, ]  # plan to be elegant

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        buffer.update_now_len()

        alpha = self.alpha_log.exp().detach()
        obj_critic = obj_actor = None
        for _ in range(int(buffer.now_len * repeat_times / batch_size)):
            '''objective of critic (loss function of critic)'''
            with torch.no_grad():
                reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
                next_a, next_log_prob = self.act_target.get_action_logprob(next_s)
                next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))
                q_label = reward + mask * (next_q + next_log_prob * alpha)
            q1, q2 = self.cri.get_q1_q2(state, action)
            obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)
            self.optim_update(self.cri_optim, obj_critic)
            self.soft_update(self.cri_target, self.cri, soft_update_tau)

            '''objective of alpha (temperature parameter automatic adjustment)'''
            action_pg, log_prob = self.act.get_action_logprob(state)  # policy gradient
            obj_alpha = (self.alpha_log * (log_prob - self.target_entropy).detach()).mean()
            self.optim_update(self.alpha_optim, obj_alpha)

            '''objective of actor'''
            alpha = self.alpha_log.exp().detach()
            with torch.no_grad():
                self.alpha_log[:] = self.alpha_log.clamp(-20, 2)
            obj_actor = -(torch.min(*self.cri_target.get_q1_q2(state, action_pg)) + log_prob * alpha).mean()
            self.optim_update(self.act_optim, obj_actor)

            self.soft_update(self.act_target, self.act, soft_update_tau)

        return obj_critic.item(), obj_actor.item(), alpha.item()

    def get_obj_critic_raw(self, buffer, batch_size, alpha):
        """
        Calculate the loss of networks with **uniform sampling**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :param alpha: the trade-off coefficient of entropy regularization.
        :return: the loss of the network and states.
        """
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)

            next_a, next_log_prob = self.act_target.get_action_logprob(next_s)  # stochastic policy
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))  # twin critics

            q_label = reward + mask * (next_q + next_log_prob * alpha)
        q1, q2 = self.cri.get_q1_q2(state, action)
        obj_critic = (self.criterion(q1, q_label) + self.criterion(q2, q_label)) / 2
        return obj_critic, state

    def get_obj_critic_per(self, buffer, batch_size, alpha):
        """
        Calculate the loss of the network with **Prioritized Experience Replay (PER)**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :param alpha: the trade-off coefficient of entropy regularization.
        :return: the loss of the network and states.
        """
        with torch.no_grad():
            reward, mask, action, state, next_s, is_weights = buffer.sample_batch(batch_size)

            next_a, next_log_prob = self.act_target.get_action_logprob(next_s)  # stochastic policy
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))  # twin critics

            q_label = reward + mask * (next_q + next_log_prob * alpha)
        q1, q2 = self.cri.get_q1_q2(state, action)
        td_error = (self.criterion(q1, q_label) + self.criterion(q2, q_label)) / 2.
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, state


class AgentModSAC(AgentSAC):  # Modified SAC using reliable_lambda and TTUR (Two Time-scale Update Rule)
    def __init__(self, net_dim, state_dim, action_dim, gpu_id=0, args=None):
        act_class = ActorFixSAC
        cri_class = CriticTwin
        self.if_act_target = True
        self.if_cri_target = True
        AgentBase.__init__(self, net_dim, state_dim, action_dim, act_class, cri_class, gpu_id, args)

        self.alpha_log = torch.tensor((-np.log(action_dim) * np.e,), dtype=torch.float32,
                                      requires_grad=True, device=self.device)  # trainable parameter
        self.alpha_optim = torch.optim.Adam((self.alpha_log,), lr=args.learning_rate)
        self.target_entropy = np.log(action_dim)
        self.obj_c = (-np.log(0.5)) ** 0.5  # for reliable_lambda

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        buffer.update_now_len()

        alpha = self.alpha_log.exp().detach()
        update_a = 0
        obj_actor = None
        for update_c in range(1, int(buffer.now_len * repeat_times / batch_size)):
            '''objective of critic (loss function of critic)'''
            with torch.no_grad():
                reward, mask, action, state, next_s = buffer.sample_batch(batch_size)

                next_a, next_log_prob = self.act_target.get_action_logprob(next_s)
                next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))
                q_label = reward + mask * (next_q + next_log_prob * alpha)
            q1, q2 = self.cri.get_q1_q2(state, action)
            obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)
            self.obj_c = 0.995 * self.obj_c + 0.0025 * obj_critic.item()  # for reliable_lambda
            self.optim_update(self.cri_optim, obj_critic)
            self.soft_update(self.cri_target, self.cri, soft_update_tau)

            a_noise_pg, log_prob = self.act.get_action_logprob(state)  # policy gradient
            '''objective of alpha (temperature parameter automatic adjustment)'''
            obj_alpha = (self.alpha_log * (log_prob - self.target_entropy).detach()).mean()
            self.optim_update(self.alpha_optim, obj_alpha)
            with torch.no_grad():
                self.alpha_log[:] = self.alpha_log.clamp(-16, 2)
            alpha = self.alpha_log.exp().detach()

            '''objective of actor using reliable_lambda and TTUR (Two Time-scales Update Rule)'''
            reliable_lambda = np.exp(-self.obj_c ** 2)  # for reliable_lambda
            if_update_a = update_a / update_c < 1 / (2 - reliable_lambda)
            if if_update_a:  # auto TTUR
                update_a += 1

                q_value_pg = torch.min(*self.cri.get_q1_q2(state, a_noise_pg))
                obj_actor = -(q_value_pg + log_prob * alpha).mean()
                self.optim_update(self.act_optim, obj_actor)
                self.soft_update(self.act_target, self.act, soft_update_tau)
        return self.obj_c, -obj_actor.item(), alpha.item()


'''on-policy'''


class AgentPPO(AgentBase):
    def __init__(self, net_dim: int, state_dim: int, action_dim: int, gpu_id=0, args=None):
        act_class = ActorPPO
        cri_class = CriticPPO
        args.if_act_target = False
        AgentBase.__init__(self, net_dim=net_dim, state_dim=state_dim, action_dim=action_dim,
                           act_class=act_class, cri_class=cri_class, gpu_id=gpu_id, args=args)

        self.if_off_policy = False
        self.ratio_clip = getattr(args, 'ratio_clip', 0.25)  # could be 0.00 ~ 0.50 `ratio.clamp(1 - clip, 1 + clip)`
        self.lambda_entropy = getattr(args, 'lambda_entropy', 0.02)  # could be 0.00~0.10
        self.lambda_gae_adv = getattr(args, 'lambda_entropy', 0.98)  # could be 0.95~0.99, GAE (ICLR.2016.)

        '''attribute'''
        if self.env_num == 1:
            self.explore_env = self.explore_one_env
        else:  # vector env
            self.explore_env = self.explore_vec_env

        if getattr(args, 'if_use_gae', False):  # GAE (Generalized Advantage Estimation) for sparse reward
            self.get_reward_sum = self.get_reward_sum_gae
        else:
            self.get_reward_sum = self.get_reward_sum_raw

    def explore_one_env(self, env, target_step) -> list:
        """
        Collect trajectories through the actor-environment interaction.

        :param env: the DRL environment instance.
        :param target_step: the total step_i for the interaction.
        :return: a list of trajectories [traj, ...] where `traj = [(state, other), ...]`.
        """
        buf_items = list()
        last_done = 0
        state = self.states[0]

        '''get buf_items and last_done'''
        step_i = 0
        done = False
        while step_i < target_step or not done:
            ten_s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            ten_a, ten_n = [ten.cpu() for ten in self.act.get_action(ten_s.to(self.device))]  # different
            next_s, reward, done, _ = env.step(ten_a[0].tanh().numpy())

            buf_items.append((ten_s, reward, done, ten_a, ten_n))  # different

            step_i += 1
            if done:
                state = env.reset()
                last_done = step_i  # behind `step_i += 1`
            else:
                state = next_s
        last_done = (last_done,)
        # assert len(buf_items) == step_i
        # assert len(buf_items[0]) == 5  # different
        # assert len(buf_items[0][0]) == self.env_num
        self.states[0] = state

        out_items = self.convert_trajectory(buf_items, last_done)
        return [out_items, ]  # traj_list

    def explore_vec_env(self, env, target_step) -> list:
        buf_items = list()
        last_done = torch.zeros(self.env_num, dtype=torch.int, device=self.device)

        '''get buf_items and last_done'''
        ten_s = self.states
        step_i = 0
        ten_dones = torch.zeros(self.env_num, dtype=torch.int, device=self.device)
        while step_i < target_step or not any(ten_dones):
            ten_a, ten_n = self.act.get_action(ten_s)  # different
            ten_s_next, ten_rewards, ten_dones, _ = env.step(ten_a.tanh())

            buf_items.append((ten_s.clone(), ten_rewards.clone(), ten_dones.clone(), ten_a, ten_n))  # different

            ten_s = ten_s_next

            step_i += 1
            last_done[torch.where(ten_dones)[0]] = step_i  # behind `step_i+=1`
        # assert len(buf_items) == step_i
        # assert len(buf_items[0]) in {4, 5}
        # assert len(buf_items[0][0]) == self.env_num
        self.states = ten_s

        out_items = self.convert_trajectory(buf_items, last_done)
        return [out_items, ]  # traj_list

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        """
        Update the neural networks by sampling batch data from `ReplayBuffer`.

        .. note::
            Using advantage normalization and entropy loss.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :param repeat_times: the re-using times of each trajectory.
        :param soft_update_tau: the soft update parameter.
        :return: a tuple of the log information.
        """
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
        # with torch.enable_grad():
        # torch.set_grad_enabled(True)
        obj_critic = None
        obj_actor = None
        assert buf_len >= batch_size
        update_times = int(buf_len / batch_size * repeat_times)
        for update_i in range(1, update_times + 1):
            indices = torch.randint(buf_len, size=(batch_size,), requires_grad=False, device=self.device)

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
            self.optim_update(self.act_optim, obj_actor)

            value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, r_sum)
            self.optim_update(self.cri_optim, obj_critic / (r_sum.std() + 1e-6))
            if self.if_cri_target:
                self.soft_update(self.cri_target, self.cri, soft_update_tau)
        # torch.set_grad_enabled(False)

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
        act_class = ActorDiscretePPO
        cri_class = CriticPPO
        args.if_act_target = False
        AgentBase.__init__(self, net_dim=net_dim, state_dim=state_dim, action_dim=action_dim,
                           act_class=act_class, cri_class=cri_class, gpu_id=gpu_id, args=args)

        self.if_off_policy = False
        self.ratio_clip = getattr(args, 'ratio_clip', 0.25)  # could be 0.00 ~ 0.50 `ratio.clamp(1 - clip, 1 + clip)`
        self.lambda_entropy = getattr(args, 'lambda_entropy', 0.02)  # could be 0.00~0.10
        self.lambda_gae_adv = getattr(args, 'lambda_entropy', 0.98)  # could be 0.95~0.99, GAE (ICLR.2016.)

        '''attribute'''
        if self.env_num == 1:
            self.explore_env = self.explore_one_env
        else:  # vector env
            self.explore_env = self.explore_vec_env

        if getattr(args, 'if_use_gae', False):  # GAE (Generalized Advantage Estimation) for sparse reward
            self.get_reward_sum = self.get_reward_sum_gae
        else:
            self.get_reward_sum = self.get_reward_sum_raw

    def explore_one_env(self, env, target_step) -> list:
        """
        Collect trajectories through the actor-environment interaction.

        :param env: the DRL environment instance.
        :param target_step: the total step_i for the interaction.
        :return: a list of trajectories [traj, ...] where `traj = [(state, other), ...]`.
        """
        buf_items = list()
        last_done = 0
        state = self.states[0]

        '''get buf_items and last_done'''
        step_i = 0
        done = False
        while step_i < target_step or not done:
            ten_s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            ten_a, ten_n = [ten.cpu() for ten in self.act.get_action(ten_s.to(self.device))]  # different
            next_s, reward, done, _ = env.step(ten_a[0].int().numpy())  # different

            buf_items.append((ten_s, reward, done, ten_a, ten_n))  # different

            step_i += 1
            if done:
                state = env.reset()
                last_done = step_i  # behind `step_i += 1`
            else:
                state = next_s
        last_done = (last_done,)
        # assert len(buf_items) == step_i
        # assert len(buf_items[0]) == 5  # different
        # assert len(buf_items[0][0]) == self.env_num
        self.states[0] = state

        out_items = self.convert_trajectory(buf_items, last_done)
        return [out_items, ]  # traj_list

    def explore_vec_env(self, env, target_step) -> list:
        buf_items = list()
        last_done = torch.zeros(self.env_num, dtype=torch.int, device=self.device)

        '''get buf_items and last_done'''
        ten_s = self.states
        step_i = 0
        ten_dones = torch.zeros(self.env_num, dtype=torch.int, device=self.device)
        while step_i < target_step or not any(ten_dones):
            ten_a, ten_n = self.act.get_action(ten_s)  # different
            ten_s_next, ten_rewards, ten_dones, _ = env.step(ten_a.int())  # different

            buf_items.append((ten_s.clone(), ten_rewards.clone(), ten_dones.clone(), ten_a, ten_n))  # different

            ten_s = ten_s_next

            step_i += 1
            last_done[torch.where(ten_dones)[0]] = step_i  # behind `step_i+=1`
        # assert len(buf_items) == step_i
        # assert len(buf_items[0]) in {4, 5}
        # assert len(buf_items[0][0]) == self.env_num
        self.states = ten_s

        out_items = self.convert_trajectory(buf_items, last_done)
        return [out_items, ]  # traj_list


'''replay buffer'''


class ReplayBuffer:
    def __init__(self, max_len, state_dim, action_dim, gpu_id=0):
        self.now_len = 0
        self.next_idx = 0
        self.if_full = False
        self.max_len = max_len
        self.data_type = torch.float32
        self.action_dim = action_dim
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        other_dim = 1 + 1 + self.action_dim
        self.buf_other = torch.empty((max_len, other_dim), dtype=torch.float32, device=self.device)

        if isinstance(state_dim, int):  # state is pixel
            self.buf_state = torch.empty((max_len, state_dim), dtype=torch.float32, device=self.device)
        elif isinstance(state_dim, tuple):
            self.buf_state = torch.empty((max_len, *state_dim), dtype=torch.uint8, device=self.device)
        else:
            raise ValueError('state_dim')

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

    def sample_batch(self, batch_size) -> tuple:
        indices = rd.randint(self.now_len - 1, size=batch_size)
        r_m_a = self.buf_other[indices]
        return (r_m_a[:, 0:1],
                r_m_a[:, 1:2],
                r_m_a[:, 2:],
                self.buf_state[indices],
                self.buf_state[indices + 1])

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
