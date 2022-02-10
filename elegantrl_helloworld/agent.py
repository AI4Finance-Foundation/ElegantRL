import os
from copy import deepcopy
from typing import Tuple

from elegantrl_helloworld.net import *


class AgentBase:
    """
    Base class. Replace by different DRL algorithms.

    :param net_dim[int]: the dimension of networks (the width of neural networks)
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    :param gpu_id[int]: the gpu_id of the training device. Use CPU when cuda is not available
    """

    def __init__(
        self, net_dim: int, state_dim: int, action_dim: int, gpu_id=0, args=None
    ):
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.repeat_times = args.repeat_times
        self.reward_scale = args.reward_scale
        self.soft_update_tau = args.soft_update_tau
        self.env_num = getattr(args, "env_num", 1)
        self.lambda_gae_adv = getattr(
            args, "lambda_entropy", 0.98
        )  # could be 0.95~0.99, GAE (ICLR.2016.)
        self.if_use_old_traj = getattr(
            args, "if_use_old_traj", False
        )  # use old data to splice complete trajectory

        self.states = None
        self.device = torch.device(
            f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu"
        )
        self.traj_list = [
            [[] for _ in range(4 if args.if_off_policy else 5)]
            for _ in range(self.env_num)
        ]  # set for `self.explore_vec_env()`

        act_class = getattr(self, "act_class", None)
        cri_class = getattr(self, "cri_class", None)
        self.act = act_class(net_dim, state_dim, action_dim).to(self.device)
        self.cri = (
            cri_class(net_dim, state_dim, action_dim).to(self.device)
            if cri_class
            else self.act
        )
        self.act_target = deepcopy(self.act)
        self.cri_target = deepcopy(self.cri) if cri_class else self.cri

        self.act_optim = torch.optim.Adam(self.act.parameters(), args.learning_rate)
        self.cri_optim = (
            torch.optim.Adam(self.cri.parameters(), args.learning_rate)
            if cri_class
            else self.act_optim
        )

        """function"""
        self.criterion = torch.nn.SmoothL1Loss()

        if self.env_num == 1:
            self.explore_env = self.explore_one_env
        else:
            self.explore_env = self.explore_vec_env

        if getattr(
            args, "if_use_per", False
        ):  # PER (Prioritized Experience Replay) for sparse reward
            self.criterion = torch.nn.MSELoss(reduction="none")
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.criterion = torch.nn.MSELoss(reduction="mean")
            self.get_obj_critic = self.get_obj_critic_raw

        if getattr(
            args, "if_use_gae", False
        ):  # GAE (Generalized Advantage Estimation) for sparse reward
            self.get_reward_sum = self.get_reward_sum_gae
        else:
            self.get_reward_sum = self.get_reward_sum_raw

    def explore_one_env(self, env, target_step) -> list:
        """
        Collect trajectories through the actor-environment interaction for a **single** environment instance.

        :param env: the DRL environment instance.
        :param target_step: the total step for the interaction.
        :return: a list of trajectories [traj, ...] where each trajectory is a list of transitions [(state, other), ...].
        """
        traj_list = []
        last_done = [
            0,
        ]
        state = self.states[0]

        step_i = 0
        done = False
        while step_i < target_step or not done:
            ten_s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            ten_a = (
                self.act.get_action(ten_s.to(self.device)).detach().cpu()
            )  # different
            next_s, reward, done, _ = env.step(ten_a[0].numpy())  # different

            traj_list.append((ten_s, reward, done, ten_a))  # different

            step_i += 1
            state = env.reset() if done else next_s

        self.states[0] = state
        last_done[0] = step_i
        return self.convert_trajectory(traj_list, last_done)  # traj_list

    def explore_vec_env(self, env, target_step) -> list:
        """
        Collect trajectories through the actor-environment interaction for a **vectorized** environment instance.

        :param env: the DRL environment instance.
        :param target_step: the total step for the interaction.
        :return: a list of trajectories [traj, ...] where each trajectory is a list of transitions [(state, other), ...].
        """
        traj_list = []
        last_done = torch.zeros(self.env_num, dtype=torch.int, device=self.device)
        ten_s = self.states

        step_i = 0
        ten_dones = torch.zeros(self.env_num, dtype=torch.int, device=self.device)
        while step_i < target_step or not any(ten_dones):
            ten_a = self.act.get_action(ten_s).detach()  # different
            ten_s_next, ten_rewards, ten_dones, _ = env.step(ten_a)  # different

            traj_list.append(
                (ten_s.clone(), ten_rewards.clone(), ten_dones.clone(), ten_a)
            )  # different

            step_i += 1
            last_done[torch.where(ten_dones)[0]] = step_i  # behind `step_i+=1`
            ten_s = ten_s_next

        self.states = ten_s
        return self.convert_trajectory(traj_list, last_done)  # traj_list

    def convert_trajectory(self, buf_items, last_done):
        buf_items = list(
            map(list, zip(*buf_items))
        )  # state, reward, done, action, noise

        """stack items"""
        buf_items[0] = torch.stack(buf_items[0])
        buf_items[3:] = [torch.stack(item) for item in buf_items[3:]]
        if len(buf_items[3].shape) == 2:
            buf_items[3] = buf_items[3].unsqueeze(2)
        if self.env_num > 1:
            buf_items[1] = (torch.stack(buf_items[1]) * self.reward_scale).unsqueeze(2)
            buf_items[2] = ((1 - torch.stack(buf_items[2])) * self.gamma).unsqueeze(2)
        else:
            buf_items[1] = (
                (torch.tensor(buf_items[1], dtype=torch.float32) * self.reward_scale)
                .unsqueeze(1)
                .unsqueeze(2)
            )
            buf_items[2] = (
                ((1 - torch.tensor(buf_items[2], dtype=torch.float32)) * self.gamma)
                .unsqueeze(1)
                .unsqueeze(2)
            )

        """splice items"""
        for j in range(len(buf_items)):
            cur_item = []
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
            next_q = torch.min(*self.cri_target(next_s, next_a))
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
            reward, mask, action, state, next_s, is_weights = buffer.sample_batch(
                batch_size
            )
            next_a = self.act_target(next_s)
            next_q = torch.min(*self.cri_target(next_s, next_a))  # twin critics
            q_label = reward + mask * next_q

        q = self.cri(state, action)
        td_error = self.criterion(q, q_label)
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, state

    def get_reward_sum_raw(self, buf_len, buf_reward, buf_mask, buf_value):
        """
        Calculate the **reward-to-go** and **advantage estimation**.

        :param buf_len: the length of the ``ReplayBuffer``.
        :param buf_reward: a list of rewards for the state-action pairs.
        :param buf_mask: a list of masks computed by the product of done signal and discount factor.
        :param buf_value: a list of state values estimated by the ``Critic`` network.
        :return: the reward-to-go and advantage estimation.
        """
        buf_r_sum = torch.empty(
            buf_len, dtype=torch.float32, device=self.device
        )  # reward sum

        pre_r_sum = 0
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
        buf_adv_v = buf_r_sum - buf_value[:, 0]
        return buf_r_sum, buf_adv_v

    def get_reward_sum_gae(self, buf_len, ten_reward, ten_mask, ten_value):
        """
        Calculate the **reward-to-go** and **advantage estimation** using GAE.

        :param buf_len: the length of the ``ReplayBuffer``.
        :param ten_reward: a list of rewards for the state-action pairs.
        :param ten_mask: a list of masks computed by the product of done signal and discount factor.
        :param ten_value: a list of state values estimated by the ``Critic`` network.
        :return: the reward-to-go and advantage estimation.
        """
        buf_r_sum = torch.empty(
            buf_len, dtype=torch.float32, device=self.device
        )  # old policy value
        buf_adv_v = torch.empty(
            buf_len, dtype=torch.float32, device=self.device
        )  # advantage value

        pre_r_sum = 0
        pre_adv_v = 0  # advantage value of previous step
        for i in range(buf_len - 1, -1, -1):  # Notice: mask = (1-done) * gamma
            buf_r_sum[i] = ten_reward[i] + ten_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]

            buf_adv_v[i] = ten_reward[i] + ten_mask[i] * pre_adv_v - ten_value[i]
            pre_adv_v = ten_value[i] + buf_adv_v[i] * self.lambda_gae_adv
            # ten_mask[i] * pre_adv_v == (1-done) * gamma * pre_adv_v
        return buf_r_sum, buf_adv_v

    @staticmethod
    def optim_update(optimizer, objective):
        """
        Optimize networks through backpropagation.

        :param optimizer: the optimizer instance.
        :param objective: the loss.
        """
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    @staticmethod
    def soft_update(target_net, current_net, tau):
        """
        Soft update method for target networks.

        :param target_net: the target network.
        :param current_net: the current network.
        :param tau: the ratio for update.
        """
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

    def save_or_load_agent(self, cwd, if_save):
        """
        Save or load the models.

        :param cwd: the directory path.
        :param if_save: if true, save the model; else then load the model.
        """

        def load_torch_file(model_or_optim, _path):
            state_dict = torch.load(_path, map_location=lambda storage, loc: storage)
            model_or_optim.load_state_dict(state_dict)

        name_obj_list = [
            ("actor", self.act),
            ("act_target", self.act_target),
            ("act_optim", self.act_optim),
            ("critic", self.cri),
            ("cri_target", self.cri_target),
            ("cri_optim", self.cri_optim),
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


"""DQN"""


class AgentDQN(AgentBase):
    """
    Bases: ``AgentBase``

    Deep Q-Network algorithm. “Human-Level Control Through Deep Reinforcement Learning”. Mnih V. et al.. 2015.

    :param net_dim[int]: the dimension of networks (the width of neural networks)
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    :param gpu_id[int]: gpu id
    """

    def __init__(self, net_dim, state_dim, action_dim, gpu_id=0, args=None):
        self.if_off_policy = True
        self.act_class = getattr(self, "act_class", QNet)
        self.cri_class = None  # = act_class
        AgentBase.__init__(self, net_dim, state_dim, action_dim, gpu_id, args)
        self.act.explore_rate = getattr(args, "explore_rate", 0.125)
        # the probability of choosing action randomly in epsilon-greedy

    def explore_one_env(self, env, target_step) -> list:
        """
        Collect trajectories through the actor-environment interaction for a **single** environment instance.

        :param env: the DRL environment instance.
        :param target_step: the total step for the interaction.
        :return: a list of trajectories [traj, ...] where each trajectory is a list of transitions [(state, other), ...].
        """
        traj_list = []
        last_done = [
            0,
        ]
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
        """
        Collect trajectories through the actor-environment interaction for a **vectorized** environment instance.

        :param env: the DRL environment instance.
        :param target_step: the total step for the interaction.
        :return: a list of trajectories [traj, ...] where each trajectory is a list of transitions [(state, other), ...].
        """
        traj_list = []
        last_done = torch.zeros(self.env_num, dtype=torch.int, device=self.device)
        ten_s = self.states

        step_i = 0
        ten_dones = torch.zeros(self.env_num, dtype=torch.int, device=self.device)
        while step_i < target_step or not any(ten_dones):
            ten_a = self.act.get_action(ten_s).detach()
            ten_s_next, ten_rewards, ten_dones, _ = env.step(ten_a)  # different

            traj_list.append(
                (ten_s.clone(), ten_rewards.clone(), ten_dones.clone(), ten_a)
            )  # different

            step_i += 1
            last_done[torch.where(ten_dones)[0]] = step_i  # behind `step_i+=1`
            ten_s = ten_s_next

        self.states = ten_s
        return self.convert_trajectory(traj_list, last_done)  # traj_list

    def update_net(self, buffer) -> tuple:
        """
        Update the neural networks by sampling batch data from ``ReplayBuffer``.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :return: a tuple of the log information.
        """
        buffer.update_now_len()
        obj_critic = q_value = None
        for _ in range(int(1 + buffer.now_len * self.repeat_times / self.batch_size)):
            obj_critic, q_value = self.get_obj_critic(buffer, self.batch_size)
            self.optim_update(self.cri_optim, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
        return obj_critic.item(), q_value.mean().item()

    def get_obj_critic_raw(self, buffer, batch_size):
        """
        Calculate the loss of the network and predict Q values with **uniform sampling**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and Q values.
        """
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_q = self.cri_target(next_s).max(dim=1, keepdim=True)[0]
            q_label = reward + mask * next_q

        q_value = self.cri(state).gather(1, action.long())
        obj_critic = self.criterion(q_value, q_label)
        return obj_critic, q_value

    def get_obj_critic_per(self, buffer, batch_size):
        """
        Calculate the loss of the network and predict Q values with **Prioritized Experience Replay (PER)**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and Q values.
        """
        with torch.no_grad():
            reward, mask, action, state, next_s, is_weights = buffer.sample_batch(
                batch_size
            )
            next_q = self.cri_target(next_s).max(dim=1, keepdim=True)[0]
            q_label = reward + mask * next_q

        q_value = self.cri(state).gather(1, action.long())
        td_error = self.criterion(
            q_value, q_label
        )  # or td_error = (q_value - q_label).abs()
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, q_value


"""off-policy"""


class AgentSAC(AgentBase):
    """
    Bases: ``AgentBase``

    Soft Actor-Critic algorithm. “Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor”. Tuomas Haarnoja et al.. 2018.

    :param net_dim[int]: the dimension of networks (the width of neural networks)
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    :param gpu_id[int]: gpu id
    """

    def __init__(self, net_dim, state_dim, action_dim, gpu_id=0, args=None):
        self.if_off_policy = True
        self.act_class = getattr(self, "act_class", ActorSAC)
        self.cri_class = getattr(self, "cri_class", CriticTwin)
        super().__init__(net_dim, state_dim, action_dim, gpu_id, args)

        self.alpha_log = torch.tensor(
            (-np.log(action_dim) * np.e,),
            dtype=torch.float32,
            requires_grad=True,
            device=self.device,
        )  # trainable parameter
        self.alpha_optim = torch.optim.Adam((self.alpha_log,), lr=args.learning_rate)
        self.target_entropy = np.log(action_dim)

    def update_net(self, buffer):
        """
        Update the neural networks by sampling batch data from ``ReplayBuffer``.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :return: a tuple of the log information.
        """
        buffer.update_now_len()

        obj_critic = obj_actor = None
        for _ in range(int(1 + buffer.now_len * self.repeat_times / self.batch_size)):
            """objective of critic (loss function of critic)"""
            obj_critic, state = self.get_obj_critic(buffer, self.batch_size)
            self.optim_update(self.cri_optim, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

            """objective of alpha (temperature parameter automatic adjustment)"""
            action_pg, log_prob = self.act.get_action_logprob(state)  # policy gradient
            obj_alpha = (
                self.alpha_log * (log_prob - self.target_entropy).detach()
            ).mean()
            self.optim_update(self.alpha_optim, obj_alpha)

            """objective of actor"""
            alpha = self.alpha_log.exp().detach()
            with torch.no_grad():
                self.alpha_log[:] = self.alpha_log.clamp(-20, 2)
            obj_actor = -(
                torch.min(*self.cri_target.get_q1_q2(state, action_pg))
                + log_prob * alpha
            ).mean()
            self.optim_update(self.act_optim, obj_actor)
            # self.soft_update(self.act_target, self.act, self.soft_update_tau)

        return (
            obj_critic.item(),
            -obj_actor.item(),
            self.alpha_log.exp().detach().item(),
        )

    def get_obj_critic_raw(self, buffer, batch_size):
        """
        Calculate the loss of networks with **uniform sampling**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and states.
        """
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)

            next_a, next_log_prob = self.act_target.get_action_logprob(
                next_s
            )  # stochastic policy
            next_q = torch.min(
                *self.cri_target.get_q1_q2(next_s, next_a)
            )  # twin critics

            alpha = self.alpha_log.exp().detach()
            q_label = reward + mask * (next_q + next_log_prob * alpha)
        q1, q2 = self.cri.get_q1_q2(state, action)
        obj_critic = (self.criterion(q1, q_label) + self.criterion(q2, q_label)) / 2
        return obj_critic, state

    def get_obj_critic_per(self, buffer, batch_size):
        """
        Calculate the loss of the network with **Prioritized Experience Replay (PER)**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and states.
        """
        with torch.no_grad():
            reward, mask, action, state, next_s, is_weights = buffer.sample_batch(
                batch_size
            )

            next_a, next_log_prob = self.act_target.get_action_logprob(
                next_s
            )  # stochastic policy
            next_q = torch.min(
                *self.cri_target.get_q1_q2(next_s, next_a)
            )  # twin critics

            alpha = self.alpha_log.exp().detach()
            q_label = reward + mask * (next_q + next_log_prob * alpha)
        q1, q2 = self.cri.get_q1_q2(state, action)
        td_error = (self.criterion(q1, q_label) + self.criterion(q2, q_label)) / 2.0
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, state


"""on-policy"""


class AgentPPO(AgentBase):
    """
    Bases: ``AgentBase``

    PPO algorithm. “Proximal Policy Optimization Algorithms”. John Schulman. et al.. 2017.

    :param net_dim[int]: the dimension of networks (the width of neural networks)
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    :param gpu_id[int]: gpu id
    """

    def __init__(
        self, net_dim: int, state_dim: int, action_dim: int, gpu_id=0, args=None
    ):
        self.if_off_policy = False
        self.act_class = getattr(self, "act_class", ActorPPO)
        self.cri_class = getattr(self, "cri_class", CriticPPO)
        self.if_cri_target = getattr(args, "if_cri_target", False)
        AgentBase.__init__(self, net_dim, state_dim, action_dim, gpu_id, args)

        self.ratio_clip = getattr(
            args, "ratio_clip", 0.25
        )  # could be 0.00 ~ 0.50 `ratio.clamp(1 - clip, 1 + clip)`
        self.lambda_entropy = getattr(
            args, "lambda_entropy", 0.02
        )  # could be 0.00~0.10
        self.lambda_gae_adv = getattr(
            args, "lambda_entropy", 0.98
        )  # could be 0.95~0.99, GAE (ICLR.2016.)

    def explore_one_env(self, env, target_step) -> list:
        """
        Collect trajectories through the actor-environment interaction.

        :param env: the DRL environment instance.
        :param target_step: the total step for the interaction.
        :return: a list of trajectories [traj, ...] where `traj = [(state, other), ...]`.
        """
        traj_list = []
        last_done = [
            0,
        ]
        state = self.states[0]

        step_i = 0
        done = False
        get_action = self.act.get_action
        get_a_to_e = self.act.get_a_to_e
        while step_i < target_step or not done:
            ten_s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            ten_a, ten_n = [
                ten.cpu() for ten in get_action(ten_s.to(self.device))
            ]  # different
            next_s, reward, done, _ = env.step(get_a_to_e(ten_a)[0].numpy())

            traj_list.append((ten_s, reward, done, ten_a, ten_n))  # different

            step_i += 1
            state = env.reset() if done else next_s
        self.states[0] = state
        last_done[0] = step_i
        return self.convert_trajectory(traj_list, last_done)

    def explore_vec_env(self, env, target_step) -> list:
        """
        Collect trajectories through the actor-environment interaction for a **vectorized** environment instance.

        :param env: the DRL environment instance.
        :param target_step: the total step for the interaction.
        :return: a list of trajectories [traj, ...] where each trajectory is a list of transitions [(state, other), ...].
        """
        traj_list = []
        last_done = torch.zeros(self.env_num, dtype=torch.int, device=self.device)
        ten_s = self.states

        step_i = 0
        ten_dones = torch.zeros(self.env_num, dtype=torch.int, device=self.device)
        get_action = self.act.get_action
        get_a_to_e = self.act.get_a_to_e
        while step_i < target_step or not any(ten_dones):
            ten_a, ten_n = get_action(ten_s)  # different
            ten_s_next, ten_rewards, ten_dones, _ = env.step(get_a_to_e(ten_a))

            traj_list.append(
                (ten_s.clone(), ten_rewards.clone(), ten_dones.clone(), ten_a, ten_n)
            )  # different

            step_i += 1
            last_done[torch.where(ten_dones)[0]] = step_i  # behind `step_i+=1`
            ten_s = ten_s_next

        self.states = ten_s
        return self.convert_trajectory(traj_list, last_done)  # traj_list

    def update_net(self, buffer):
        """
        Update the neural networks by sampling batch data from `ReplayBuffer`.

        .. note::
            Using advantage normalization and entropy loss.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :return: a tuple of the log information.
        """
        with torch.no_grad():
            buf_state, buf_reward, buf_mask, buf_action, buf_noise = [
                ten.to(self.device) for ten in buffer
            ]
            buf_len = buf_state.shape[0]

            """get buf_r_sum, buf_logprob"""
            bs = 2**10  # set a smaller 'BatchSize' when out of GPU memory.
            buf_value = [
                self.cri_target(buf_state[i : i + bs]) for i in range(0, buf_len, bs)
            ]
            buf_value = torch.cat(buf_value, dim=0)
            buf_logprob = self.act.get_old_logprob(buf_action, buf_noise)

            buf_r_sum, buf_adv_v = self.get_reward_sum(
                buf_len, buf_reward, buf_mask, buf_value
            )  # detach()
            buf_adv_v = (buf_adv_v - buf_adv_v.mean()) / (buf_adv_v.std() + 1e-5)
            # buf_adv_v: buffer data of adv_v value
            del buf_noise

        """update network"""
        # with torch.enable_grad():  # todo
        # torch.set_grad_enabled(True)
        obj_critic = None
        obj_actor = None
        assert buf_len >= self.batch_size
        for _ in range(int(1 + buf_len * self.repeat_times / self.batch_size)):
            indices = torch.randint(
                buf_len,
                size=(self.batch_size,),
                requires_grad=False,
                device=self.device,
            )

            state = buf_state[indices]
            r_sum = buf_r_sum[indices]
            adv_v = buf_adv_v[indices]
            action = buf_action[indices]
            logprob = buf_logprob[indices]

            """PPO: Surrogate objective of Trust Region"""
            new_logprob, obj_entropy = self.act.get_logprob_entropy(
                state, action
            )  # it is obj_actor
            ratio = (new_logprob - logprob.detach()).exp()
            surrogate1 = adv_v * ratio
            surrogate2 = adv_v * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy
            self.optim_update(self.act_optim, obj_actor)

            value = self.cri(state).squeeze(
                1
            )  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, r_sum)
            self.optim_update(self.cri_optim, obj_critic / (r_sum.std() + 1e-6))
            if self.if_cri_target:
                self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
        # torch.set_grad_enabled(False) # todo

        a_std_log = getattr(self.act, "a_std_log", torch.zeros(1)).mean()
        return obj_critic.item(), -obj_actor.item(), a_std_log.item()  # logging_tuple

    def get_reward_sum_raw(
        self, buf_len, buf_reward, buf_mask, buf_value
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the **reward-to-go** and **advantage estimation**.

        :param buf_len: the length of the ``ReplayBuffer``.
        :param buf_reward: a list of rewards for the state-action pairs.
        :param buf_mask: a list of masks computed by the product of done signal and discount factor.
        :param buf_value: a list of state values estimated by the ``Critic`` network.
        :return: the reward-to-go and advantage estimation.
        """
        buf_r_sum = torch.empty(
            buf_len, dtype=torch.float32, device=self.device
        )  # reward sum

        pre_r_sum = 0
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
        buf_adv_v = buf_r_sum - buf_value[:, 0]
        return buf_r_sum, buf_adv_v

    def get_reward_sum_gae(
        self, buf_len, ten_reward, ten_mask, ten_value
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the **reward-to-go** and **advantage estimation** using GAE.

        :param buf_len: the length of the ``ReplayBuffer``.
        :param ten_reward: a list of rewards for the state-action pairs.
        :param ten_mask: a list of masks computed by the product of done signal and discount factor.
        :param ten_value: a list of state values estimated by the ``Critic`` network.
        :return: the reward-to-go and advantage estimation.
        """
        buf_r_sum = torch.empty(
            buf_len, dtype=torch.float32, device=self.device
        )  # old policy value
        buf_adv_v = torch.empty(
            buf_len, dtype=torch.float32, device=self.device
        )  # advantage value

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
    """
    Bases: ``AgentPPO``

    PPO algorithm for discrete action space.

    :param net_dim[int]: the dimension of networks (the width of neural networks)
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    :param gpu_id[int]: gpu id
    """

    def __init__(
        self, net_dim: int, state_dim: int, action_dim: int, gpu_id=0, args=None
    ):
        self.act_class = getattr(self, "act_class", ActorDiscretePPO)
        self.cri_class = getattr(self, "cri_class", CriticPPO)
        super().__init__(net_dim, state_dim, action_dim, gpu_id, args)


"""replay buffer"""


class ReplayBuffer:  # for off-policy
    """
    Experience Replay Buffer for off-policy algorithms.

    Save environment transition in a continuous RAM for high performance training.

    :param max_len[int]: the maximum capacity of ReplayBuffer. First In First Out
    :param state_dim[int]: the dimension of state
    :param action_dim[int]: the dimension of action (action_dim==1 for discrete action)
    :param gpu_id[int]: create buffer space on CPU RAM or GPU, `-1` denotes create on CPU
    """

    def __init__(self, max_len, state_dim, action_dim, gpu_id=0):
        self.now_len = 0
        self.next_idx = 0
        self.if_full = False
        self.max_len = max_len
        self.action_dim = action_dim
        self.device = torch.device(
            f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu"
        )

        other_dim = 1 + 1 + self.action_dim
        self.buf_other = torch.empty(
            (max_len, other_dim), dtype=torch.float32, device=self.device
        )

        buf_state_size = (
            (max_len, state_dim)
            if isinstance(state_dim, int)
            else (max_len, *state_dim)
        )
        self.buf_state = torch.empty(
            buf_state_size, dtype=torch.float32, device=self.device
        )

    def extend_buffer(self, state, other):  # CPU array to CPU array
        """
        Add a transition to buffer.

        :param state: the current state.
        :param other: a tuple contains reward, action, and done signal.
        """
        size = len(other)
        next_idx = self.next_idx + size

        if next_idx > self.max_len:
            self.buf_state[self.next_idx : self.max_len] = state[
                : self.max_len - self.next_idx
            ]
            self.buf_other[self.next_idx : self.max_len] = other[
                : self.max_len - self.next_idx
            ]
            self.if_full = True

            next_idx = next_idx - self.max_len
            self.buf_state[0:next_idx] = state[-next_idx:]
            self.buf_other[0:next_idx] = other[-next_idx:]
        else:
            self.buf_state[self.next_idx : next_idx] = state
            self.buf_other[self.next_idx : next_idx] = other
        self.next_idx = next_idx

    def update_buffer(self, traj_lists):
        """
        Add a list of transitions to buffer.

        :param traj_lists: a list of transitions.
        :return: the added steps and average reward.
        """
        steps = 0
        r_exp = 0.0
        for traj_list in traj_lists:
            self.extend_buffer(state=traj_list[0], other=torch.hstack(traj_list[1:]))

            steps += traj_list[1].shape[0]
            r_exp += traj_list[1].mean().item()
        return steps, r_exp / len(traj_lists)

    def sample_batch(self, batch_size) -> tuple:
        """
        Randomly sample a batch of data for training.

        :param batch_size: the number of transitions in a batch for Stochastic Gradient Descent.
        :return reward, mask, action, state, and next_state
        """
        indices = rd.randint(self.now_len - 1, size=batch_size)
        r_m_a = self.buf_other[indices]
        return (
            r_m_a[:, 0:1],
            r_m_a[:, 1:2],
            r_m_a[:, 2:],
            self.buf_state[indices],
            self.buf_state[indices + 1],
        )

    def update_now_len(self):
        """
        Update the size of the ``ReplayBuffer``.
        """
        self.now_len = self.max_len if self.if_full else self.next_idx

    def save_or_load_history(self, cwd, if_save, buffer_id=0):
        """
        ave or load the ``ReplayBuffer``.

        :param cwd: the directory path.
        :param if_save: if true, save the buffer; else then load the buffer.
        :param buffer_id: the buffer id.
        """
        save_path = f"{cwd}/replay_{buffer_id}.npz"

        if if_save:
            self.update_now_len()
            state_dim = self.buf_state.shape[1]
            other_dim = self.buf_other.shape[1]
            buf_state = np.empty(
                (self.max_len, state_dim), dtype=np.float16
            )  # sometimes np.uint8
            buf_other = np.empty((self.max_len, other_dim), dtype=np.float16)

            temp_len = self.max_len - self.now_len
            buf_state[0:temp_len] = (
                self.buf_state[self.now_len : self.max_len].detach().cpu().numpy()
            )
            buf_other[0:temp_len] = (
                self.buf_other[self.now_len : self.max_len].detach().cpu().numpy()
            )

            buf_state[temp_len:] = self.buf_state[: self.now_len].detach().cpu().numpy()
            buf_other[temp_len:] = self.buf_other[: self.now_len].detach().cpu().numpy()

            np.savez_compressed(save_path, buf_state=buf_state, buf_other=buf_other)
            print(f"| ReplayBuffer save in: {save_path}")
        elif os.path.isfile(save_path):
            buf_dict = np.load(save_path)
            buf_state = buf_dict["buf_state"]
            buf_other = buf_dict["buf_other"]

            buf_state = torch.as_tensor(
                buf_state, dtype=torch.float32, device=self.device
            )
            buf_other = torch.as_tensor(
                buf_other, dtype=torch.float32, device=self.device
            )
            self.extend_buffer(buf_state, buf_other)
            self.update_now_len()
            print(f"| ReplayBuffer load: {save_path}")


class ReplayBufferList(list):  # for on-policy
    """
    Experience Replay Buffer for on-policy algorithms (a list).
    """

    def __init__(self):
        list.__init__(self)

    def update_buffer(self, traj_lists):
        """
        Add a list of transitions to buffer.

        :param traj_lists: a list of transitions.
        :return: the added steps and average reward.
        """
        cur_items = list(map(list, zip(*traj_lists)))
        self[:] = [torch.cat(item, dim=0) for item in cur_items]

        steps = self[1].shape[0]
        r_exp = self[1].mean().item()
        return steps, r_exp
