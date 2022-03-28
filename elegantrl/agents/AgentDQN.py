import torch
from elegantrl.agents.AgentBase import AgentBase
from elegantrl.agents.net import QNet, QNetDuel


class AgentDQN(AgentBase):
    """
    Bases: ``AgentBase``

    Deep Q-Network algorithm. “Human-Level Control Through Deep Reinforcement Learning”. Mnih V. et al.. 2015.

    :param net_dim[int]: the dimension of networks (the width of neural networks)
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    :param learning_rate[float]: learning rate of optimizer
    :param if_per_or_gae[bool]: PER (off-policy) or GAE (on-policy) for sparse reward
    :param env_num[int]: the env number of VectorEnv. env_num == 1 means don't use VectorEnv
    :param agent_id[int]: if the visible_gpu is '1,9,3,4', agent_id=1 means (1,9,4,3)[agent_id] == 9
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
        :param reward_scale: a reward scalar to clip the reward.
        :param gamma: the discount factor.
        :return: a list of trajectories [traj, ...] where each trajectory is a list of transitions [(state, other), ...].
        """
        traj_list = list()
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
        :param reward_scale: a reward scalar to clip the reward.
        :param gamma: the discount factor.
        :return: a list of trajectories [traj, ...] where each trajectory is a list of transitions [(state, other), ...].
        """
        traj_list = list()
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
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :param repeat_times: the re-using times of each trajectory.
        :param soft_update_tau: the soft update parameter.
        :return: a tuple of the log information.
        """
        buffer.update_now_len()
        obj_critic = q_value = None
        for _ in range(int(1 + buffer.now_len * self.repeat_times / self.batch_size)):
            obj_critic, q_value = self.get_obj_critic(buffer, self.batch_size)
            self.optimizer_update(self.cri_optimizer, obj_critic)
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


class AgentDuelingDQN(AgentDQN):
    """
    Bases: ``AgentDQN``

    Dueling network.

    """

    def __init__(self, net_dim, state_dim, action_dim, gpu_id=0, args=None):
        self.act_class = getattr(self, "act_class", QNetDuel)
        super().__init__(net_dim, state_dim, action_dim, gpu_id, args)
