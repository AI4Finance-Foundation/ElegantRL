import torch

from torch import Tensor
from elegantrl.agents.AgentBase import AgentBase
from elegantrl.agents.net import QNet
from elegantrl.train.replay_buffer import ReplayBuffer
from elegantrl.train.config import Arguments

'''[ElegantRL.2022.05.05](github.com/AI4Fiance-Foundation/ElegantRL)'''


class AgentDQN(AgentBase):  # [ElegantRL.2022.04.18]
    """
    Deep Q-Network algorithm. “Human-Level Control Through Deep Reinforcement Learning”. Mnih V. et al.. 2015.

    :param net_dim: the dimension of networks (the width of neural networks)
    :param state_dim: the dimension of state (the number of state vector)
    :param action_dim: the dimension of action (the number of discrete action)
    :param gpu_id: the gpu_id of the training device. Use CPU when cuda is not available.
    :param args: the arguments for agent training. `args = Arguments()`
    """

    def __init__(self, net_dim: int, state_dim: int, action_dim: int, gpu_id: int = 0, args: Arguments = None):
        self.act_class = getattr(self, "act_class", QNet)
        self.cri_class = None  # means `self.cri = self.act`
        args.if_act_target = getattr(args, 'if_act_target', True)
        args.if_cri_target = getattr(args, 'if_cri_target', True)
        AgentBase.__init__(self, net_dim, state_dim, action_dim, gpu_id, args)

        self.act.explore_rate = getattr(args, "explore_rate", 0.25)
        # Using ϵ-greedy to select uniformly random actions for exploration with `explore_rate` probability.

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
        get_action = self.act.get_action
        while i < target_step or not done:
            tensor_state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            tensor_action = get_action(tensor_state.to(self.device)).detach().cpu()
            next_state, reward, done, _ = env.step(tensor_action[0, 0].numpy())

            traj_list.append((tensor_state, reward, done, tensor_action))

            i += 1
            state = env.reset() if done else next_state

        self.states[0] = state
        last_done[0] = i
        return self.convert_trajectory(traj_list, last_done)

    def explore_vec_env(self, env, target_step: int) -> list:
        """
        Collect trajectories through the actor-environment interaction for a **vectorized** environment instance.

        :param env: the DRL environment instance.
        :param target_step: the total step for the interaction.
        :return: a list of trajectories [traj, ...] where each trajectory is a list of transitions [(state, other), ...].
        """
        traj_list = []
        last_done = torch.zeros(self.env_num, dtype=torch.int, device=self.device)
        states = self.states

        i = 0
        dones = torch.zeros(self.env_num, dtype=torch.int, device=self.device)
        while i < target_step or not any(dones):
            ten_action = self.act.get_action(states).detach()
            next_states, rewards, dones, _ = env.step(ten_action)  # different

            traj_list.append((states.clone(), rewards.clone(), dones.clone(), ten_action))  # different

            i += 1
            last_done[torch.where(dones)[0]] = i  # behind `i+=1`
            states = next_states

        self.states = states
        return self.convert_trajectory(traj_list, last_done)  # traj_list

    def update_net(self, buffer: ReplayBuffer) -> tuple:
        """
        Update the neural networks by sampling batch data from ``ReplayBuffer``.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :return: a tuple of the log information.
        """
        obj_critic = q_value = torch.zeros(1)
        update_times = int(buffer.cur_capacity / self.batch_size * self.repeat_times)
        for i in range(update_times):
            obj_critic, q_value = self.get_obj_critic(buffer, self.batch_size)
            self.optimizer_update(self.cri_optimizer, obj_critic)
            if self.if_cri_target:
                self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
        return obj_critic.item(), q_value.mean().item()

    def get_obj_critic_raw(self, buffer: ReplayBuffer, batch_size: int) -> (Tensor, Tensor):
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

    def get_obj_critic_per(self, buffer: ReplayBuffer, batch_size: int) -> (Tensor, Tensor):
        """
        Calculate the loss of the network and predict Q values with **Prioritized Experience Replay (PER)**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and Q values.
        """
        with torch.no_grad():
            reward, mask, action, state, next_s, is_weights = buffer.sample_batch(batch_size)
            next_q = self.cri_target(next_s).max(dim=1, keepdim=True)[0]
            q_label = reward + mask * next_q

        q_value = self.cri(state).gather(1, action.long())
        td_error = self.criterion(q_value, q_label)  # or td_error = (q_value - q_label).abs()
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, q_value
