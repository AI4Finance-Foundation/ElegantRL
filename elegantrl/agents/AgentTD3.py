import torch

from elegantrl.agents.AgentBase import AgentBase
from elegantrl.agents.net import Actor, CriticTwin
from elegantrl.train.replay_buffer import ReplayBuffer
from elegantrl.train.config import Arguments

'''[ElegantRL.2022.05.05](github.com/AI4Fiance-Foundation/ElegantRL)'''

Tensor = torch.Tensor


class AgentTD3(AgentBase):
    """
    Bases: ``AgentBase``

    Twin Delayed DDPG algorithm. “Addressing Function Approximation Error in Actor-Critic Methods”. Scott Fujimoto. et al.. 2015.

    :param net_dim[int]: the dimension of networks (the width of neural networks)
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    :param learning_rate[float]: learning rate of optimizer
    :param if_per_or_gae[bool]: PER (off-policy) or GAE (on-policy) for sparse reward
    :param env_num[int]: the env number of VectorEnv. env_num == 1 means don't use VectorEnv
    :param agent_id[int]: if the visible_gpu is '1,9,3,4', agent_id=1 means (1,9,4,3)[agent_id] == 9
    """

    def __init__(self, net_dim: int, state_dim: int, action_dim: int, gpu_id: int = 0, args: Arguments = None):
        self.if_off_policy = True
        self.act_class = getattr(self, 'act_class', Actor)
        self.cri_class = getattr(self, 'cri_class', CriticTwin)
        args.if_act_target = getattr(args, 'if_act_target', True)
        args.if_cri_target = getattr(args, 'if_cri_target', True)
        super().__init__(net_dim, state_dim, action_dim, gpu_id, args)

        self.explore_noise_std = 0.06  # standard deviation of exploration noise
        self.policy_noise_std = 0.12  # standard deviation of policy noise
        self.update_freq = 2  # delay update frequency

    def update_net(self, buffer: ReplayBuffer) -> tuple:
        obj_critic = torch.zeros(1)
        obj_actor = torch.zeros(1)
        update_times = int(1 + buffer.max_capacity * self.repeat_times / self.batch_size)
        for update_c in range(update_times):
            obj_critic, state = self.get_obj_critic(buffer, self.batch_size)
            self.optimizer_update(self.cri_optimizer, obj_critic)

            if update_c % self.update_freq == 0:  # delay update
                action_pg = self.act(state)  # policy gradient
                obj_actor = -self.cri_target(state, action_pg).mean()  # use cri_target is more stable than cri
                self.optimizer_update(self.act_optimizer, obj_actor)
                if self.if_cri_target:
                    self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
                if self.if_act_target:
                    self.soft_update(self.act_target, self.act, self.soft_update_tau)
        return obj_critic.item() / 2, obj_actor.item()

    def get_obj_critic_raw(self, buffer: ReplayBuffer, batch_size: int) -> (Tensor, Tensor):
        """
        Calculate the loss of networks with **uniform sampling**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and states.
        """
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_a = self.act_target.get_action_noise(next_s, self.policy_noise_std)  # policy noise
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))  # twin critics
            q_label = reward + mask * next_q

        q1, q2 = self.cri.get_q1_q2(state, action)
        obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)  # twin critics
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
            next_a = self.act_target.get_action_noise(next_s, self.policy_noise_std)  # policy noise
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s, next_a))  # twin critics
            q_label = reward + mask * next_q

        q1, q2 = self.cri.get_q1_q2(state, action)
        td_error = self.criterion(q1, q_label) + self.criterion(q2, q_label)
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, state
