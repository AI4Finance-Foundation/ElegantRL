import numpy as np
import numpy.random as rd
import torch
from torch import Tensor

from elegantrl.agents.AgentBase import AgentBase
from elegantrl.agents.net import Actor, Critic
from elegantrl.train.replay_buffer import ReplayBuffer
from elegantrl.train.config import Arguments

'''[ElegantRL.2022.05.05](github.com/AI4Fiance-Foundation/ElegantRL)'''


class AgentDDPG(AgentBase):
    """
    Bases: ``AgentBase``

    Deep Deterministic Policy Gradient algorithm.
    “Continuous control with deep reinforcement learning”. T. Lillicrap et al.. 2015.

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
        self.cri_class = getattr(self, 'cri_class', Critic)
        args.if_act_target = getattr(args, 'if_act_target', True)
        args.if_cri_target = getattr(args, 'if_cri_target', True)
        super().__init__(net_dim, state_dim, action_dim, gpu_id, args)

        self.act_update_gap = getattr(args, 'act_update_gap', 2)
        self.explore_noise_std = getattr(args, 'explore_noise_std', 0.1)  # explore noise of action
        self.act.explore_noise_std = self.explore_noise_std
        self.act_target.explore_noise_std = self.act.explore_noise_std
        # self.ou_noise = OrnsteinUhlenbeckNoise(size=action_dim, sigma=self.explore_noise_std)

    def update_net(self, buffer: ReplayBuffer) -> tuple:
        obj_critic = torch.zeros(1)
        obj_actor = torch.zeros(1)
        update_times = int(buffer.cur_capacity * self.repeat_times / self.batch_size)
        for i in range(update_times):
            obj_critic, state = self.get_obj_critic(buffer, self.batch_size)
            self.optimizer_update(self.cri_optimizer, obj_critic)
            if self.if_cri_target:
                self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

            if i % self.act_update_gap == 0:
                action = self.act(state)  # policy gradient
                obj_actor = -self.cri(state, action).mean()
                self.optimizer_update(self.act_optimizer, obj_actor)
                if self.if_act_target:
                    self.soft_update(self.act_target, self.act, self.soft_update_tau)
        return obj_critic.item(), obj_actor.item()

    def get_obj_critic_raw(self, buffer: ReplayBuffer, batch_size: int) -> (Tensor, Tensor):
        """
        Calculate the loss of networks with **uniform sampling**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and states.
        """
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_action = self.act_target(next_s)
            next_q = self.cri_target(next_s, next_action)
            q_label = reward + mask * next_q
        q_value = self.cri(state, action)
        obj_critic = self.criterion(q_value, q_label)
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
            next_action = self.act_target(next_s)
            next_q = self.cri_target(next_s, next_action)
            q_label = reward + mask * next_q

        q_value = self.cri(state, action)
        td_error = self.criterion(q_value, q_label)  # or td_error = (q_value - q_label).abs()
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, state


class AgentDDPGHterm(AgentDDPG):
    def update_net(self, buffer: ReplayBuffer) -> tuple:
        if (buffer.next_p - buffer.prev_p) % buffer.max_capacity > 2 ** 11:
            with torch.no_grad():  # H term
                buf_state, buf_action, buf_reward, buf_mask = buffer.concatenate_buffer()
                buf_r_sum = self.get_r_sum_h_term(buf_reward, buf_mask)

                self.get_buf_h_term(buf_state, buf_action, buf_r_sum, buf_mask, buf_reward)  # todo H-term
                del buf_state, buf_action, buf_reward, buf_mask, buf_r_sum

        obj_critic = torch.zeros(1)
        obj_actor = torch.zeros(1)
        update_times = int(buffer.cur_capacity * self.repeat_times / self.batch_size)
        for i in range(update_times):
            obj_critic, state = self.get_obj_critic(buffer, self.batch_size)
            self.optimizer_update(self.cri_optimizer, obj_critic)
            if self.if_cri_target:
                self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

            action = self.act(state)  # policy gradient
            if i % self.act_update_gap == 0:
                obj_actor = -self.cri(state, action).mean() + self.get_obj_h_term()  # todo H-term
            else:
                obj_actor = -self.cri(state, action).mean()

            self.optimizer_update(self.act_optimizer, obj_actor)
            if self.if_act_target:
                self.soft_update(self.act_target, self.act, self.soft_update_tau)
        return obj_critic.item(), obj_actor.item()


class OrnsteinUhlenbeckNoise:
    def __init__(self, size: int, theta=0.15, sigma=0.3, ou_noise=0.0, dt=1e-2):
        """
        The noise of Ornstein-Uhlenbeck Process

        Source: https://github.com/slowbull/DDPG/blob/master/src/explorationnoise.py
        It makes Zero-mean Gaussian Noise more stable.
        It helps agent explore better in a inertial system.
        Don't abuse OU Process. OU process has too much hyper-parameters and over fine-tuning make no sense.

        :int size: the size of noise, noise.shape==(-1, action_dim)
        :float theta: related to the not independent of OU-noise
        :float sigma: related to action noise std
        :float ou_noise: initialize OU-noise
        :float dt: derivative
        """
        self.theta = theta
        self.sigma = sigma
        self.ou_noise = ou_noise
        self.dt = dt
        self.size = size

    def __call__(self) -> float:
        """
        output a OU-noise

        :return array ou_noise: a noise generated by Ornstein-Uhlenbeck Process
        """
        noise = self.sigma * np.sqrt(self.dt) * rd.normal(size=self.size)
        self.ou_noise -= self.theta * self.ou_noise * self.dt + noise
        return self.ou_noise
