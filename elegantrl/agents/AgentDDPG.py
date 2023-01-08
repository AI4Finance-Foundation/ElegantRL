import numpy as np
import numpy.random as rd
import torch
from copy import deepcopy
from torch import Tensor

from elegantrl.train.config import Config
from elegantrl.train.replay_buffer import ReplayBuffer
from elegantrl.agents.AgentBase import AgentBase
from elegantrl.agents.net import Actor, Critic

'''[ElegantRL.2022.12.12](github.com/AI4Fiance-Foundation/ElegantRL)'''


class AgentDDPG(AgentBase):
    """DDPG(Deep Deterministic Policy Gradient)
    “Continuous control with deep reinforcement learning”. T. Lillicrap et al.. 2015.”

    net_dims: the middle layer dimension of MLP (MultiLayer Perceptron)
    state_dim: the dimension of state (the number of state vector)
    action_dim: the dimension of action (or the number of discrete action)
    gpu_id: the gpu_id of the training device. Use CPU when cuda is not available.
    args: the arguments for agent training. `args = Config()`
    """

    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.act_class = getattr(self, 'act_class', Actor)
        self.cri_class = getattr(self, 'cri_class', Critic)
        super().__init__(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, gpu_id=gpu_id, args=args)
        self.act_target = deepcopy(self.act)
        self.cri_target = deepcopy(self.cri)

        self.explore_noise_std = getattr(args, 'explore_noise_std', 0.05)  # standard deviation of exploration noise

        self.act.explore_noise_std = self.explore_noise_std  # assign explore_noise_std for agent.act.get_action(state)

    def update_net(self, buffer: ReplayBuffer) -> tuple:
        with torch.no_grad():
            states, actions, rewards, undones = buffer.add_item
            self.update_avg_std_for_normalization(
                states=states.reshape((-1, self.state_dim)),
                returns=self.get_returns(rewards=rewards, undones=undones).reshape((-1,))
            )

        '''update network'''
        obj_critics = 0.0
        obj_actors = 0.0

        update_times = int(buffer.add_size * self.repeat_times)
        assert update_times >= 1
        for update_c in range(update_times):
            obj_critic, state = self.get_obj_critic(buffer, self.batch_size)
            obj_critics += obj_critic.item()
            self.optimizer_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

            action_pg = self.act(state)  # policy gradient
            obj_actor = self.cri_target(state, action_pg).mean()  # use cri_target is more stable than cri
            obj_actors += obj_actor.item()
            self.optimizer_update(self.act_optimizer, -obj_actor)
            self.soft_update(self.act_target, self.act, self.soft_update_tau)
        return obj_critics / update_times, obj_actors / update_times

    def get_obj_critic_raw(self, buffer: ReplayBuffer, batch_size: int) -> (Tensor, Tensor):
        with torch.no_grad():
            state, action, reward, undone, next_s = buffer.sample(batch_size)
            next_a = self.act_target(next_s)  # policy noise
            next_q = self.cri_target.get_q_min(next_s, next_a)  # twin critics
            q_label = reward + undone * self.gamma * next_q

        q_value = self.cri(state, action)
        obj_critic = self.criterion(q_value, q_label)
        return obj_critic, state

    def get_obj_critic_per(self, buffer: ReplayBuffer, batch_size: int) -> (Tensor, Tensor):
        with torch.no_grad():
            state, action, reward, undone, next_s, is_weight = buffer.sample(batch_size)
            next_a = self.act_target(next_s)  # policy noise
            next_q = self.cri_target.get_q_min(next_s, next_a)  # twin critics
            q_label = reward + undone * self.gamma * next_q

        q_value = self.cri(state, action)
        td_error = self.criterion(q_value, q_label)
        obj_critic = (td_error * is_weight).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, state


class OrnsteinUhlenbeckNoise:
    def __init__(self, size: int, theta=0.15, sigma=0.3, ou_noise=0.0, dt=1e-2):
        """
        The noise of Ornstein-Uhlenbeck Process

        Source: https://github.com/slowbull/DDPG/blob/master/src/explorationnoise.py
        It makes Zero-mean Gaussian Noise more stable.
        It helps agent explore better in a inertial system.
        Don't abuse OU Process. OU process has too much hyper-parameters and over fine-tuning make no sense.

        int size: the size of noise, noise.shape==(-1, action_dim)
        float theta: related to the not independent of OU-noise
        float sigma: related to action noise std
        float ou_noise: initialize OU-noise
        float dt: derivative
        """
        self.theta = theta
        self.sigma = sigma
        self.ou_noise = ou_noise
        self.dt = dt
        self.size = size

    def __call__(self) -> float:
        """
        output a OU-noise

        return array ou_noise: a noise generated by Ornstein-Uhlenbeck Process
        """
        noise = self.sigma * np.sqrt(self.dt) * rd.normal(size=self.size)
        self.ou_noise -= self.theta * self.ou_noise * self.dt + noise
        return self.ou_noise
