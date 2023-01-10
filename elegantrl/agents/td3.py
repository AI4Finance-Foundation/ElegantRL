import torch
from typing import Tuple
from copy import deepcopy
from torch import Tensor

from elegantrl.train.config import Config
from elegantrl.train.replay_buffer import ReplayBuffer
from elegantrl.agents.base import AgentBase
from elegantrl.agents.net import Actor, CriticTwin

'''[ElegantRL.2022.12.12](github.com/AI4Fiance-Foundation/ElegantRL)'''


class AgentTD3(AgentBase):
    """Twin Delayed DDPG algorithm.
    Addressing Function Approximation Error in Actor-Critic Methods. 2018.
    """

    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.act_class = getattr(self, 'act_class', Actor)
        self.cri_class = getattr(self, 'cri_class', CriticTwin)
        super().__init__(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, gpu_id=gpu_id, args=args)
        self.act_target = deepcopy(self.act)
        self.cri_target = deepcopy(self.cri)

        self.explore_noise_std = getattr(args, 'explore_noise_std', 0.05)  # standard deviation of exploration noise
        self.policy_noise_std = getattr(args, 'policy_noise_std', 0.10)  # standard deviation of exploration noise
        self.update_freq = getattr(args, 'update_freq', 2)  # delay update frequency

        self.act.explore_noise_std = self.explore_noise_std  # assign explore_noise_std for agent.act.get_action(state)

    def update_net(self, buffer: ReplayBuffer) -> Tuple[float, ...]:
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

            if update_c % self.update_freq == 0:  # delay update
                action_pg = self.act(state)  # policy gradient
                obj_actor = self.cri_target(state, action_pg).mean()  # use cri_target is more stable than cri
                obj_actors += obj_actor.item()
                self.optimizer_update(self.act_optimizer, -obj_actor)
                self.soft_update(self.act_target, self.act, self.soft_update_tau)
        return obj_critics / update_times, obj_actors / update_times

    def get_obj_critic_raw(self, buffer: ReplayBuffer, batch_size: int) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            state, action, reward, undone, next_s = buffer.sample(batch_size)
            next_a = self.act_target.get_action_noise(next_s, self.policy_noise_std)  # policy noise
            next_q = self.cri_target.get_q_min(next_s, next_a)  # twin critics
            q_label = reward + undone * self.gamma * next_q

        q1, q2 = self.cri.get_q1_q2(state, action)
        obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)  # twin critics
        return obj_critic, state

    def get_obj_critic_per(self, buffer: ReplayBuffer, batch_size: int) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            state, action, reward, undone, next_s, is_weight = buffer.sample(batch_size)
            next_a = self.act_target.get_action_noise(next_s, self.policy_noise_std)  # policy noise
            next_q = self.cri_target.get_q_min(next_s, next_a)  # twin critics
            q_label = reward + undone * self.gamma * next_q

        q1, q2 = self.cri.get_q1_q2(state, action)
        td_error = self.criterion(q1, q_label) + self.criterion(q2, q_label)
        obj_critic = (td_error * is_weight).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, state
