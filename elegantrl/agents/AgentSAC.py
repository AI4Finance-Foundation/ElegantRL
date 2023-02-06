import math
import torch
from typing import Tuple
from copy import deepcopy
from torch import Tensor

from elegantrl.agents.AgentBase import AgentBase
from elegantrl.agents.net import ActorSAC, ActorFixSAC, CriticTwin
from elegantrl.train.config import Config
from elegantrl.train.replay_buffer import ReplayBuffer


class AgentSAC(AgentBase):
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.act_class = getattr(self, 'act_class', ActorSAC)
        self.cri_class = getattr(self, 'cri_class', CriticTwin)
        super().__init__(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, gpu_id=gpu_id, args=args)
        self.cri_target = deepcopy(self.cri)

        self.alpha_log = torch.tensor((-1,), dtype=torch.float32, requires_grad=True, device=self.device)  # trainable
        self.alpha_optimizer = torch.optim.AdamW((self.alpha_log,), lr=self.learning_rate * 4)
        self.target_entropy = getattr(args, 'target_entropy', action_dim)

    def update_net(self, buffer: ReplayBuffer) -> Tuple[float, ...]:
        with torch.no_grad():
            states, actions, rewards, undones = buffer.add_item
            self.update_avg_std_for_normalization(
                states=states.reshape((-1, self.state_dim)),
                returns=self.get_cumulative_rewards(rewards=rewards, undones=undones).reshape((-1,))
            )

        '''update network'''
        obj_critics = 0.0
        obj_actors = 0.0
        alphas = 0.0

        update_times = int(buffer.add_size * self.repeat_times)
        assert update_times >= 1
        for _ in range(update_times):
            '''objective of critic (loss function of critic)'''
            obj_critic, state = self.get_obj_critic(buffer, self.batch_size)
            obj_critics += obj_critic.item()
            self.optimizer_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

            '''objective of alpha (temperature parameter automatic adjustment)'''
            action_pg, log_prob = self.act.get_action_logprob(state)  # policy gradient
            obj_alpha = (self.alpha_log * (self.target_entropy - log_prob).detach()).mean()
            self.optimizer_update(self.alpha_optimizer, obj_alpha)

            '''objective of actor'''
            alpha = self.alpha_log.exp().detach()
            alphas += alpha.item()
            with torch.no_grad():
                self.alpha_log[:] = self.alpha_log.clamp(-16, 2)

            q_value_pg = self.cri_target(state, action_pg).mean()
            obj_actor = (q_value_pg - log_prob * alpha).mean()
            obj_actors += obj_actor.item()
            self.optimizer_update(self.act_optimizer, -obj_actor)

        return obj_critics / update_times, obj_actors / update_times, alphas / update_times

    def get_obj_critic_raw(self, buffer, batch_size: int) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            states, actions, rewards, undones, next_ss = buffer.sample(batch_size)  # next_ss: next states
            next_as, next_logprobs = self.act.get_action_logprob(next_ss)  # next actions
            next_qs = self.cri_target.get_q_min(next_ss, next_as)  # next q values

            alpha = self.alpha_log.exp().detach()
            q_labels = rewards + undones * self.gamma * (next_qs - next_logprobs * alpha)

        q1, q2 = self.cri.get_q1_q2(states, actions)
        obj_critic = self.criterion(q1, q_labels) + self.criterion(q2, q_labels)  # twin critics
        return obj_critic, states

    def get_obj_critic_per(self, buffer: ReplayBuffer, batch_size: int) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            states, actions, rewards, undones, next_ss, is_weights, is_indices = buffer.sample_for_per(batch_size)
            next_as, next_logprobs = self.act.get_action_logprob(next_ss)
            next_qs = self.cri_target.get_q_min(next_ss, next_as)

            alpha = self.alpha_log.exp().detach()
            q_labels = rewards + undones * self.gamma * (next_qs - next_logprobs * alpha)

        q1, q2 = self.cri.get_q1_q2(states, actions)
        td_errors = self.criterion(q1, q_labels) + self.criterion(q2, q_labels)
        obj_critic = (td_errors * is_weights).mean()

        buffer.td_error_update_for_per(is_indices.detach(), td_errors.detach())
        return obj_critic, states


class AgentModSAC(AgentSAC):  # Modified SAC using reliable_lambda and Two Time-scale Update Rule
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.act_class = getattr(self, "act_class", ActorFixSAC)
        super().__init__(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, gpu_id=gpu_id, args=args)
        self.obj_c = 1.0  # for reliable_lambda

    def update_net(self, buffer: ReplayBuffer) -> Tuple[float, ...]:
        with torch.no_grad():
            states, actions, rewards, undones = buffer.add_item
            self.update_avg_std_for_normalization(
                states=states.reshape((-1, self.state_dim)),
                returns=self.get_cumulative_rewards(rewards=rewards, undones=undones).reshape((-1,))
            )

        '''update network'''
        obj_critics = 0.0
        obj_actors = 0.0
        alphas = 0.0

        update_times = int(buffer.add_size * self.repeat_times)
        assert update_times >= 1
        update_a = 0
        for update_c in range(1, update_times + 1):
            '''objective of critic (loss function of critic)'''
            obj_critic, state = self.get_obj_critic(buffer, self.batch_size)
            obj_critics += obj_critic.item()
            self.optimizer_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
            self.obj_c = 0.995 * self.obj_c + 0.005 * obj_critic.item()  # for reliable_lambda

            reliable_lambda = math.exp(-self.obj_c ** 2)  # for reliable_lambda
            if update_a / update_c < 1 / (2 - reliable_lambda):  # auto TTUR
                '''objective of alpha (temperature parameter automatic adjustment)'''
                action_pg, log_prob = self.act.get_action_logprob(state)  # policy gradient
                obj_alpha = (self.alpha_log * (self.target_entropy - log_prob).detach()).mean()
                self.optimizer_update(self.alpha_optimizer, obj_alpha)

                '''objective of actor'''
                alpha = self.alpha_log.exp().detach()
                alphas += alpha.item()
                with torch.no_grad():
                    self.alpha_log[:] = self.alpha_log.clamp(-16, 2)

                q_value_pg = self.cri_target(state, action_pg).mean()
                obj_actor = (q_value_pg - log_prob * alpha).mean()
                obj_actors += obj_actor.item()
                self.optimizer_update(self.act_optimizer, -obj_actor)

        return obj_critics / update_times, obj_actors / update_times, alphas / update_times
