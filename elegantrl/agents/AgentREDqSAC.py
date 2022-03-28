import numpy as np
import torch
from elegantrl.agents.net import ActorFixSAC, CriticREDq
from elegantrl.agents.AgentSAC import AgentSAC


class AgentREDqSAC(
    AgentSAC
):  # Modified SAC using reliable_lambda and TTUR (Two Time-scale Update Rule)
    def __init__(self, net_dim, state_dim, action_dim, gpu_id=0, args=None):
        self.act_class = getattr(self, "act_class", ActorFixSAC)
        self.cri_class = getattr(self, "cri_class", CriticREDq)
        super().__init__(net_dim, state_dim, action_dim, gpu_id, args)
        self.obj_c = (-np.log(0.5)) ** 0.5  # for reliable_lambda

    def get_obj_critic_raw(self, buffer, batch_size):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)

            next_a, next_log_prob = self.act_target.get_action_logprob(
                next_s
            )  # stochastic policy
            next_q = self.cri_target.get_q_min(next_s, next_a)

            alpha = self.alpha_log.exp().detach()
            q_label = reward + mask * (next_q + next_log_prob * alpha)
        qs = self.cri.get_q_values(state, action)
        obj_critic = self.criterion(qs, q_label * torch.ones_like(qs))
        return obj_critic, state

    def get_obj_critic_per(self, buffer, batch_size):
        with torch.no_grad():
            reward, mask, action, state, next_s, is_weights = buffer.sample_batch(
                batch_size
            )

            next_a, next_log_prob = self.act_target.get_action_logprob(
                next_s
            )  # stochastic policy
            next_q = self.cri_target.get_q_min(next_s, next_a)

            alpha = self.alpha_log.exp().detach()
            q_label = reward + mask * (next_q + next_log_prob * alpha)
        qs = self.cri.get_q_values(state, action)
        td_error = self.criterion(qs, q_label * torch.ones_like(qs)).mean(dim=1)
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, state
