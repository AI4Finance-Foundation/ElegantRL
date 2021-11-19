import torch
import numpy as np
from elegantrl.agents.AgentSAC import AgentSAC
from elegantrl.agents.net import CriticMultiple


class AgentModSAC(AgentSAC):  # [ElegantRL.2021.11.11]
    """Modified SAC
    - reliable_lambda and TTUR (Two Time-scale Update Rule)
    - modified REDQ (Randomized Ensemble Double Q-learning)
    """

    def __init__(self):
        AgentSAC.__init__(self)
        self.ClassCri = CriticMultiple  # REDQ ensemble (parameter sharing)
        # self.ClassCri = CriticEnsemble  # REDQ ensemble  # todo ensemble
        self.if_use_cri_target = True
        self.if_use_act_target = True

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        buffer.update_now_len()

        obj_actor = None
        update_a = 0
        alpha = None
        for update_c in range(1, int(buffer.now_len * repeat_times / batch_size)):
            alpha = self.alpha_log.exp()

            '''objective of critic (loss function of critic)'''
            obj_critic, state = self.get_obj_critic(buffer, batch_size, alpha)
            self.obj_critic = 0.995 * self.obj_critic + 0.005 * obj_critic.item()  # for reliable_lambda
            self.optim_update(self.cri_optim, obj_critic)
            if self.if_use_cri_target:
                self.soft_update(self.cri_target, self.cri, soft_update_tau)

            a_noise_pg, logprob = self.act.get_action_logprob(state)  # policy gradient
            '''objective of alpha (temperature parameter automatic adjustment)'''
            obj_alpha = (self.alpha_log * (logprob - self.target_entropy).detach()).mean()
            self.optim_update(self.alpha_optim, obj_alpha)
            with torch.no_grad():
                self.alpha_log[:] = self.alpha_log.clamp(-16, 2).detach()

            '''objective of actor using reliable_lambda and TTUR (Two Time-scales Update Rule)'''
            reliable_lambda = np.exp(-self.obj_critic ** 2)  # for reliable_lambda
            if_update_a = update_a / update_c < 1 / (2 - reliable_lambda)
            if if_update_a:  # auto TTUR
                update_a += 1

                q_value_pg = self.cri(state, a_noise_pg)
                obj_actor = -(q_value_pg + logprob * alpha).mean()  # todo ensemble
                self.optim_update(self.act_optim, obj_actor)
                if self.if_use_act_target:
                    self.soft_update(self.act_target, self.act, soft_update_tau)

        return self.obj_critic, obj_actor.item(), alpha.item()

    def get_obj_critic_raw(self, buffer, batch_size, alpha):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)

            next_a, next_log_prob = self.act_target.get_action_logprob(next_s)  # stochastic policy
            next_q = torch.min(self.cri_target.get_q_values(next_s, next_a), dim=1, keepdim=True)[0]  # multiple critics

            # todo ensemble
            q_label = (reward + mask * (next_q + next_log_prob * alpha))
            q_labels = q_label * torch.ones((1, self.cri.q_values_num), dtype=torch.float32, device=self.device)
        q_values = self.cri.get_q_values(state, action)  # todo ensemble

        obj_critic = self.criterion(q_values, q_labels)
        return obj_critic, state

    def get_obj_critic_per(self, buffer, batch_size, alpha):
        with torch.no_grad():
            # reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            reward, mask, action, state, next_s, is_weights = buffer.sample_batch(batch_size)

            next_a, next_log_prob = self.act_target.get_action_logprob(next_s)  # stochastic policy
            next_q = torch.min(self.cri_target.get_q_values(next_s, next_a), dim=1, keepdim=True)[0]  # multiple critics

            q_label = (reward + mask * (next_q + next_log_prob * alpha))
            q_labels = q_label * torch.ones((1, self.cri.q_values_num), dtype=torch.float32, device=self.device)
        q_values = self.cri.get_q_values(state, action)

        # obj_critic = self.criterion(q_values, q_labels)
        td_error = self.criterion(q_values, q_labels).mean(dim=1, keepdim=True)
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, state
