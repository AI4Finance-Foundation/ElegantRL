import torch
import numpy.random as rd
from elegantrl.agents.AgentDQN import AgentDQN
from elegantrl.agents.net import QNetTwin, QNetTwinDuel


class AgentDoubleDQN(AgentDQN):  # [ElegantRL.2021.10.25]
    def __init__(self):
        AgentDQN.__init__(self)
        self.soft_max = torch.nn.Softmax(dim=1)

    def init(self, net_dim=256, state_dim=8, action_dim=2, reward_scale=1.0, gamma=0.99,
             learning_rate=1e-4, if_per_or_gae=False, env_num=1, gpu_id=0):
        self.ClassCri = QNetTwinDuel if self.if_use_dueling else QNetTwin
        AgentDQN.init(self, net_dim, state_dim, action_dim, learning_rate, reward_scale, gamma,
                      if_per_or_gae, env_num, gpu_id)

        if if_per_or_gae:  # if_use_per
            self.criterion = torch.nn.SmoothL1Loss(reduction='none')
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.criterion = torch.nn.SmoothL1Loss(reduction='mean')
            self.get_obj_critic = self.get_obj_critic_raw

    def select_actions(self, states: torch.Tensor) -> torch.Tensor:  # for discrete action space
        actions = self.act(states.to(self.device))
        if rd.rand() < self.explore_rate:  # epsilon-greedy
            a_prob = self.soft_max(actions)
            a_ints = torch.multinomial(a_prob, num_samples=1, replacement=True)[:, 0]
            # a_int = rd.choice(self.action_dim, prob=a_prob)  # numpy version
        else:
            a_ints = actions.argmax(dim=1)
        return a_ints.detach().cpu()

    def get_obj_critic_raw(self, buffer, batch_size) -> (torch.Tensor, torch.Tensor):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s)).max(dim=1, keepdim=True)[0]
            q_label = reward + mask * next_q

        q1, q2 = [qs.gather(1, action.long()) for qs in self.act.get_q1_q2(state)]
        obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)
        return obj_critic, q1

    def get_obj_critic_per(self, buffer, batch_size):
        with torch.no_grad():
            reward, mask, action, state, next_s, is_weights = buffer.sample_batch(batch_size)
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s)).max(dim=1, keepdim=True)[0]
            q_label = reward + mask * next_q

        q1, q2 = [qs.gather(1, action.long()) for qs in self.act.get_q1_q2(state)]
        td_error = self.criterion(q1, q_label) + self.criterion(q2, q_label)
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, q1
