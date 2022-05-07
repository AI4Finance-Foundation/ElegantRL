import torch
from torch import Tensor

from elegantrl.agents.AgentDQN import AgentDQN
from elegantrl.agents.net import QNetTwin
from elegantrl.train.replay_buffer import ReplayBuffer
from elegantrl.train.config import Arguments

'''[ElegantRL.2022.05.05](github.com/AI4Fiance-Foundation/ElegantRL)'''


class AgentDoubleDQN(AgentDQN):
    """
    Double Deep Q-Network algorithm. “Deep Reinforcement Learning with Double Q-learning”. H. V. Hasselt et al.. 2015.

    :param net_dim: the dimension of networks (the width of neural networks)
    :param state_dim: the dimension of state (the number of state vector)
    :param action_dim: the dimension of action (the number of discrete action)
    :param gpu_id: the gpu_id of the training device. Use CPU when cuda is not available.
    :param args: the arguments for agent training. `args = Arguments()`
    """

    def __init__(self, net_dim: int, state_dim: int, action_dim: int, gpu_id: int = 0, args: Arguments = None):
        self.act_class = getattr(self, "act_class", QNetTwin)
        self.cri_class = getattr(self, "cri_class", None)  # means `self.cri = self.act`
        AgentDQN.__init__(self, net_dim, state_dim, action_dim, gpu_id, args)

    def get_obj_critic_raw(self, buffer: ReplayBuffer, batch_size: int) -> (Tensor, Tensor):
        """
        Calculate the loss of the network and predict Q values with **uniform sampling**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and Q values.
        """
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s)).max(dim=1, keepdim=True)[0]
            q_label = reward + mask * next_q

        q1, q2 = [qs.gather(1, action.long()) for qs in self.act.get_q1_q2(state)]
        obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)
        return obj_critic, q1

    def get_obj_critic_per(self, buffer: ReplayBuffer, batch_size: int) -> (Tensor, Tensor):
        """
        Calculate the loss of the network and predict Q values with **Prioritized Experience Replay (PER)**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and Q values.
        """
        with torch.no_grad():
            reward, mask, action, state, next_s, is_weights = buffer.sample_batch(batch_size)
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s)).max(dim=1, keepdim=True)[0]
            q_label = reward + mask * next_q

        q1, q2 = [qs.gather(1, action.long()) for qs in self.act.get_q1_q2(state)]
        td_error = self.criterion(q1, q_label) + self.criterion(q2, q_label)
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, q1
