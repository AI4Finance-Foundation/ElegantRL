import numpy.random as rd
import torch

from elegantrl.agents.AgentDQN import AgentDQN
from elegantrl.agents.net import QNetTwin, QNetTwinDuel
from typing import Tuple


class AgentDoubleDQN(AgentDQN):  # [ElegantRL.2021.10.25]
    """
    Bases: ``AgentDQN``

    Double Deep Q-Network algorithm. “Deep Reinforcement Learning with Double Q-learning”. H. V. Hasselt et al.. 2015.

    :param net_dim[int]: the dimension of networks (the width of neural networks)
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    :param learning_rate[float]: learning rate of optimizer
    :param if_per_or_gae[bool]: PER (off-policy) or GAE (on-policy) for sparse reward
    :param env_num[int]: the env number of VectorEnv. env_num == 1 means don't use VectorEnv
    :param agent_id[int]: if the visible_gpu is '1,9,3,4', agent_id=1 means (1,9,4,3)[agent_id] == 9
    """

    def __init__(self):
        AgentDQN.__init__(self)
        self.ClassCri = QNetTwin
        self.soft_max = torch.nn.Softmax(dim=1)

    def select_actions(
        self, states: torch.Tensor
    ) -> torch.Tensor:  # for discrete action space
        """
        Select discrete actions given an array of states.

        .. note::
            Using ϵ-greedy to select uniformly random actions for exploration.

        :param states: an array of states in a shape (batch_size, state_dim, ).
        :return: an array of actions in a shape (batch_size, action_dim, ) where each action is clipped into range(-1, 1).
        """
        actions = self.act(states.to(self.device))
        if rd.rand() < self.explore_rate:  # epsilon-greedy
            a_prob = self.soft_max(actions)
            a_ints = torch.multinomial(a_prob, num_samples=1, replacement=True)[:, 0]
            # a_int = rd.choice(self.action_dim, prob=a_prob)  # numpy version
        else:
            a_ints = actions.argmax(dim=1)
        return a_ints.detach().cpu()

    def get_obj_critic_raw(
        self, buffer, batch_size
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the loss of the network and predict Q values with **uniform sampling**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and Q values.
        """
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s)).max(
                dim=1, keepdim=True
            )[0]
            q_label = reward + mask * next_q

        q1, q2 = [qs.gather(1, action.long()) for qs in self.act.get_q1_q2(state)]
        obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)
        return obj_critic, q1

    def get_obj_critic_per(self, buffer, batch_size):
        """
        Calculate the loss of the network and predict Q values with **Prioritized Experience Replay (PER)**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and Q values.
        """
        with torch.no_grad():
            reward, mask, action, state, next_s, is_weights = buffer.sample_batch(
                batch_size
            )
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s)).max(
                dim=1, keepdim=True
            )[0]
            q_label = reward + mask * next_q

        q1, q2 = [qs.gather(1, action.long()) for qs in self.act.get_q1_q2(state)]
        td_error = self.criterion(q1, q_label) + self.criterion(q2, q_label)
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, q1


class AgentD3QN(AgentDoubleDQN):  # D3QN: DuelingDoubleDQN
    def __init__(self):
        AgentDoubleDQN.__init__(self)
        self.ClassCri = QNetTwinDuel
