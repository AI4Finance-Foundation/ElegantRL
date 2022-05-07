import numpy as np
import numpy.random as rd
import torch
import torch.nn as nn


class QNet(nn.Module):  # nn.Module is a standard PyTorch Network
    """
    Class for **Q-network**.

    :param mid_dim[int]: the middle dimension of networks
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    """

    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, action_dim),
        )
        self.explore_rate = 0.125
        self.action_dim = action_dim

    def forward(self, state):
        """
        The forward function.

        :param state: the input state.
        :return: the output tensor.
        """
        return self.net(state)  # Q values for multiple actions

    def get_action(self, state):
        """
        The forward function with ϵ-greedy.

        :param state: the input state.
        :return: the action.
        """
        return (
            self.net(state).argmax(dim=1, keepdim=True)
            if rd.rand() > self.explore_rate
            else torch.randint(self.action_dim, size=(state.shape[0], 1))
        )


class ActorSAC(nn.Module):
    """
    Actor class for **SAC** with stochastic, learnable, **state-dependent** log standard deviation..

    :param mid_dim[int]: the middle dimension of networks
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    """

    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net_state = nn.Sequential(
            nn.Linear(state_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
        )
        self.net_a_avg = nn.Sequential(
            nn.Linear(mid_dim, mid_dim), nn.ReLU(), nn.Linear(mid_dim, action_dim)
        )  # the average of action
        self.net_a_std = nn.Sequential(
            nn.Linear(mid_dim, mid_dim), nn.ReLU(), nn.Linear(mid_dim, action_dim)
        )  # the log_std of action
        self.log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))

    def forward(self, state):
        """
        The forward function for **SAC**.

        :param state: the input state.
        :return: the output tensor.
        """
        tmp = self.net_state(state)
        return self.net_a_avg(tmp).tanh()  # action

    def get_action(self, state):
        """
        The forward function with noise.

        :param state: the input state.
        :return: the action and added noise.
        """
        t_tmp = self.net_state(state)
        a_avg = self.net_a_avg(t_tmp)  # NOTICE! it is a_avg without .tanh()
        a_std = self.net_a_std(t_tmp).clamp(-20, 2).exp()
        return torch.normal(a_avg, a_std).tanh()  # re-parameterize

    def get_action_logprob(self, state):
        """
        Compute the action and log of probability with current network.

        :param state: the input state.
        :return: the action and log of probability.
        """
        t_tmp = self.net_state(state)
        a_avg = self.net_a_avg(t_tmp)  # NOTICE! it needs a_avg.tanh()
        a_std_log = self.net_a_std(t_tmp).clamp(-20, 2)
        a_std = a_std_log.exp()

        noise = torch.randn_like(a_avg, requires_grad=True)
        a_tan = (a_avg + a_std * noise).tanh()  # action.tanh()

        log_prob = (
            a_std_log + self.log_sqrt_2pi + noise.pow(2).__mul__(0.5)
        )  # noise.pow(2) * 0.5
        log_prob = (
            log_prob + (-a_tan.pow(2) + 1.000001).log()
        )  # fix log_prob using the derivative of action.tanh()
        return a_tan, log_prob.sum(1, keepdim=True)


class ActorPPO(nn.Module):
    """
    Actor class for **PPO** with stochastic, learnable, **state-independent** log standard deviation.

    :param mid_dim[int]: the middle dimension of networks
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    """

    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, action_dim),
        )

        # the logarithm (log) of standard deviation (std) of action, it is a trainable parameter
        self.a_std_log = nn.Parameter(
            torch.zeros((1, action_dim)) - 0.5, requires_grad=True
        )
        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))

    def forward(self, state):
        """
        The forward function for **PPO**.

        :param state: the input state.
        :return: the output tensor.
        """
        return self.net(state).tanh()  # action.tanh()

    def get_action(self, state):
        """
        The forward function with Gaussian noise.

        :param state: the input state.
        :return: the action and added noise.
        """
        a_avg = self.net(state)
        a_std = self.a_std_log.exp()

        noise = torch.randn_like(a_avg)
        action = a_avg + noise * a_std
        return action, noise

    def get_logprob_entropy(self, state, action):
        """
        Compute the log of probability with current network.

        :param state: the input state.
        :param action: the action.
        :return: the log of probability and entropy.
        """
        a_avg = self.net(state)
        a_std = self.a_std_log.exp()

        delta = ((a_avg - action) / a_std).pow(2) * 0.5
        logprob = -(self.a_std_log + self.sqrt_2pi_log + delta).sum(1)  # new_logprob

        dist_entropy = (logprob.exp() * logprob).mean()  # policy entropy
        return logprob, dist_entropy

    def get_old_logprob(self, _action, noise):  # noise = action - a_noise
        """
        Compute the log of probability with old network.

        :param _action: the action made.
        :param noise: the noised added.
        :return: the log of probability of old network.
        """
        delta = noise.pow(2) * 0.5
        return -(self.a_std_log + self.sqrt_2pi_log + delta).sum(1)  # old_logprob

    @staticmethod
    def get_a_to_e(action):
        return action.tanh()


class ActorDiscretePPO(nn.Module):
    """
    Actor class for **Discrete PPO**.

    :param mid_dim[int]: the middle dimension of networks
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    """

    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, action_dim),
        )
        self.action_dim = action_dim
        self.soft_max = nn.Softmax(dim=-1)
        self.Categorical = torch.distributions.Categorical

    def forward(self, state):
        """
        The forward function for **Q-network**.

        :param state: the input state.
        :return: the output tensor.
        """
        return self.net(state)  # action_prob without softmax

    def get_action(self, state):
        """
        The forward function with SoftMax.

        :param state: the input state.
        :return: the action and added noise.
        """
        a_prob = self.soft_max(self.net(state))
        # action = Categorical(a_prob).sample()
        samples_2d = torch.multinomial(a_prob, num_samples=1, replacement=True)
        action = samples_2d.reshape(state.size(0))
        return action, a_prob

    def get_logprob_entropy(self, state, a_int):
        # assert a_int.shape == (batch_size, 1)
        a_prob = self.soft_max(self.net(state))
        dist = self.Categorical(a_prob)
        return dist.log_prob(a_int.squeeze(1)), dist.entropy().mean()

    def get_old_logprob(self, a_int, a_prob):
        # assert a_int.shape == (batch_size, 1)
        dist = self.Categorical(a_prob)
        return dist.log_prob(a_int.squeeze(1))

    @staticmethod
    def get_a_to_e(action):
        return action.int()


class CriticPPO(nn.Module):
    """
    The Critic class for **PPO**.

    :param mid_dim[int]: the middle dimension of networks
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    """

    def __init__(self, mid_dim, state_dim, _action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, 1),
        )

    def forward(self, state):
        """
        The forward function to ouput the value of the state.

        :param state: the input state.
        :return: the output tensor.
        """
        return self.net(state)  # advantage value


class CriticTwin(nn.Module):  # shared parameter
    """
    The Critic class for **Clipped Double DQN**.

    :param mid_dim[int]: the middle dimension of networks
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    """

    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net_sa = nn.Sequential(
            nn.Linear(state_dim + action_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
        )  # concat(state, action)
        self.net_q1 = nn.Sequential(
            nn.Linear(mid_dim, mid_dim), nn.ReLU(), nn.Linear(mid_dim, 1)
        )  # q1 value
        self.net_q2 = nn.Sequential(
            nn.Linear(mid_dim, mid_dim), nn.ReLU(), nn.Linear(mid_dim, 1)
        )  # q2 value

    def forward(self, state, action):
        """
        The forward function to ouput the value of the state.

        :param state: the input state.
        :param action: the input action.
        :return: the output tensor.
        """
        tmp = self.net_sa(torch.cat((state, action), dim=1))
        return self.net_q1(tmp)  # one Q value

    def get_q1_q2(self, state, action):
        """
        The forward function to output two Q-values from two shared-paramter networks.

        :param state: the input state.
        :param action: the input action.
        :return: the output tensor.
        """
        tmp = self.net_sa(torch.cat((state, action), dim=1))
        return self.net_q1(tmp), self.net_q2(tmp)  # two Q values
