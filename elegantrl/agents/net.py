import numpy as np
import torch
import torch.nn as nn

"""[ElegantRL.2021.12.12](https://github.com/AI4Finance-Foundation/ElegantRL)"""

"""Q Network"""


class QNet(nn.Module):  # nn.Module is a standard PyTorch Network
    """
    Critic class for **Q-network**.

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
            nn.Hardswish(),
            nn.Linear(mid_dim, action_dim),
        )

    def forward(self, state):
        """
        The forward function for **Q-network**.

        :param state: [tensor] the input state.
        :return: the output tensor.
        """
        return self.net(state)  # Q value


class QNetDuel(nn.Module):  # Dueling DQN
    """
    Critic class for **Dueling Q-network**.

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
        self.net_adv = nn.Sequential(
            nn.Linear(mid_dim, mid_dim), nn.Hardswish(), nn.Linear(mid_dim, 1)
        )  # advantage function value 1
        self.net_val = nn.Sequential(
            nn.Linear(mid_dim, mid_dim), nn.Hardswish(), nn.Linear(mid_dim, action_dim)
        )  # Q value

    def forward(self, state):
        """
        The forward function for **Dueling Q-network**.

        :param state: [tensor] the input state.
        :return: the output tensor.
        """
        t_tmp = self.net_state(state)  # tensor of encoded state
        q_adv = self.net_adv(t_tmp)
        q_val = self.net_val(t_tmp)
        return q_adv + q_val - q_val.mean(dim=1, keepdim=True)  # dueling Q value


class QNetTwin(nn.Module):  # Double DQN
    """
    Critic class for **Double DQN**.

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
        self.net_q1 = nn.Sequential(
            nn.Linear(mid_dim, mid_dim), nn.Hardswish(), nn.Linear(mid_dim, action_dim)
        )  # q1 value
        self.net_q2 = nn.Sequential(
            nn.Linear(mid_dim, mid_dim), nn.Hardswish(), nn.Linear(mid_dim, action_dim)
        )  # q2 value

    def forward(self, state):
        """
        The forward function for **Double DQN**.

        :param state: [tensor] the input state.
        :return: the output tensor.
        """
        tmp = self.net_state(state)
        return self.net_q1(tmp)  # one Q value

    def get_q1_q2(self, state):
        """
        TBD
        """
        tmp = self.net_state(state)
        return self.net_q1(tmp), self.net_q2(tmp)  # two Q values


class QNetTwinDuel(nn.Module):  # D3QN: Dueling Double DQN
    """
    Critic class for **Dueling Double DQN**.

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
        self.net_adv1 = nn.Sequential(
            nn.Linear(mid_dim, mid_dim), nn.Hardswish(), nn.Linear(mid_dim, 1)
        )  # q1 value
        self.net_adv2 = nn.Sequential(
            nn.Linear(mid_dim, mid_dim), nn.Hardswish(), nn.Linear(mid_dim, 1)
        )  # q2 value
        self.net_val1 = nn.Sequential(
            nn.Linear(mid_dim, mid_dim), nn.Hardswish(), nn.Linear(mid_dim, action_dim)
        )  # advantage function value 1
        self.net_val2 = nn.Sequential(
            nn.Linear(mid_dim, mid_dim), nn.Hardswish(), nn.Linear(mid_dim, action_dim)
        )  # advantage function value 1

    def forward(self, state):
        """
        The forward function for **Dueling Double DQN**.

        :param state: [tensor] the input state.
        :return: the output tensor.
        """
        t_tmp = self.net_state(state)
        q_adv = self.net_adv1(t_tmp)
        q_val = self.net_val1(t_tmp)
        return q_adv + q_val - q_val.mean(dim=1, keepdim=True)  # one dueling Q value

    def get_q1_q2(self, state):
        """
        TBD
        """
        tmp = self.net_state(state)

        adv1 = self.net_adv1(tmp)
        val1 = self.net_val1(tmp)
        q1 = adv1 + val1 - val1.mean(dim=1, keepdim=True)

        adv2 = self.net_adv2(tmp)
        val2 = self.net_val2(tmp)
        q2 = adv2 + val2 - val2.mean(dim=1, keepdim=True)
        return q1, q2  # two dueling Q values


"""Policy Network (Actor)"""


class Actor(nn.Module):
    """
    A simple Actor class.

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
            nn.Hardswish(),
            nn.Linear(mid_dim, action_dim),
        )

    def forward(self, state):
        """
        The forward function.

        :param state: [tensor] the input state.
        :return: the output tensor.
        """
        return self.net(state).tanh()  # action.tanh()

    def get_action(self, state, action_std):
        """
        The forward function with Gaussian noise.

        :param state: [tensor] the input state.
        :param action_std: [float] the standard deviation of the Gaussian distribution.
        :return: the output tensor.
        """
        action = self.net(state).tanh()
        noise = (torch.randn_like(action) * action_std).clamp(-0.5, 0.5)
        return (action + noise).clamp(-1.0, 1.0)


class ActorSAC(nn.Module):
    """
    Actor class for **SAC** with stochastic, learnable, **state-dependent** log standard deviation..

    :param mid_dim[int]: the middle dimension of networks
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    """

    def __init__(self, mid_dim, state_dim, action_dim, if_use_dn=False):
        super().__init__()
        if if_use_dn:
            nn_middle = DenseNet(mid_dim // 2)
            inp_dim = nn_middle.inp_dim
            out_dim = nn_middle.out_dim
        else:
            nn_middle = nn.Sequential(
                nn.Linear(mid_dim, mid_dim),
                nn.ReLU(),
                nn.Linear(mid_dim, mid_dim),
                nn.ReLU(),
            )
            inp_dim = mid_dim
            out_dim = mid_dim

        self.net_state = nn.Sequential(
            nn.Linear(state_dim, inp_dim),
            nn.ReLU(),
            nn_middle,
        )
        self.net_a_avg = nn.Sequential(
            nn.Linear(out_dim, mid_dim), nn.Hardswish(), nn.Linear(mid_dim, action_dim)
        )  # the average of action
        self.net_a_std = nn.Sequential(
            nn.Linear(out_dim, mid_dim), nn.Hardswish(), nn.Linear(mid_dim, action_dim)
        )  # the log_std of action
        self.log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))
        self.soft_plus = nn.Softplus()

    def forward(self, state):
        """
        The forward function.

        :param state: [tensor] the input state.
        :return: the output tensor.
        """
        tmp = self.net_state(state)
        return self.net_a_avg(tmp).tanh()  # action

    def get_action(self, state):
        """
        The forward function with noise.

        :param state: [tensor] the input state.
        :return: the action and added noise.
        """
        t_tmp = self.net_state(state)
        a_avg = self.net_a_avg(t_tmp)  # NOTICE! it is a_avg without .tanh()
        a_std = self.net_a_std(t_tmp).clamp(-20, 2).exp()
        return torch.normal(a_avg, a_std).tanh()  # re-parameterize

    def get_action_logprob(self, state):
        """
        Compute the action and log of probability with current network.

        :param state: [tensor] the input state.
        :return: the action and log of probability.
        """
        t_tmp = self.net_state(state)
        a_avg = self.net_a_avg(t_tmp)  # NOTICE! it needs a_avg.tanh()
        a_std_log = self.net_a_std(t_tmp).clamp(-20, 2)
        a_std = a_std_log.exp()

        """add noise to a_noise in stochastic policy"""
        noise = torch.randn_like(a_avg, requires_grad=True)
        a_noise = a_avg + a_std * noise
        # Can only use above code instead of below, because the tensor need gradients here.
        # a_noise = torch.normal(a_avg, a_std, requires_grad=True)

        """compute log_prob according to mean and std of a_noise (stochastic policy)"""
        # self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))
        log_prob = (
            a_std_log + self.log_sqrt_2pi + noise.pow(2).__mul__(0.5)
        )  # noise.pow(2) * 0.5
        """same as below:
        from torch.distributions.normal import Normal
        log_prob = Normal(a_avg, a_std).log_prob(a_noise)
        # same as below:
        a_delta = (a_avg - a_noise).pow(2) /(2*a_std.pow(2))
        log_prob = -a_delta - a_std.log() - np.log(np.sqrt(2 * np.pi))
        """

        """fix log_prob of action.tanh"""
        log_prob += (
            np.log(2.0) - a_noise - self.soft_plus(-2.0 * a_noise)
        ) * 2.0  # better than below
        """same as below:
        epsilon = 1e-6
        a_noise_tanh = a_noise.tanh()
        log_prob = log_prob - (1 - a_noise_tanh.pow(2) + epsilon).log()

        Thanks for:
        https://github.com/denisyarats/pytorch_sac/blob/81c5b536d3a1c5616b2531e446450df412a064fb/agent/actor.py#L37
        ↑ MIT License， Thanks for https://www.zhihu.com/people/Z_WXCY 2ez4U
        They use action formula that is more numerically stable, see details in the following link
        https://pytorch.org/docs/stable/_modules/torch/distributions/transforms.html#TanhTransform
        https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f
        """
        return a_noise.tanh(), log_prob.sum(1, keepdim=True)


class ActorPPO(nn.Module):
    """
    Actor class for **PPO** with stochastic, learnable, **state-independent** log standard deviation.

    :param mid_dim[int]: the middle dimension of networks
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    """

    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        if isinstance(state_dim, int):
            nn_middle = nn.Sequential(
                nn.Linear(state_dim, mid_dim),
                nn.ReLU(),
                nn.Linear(mid_dim, mid_dim),
                nn.ReLU(),
            )
        else:
            nn_middle = ConvNet(
                inp_dim=state_dim[2], out_dim=mid_dim, image_size=state_dim[0]
            )

        self.net = nn.Sequential(
            nn_middle,
            nn.Linear(mid_dim, mid_dim),
            nn.Hardswish(),
            nn.Linear(mid_dim, action_dim),
        )
        layer_norm(self.net[-1], std=0.1)  # output layer for action

        # the logarithm (log) of standard deviation (std) of action, it is a trainable parameter
        self.a_std_log = nn.Parameter(
            torch.zeros((1, action_dim)) - 0.5, requires_grad=True
        )
        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))

    def forward(self, state):
        """
        The forward function.

        :param state: [tensor] the input state.
        :return: the output tensor.
        """
        return self.net(state).tanh()  # action.tanh()

    def get_action(self, state):
        """
        The forward function with Gaussian noise.

        :param state: [tensor] the input state.
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

        :param state: [tensor] the input state.
        :param action: [tensor] the action.
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

        :param _action: [tensor] the action.
        :param noise: [tensor] the added noise when exploring.
        :return: the log of probability with old network.
        """
        delta = noise.pow(2) * 0.5
        return -(self.a_std_log + self.sqrt_2pi_log + delta).sum(1)  # old_logprob


class ActorDiscreteSAC(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.Hardswish(),
            nn.Linear(mid_dim, action_dim),
        )
        self.action_dim = action_dim

        self.soft_max = nn.Softmax(dim=-1)
        self.Categorical = torch.distributions.Categorical

    def forward(self, state):
        return self.net(state)  # action_prob without softmax

    def get_action(self, state):
        # print(self.net(state).shape)
        # exit()
        a_prob = self.soft_max(self.net(state))
        # action = Categorical(a_prob).sample()
        # print(a_prob.shape)
        samples_2d = torch.multinomial(a_prob, num_samples=1, replacement=True)
        action = samples_2d.reshape(state.size(0))
        z = a_prob == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(a_prob + z)
        return action, a_prob, log_action_probabilities

    def get_logprob_entropy(self, state, a_int):
        a_prob = self.soft_max(self.net(state))
        dist = self.Categorical(a_prob)
        return dist.log_prob(a_int), dist.entropy().mean()

    def get_old_logprob(self, a_int, a_prob):
        dist = self.Categorical(a_prob)
        return dist.log_prob(a_int)


class ActorDiscretePPO(nn.Module):
    """
    Actor class for **Discrete PPO**.

    :param mid_dim[int]: the middle dimension of networks
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    """

    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        if isinstance(state_dim, int):
            nn_middle = nn.Sequential(
                nn.Linear(state_dim, mid_dim),
                nn.ReLU(),
            )
        else:
            nn_middle = ConvNet(
                inp_dim=state_dim[2], out_dim=mid_dim, image_size=state_dim[0]
            )

        self.net = nn.Sequential(
            nn_middle,
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.Hardswish(),
            nn.Linear(mid_dim, action_dim),
        )
        layer_norm(self.net[-1], std=0.1)  # output layer for action

        self.action_dim = action_dim
        self.soft_max = nn.Softmax(dim=-1)
        self.Categorical = torch.distributions.Categorical

    def forward(self, state):
        """
        The forward function.

        :param state: [tensor] the input state.
        :return: the output tensor.
        """
        return self.net(state)  # action_prob without softmax

    def get_action(self, state):
        """
        The forward function with Softmax.

        :param state: [tensor] the input state.
        :return: the action index and probabilities.
        """
        a_prob = self.soft_max(self.net(state))
        # dist = Categorical(a_prob)
        # a_int = dist.sample()
        a_int = torch.multinomial(a_prob, num_samples=1, replacement=True)[:, 0]
        # samples_2d = torch.multinomial(a_prob, num_samples=1, replacement=True)
        # samples_2d.shape == (batch_size, num_samples)
        return a_int, a_prob

    def get_logprob_entropy(self, state, a_int):
        """
        Compute the log of probability with current network.

        :param state: [tensor] the input state.
        :param a_int: [tensor] the action.
        :return: the log of probability and entropy.
        """
        a_prob = self.soft_max(self.net(state))
        dist = self.Categorical(a_prob)
        return dist.log_prob(a_int), dist.entropy().mean()

    def get_old_logprob(self, a_int, a_prob):
        """
        Compute the log of probability with old network.

        :param a_int: [tensor] the action.
        :param a_prob: [tensor] the action probability.
        :return: the log of probability with old network.
        """
        dist = self.Categorical(a_prob)
        return dist.log_prob(a_int)


class ActorBiConv(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            BiConvNet(mid_dim, state_dim, mid_dim * 4),
            nn.Linear(mid_dim * 4, mid_dim * 1),
            nn.ReLU(),
            DenseNet(mid_dim * 1),
            nn.ReLU(),
            nn.Linear(mid_dim * 4, action_dim),
        )
        layer_norm(self.net[-1], std=0.1)  # output layer for action

    def forward(self, state):
        action = self.net(state)
        return action * torch.pow(
            (action**2).sum(), -0.5
        )  # action / sqrt(L2_norm(action))

    def get_action(self, state, action_std):
        action = self.net(state)
        action = action + (torch.randn_like(action) * action_std)
        return action * torch.pow(
            (action**2).sum(), -0.5
        )  # action / sqrt(L2_norm(action))


"""Value Network (Critic)"""


class Critic(nn.Module):
    """
    A simple Critic class.

    :param mid_dim[int]: the middle dimension of networks
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    """

    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.Hardswish(),
            nn.Linear(mid_dim, 1),
        )

    def forward(self, state, action):
        """
        The forward function.

        :param state: [tensor] the input state.
        :param action: [tensor] the input action.
        :return: the output tensor.
        """
        return self.net(torch.cat((state, action), dim=1))  # Q value


class CriticTwin(nn.Module):  # shared parameter
    """
    The Critic class for **Clipped Double DQN**.

    :param mid_dim[int]: the middle dimension of networks
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    """

    def __init__(self, mid_dim, state_dim, action_dim, if_use_dn=False):
        super().__init__()
        if if_use_dn:
            nn_middle = DenseNet(mid_dim // 2)
            inp_dim = nn_middle.inp_dim
            out_dim = nn_middle.out_dim
        else:
            nn_middle = nn.Sequential(
                nn.Linear(mid_dim, mid_dim),
                nn.ReLU(),
                nn.Linear(mid_dim, mid_dim),
                nn.ReLU(),
            )
            inp_dim = mid_dim
            out_dim = mid_dim

        self.net_sa = nn.Sequential(
            nn.Linear(state_dim + action_dim, inp_dim),
            nn.ReLU(),
            nn_middle,
        )  # concat(state, action)
        self.net_q1 = nn.Sequential(
            nn.Linear(out_dim, mid_dim), nn.Hardswish(), nn.Linear(mid_dim, 1)
        )  # q1 value
        self.net_q2 = nn.Sequential(
            nn.Linear(out_dim, mid_dim), nn.Hardswish(), nn.Linear(mid_dim, 1)
        )  # q2 value

    def forward(self, state, action):
        """
        The forward function to ouput a single Q-value.

        :param state: [tensor] the input state.
        :param action: [tensor] the input action.
        :return: the output tensor.
        """
        tmp = self.net_sa(torch.cat((state, action), dim=1))
        return self.net_q1(tmp)  # one Q value

    def get_q1_q2(self, state, action):
        """
        The forward function to output two Q-values from two shared-paramter networks.

        :param state: [tensor] the input state.
        :param action: [tensor] the input action.
        :return: the output tensor.
        """
        tmp = self.net_sa(torch.cat((state, action), dim=1))
        return self.net_q1(tmp), self.net_q2(tmp)  # two Q values


class CriticEnsemble(nn.Module):  # todo ensemble
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.q_values_num = 8  # modified REDQ (Randomized Ensemble Double Q-learning)
        for critic_id in range(self.q_values_num):
            setattr(
                self, f"critic{critic_id:02}", Critic(mid_dim, state_dim, action_dim)
            )

    def forward(self, state, action):
        tensor_sa = torch.cat((state, action), dim=1)
        tensor_qs = [
            getattr(self, f"critic{critic_id:02}").net(tensor_sa)  # criticID(tensor_sa)
            for critic_id in range(self.q_values_num)
        ]
        tensor_qs = torch.cat(tensor_qs, dim=1)
        return tensor_qs.mean(dim=1, keepdim=True)  # the mean of multiple Q values

    def get_q_values(self, state, action):  # todo ensemble
        tensor_sa = torch.cat((state, action), dim=1)
        tensor_qs = [
            getattr(self, f"critic{critic_id:02}").net(
                tensor_sa
            )  # criticID.net(tensor_sa)
            for critic_id in range(self.q_values_num)
        ]
        tensor_qs = torch.cat(tensor_qs, dim=1)
        return tensor_qs  # multiple Q values


class CriticMultiple(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim, if_use_dn=False):
        super().__init__()
        self.q_values_num = 16  # modified REDQ (Randomized Ensemble Double Q-learning)
        if if_use_dn:
            nn_middle = DenseNet(mid_dim)
            out_dim = nn_middle.out_dim
        else:
            nn_middle = nn.Sequential(
                nn.Linear(mid_dim, mid_dim),
                nn.ReLU(),
                nn.Linear(mid_dim, mid_dim),
                nn.ReLU(),
            )
            out_dim = mid_dim

        self.enc_s = nn.Sequential(
            nn.Linear(state_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
        )
        self.enc_a = nn.Sequential(
            nn.Linear(action_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
        )
        self.mid_n = nn_middle

        self.net_q = nn.Sequential(
            nn.Linear(out_dim, mid_dim),
            nn.Hardswish(),
            nn.Linear(mid_dim, self.q_values_num),
        )

    def forward(self, state, action):
        x = self.mid_n(
            self.enc_s(state) + self.enc_a(action)
        )  # use add instead of concatenate
        return self.net_q(x).mean(dim=1, keepdim=True)

    def get_q_values(self, state, action):
        x = self.mid_n(self.enc_s(state) + self.enc_a(action))
        return self.net_q(x)  # multiple Q values


class CriticPPO(nn.Module):
    """
    The Critic class for **PPO**.

    :param mid_dim[int]: the middle dimension of networks
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    """

    def __init__(self, mid_dim, state_dim, _action_dim):
        super().__init__()
        if isinstance(state_dim, int):
            nn_middle = nn.Sequential(
                nn.Linear(state_dim, mid_dim),
                nn.ReLU(),
            )
        else:
            nn_middle = ConvNet(
                inp_dim=state_dim[2], out_dim=mid_dim, image_size=state_dim[0]
            )

        self.net = nn.Sequential(
            nn_middle,
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.Hardswish(),
            nn.Linear(mid_dim, 1),
        )
        layer_norm(self.net[-1], std=0.5)  # output layer for advantage value

    def forward(self, state):
        """
        The forward function to ouput the value of the state.

        :param state: [tensor] the input state.
        :return: the output tensor.
        """
        return self.net(state)  # advantage value


class CriticDiscreteTwin(nn.Module):  # shared parameter
    def __init__(self, mid_dim, state_dim, action_dim, if_use_dn=False):
        super().__init__()
        if if_use_dn:
            nn_middle = DenseNet(mid_dim // 2)
            inp_dim = nn_middle.inp_dim
            out_dim = nn_middle.out_dim
        else:
            nn_middle = nn.Sequential(
                nn.Linear(mid_dim, mid_dim),
                nn.ReLU(),
                nn.Linear(mid_dim, mid_dim),
                nn.ReLU(),
            )
            inp_dim = mid_dim
            out_dim = mid_dim

        self.net_sa = nn.Sequential(
            nn.Linear(state_dim + action_dim, inp_dim),
            nn.ReLU(),
            nn_middle,
        )  # concat(state, action)
        self.net_q1 = nn.Sequential(
            nn.Linear(out_dim, mid_dim), nn.Hardswish(), nn.Linear(mid_dim, action_dim)
        )  # q1 value
        self.net_q2 = nn.Sequential(
            nn.Linear(out_dim, mid_dim), nn.Hardswish(), nn.Linear(mid_dim, action_dim)
        )  # q2 value

    def forward(self, state, action):
        tmp = self.net_sa(torch.cat((state, action), dim=1))
        return self.net_q1(tmp)  # one Q value

    def get_q1_q2(self, state):
        tmp = self.net_sa(state)
        return self.net_q1(tmp), self.net_q2(tmp)  # two Q values


class CriticBiConv(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        i_c_dim, i_h_dim, i_w_dim = state_dim  # inp_for_cnn.shape == (N, C, H, W)
        assert action_dim == int(np.prod((i_c_dim, i_w_dim, i_h_dim)))
        action_dim = (i_c_dim, i_w_dim, i_h_dim)  # (2, bs_n, ur_n)

        self.cnn_s = nn.Sequential(
            BiConvNet(mid_dim, state_dim, mid_dim * 4),
            nn.Linear(mid_dim * 4, mid_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim * 2, mid_dim * 1),
        )
        self.cnn_a = nn.Sequential(
            NnReshape(*action_dim),
            BiConvNet(mid_dim, action_dim, mid_dim * 4),
            nn.Linear(mid_dim * 4, mid_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim * 2, mid_dim * 1),
        )

        self.out_net = nn.Sequential(
            nn.Linear(mid_dim * 1, mid_dim * 1),
            nn.Hardswish(),
            nn.Linear(mid_dim * 1, 1),
        )
        layer_norm(self.out_net[-1], std=0.1)  # output layer for action

    def forward(self, state, action):
        xs = self.cnn_s(state)
        xa = self.cnn_a(action)
        return self.out_net(xs + xa)  # Q value


"""Parameter Sharing Network"""


class ShareDPG(nn.Module):  # DPG means deterministic policy gradient
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()
        nn_dense = DenseNet(mid_dim // 2)
        inp_dim = nn_dense.inp_dim
        out_dim = nn_dense.out_dim

        self.enc_s = nn.Sequential(
            nn.Linear(state_dim, mid_dim), nn.ReLU(), nn.Linear(mid_dim, inp_dim)
        )
        self.enc_a = nn.Sequential(
            nn.Linear(action_dim, mid_dim), nn.ReLU(), nn.Linear(mid_dim, inp_dim)
        )

        self.net = nn_dense

        self.dec_a = nn.Sequential(
            nn.Linear(out_dim, mid_dim),
            nn.Hardswish(),
            nn.Linear(mid_dim, action_dim),
            nn.Tanh(),
        )
        self.dec_q = nn.Sequential(
            nn.Linear(out_dim, mid_dim),
            nn.Hardswish(),
            nn.utils.spectral_norm(nn.Linear(mid_dim, 1)),
        )

    @staticmethod
    def add_noise(a, noise_std):
        a_temp = torch.normal(a, noise_std)

        mask = torch.lt(a_temp, -1) + torch.gt(
            a_temp, 1
        )  # mask = (a_temp < -1.0) + (a_temp > 1.0)
        mask = torch.tensor(mask, dtype=torch.float32).cuda()

        noise_uniform = torch.rand_like(a)
        return noise_uniform * mask + a_temp * (-mask + 1)

    def forward(self, s, noise_std=0.0):  # actor
        s_ = self.enc_s(s)
        a_ = self.net(s_)
        a = self.dec_a(a_)
        return a if noise_std == 0.0 else self.add_noise(a, noise_std)

    def critic(self, s, a):
        s_ = self.enc_s(s)
        a_ = self.enc_a(a)
        q_ = self.net(s_ + a_)
        return self.dec_q(q_)

    def next_q_action(self, s, s_next, noise_std):
        s_ = self.enc_s(s)
        a_ = self.net(s_)
        a = self.dec_a(a_)

        """q_target (without noise)"""
        a_ = self.enc_a(a)
        s_next_ = self.enc_s(s_next)
        q_target0_ = self.net(s_next_ + a_)
        q_target0 = self.dec_q(q_target0_)

        """q_target (with noise)"""
        a_noise = self.add_noise(a, noise_std)
        a_noise_ = self.enc_a(a_noise)
        q_target1_ = self.net(s_next_ + a_noise_)
        q_target1 = self.dec_q(q_target1_)

        q_target = (q_target0 + q_target1) * 0.5
        return q_target, a


class ShareSPG(nn.Module):  # SPG means stochastic policy gradient
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.log_sqrt_2pi_sum = np.log(np.sqrt(2 * np.pi)) * action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        nn_dense = DenseNet(mid_dim // 2)
        inp_dim = nn_dense.inp_dim
        out_dim = nn_dense.out_dim

        self.enc_s = nn.Sequential(
            nn.Linear(state_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, inp_dim),
        )  # state
        self.enc_a = nn.Sequential(
            nn.Linear(action_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, inp_dim),
        )  # action without nn.Tanh()

        self.net = nn_dense

        self.dec_a = nn.Sequential(
            nn.Linear(out_dim, mid_dim // 2),
            nn.ReLU(),
            nn.Linear(mid_dim // 2, action_dim),
        )  # action_mean
        self.dec_d = nn.Sequential(
            nn.Linear(out_dim, mid_dim // 2),
            nn.ReLU(),
            nn.Linear(mid_dim // 2, action_dim),
        )  # action_std_log (d means standard deviation)
        self.dec_q1 = nn.Sequential(
            nn.Linear(out_dim, mid_dim // 2),
            nn.ReLU(),
            nn.Linear(mid_dim // 2, 1),
        )  # q1 value
        self.dec_q2 = nn.Sequential(
            nn.Linear(out_dim, mid_dim // 2),
            nn.ReLU(),
            nn.Linear(mid_dim // 2, 1),
        )  # q2 value
        self.log_alpha = nn.Parameter(
            torch.zeros((1, action_dim)) - np.log(action_dim), requires_grad=True
        )

        layer_norm(self.dec_a[-1], std=0.5)
        layer_norm(self.dec_d[-1], std=0.1)
        layer_norm(self.dec_q1[-1], std=0.5)
        layer_norm(self.dec_q2[-1], std=0.5)

    def forward(self, s):
        x = self.enc_s(s)
        x = self.net(x)
        a_avg = self.dec_a(x)
        return a_avg.tanh()

    def get_action(self, s):
        s_ = self.enc_s(s)
        a_ = self.net(s_)
        a_avg = self.dec_a(a_)  # NOTICE! it is a_avg without tensor.tanh()

        a_std_log = self.dec_d(a_).clamp(-20, 2)
        a_std = a_std_log.exp()

        action = torch.normal(a_avg, a_std)  # NOTICE! it is action without .tanh()
        return action.tanh()

    def get_action_logprob(self, state):  # actor
        s_ = self.enc_s(state)
        a_ = self.net(s_)

        """add noise to action, stochastic policy"""
        a_avg = self.dec_a(a_)  # NOTICE! it is action without .tanh()
        a_std_log = self.dec_d(a_).clamp(-20, 2)
        a_std = a_std_log.exp()

        noise = torch.randn_like(a_avg, requires_grad=True)
        a_noise = a_avg + a_std * noise

        a_noise_tanh = a_noise.tanh()
        fix_term = (-a_noise_tanh.pow(2) + 1.00001).log()
        logprob = (noise.pow(2) / 2 + a_std_log + fix_term).sum(
            1, keepdim=True
        ) + self.log_sqrt_2pi_sum
        return a_noise_tanh, logprob

    def get_q_logprob(self, state):
        s_ = self.enc_s(state)
        a_ = self.net(s_)

        """add noise to action, stochastic policy"""
        a_avg = self.dec_a(a_)  # NOTICE! it is action without .tanh()
        a_std_log = self.dec_d(a_).clamp(-20, 2)
        a_std = a_std_log.exp()

        noise = torch.randn_like(a_avg, requires_grad=True)
        a_noise = a_avg + a_std * noise

        a_noise_tanh = a_noise.tanh()
        fix_term = (-a_noise_tanh.pow(2) + 1.00001).log()
        logprob = (noise.pow(2) / 2 + a_std_log + fix_term).sum(
            1, keepdim=True
        ) + self.log_sqrt_2pi_sum

        """get q"""
        a_ = self.enc_a(a_noise_tanh)
        q_ = self.net(s_ + a_)
        q = torch.min(self.dec_q1(q_), self.dec_q2(q_))
        return q, logprob

    def get_q1_q2(self, s, a):  # critic
        s_ = self.enc_s(s)
        a_ = self.enc_a(a)
        q_ = self.net(s_ + a_)
        q1 = self.dec_q1(q_)
        q2 = self.dec_q2(q_)
        return q1, q2


class SharePPO(nn.Module):  # Pixel-level state version
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()
        if isinstance(state_dim, int):
            self.enc_s = nn.Sequential(
                nn.Linear(state_dim, mid_dim), nn.ReLU(), nn.Linear(mid_dim, mid_dim)
            )  # the only difference.
        else:
            self.enc_s = ConvNet(
                inp_dim=state_dim[2], out_dim=mid_dim, image_size=state_dim[0]
            )
        out_dim = mid_dim

        self.dec_a = nn.Sequential(
            nn.Linear(out_dim, mid_dim), nn.ReLU(), nn.Linear(mid_dim, action_dim)
        )
        self.a_std_log = nn.Parameter(
            torch.zeros(1, action_dim) - 0.5, requires_grad=True
        )

        self.dec_q1 = nn.Sequential(
            nn.Linear(out_dim, mid_dim), nn.ReLU(), nn.Linear(mid_dim, 1)
        )
        self.dec_q2 = nn.Sequential(
            nn.Linear(out_dim, mid_dim), nn.ReLU(), nn.Linear(mid_dim, 1)
        )

        layer_norm(self.dec_a[-1], std=0.01)
        layer_norm(self.dec_q1[-1], std=0.01)
        layer_norm(self.dec_q2[-1], std=0.01)

        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))

    def forward(self, s):
        s_ = self.enc_s(s)
        a_avg = self.dec_a(s_)
        return a_avg.tanh()

    def get_action_noise(self, state):
        s_ = self.enc_s(state)
        a_avg = self.dec_a(s_)
        a_std = self.a_std_log.exp()

        # a_noise = torch.normal(a_avg, a_std) # same as below
        noise = torch.randn_like(a_avg)
        a_noise = a_avg + noise * a_std
        return a_noise, noise

    def get_q_logprob(self, state, noise):
        s_ = self.enc_s(state)

        q = torch.min(self.dec_q1(s_), self.dec_q2(s_))
        logprob = -(noise.pow(2) / 2 + self.a_std_log + self.sqrt_2pi_log).sum(1)
        return q, logprob

    def get_q1_q2_logprob(self, state, action):
        s_ = self.enc_s(state)

        q1, q2, a_avg, a_std = (
            self.dec_q1(s_),
            self.dec_q2(s_),
            self.dec_a(s_),
            self.a_std_log.exp(),
        )

        logprob = -(
            ((a_avg - action) / a_std).pow(2) / 2 + self.a_std_log + self.sqrt_2pi_log
        ).sum(1)
        return q1, q2, logprob


class ShareBiConv(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        i_c_dim, i_h_dim, i_w_dim = state_dim  # inp_for_cnn.shape == (N, C, H, W)
        assert action_dim == int(np.prod((i_c_dim, i_w_dim, i_h_dim)))

        state_tuple = (i_c_dim, i_h_dim, i_w_dim)
        self.enc_s = nn.Sequential(
            # NnReshape(*state_tuple),
            BiConvNet(mid_dim, state_tuple, mid_dim * 4),
        )
        action_tuple = (i_c_dim, i_w_dim, i_h_dim)
        self.enc_a = nn.Sequential(
            NnReshape(*action_tuple),
            BiConvNet(mid_dim, action_tuple, mid_dim * 4),
        )

        self.mid_n = nn.Sequential(
            nn.Linear(mid_dim * 4, mid_dim * 2),
            nn.ReLU(),
            nn.Linear(mid_dim * 2, mid_dim * 1),
            nn.ReLU(),
            DenseNet(mid_dim),
        )

        self.dec_a = nn.Sequential(
            nn.Linear(mid_dim * 4, mid_dim * 2),
            nn.Hardswish(),
            nn.Linear(mid_dim * 2, action_dim),
        )
        layer_norm(self.dec_a[-1], std=0.1)  # output layer for action
        self.dec_q = nn.Sequential(
            nn.Linear(mid_dim * 4, mid_dim * 2),
            nn.Hardswish(),
            nn.Linear(mid_dim * 2, 1),
        )
        layer_norm(self.dec_q[-1], std=0.1)  # output layer for action

    def forward(self, state):  # actor
        xs = self.enc_s(state)
        xn = self.mid_n(xs)
        action = self.dec_a(xn)
        return action * torch.pow(
            (action**2).sum(), -0.5
        )  # action / sqrt(L2_norm(action))

    def critic(self, state, action):
        xs = self.enc_s(state)
        xa = self.enc_a(action)
        xn = self.mid_n(xs + xa)
        return self.dec_q(xn)  # Q value

    def get_action(self, state, action_std):  # actor, get noisy action
        xs = self.enc_s(state)
        xn = self.mid_n(xs)
        action = self.dec_a(xn)
        action = action + (torch.randn_like(action) * action_std)
        return action * torch.pow(
            (action**2).sum(), -0.5
        )  # action / sqrt(L2_norm(action))


"""MARL (QMix: Q value and Mixing network)"""


class QMix(nn.Module):
    """
    Mixer network for QMix. Outputs total q value given independent q value and states.
    """

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim
        self.abs = getattr(self.args, "abs", True)

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(
                nn.Linear(self.state_dim, hypernet_embed),
                nn.ReLU(inplace=True),
                nn.Linear(hypernet_embed, self.embed_dim * self.n_agents),
            )
            self.hyper_w_final = nn.Sequential(
                nn.Linear(self.state_dim, hypernet_embed),
                nn.ReLU(inplace=True),
                nn.Linear(hypernet_embed, self.embed_dim),
            )
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, 1),
        )

    def forward(self, agent_qs, states):
        """
        Compute total q value for QMix.

        :param agent_qs: independent q value.
        :param states: state of agents
        :return total q value:
        """
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.reshape(-1, 1, self.n_agents)
        # First layer
        w1 = self.hyper_w_1(states).abs() if self.abs else self.hyper_w_1(states)
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)

        # Second layer
        w_final = (
            self.hyper_w_final(states).abs() if self.abs else self.hyper_w_final(states)
        )
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = torch.bmm(hidden, w_final) + v
        # Reshape and return
        return y.view(bs, -1, 1)

    def k(self, states):
        bs = states.size(0)
        w1 = torch.abs(self.hyper_w_1(states))
        w_final = torch.abs(self.hyper_w_final(states))
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        w_final = w_final.view(-1, self.embed_dim, 1)
        k = torch.bmm(w1, w_final).view(bs, -1, self.n_agents)
        k = k / torch.sum(k, dim=2, keepdim=True)
        return k

    def b(self, states):
        # bs = states.size(0)
        w_final = torch.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        b1 = self.hyper_b_1(states)
        b1 = b1.view(-1, 1, self.embed_dim)
        v = self.V(states).view(-1, 1, 1)
        return torch.bmm(b1, w_final) + v


class VDN(nn.Module):
    """
    Mixer network for VDN. Outputs total q value given independent q value.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(agent_qs, _batch):
        """
        Compute total q value for VDN.

        :param agent_qs: independent q value.
        :return total q value:
        """
        return torch.sum(agent_qs, dim=2, keepdim=True)


"""MARL (CTDE: Centralized Training with Decentralized Execution)"""


class ActorMAPPO(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.

    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super().__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_size,
                self.hidden_size,
                self._recurrent_N,
                self._use_orthogonal,
            )

        self.act = ACTLayer(
            action_space, self.hidden_size, self._use_orthogonal, self._gain
        )

        self.to(device)

    def forward(
        self, obs, rnn_states, masks, available_actions=None, deterministic=False
    ):
        """
        Compute actions from the given inputs.

        :param obs: [tensor] observation inputs into network.
        :param rnn_states: [tensor] if RNN network, hidden states for RNN.
        :param masks: [tensor] mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: [tensor] denotes which actions are available to agent (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actions, action_log_probs = self.act(
            actor_features, available_actions, deterministic
        )

        return actions, action_log_probs, rnn_states

    def evaluate_actions(
        self, obs, rnn_states, action, masks, available_actions=None, active_masks=None
    ):
        """
        Compute log probability and entropy of given actions.

        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy = self.act.evaluate_actions(
            actor_features,
            action,
            available_actions,
            active_masks=active_masks if self._use_policy_active_masks else None,
        )

        return action_log_probs, dist_entropy


class CriticMAPPO(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or local observations (IPPO).

    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super().__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][
            self._use_orthogonal
        ]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        base = CNNBase if len(cent_obs_shape) == 3 else MLPBase
        self.base = base(args, cent_obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_size,
                self.hidden_size,
                self._recurrent_N,
                self._use_orthogonal,
            )

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        """
        Compute actions from the given inputs.

        :param cent_obs: [tensor] observation inputs into network.
        :param rnn_states: [tensor] if RNN network, hidden states for RNN.
        :param masks: [tensor] mask tensor denoting if RNN states should be reinitialized to zeros.
        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(cent_obs)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        values = self.v_out(critic_features)

        return values, rnn_states


"""utils"""


class NnReshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.view((x.size(0),) + self.args)


class DenseNet(nn.Module):  # plan to hyper-param: layer_number
    def __init__(self, lay_dim):
        super().__init__()
        self.dense1 = nn.Sequential(nn.Linear(lay_dim * 1, lay_dim * 1), nn.Hardswish())
        self.dense2 = nn.Sequential(nn.Linear(lay_dim * 2, lay_dim * 2), nn.Hardswish())
        self.inp_dim = lay_dim
        self.out_dim = lay_dim * 4

    def forward(self, x1):  # x1.shape==(-1, lay_dim*1)
        x2 = torch.cat((x1, self.dense1(x1)), dim=1)
        return torch.cat(
            (x2, self.dense2(x2)), dim=1
        )  # x3  # x2.shape==(-1, lay_dim*4)


class ConcatNet(nn.Module):  # concatenate
    def __init__(self, lay_dim):
        super().__init__()
        self.dense1 = nn.Sequential(
            nn.Linear(lay_dim, lay_dim),
            nn.ReLU(),
            nn.Linear(lay_dim, lay_dim),
            nn.Hardswish(),
        )
        self.dense2 = nn.Sequential(
            nn.Linear(lay_dim, lay_dim),
            nn.ReLU(),
            nn.Linear(lay_dim, lay_dim),
            nn.Hardswish(),
        )
        self.dense3 = nn.Sequential(
            nn.Linear(lay_dim, lay_dim),
            nn.ReLU(),
            nn.Linear(lay_dim, lay_dim),
            nn.Hardswish(),
        )
        self.dense4 = nn.Sequential(
            nn.Linear(lay_dim, lay_dim),
            nn.ReLU(),
            nn.Linear(lay_dim, lay_dim),
            nn.Hardswish(),
        )
        self.inp_dim = lay_dim
        self.out_dim = lay_dim * 4

    def forward(self, x0):
        x1 = self.dense1(x0)
        x2 = self.dense2(x0)
        x3 = self.dense3(x0)
        x4 = self.dense4(x0)
        return torch.cat((x1, x2, x3, x4), dim=1)


class ConvNet(nn.Module):  # pixel-level state encoder
    def __init__(self, inp_dim, out_dim, image_size=224):
        super().__init__()
        if image_size == 224:
            self.net = nn.Sequential(  # size==(batch_size, inp_dim, 224, 224)
                nn.Conv2d(inp_dim, 32, (5, 5), stride=(2, 2), bias=False),
                nn.ReLU(inplace=True),  # size=110
                nn.Conv2d(32, 48, (3, 3), stride=(2, 2)),
                nn.ReLU(inplace=True),  # size=54
                nn.Conv2d(48, 64, (3, 3), stride=(2, 2)),
                nn.ReLU(inplace=True),  # size=26
                nn.Conv2d(64, 96, (3, 3), stride=(2, 2)),
                nn.ReLU(inplace=True),  # size=12
                nn.Conv2d(96, 128, (3, 3), stride=(2, 2)),
                nn.ReLU(inplace=True),  # size=5
                nn.Conv2d(128, 192, (5, 5), stride=(1, 1)),
                nn.ReLU(inplace=True),  # size=1
                NnReshape(-1),  # size (batch_size, 1024, 1, 1) ==> (batch_size, 1024)
                nn.Linear(192, out_dim),  # size==(batch_size, out_dim)
            )
        elif image_size == 112:
            self.net = nn.Sequential(  # size==(batch_size, inp_dim, 112, 112)
                nn.Conv2d(inp_dim, 32, (5, 5), stride=(2, 2), bias=False),
                nn.ReLU(inplace=True),  # size=54
                nn.Conv2d(32, 48, (3, 3), stride=(2, 2)),
                nn.ReLU(inplace=True),  # size=26
                nn.Conv2d(48, 64, (3, 3), stride=(2, 2)),
                nn.ReLU(inplace=True),  # size=12
                nn.Conv2d(64, 96, (3, 3), stride=(2, 2)),
                nn.ReLU(inplace=True),  # size=5
                nn.Conv2d(96, 128, (5, 5), stride=(1, 1)),
                nn.ReLU(inplace=True),  # size=1
                NnReshape(-1),  # size (batch_size, 1024, 1, 1) ==> (batch_size, 1024)
                nn.Linear(128, out_dim),  # size==(batch_size, out_dim)
            )
        else:
            assert image_size in {224, 112}

    def forward(self, x):
        # assert x.shape == (batch_size, inp_dim, image_size, image_size)
        x = x.permute(0, 3, 1, 2)
        x = x / 128.0 - 1.0
        return self.net(x)

    # @staticmethod
    # def check():
    #     inp_dim = 3
    #     out_dim = 32
    #     batch_size = 2
    #     image_size = [224, 112][1]
    #     # from elegantrl.net import Conv2dNet
    #     net = Conv2dNet(inp_dim, out_dim, image_size)
    #
    #     x = torch.ones((batch_size, image_size, image_size, inp_dim), dtype=torch.uint8) * 255
    #     print(x.shape)
    #     y = net(x)
    #     print(y.shape)


class BiConvNet(nn.Module):
    def __init__(self, mid_dim, inp_dim, out_dim):
        super().__init__()
        i_c_dim, i_h_dim, i_w_dim = inp_dim  # inp_for_cnn.shape == (N, C, H, W)

        self.cnn_h = nn.Sequential(
            nn.Conv2d(i_c_dim * 1, mid_dim * 2, (1, i_w_dim), bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_dim * 2, mid_dim * 1, (1, 1), bias=True),
            nn.ReLU(inplace=True),
            NnReshape(-1),  # shape=(-1, i_h_dim * mid_dim)
            nn.Linear(i_h_dim * mid_dim, out_dim),
        )
        self.cnn_w = nn.Sequential(
            nn.Conv2d(i_c_dim * 1, mid_dim * 2, (i_h_dim, 1), bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_dim * 2, mid_dim * 1, (1, 1), bias=True),
            nn.ReLU(inplace=True),
            NnReshape(-1),  # shape=(-1, i_w_dim * mid_dim)
            nn.Linear(i_w_dim * mid_dim, out_dim),
        )

    def forward(self, state):
        xh = self.cnn_h(state)
        xw = self.cnn_w(state)
        return xw + xh


class ActorSimplify:
    def __init__(self, gpu_id, actor_net):
        self.device = torch.device(
            f"cuda:{gpu_id}" if (torch.cuda.is_available() and gpu_id >= 0) else "cpu"
        )
        self.actor_net = actor_net.to(self.device)

    def get_action(self, state: np.ndarray) -> np.ndarray:
        states = torch.as_tensor(
            state[np.newaxis], dtype=torch.float32, device=self.device
        )
        action = self.actor_net(states)[0]
        return action.detach().cpu().numpy()


class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        if getattr(args, "use_layer_norm", False):
            self.layer_norm = LayerNorm(args.rnn_hidden_dim)

        if getattr(args, "use_orthogonal", False):
            orthogonal_init_(self.fc1)
            orthogonal_init_(self.fc2, gain=args.gain)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        b, a, e = inputs.size()

        inputs = inputs.view(-1, e)
        x = F.relu(self.fc1(inputs), inplace=True)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        hh = self.rnn(x, h_in)

        if getattr(self.args, "use_layer_norm", False):
            q = self.fc2(self.layer_norm(hh))
        else:
            q = self.fc2(hh)

        return q.view(b, a, -1), hh.view(b, a, -1)


def layer_norm(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
