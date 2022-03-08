import torch
import torch.nn as nn
import numpy as np
import numpy.random as rd

"""DQN"""


class QNet(nn.Module):  # nn.Module is a standard PyTorch Network
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
        return self.net(state)  # Q values for multiple actions

    def get_action(self, state):
        if rd.rand() > self.explore_rate:
            action = self.net(state).argmax(dim=1, keepdim=True)
        else:
            action = torch.randint(self.action_dim, size=(state.shape[0], 1))
        return action


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
        self.explore_rate = 0.125
        self.action_dim = action_dim

    def forward(self, state):
        """
        The forward function for **Dueling Q-network**.

        :param state: [tensor] the input state.
        :return: the output tensor.
        """
        s_tmp = self.net_state(state)  # encoded state
        q_val = self.net_val(s_tmp)
        q_adv = self.net_adv(s_tmp)
        return q_val - q_val.mean(dim=1, keepdim=True) + q_adv  # dueling Q value

    def get_action(self, state):
        if rd.rand() > self.explore_rate:
            s_tmp = self.net_state(state)
            q_val = self.net_val(s_tmp)
            action = q_val.argmax(dim=1, keepdim=True)
        else:
            action = torch.randint(self.action_dim, size=(state.shape[0], 1))
        return action


class QNetTwin(nn.Module):  # Double DQN
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net_state = nn.Sequential(
            nn.Linear(state_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
        )
        self.net_q1 = nn.Sequential(
            nn.Linear(mid_dim, mid_dim), nn.ReLU(), nn.Linear(mid_dim, action_dim)
        )  # q1 value
        self.net_q2 = nn.Sequential(
            nn.Linear(mid_dim, mid_dim), nn.ReLU(), nn.Linear(mid_dim, action_dim)
        )  # q2 value
        self.explore_rate = 0.125
        self.action_dim = action_dim
        self.soft_max = nn.Softmax(dim=1)

    def forward(self, state):
        tmp = self.net_state(state)
        return self.net_q1(tmp)  # one group of Q values

    def get_q1_q2(self, state):
        tmp = self.net_state(state)
        return self.net_q1(tmp), self.net_q2(tmp)  # two groups of Q values

    def get_action(self, state):
        s = self.net_state(state)
        q = self.net_q1(s)
        if rd.rand() > self.explore_rate:
            action = q.argmax(dim=1, keepdim=True)
        else:
            a_prob = self.soft_max(q)
            action = torch.multinomial(a_prob, num_samples=1)
        return action


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
        self.net_val1 = nn.Sequential(
            nn.Linear(mid_dim, mid_dim), nn.Hardswish(), nn.Linear(mid_dim, action_dim)
        )  # q1 value
        self.net_val2 = nn.Sequential(
            nn.Linear(mid_dim, mid_dim), nn.Hardswish(), nn.Linear(mid_dim, action_dim)
        )  # q2 value
        self.net_adv1 = nn.Sequential(
            nn.Linear(mid_dim, mid_dim), nn.Hardswish(), nn.Linear(mid_dim, 1)
        )  # advantage function value 1
        self.net_adv2 = nn.Sequential(
            nn.Linear(mid_dim, mid_dim), nn.Hardswish(), nn.Linear(mid_dim, 1)
        )  # advantage function value 1
        self.explore_rate = 0.125
        self.action_dim = action_dim
        self.soft_max = nn.Softmax(dim=1)

    def forward(self, state):
        """
        The forward function for **Dueling Double DQN**.

        :param state: [tensor] the input state.
        :return: the output tensor.
        """
        t_tmp = self.net_state(state)
        q_val = self.net_val1(t_tmp)
        q_adv = self.net_adv1(t_tmp)
        return q_val - q_val.mean(dim=1, keepdim=True) + q_adv  # one dueling Q value

    def get_q1_q2(self, state):
        """
        TBD
        """
        s_tmp = self.net_state(state)

        q_val1 = self.net_val1(s_tmp)
        q_adv1 = self.net_adv1(s_tmp)
        q_duel1 = q_val1 - q_val1.mean(dim=1, keepdim=True) + q_adv1

        q_val2 = self.net_val2(s_tmp)
        q_adv2 = self.net_adv2(s_tmp)
        q_duel2 = q_val2 - q_val2.mean(dim=1, keepdim=True) + q_adv2
        return q_duel1, q_duel2  # two dueling Q values

    def get_action(self, state):
        s = self.net_state(state)
        q = self.net_val1(s)
        if rd.rand() > self.explore_rate:
            action = q.argmax(dim=1, keepdim=True)
        else:
            a_prob = self.soft_max(q)
            action = torch.multinomial(a_prob, num_samples=1)
        return action


"""Actor (policy network)"""


class Actor(nn.Module):
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
        self.explore_noise = 0.1  # standard deviation of exploration action noise

    def forward(self, state):
        return self.net(state).tanh()  # action.tanh()

    def get_action(self, state):  # for exploration
        action = self.net(state).tanh()
        noise = (torch.randn_like(action) * self.explore_noise).clamp(-0.5, 0.5)
        return (action + noise).clamp(-1.0, 1.0)

    def get_action_noise(self, state, action_std):
        action = self.net(state).tanh()
        noise = (torch.randn_like(action) * action_std).clamp(-0.5, 0.5)
        return (action + noise).clamp(-1.0, 1.0)


class ActorSAC(nn.Module):
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
        tmp = self.net_state(state)
        return self.net_a_avg(tmp).tanh()  # action

    def get_action(self, state):
        t_tmp = self.net_state(state)
        a_avg = self.net_a_avg(t_tmp)  # NOTICE! it is a_avg without .tanh()
        a_std = self.net_a_std(t_tmp).clamp(-20, 2).exp()
        return torch.normal(a_avg, a_std).tanh()  # re-parameterize

    def get_action_logprob(self, state):
        t_tmp = self.net_state(state)
        a_avg = self.net_a_avg(t_tmp)  # NOTICE! it needs a_avg.tanh()
        a_std_log = self.net_a_std(t_tmp).clamp(-20, 2)
        a_std = a_std_log.exp()

        noise = torch.randn_like(a_avg, requires_grad=True)
        a_tan = (a_avg + a_std * noise).tanh()  # action.tanh()

        logprob = (
            a_std_log + self.log_sqrt_2pi + noise.pow(2).__mul__(0.5)
        )  # noise.pow(2) * 0.5
        logprob = (
            logprob + (-a_tan.pow(2) + 1.000001).log()
        )  # fix logprob using the derivative of action.tanh()
        return a_tan, logprob.sum(1, keepdim=True)  # todo negative logprob


class ActorFixSAC(nn.Module):
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
        self.soft_plus = nn.Softplus()

    def forward(self, state):
        tmp = self.net_state(state)
        return self.net_a_avg(tmp).tanh()  # action

    def get_action(self, state):
        t_tmp = self.net_state(state)
        a_avg = self.net_a_avg(t_tmp)  # NOTICE! it is a_avg without .tanh()
        a_std = self.net_a_std(t_tmp).clamp(-20, 2).exp()
        return torch.normal(a_avg, a_std).tanh()  # re-parameterize

    def get_a_log_std(self, state):
        t_tmp = self.net_state(state)
        a_log_std = self.net_a_std(t_tmp).clamp(-20, 2).exp()
        return a_log_std

    def get_logprob(self, state, action):
        t_tmp = self.net_state(state)
        a_avg = self.net_a_avg(t_tmp)  # NOTICE! it needs a_avg.tanh()
        a_std_log = self.net_a_std(t_tmp).clamp(-20, 2)
        a_std = a_std_log.exp()

        """add noise to a_noise in stochastic policy"""
        a_noise = a_avg + a_std * torch.randn_like(a_avg, requires_grad=True)
        noise = a_noise - action  # todo

        log_prob = (
            a_std_log + self.log_sqrt_2pi + noise.pow(2).__mul__(0.5)
        )  # noise.pow(2) * 0.5
        log_prob += (
            np.log(2.0) - a_noise - self.soft_plus(-2.0 * a_noise)
        ) * 2.0  # better than below
        return log_prob

    def get_action_logprob(self, state):
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
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, mid_dim),
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
        return self.net(state).tanh()  # action.tanh()

    def get_action(self, state):
        a_avg = self.net(state)
        a_std = self.a_std_log.exp()

        noise = torch.randn_like(a_avg)
        action = a_avg + noise * a_std
        return action, noise

    def get_logprob(self, state, action):
        a_avg = self.net(state)
        a_std = self.a_std_log.exp()

        delta = ((a_avg - action) / a_std).pow(2) * 0.5
        log_prob = -(self.a_std_log + self.sqrt_2pi_log + delta)  # new_logprob

        return log_prob

    def get_logprob_entropy(self, state, action):
        a_avg = self.net(state)
        a_std = self.a_std_log.exp()

        delta = ((a_avg - action) / a_std).pow(2) * 0.5
        logprob = -(self.a_std_log + self.sqrt_2pi_log + delta).sum(1)  # new_logprob

        dist_entropy = (logprob.exp() * logprob).mean()  # policy entropy
        return logprob, dist_entropy

    def get_old_logprob(self, _action, noise):  # noise = action - a_noise
        delta = noise.pow(2) * 0.5
        return -(self.a_std_log + self.sqrt_2pi_log + delta).sum(1)  # old_logprob

    @staticmethod
    def get_a_to_e(action):
        return action.tanh()


class ActorDiscretePPO(nn.Module):
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
        return self.net(state)  # action_prob without softmax

    def get_action(self, state):
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


"""Critic (value network)"""


class Critic(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat((state, action), dim=1))  # q value


class CriticPPO(nn.Module):
    def __init__(self, mid_dim, state_dim, _action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, 1),
        )

    def forward(self, state):
        return self.net(state)  # advantage value


class CriticTwin(nn.Module):  # shared parameter
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
        return torch.add(*self.get_q1_q2(state, action)) / 2.0  # mean Q value

    def get_q_min(self, state, action):
        return torch.min(*self.get_q1_q2(state, action))  # min Q value

    def get_q1_q2(self, state, action):
        tmp = self.net_sa(torch.cat((state, action), dim=1))
        return self.net_q1(tmp), self.net_q2(tmp)  # two Q values


class CriticREDq(nn.Module):  # modified REDQ (Randomized Ensemble Double Q-learning)
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.critic_num = 8
        self.critic_list = list()
        for critic_id in range(self.critic_num):
            child_cri_net = Critic(mid_dim, state_dim, action_dim).net
            setattr(self, f"critic{critic_id:02}", child_cri_net)
            self.critic_list.append(child_cri_net)

    def forward(self, state, action):
        return self.get_q_values(state, action).mean(
            dim=1, keepdim=True
        )  # mean Q value

    def get_q_min(self, state, action):
        tensor_qs = self.get_q_values(state, action)
        q_min = torch.min(tensor_qs, dim=1, keepdim=True)[0]  # min Q value
        q_sum = tensor_qs.sum(dim=1, keepdim=True)  # mean Q value
        return (q_min * (self.critic_num * 0.5) + q_sum) / (
            self.critic_num * 1.5
        )  # better than min

    def get_q_values(self, state, action):
        tensor_sa = torch.cat((state, action), dim=1)
        tensor_qs = [cri_net(tensor_sa) for cri_net in self.critic_list]
        tensor_qs = torch.cat(tensor_qs, dim=1)
        return tensor_qs  # multiple Q values


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


class NnReshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.view((x.size(0),) + self.args)


def layer_norm(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
