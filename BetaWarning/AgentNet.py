import torch
import torch.nn as nn  # import torch.nn.functional as F
import numpy as np  # import numpy.random as rd

"""ZenJiaHao, GitHub: YonV1943 ElegantRL (Pytorch model-free DRL)
Issay, Easy Essay, EAsy esSAY 谐音: 意识
"""


class InterDPG(nn.Module):  # class AgentIntelAC
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()
        self.enc_s = nn.Sequential(
            nn.Linear(state_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
        )
        self.enc_a = nn.Sequential(
            nn.Linear(action_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
        )

        self.net = DenseNet(mid_dim)
        net_out_dim = self.net.out_dim

        self.dec_a = nn.Sequential(nn.Linear(net_out_dim, mid_dim), HardSwish(),
                                   nn.Linear(mid_dim, action_dim), nn.Tanh(), )
        self.dec_q = nn.Sequential(nn.Linear(net_out_dim, mid_dim), HardSwish(),
                                   nn.utils.spectral_norm(nn.Linear(mid_dim, 1)), )

    @staticmethod
    def add_noise(a, noise_std):  # 2020-03-03
        # noise_normal = torch.randn_like(a) * noise_std
        # a_temp = a + noise_normal
        a_temp = torch.normal(a, noise_std)
        mask = ((a_temp < -1.0) + (a_temp > 1.0)).type(torch.float32)  # 2019-12-30

        noise_uniform = torch.rand_like(a)
        a_noise = noise_uniform * mask + a_temp * (-mask + 1)
        return a_noise

    def forward(self, s, noise_std=0.0):  # actor
        s_ = self.enc_s(s)
        a_ = self.net(s_)
        a = self.dec_a(a_)
        return a if noise_std == 0.0 else self.add_noise(a, noise_std)

    def critic(self, s, a):
        s_ = self.enc_s(s)
        a_ = self.enc_a(a)
        q_ = self.net(s_ + a_)
        q = self.dec_q(q_)
        return q

    def next__q_a(self, s, s_next, noise_std):
        s_ = self.enc_s(s)
        a_ = self.net(s_)
        a = self.dec_a(a_)

        '''q_target (without noise)'''
        a_ = self.enc_a(a)
        s_next_ = self.enc_s(s_next)
        q_target0_ = self.net(s_next_ + a_)
        q_target0 = self.dec_q(q_target0_)

        '''q_target (with noise)'''
        a_noise = self.add_noise(a, noise_std)
        a_noise_ = self.enc_a(a_noise)
        q_target1_ = self.net(s_next_ + a_noise_)
        q_target1 = self.dec_q(q_target1_)

        q_target = (q_target0 + q_target1) * 0.5
        return q_target, a


class InterSPG(nn.Module):  # 2020-1111 class AgentIntelAC for SAC (SPG means stochastic policy gradient)
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()
        self.log_std_min = -20
        self.log_std_max = 2
        self.constant_log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.enc_s = nn.Sequential(
            nn.Linear(state_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
        )  # state
        self.enc_a = nn.Sequential(
            nn.Linear(action_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
        )  # action without nn.Tanh()

        self.net = DenseNet(mid_dim)
        net_out_dim = self.net.out_dim

        self.dec_a = nn.Sequential(
            nn.Linear(net_out_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, action_dim),
        )  # action_mean
        self.dec_d = nn.Sequential(
            nn.Linear(net_out_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, action_dim),
        )  # action_std_log (d means standard deviation)
        self.dec_q1 = nn.Sequential(
            nn.Linear(net_out_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, 1),
        )  # q_value1 SharedTwinCritic
        self.dec_q2 = nn.Sequential(
            nn.Linear(net_out_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, 1),
        )  # q_value2 SharedTwinCritic

        layer_norm(self.dec_a[-1], std=0.1)  # net[-1] is output layer for action, it is no necessary.
        layer_norm(self.dec_q1[-1], std=0.1)
        layer_norm(self.dec_q2[-1], std=0.1)

    def forward(self, s):
        x = self.enc_s(s)
        x = self.net(x)
        a_avg = self.dec_a(x)
        return a_avg.tanh()

    def get_explored_action(self, s):
        s_ = self.enc_s(s)
        a_ = self.net(s_)
        a_avg = self.dec_a(a_)  # NOTICE! it is a_avg without tensor.tanh()

        a_std_log = self.dec_d(a_).clamp(self.log_std_min, self.log_std_max)
        a_std = a_std_log.exp()

        action = torch.normal(a_avg, a_std)  # NOTICE! it is action without .tanh()
        return action.tanh()

    def get__a__log_prob(self, state):  # actor
        s_ = self.enc_s(state)
        a_ = self.net(s_)

        """add noise to action, stochastic policy"""
        a_avg = self.dec_a(a_)  # NOTICE! it is action without .tanh()
        a_std_log = self.dec_d(a_).clamp(self.log_std_min, self.log_std_max)
        a_std = a_std_log.exp()

        noise = torch.randn_like(a_avg, requires_grad=True)
        a_noise = a_avg + a_std * noise

        a_noise_tanh = a_noise.tanh()
        fix_term = (-a_noise_tanh.pow(2) + 1.00001).log()
        log_prob = (noise.pow(2) / 2 + a_std_log + fix_term).sum(1, keepdim=True)
        return a_noise_tanh, log_prob

    def get__q__log_prob(self, state):
        s_ = self.enc_s(state)
        a_ = self.net(s_)

        """add noise to action, stochastic policy"""
        a_avg = self.dec_a(a_)  # NOTICE! it is action without .tanh()
        a_std_log = self.dec_d(a_).clamp(self.log_std_min, self.log_std_max)
        a_std = a_std_log.exp()

        noise = torch.randn_like(a_avg, requires_grad=True)
        a_noise = a_avg + a_std * noise

        a_noise_tanh = a_noise.tanh()
        fix_term = (-a_noise_tanh.pow(2) + 1.00001).log()
        log_prob = (noise.pow(2) / 2 + a_std_log + fix_term).sum(1, keepdim=True)

        '''get q'''
        a_ = self.enc_a(a_noise_tanh)
        q_ = self.net(s_ + a_)
        q = torch.min(self.dec_q1(q_), self.dec_q2(q_))
        return q, log_prob

    def get__q1_q2(self, s, a):  # critic
        s_ = self.enc_s(s)
        a_ = self.enc_a(a)
        q_ = self.net(s_ + a_)
        q1 = self.dec_q1(q_)
        q2 = self.dec_q2(q_)
        return q1, q2


class InterGAE(nn.Module):  # 2020-10-10
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()
        self.log_std_min = -20
        self.log_std_max = 2
        self.constant_log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # encoder
        self.enc_s = nn.Sequential(
            nn.Linear(state_dim, mid_dim), nn.ReLU(),
        )  # state

        '''use densenet'''
        self.net = DenseNet(mid_dim)
        net_out_dim = self.net.out_dim
        # '''not use densenet'''
        # self.net = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(), )
        # net_out_dim = mid_dim

        '''todo two layer'''
        self.dec_a = nn.Sequential(
            nn.Linear(net_out_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, action_dim),
        )  # action_mean
        self.dec_d = nn.Sequential(
            nn.Linear(net_out_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, action_dim),
        )  # action_std_log (d means standard dev.)

        self.dec_q1 = nn.Sequential(
            nn.Linear(net_out_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, 1),
        )  # q_value1 SharedTwinCritic
        self.dec_q2 = nn.Sequential(
            nn.Linear(net_out_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, 1),
        )  # q_value2 SharedTwinCritic
        '''todo one layer'''
        # self.dec_a = nn.Sequential(
        #     nn.Linear(net_out_dim, action_dim),
        # )  # action_mean
        # self.dec_d = nn.Sequential(
        #     nn.Linear(net_out_dim, action_dim),
        # )  # action_std_log (d means standard dev.)
        #
        # self.dec_q1 = nn.Sequential(
        #     nn.Linear(net_out_dim, 1),
        # )  # q_value1 SharedTwinCritic
        # self.dec_q2 = nn.Sequential(
        #     nn.Linear(net_out_dim, 1),
        # )  # q_value2 SharedTwinCritic

        # layer_norm(self.net[0], std=1.0)
        layer_norm(self.dec_a[-1], std=0.01)  # output layer for action
        layer_norm(self.dec_d[-1], std=0.01)  # output layer for std_log
        layer_norm(self.dec_q1[-1], std=0.1)  # output layer for q value
        layer_norm(self.dec_q1[-1], std=0.1)  # output layer for q value

    def forward(self, s):
        x = self.enc_s(s)
        x = self.net(x)
        a_mean = self.dec_a(x)
        return a_mean.tanh()

    def get__a__log_prob(self, state):
        x = self.enc_s(state)
        x = self.net(x)
        a_mean = self.dec_a(x)
        a_log_std = self.dec_d(x).clamp(self.log_std_min, self.log_std_max)
        a_std = torch.exp(a_log_std)

        a_noise = a_mean + a_std * torch.randn_like(a_mean, requires_grad=True, device=self.device)

        a_delta = (a_noise - a_mean).pow(2) / (2 * a_std.pow(2))
        log_prob = -(a_delta + a_log_std + self.constant_log_sqrt_2pi)
        return a_noise, log_prob.sum(1)

    def compute__log_prob(self, state, a_noise):
        x = self.enc_s(state)
        x = self.net(x)
        a_mean = self.dec_a(x)
        a_log_std = self.dec_d(x).clamp(self.log_std_min, self.log_std_max)
        a_std = torch.exp(a_log_std)

        a_delta = (a_noise - a_mean).pow(2) / (2 * a_std.pow(2))
        log_prob = -(a_delta + a_log_std + self.constant_log_sqrt_2pi)
        log_prob = log_prob.sum(1)
        return log_prob

    def get__q1_q2(self, s):
        x = self.enc_s(s)
        x = self.net(x)
        q1 = self.dec_q1(x)
        q2 = self.dec_q2(x)
        return q1, q2


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 # nn.BatchNorm1d(mid_dim),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, action_dim), nn.Tanh(), )

    def forward(self, s, noise_std=0.0):
        a = self.net(s)
        return a if noise_std == 0 else self.add_noise(a, noise_std)

    @staticmethod
    def add_noise(action, noise_std):
        normal_noise = (torch.randn_like(action) * noise_std).clamp(-0.5, 0.5)
        a_noise = (action + normal_noise).clamp(-1.0, 1.0)
        return a_noise


class ActorDN(nn.Module):  # dn: DenseNet
    def __init__(self, state_dim, action_dim, mid_dim, use_dn):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if use_dn:  # use DenseNet (there are both shallow and deep network in DenseNet)
            nn_dense_net = DenseNet(mid_dim)
            self.net = nn.Sequential(
                nn.Linear(state_dim, mid_dim), nn.ReLU(),
                nn_dense_net,
                nn.Linear(nn_dense_net.out_dim, action_dim),
            )
        else:  # use a simple network for actor. In RL, deeper network does not mean better performance.
            self.net = nn.Sequential(
                nn.Linear(state_dim, mid_dim), nn.ReLU(),
                nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                nn.Linear(mid_dim, action_dim),
            )

        layer_norm(self.net[-1], std=0.01)  # net[-1] is output layer for action, it is no necessary.

    def forward(self, s, noise_std=0.0):
        a = self.net(s)
        return a if noise_std == 0.0 else self.add_noise(a, noise_std)

    @staticmethod
    def add_noise(a, noise_std):  # 2020-03-03
        # noise_normal = torch.randn_like(a, device=self.device) * noise_std
        # a_temp = a + noise_normal
        a_temp = torch.normal(a, noise_std)
        mask = ((a_temp < -1.0) + (a_temp > 1.0)).type(torch.float32)  # 2019-12-30

        noise_uniform = torch.rand_like(a)  # , device=self.device)
        a_noise = noise_uniform * mask + a_temp * (-mask + 1)
        return a_noise


class ActorSAC(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim, use_dn):
        super().__init__()

        if use_dn:  # use DenseNet (DenseNet has both shallow and deep linear layer)
            nn_dense_net = DenseNet(mid_dim)
            self.net__mid = nn.Sequential(
                nn.Linear(state_dim, mid_dim), nn.ReLU(),
                nn_dense_net,
            )
            lay_dim = nn_dense_net.out_dim
        else:  # use a simple network for actor. Deeper network does not mean better performance in RL.
            self.net__mid = nn.Sequential(
                nn.Linear(state_dim, mid_dim), nn.ReLU(),
                nn.Linear(mid_dim, mid_dim),
            )
            lay_dim = mid_dim

        self.net__mean = nn.Linear(lay_dim, action_dim)
        self.net__std_log = nn.Linear(lay_dim, action_dim)

        layer_norm(self.net__mean, std=0.01)  # net[-1] is output layer for action, it is no necessary.

        self.log_std_min = -20
        self.log_std_max = 2
        self.constant_log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, state, noise_std=0.0):  # in fact, noise_std is a boolean
        x = self.net__mid(state)
        a_mean = self.net__mean(x)  # NOTICE! it is a_mean without .tanh()

        if noise_std != 0.0:
            a_std_log = self.net__std_log(x).clamp(self.log_std_min, self.log_std_max)
            a_std = a_std_log.exp()
            a_mean = torch.normal(a_mean, a_std)  # NOTICE! it needs .tanh()

        return a_mean.tanh()

    def get__a__log_prob(self, state):
        x = self.net__mid(state)
        a_mean = self.net__mean(x)  # NOTICE! it needs a_mean.tanh()
        a_std_log = self.net__std_log(x).clamp(self.log_std_min, self.log_std_max)
        a_std = a_std_log.exp()

        """add noise to action in stochastic policy"""
        a_noise = a_mean + a_std * torch.randn_like(a_mean, requires_grad=True, device=self.device)
        # Can only use above code instead of below, because the tensor need gradients here.
        # a_noise = torch.normal(a_mean, a_std, requires_grad=True)

        '''compute log_prob according to mean and std of action (stochastic policy)'''
        a_delta = ((a_noise - a_mean) / a_std).pow(2) * 0.5
        # self.constant_log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))
        log_prob_noise = a_delta + a_std_log + self.constant_log_sqrt_2pi

        # same as below:
        # from torch.distributions.normal import Normal
        # log_prob_noise = Normal(a_mean, a_std).log_prob(a_noise)
        # same as below:
        # a_delta = a_noise - a_mean).pow(2) /(2* a_std.pow(2)
        # log_prob_noise = -a_delta - a_std.log() - np.log(np.sqrt(2 * np.pi))

        a_noise_tanh = a_noise.tanh()
        log_prob = log_prob_noise + (-a_noise_tanh.pow(2) + 1.000001).log()

        # same as below:
        # epsilon = 1e-6
        # log_prob = log_prob_noise - (1 - a_noise_tanh.pow(2) + epsilon).log()
        return a_noise_tanh, log_prob.sum(1, keepdim=True)


class ActorPPO(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()

        self.net__mean = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                       nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                       nn.Linear(mid_dim, action_dim), )
        self.net__std_log = nn.Parameter(torch.zeros(1, action_dim) - 0.5, requires_grad=True)
        self.constant_log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))

        layer_norm(self.net__mean[0], std=1.0)
        layer_norm(self.net__mean[2], std=1.0)
        layer_norm(self.net__mean[4], std=0.01)  # output layer for action

    def forward(self, s):
        a_mean = self.net__mean(s)
        return a_mean.tanh()

    def get__a__log_prob(self, state):  # todo plan to cancel
        a_mean = self.net__mean(state)
        a_log_std = self.net__std_log.expand_as(a_mean)
        a_std = torch.exp(a_log_std)
        a_noise = torch.normal(a_mean, a_std)

        a_delta = (a_noise - a_mean).pow(2) / (2 * a_std.pow(2))
        log_prob = -(a_delta + a_log_std + self.constant_log_sqrt_2pi)
        log_prob = log_prob.sum(1)
        return a_noise, log_prob

    def compute__log_prob(self, state, a_noise):  # todo may plan to cancel
        a_mean = self.net__mean(state)
        a_log_std = self.net__std_log.expand_as(a_mean)
        a_std = torch.exp(a_log_std)

        a_delta = (a_noise - a_mean).pow(2) / (2 * a_std.pow(2))
        log_prob = -(a_delta + a_log_std + self.constant_log_sqrt_2pi)
        return log_prob.sum(1)


class ActorGAE(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()

        nn_dense = DenseNet(mid_dim)
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn_dense, )

        layer_dim = nn_dense.out_dim
        self.net__mean = nn.Linear(layer_dim, action_dim)
        self.net__std_log = nn.Linear(layer_dim, action_dim)

        self.log_std_min = -20
        self.log_std_max = 2
        self.constant_log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        layer_norm(self.net[0], std=1.0)
        layer_norm(self.net__mean, std=0.01)  # output layer for action
        layer_norm(self.net__std_log, std=0.01)  # output layer for std_log

    def forward(self, s):
        x = self.net(s)
        a_mean = self.net__mean(x)
        return a_mean.tanh()

    def get__a__log_prob(self, state):
        x = self.net(state)
        a_mean = self.net__mean(x)
        a_log_std = self.net__std_log(x).clamp(self.log_std_min, self.log_std_max)
        a_std = torch.exp(a_log_std)

        a_noise = a_mean + a_std * torch.randn_like(a_mean, requires_grad=True, device=self.device)

        a_delta = (a_noise - a_mean).pow(2) / (2 * a_std.pow(2))
        log_prob = -(a_delta + a_log_std + self.constant_log_sqrt_2pi)
        return a_noise, log_prob.sum(1)

    def compute__log_prob(self, state, a_noise):
        x = self.net(state)
        a_mean = self.net__mean(x)
        a_log_std = self.net__std_log(x).clamp(self.log_std_min, self.log_std_max)
        a_std = torch.exp(a_log_std)

        a_delta = (a_noise - a_mean).pow(2) / (2 * a_std.pow(2))
        log_prob = -(a_delta + a_log_std + self.constant_log_sqrt_2pi)
        log_prob = log_prob.sum(1)
        return log_prob


class Critic(nn.Module):  # 2020-05-05 fix bug
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()

        self.net = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, 1), )

    def forward(self, s, a):
        x = torch.cat((s, a), dim=1)
        q = self.net(x)
        return q


class CriticTwin(nn.Module):  # TwinSAC <- TD3(TwinDDD) <- DoubleDQN <- Double Q-learning
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()

        def build_critic_network():
            net = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                nn.Linear(mid_dim, 1), )
            layer_norm(net[-1], std=0.01)  # It is no necessary.
            return net

        self.net1 = build_critic_network()
        self.net2 = build_critic_network()

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        q_value = self.net1(x)
        return q_value

    def get__q1_q2(self, state, action):
        x = torch.cat((state, action), dim=1)
        q_value1 = self.net1(x)
        q_value2 = self.net2(x)
        return q_value1, q_value2


class CriticTwinShared(nn.Module):  # 2020-06-18
    def __init__(self, state_dim, action_dim, mid_dim, use_dn):
        super().__init__()

        nn_dense = DenseNet(mid_dim)
        if use_dn:  # use DenseNet (DenseNet has both shallow and deep linear layer)
            self.net__mid = nn.Sequential(
                nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                nn_dense,
            )
            lay_dim = nn_dense.out_dim
        else:  # use a simple network for actor. Deeper network does not mean better performance in RL.
            self.net__mid = nn.Sequential(
                nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                nn.Linear(mid_dim, mid_dim),
            )
            lay_dim = mid_dim

        self.net__q1 = nn.Linear(lay_dim, 1)
        self.net__q2 = nn.Linear(lay_dim, 1)
        layer_norm(self.net__q1, std=0.1)
        layer_norm(self.net__q2, std=0.1)

        '''Not need to use both SpectralNorm and TwinCritic
        I choose TwinCritc instead of SpectralNorm, 
        because SpectralNorm is conflict with soft target update,

        if is_spectral_norm:
            self.net1[1] = nn.utils.spectral_norm(self.dec_q1[1])
            self.net2[1] = nn.utils.spectral_norm(self.dec_q2[1])
        '''

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = self.net_mid(x)
        q_value = self.net__q1(x)
        return q_value

    def get__q1_q2(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = self.net__mid(x)
        q_value1 = self.net__q1(x)
        q_value2 = self.net__q2(x)
        return q_value1, q_value2


class CriticAdv(nn.Module):  # 2020-05-05 fix bug
    def __init__(self, state_dim, mid_dim):
        super().__init__()

        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, 1), )

        layer_norm(self.net[0], std=1.0)
        layer_norm(self.net[2], std=1.0)
        layer_norm(self.net[4], std=1.0)  # output layer for action

    def forward(self, s):
        q = self.net(s)
        return q


class CriticAdvTwin(nn.Module):  # 2020-05-05 fix bug
    def __init__(self, state_dim, mid_dim):
        super().__init__()

        nn_dense = DenseNet(mid_dim)
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn_dense, )
        layer_dim = nn_dense.out_dim

        self.net_q1 = nn.Linear(layer_dim, 1)
        self.net_q2 = nn.Linear(layer_dim, 1)

        layer_norm(self.net[0], std=1.0)
        layer_norm(self.net_q1, std=0.1)  # output layer for q value
        layer_norm(self.net_q2, std=0.1)  # output layer for q value

    def forward(self, s):
        x = self.net(s)
        q1 = self.net_q1(x)
        q2 = self.net_q2(x)
        return q1, q2


class QNet(nn.Module):  # class AgentQLearning
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()  # same as super(QNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, action_dim),
        )
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, s):
        q = self.net(s)
        return q


class QNetTwin(nn.Module):  # class AgentQLearning
    def __init__(self, state_dim, action_dim, mid_dim, ):
        super().__init__()
        nn_dense = DenseNet(mid_dim)
        self.net = nn.Sequential(
            nn.Linear(state_dim, mid_dim), nn.ReLU(),
            nn_dense,
        )
        layer_dim = nn_dense.out_dim
        self.net_q1 = nn.Linear(layer_dim, action_dim)
        self.net_q2 = nn.Linear(layer_dim, action_dim)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, s, noise_std=0.0):
        x = self.net(s)
        q1 = self.net_q1(x)
        q2 = self.net_q2(x)
        return torch.min(q1, q2)

    def get__q1_q2(self, s):
        x = self.net(s)
        q1 = self.net_q1(x)
        q2 = self.net_q2(x)
        return q1, q2


class QNetDuel(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()

        self.net__head = nn.Sequential(
            nn.Linear(state_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, mid_dim), nn.ReLU(),
        )
        self.net_val = nn.Sequential(  # value
            nn.Linear(mid_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, 1),
        )
        self.net_adv = nn.Sequential(  # advantage value
            nn.Linear(mid_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, action_dim),
        )

    def forward(self, state, noise_std=0.0):
        x = self.net__head(state)
        val = self.net_val(x)
        adv = self.net_adv(x)
        q = val + adv - adv.mean(dim=1, keepdim=True)
        return q


"""utils"""


class NnnReshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.view((x.size(0),) + self.args)


class DenseNet(nn.Module):  # plan to hyper-param: layer_number
    def __init__(self, mid_dim):
        super().__init__()
        assert (mid_dim / (2 ** 3)) % 1 == 0

        def id2dim(i):
            return int((3 / 2) ** i * mid_dim)

        self.dense1 = nn.Sequential(nn.Linear(id2dim(0), id2dim(0) // 2), nn.ReLU(), )
        self.dense2 = nn.Sequential(nn.Linear(id2dim(1), id2dim(1) // 2), HardSwish(), )
        self.out_dim = id2dim(2)

        layer_norm(self.dense1[0], std=1.0)
        layer_norm(self.dense2[0], std=1.0)

    def forward(self, x1):
        x2 = torch.cat((x1, self.dense1(x1)), dim=1)
        x3 = torch.cat((x2, self.dense2(x2)), dim=1)
        return x3


class HardSwish(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu6 = nn.ReLU6()

    def forward(self, x):
        return self.relu6(x + 3.) / 6. * x


def layer_norm(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
