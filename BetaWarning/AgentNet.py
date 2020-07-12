import torch
import torch.nn as nn  # import torch.nn.functional as F
import numpy as np  # import numpy.random as rd

"""
2019-07-01 ZenJiaHao, GitHub: YonV1943 DL_RL_Zoo RL
2019-11-11 Issay-0.0 [Essay Consciousness]
2020-02-02 Issay-0.1 Deep Learning Techniques (spectral norm, DenseNet, etc.) 
2020-04-04 Issay-0.1 [An Essay of Consciousness by YonV1943], IntelAC
2020-04-20 Issay-0.2 SN_AC, IntelAC_UnitedLoss
2020-05-20 Issay-0.3 [Essay, LongDear's Cerebellum (Little Brain)]
2020-06-06 Issay-0.3 check, DPG, SDG, InterAC, InterSAC

I consider that Reinforcement Learning Algorithms before 2020 have not consciousness
They feel more like a Cerebellum (Little Brain) for Machines.

[future] plan to add:
Soft Actor-Critic for Discrete Action Settings https://www.arxiv-vanity.com/papers/1910.07207/
Multi-Agent Deep RL: MADDPG, QMIX, QTRAN
some variants of DQN: Rainbow DQN, Ape-X
"""


class ActCriDPG(nn.Module):  # class AgentIntelAC
    def __init__(self, state_dim, action_dim, mid_dim):
        super(ActCriDPG, self).__init__()
        self.enc_s = nn.Sequential(
            nn.Linear(state_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
        )
        self.enc_a = nn.Sequential(
            nn.Linear(action_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
        )

        self.net = DenseNet(mid_dim)
        net_out_dim = mid_dim * 4
        self.dec_a = nn.Sequential(nn.Linear(net_out_dim, mid_dim), HardSwish(),
                                   nn.Linear(mid_dim, action_dim), nn.Tanh(), )
        self.dec_q = nn.Sequential(nn.Linear(net_out_dim, mid_dim), HardSwish(),
                                   nn.utils.spectral_norm(nn.Linear(mid_dim, 1)), )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add_noise(self, a, noise_std):  # 2020-03-03
        # noise_normal = torch.randn_like(a, device=self.device) * noise_std
        # a_temp = a + noise_normal
        a_temp = torch.normal(a, noise_std)
        mask = ((a_temp < -1.0) + (a_temp > 1.0)).type(torch.float32)  # 2019-12-30

        noise_uniform = torch.rand_like(a, device=self.device)
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


class AcrCriSPG(nn.Module):  # class AgentIntelAC for SAC (SPG means stochastic policy gradient)
    def __init__(self, state_dim, action_dim, mid_dim, use_dn=True, use_sn=True):  # plan todo use_dn
        super(AcrCriSPG, self).__init__()
        self.log_std_min = -20
        self.log_std_max = 2
        self.constant_log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # encoder
        self.enc_s = nn.Sequential(
            nn.Linear(state_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
        )  # state
        self.enc_a = nn.Sequential(
            nn.Linear(action_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
        )  # action (without nn.Tanh())

        self.net = DenseNet(mid_dim)
        net_out_dim = mid_dim * 4

        # decoder
        self.dec_a = nn.Sequential(
            nn.Linear(net_out_dim, mid_dim), HardSwish(),
            nn.Linear(mid_dim, action_dim),
        )  # action_mean
        self.dec_d = nn.Sequential(
            nn.Linear(net_out_dim, mid_dim), HardSwish(),
            nn.Linear(mid_dim, action_dim),
        )  # action_std_log (d means standard dev.)
        self.dec_q1 = nn.Sequential(
            nn.Linear(net_out_dim, mid_dim), HardSwish(),
            nn.utils.spectral_norm(nn.Linear(mid_dim, 1)),
        )  # q_value1 SharedTwinCritic
        self.dec_q2 = nn.Sequential(
            nn.Linear(net_out_dim, mid_dim), HardSwish(),
            nn.utils.spectral_norm(nn.Linear(mid_dim, 1)),
        )  # q_value2 SharedTwinCritic

    def forward(self, s, noise_std=0.0):  # actor, in fact, noise_std is a boolean
        s_ = self.enc_s(s)
        a_ = self.net(s_)
        a_mean = self.dec_a(a_)  # NOTICE! it is a_mean without .tanh()

        if noise_std != 0.0:
            a_std_log = self.dec_d(a_).clamp(self.log_std_min, self.log_std_max)
            a_std = a_std_log.exp()
            a_mean = torch.normal(a_mean, a_std)  # NOTICE! it is a_mean without .tanh()

        return a_mean.tanh()

    def get__a__log_prob(self, state):  # actor
        s_ = self.enc_s(state)
        a_ = self.net(s_)
        a_mean = self.dec_a(a_)  # NOTICE! it is a_mean without .tanh()
        a_std_log = self.dec_d(a_).clamp(self.log_std_min, self.log_std_max)
        a_std = a_std_log.exp()

        """add noise to action, stochastic policy"""
        # a_noise = torch.normal(a_mean, a_std, requires_grad=True)
        # the above is not same as below, because it needs gradient
        noise = torch.randn_like(a_mean, requires_grad=True, device=self.device)
        a_noise = a_mean + a_std * noise

        '''compute log_prob according to mean and std of action (stochastic policy)'''
        # a_delta = a_noise - a_mean).pow(2) /(2* a_std.pow(2)
        # log_prob_noise = -a_delta - a_std.log() - np.log(np.sqrt(2 * np.pi))
        # same as:
        a_delta = ((a_noise - a_mean) / a_std).pow(2) * 0.5
        log_prob_noise = -(a_delta + a_std_log + self.constant_log_sqrt_2pi)

        a_noise_tanh = a_noise.tanh()
        # log_prob = log_prob_noise - (1 - a_noise_tanh.pow(2) + epsilon).log() # epsilon = 1e-6
        # same as:
        log_prob = log_prob_noise - (-a_noise_tanh.pow(2) + 1.000001).log()
        return a_noise_tanh, log_prob.sum(1, keepdim=True)  # todo

    def get__a__std(self, state):
        s_ = self.enc_s(state)
        a_ = self.net(s_)
        a_mean = self.dec_a(a_)  # NOTICE! it is a_mean without .tanh()
        a_std_log = self.dec_d(a_).clamp(self.log_std_min, self.log_std_max)
        a_std = a_std_log.exp()

        return a_mean.tanh(), a_std

    def get__a__avg_std_noise_prob(self, state):  # actor
        s_ = self.enc_s(state)
        a_ = self.net(s_)
        a_mean = self.dec_a(a_)  # NOTICE! it is a_mean without .tanh()
        a_std_log = self.dec_d(a_).clamp(self.log_std_min, self.log_std_max)
        a_std = a_std_log.exp()

        """add noise to action, stochastic policy"""
        # a_noise = torch.normal(a_mean, a_std, requires_grad=True)
        # the above is not same as below, because it needs gradient
        noise = torch.randn_like(a_mean, requires_grad=True, device=self.device)
        a_noise = a_mean + a_std * noise

        '''compute log_prob according to mean and std of action (stochastic policy)'''
        # a_delta = a_noise - a_mean).pow(2) /(2* a_std.pow(2)
        # log_prob_noise = -a_delta - a_std.log() - np.log(np.sqrt(2 * np.pi))
        # same as:
        a_delta = ((a_noise - a_mean) / a_std).pow(2) * 0.5
        log_prob_noise = -(a_delta + a_std_log + self.constant_log_sqrt_2pi)

        a_noise_tanh = a_noise.tanh()
        # log_prob = log_prob_noise - (1 - a_noise_tanh.pow(2) + epsilon).log() # epsilon = 1e-6
        # same as:
        log_prob = log_prob_noise - (-a_noise_tanh.pow(2) + 1.000001).log()
        return a_mean.tanh(), a_std, a_noise_tanh, log_prob.sum(1, keepdim=True)

    def get__q1_q2(self, s, a):  # critic
        s_ = self.enc_s(s)
        a_ = self.enc_a(a)
        q_ = self.net(s_ + a_)
        q1 = self.dec_q1(q_)
        q2 = self.dec_q2(q_)
        return q1, q2

class ActCriAdv(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, mid_dim), nn.ReLU(),
            # nn.Linear(mid_dim, mid_dim), nn.ReLU(),
            DenseNet(mid_dim),
        )
        self.net__mean = nn.Linear(mid_dim * 4, action_dim)
        self.net__std_log = nn.Linear(mid_dim * 4, action_dim)
        self.net__q1 = nn.Linear(mid_dim * 4, 1)
        self.net__q2 = nn.Linear(mid_dim * 4, 1)

        self.log_std_min = -20
        self.log_std_max = 2
        self.constant_log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        layer_norm(self.net[0], std=1.0)
        layer_norm(self.net[2], std=1.0)
        layer_norm(self.net__mean, std=0.01)  # output layer for action
        layer_norm(self.net__std_log, std=0.01)  # output layer for std_log
        layer_norm(self.net__q1, std=1.0)  # output layer for q value
        layer_norm(self.net__q2, std=1.0)  # TwinCritic (DoubleDQN TD3)

    def forward(self, s):
        x = self.net(s)
        a_mean = self.net__mean(x)
        return a_mean

    def get__q_min(self, s):
        x = self.net(s)
        q1 = self.net__q1(x)
        q2 = self.net__q2(x)
        return torch.min(q1, q2)

    def get__a__log_prob(self, state):
        x = self.net(state)
        a_mean = self.net__mean(x)
        a_log_std = self.net__std_log(x).clamp(self.log_std_min, self.log_std_max)
        a_std = torch.exp(a_log_std)

        # a_noise = torch.normal(a_mean, a_std, requires_grad=True)
        noise = torch.randn_like(a_mean, requires_grad=True, device=self.device)
        a_noise = a_mean + a_std * noise

        a_delta = (a_noise - a_mean).pow(2) / (2 * a_std.pow(2))
        log_prob = -(a_delta + a_log_std + self.constant_log_sqrt_2pi).sum(1)
        return a_noise, log_prob

    def compute__log_prob(self, state, a_noise):
        x = self.net(state)
        a_mean = self.net__mean(x)
        a_log_std = self.net__std_log(x).clamp(self.log_std_min, self.log_std_max)
        a_std = torch.exp(a_log_std)

        a_delta = (a_noise - a_mean).pow(2) / (2 * a_std.pow(2))
        log_prob = -(a_delta + a_log_std + self.constant_log_sqrt_2pi)
        log_prob = log_prob.sum(1)

        q1 = self.net__q1(x)
        q2 = self.net__q2(x)
        return log_prob, q1, q2

class ActorDPG(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim):
        super(ActorDPG, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, action_dim), nn.Tanh(), )

    def forward(self, s, noise_std=0.0):
        a = self.net(s)
        return a if noise_std == 0.0 else self.add_noise(a, noise_std)

    def add_noise(self, action, noise_std):  # 2020-04-04
        # return torch.normal(action, noise_std, device=self.device).clamp(-1.0, 1.0)
        normal_noise = (torch.randn_like(action, device=self.device) * noise_std).clamp_(-0.5, 0.5)
        a_noise = (action + normal_noise).clamp_(-1.0, 1.0)
        return a_noise


class ActorDN(nn.Module):  # dn: DenseNet
    def __init__(self, state_dim, action_dim, mid_dim, use_dn):
        super(ActorDN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = build_actor_net_for_dpg(state_dim, action_dim, mid_dim, use_dn)

    def forward(self, s, noise_std=0.0):
        a = self.net(s)
        return a if noise_std == 0.0 else self.add_noise(a, noise_std)

    def add_noise(self, a, noise_std):  # 2020-03-03
        # noise_normal = torch.randn_like(a, device=self.device) * noise_std
        # a_temp = a + noise_normal
        a_temp = torch.normal(a, noise_std)
        mask = ((a_temp < -1.0) + (a_temp > 1.0)).type(torch.float32)  # 2019-12-30

        noise_uniform = torch.rand_like(a, device=self.device)
        a_noise = noise_uniform * mask + a_temp * (-mask + 1)
        return a_noise


class ActorSAC(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim, use_dn):
        super(ActorSAC, self).__init__()
        self.log_std_min = -20
        self.log_std_max = 2
        self.constant_log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''network'''
        # self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
        #                          nn.Linear(mid_dim, mid_dim), nn.ReLU(), )
        # self.net__mean = nn.Linear(mid_dim, action_dim)
        # self.net__std_log = nn.Linear(mid_dim, action_dim)
        self.net, self.net__mean, self.net__std_log = build_actor_net_for_spg(
            state_dim, action_dim, mid_dim, use_dn)

    def forward(self, state, noise_std=0.0):  # in fact, noise_std is a boolean
        x = self.net(state)
        a_mean = self.net__mean(x)  # NOTICE! it is a_mean without .tanh()

        if noise_std != 0.0:
            a_std_log = self.net__std_log(x).clamp(self.log_std_min, self.log_std_max)
            a_std = a_std_log.exp()
            a_mean = torch.normal(a_mean, a_std)  # NOTICE! it needs .tanh()

        return a_mean.tanh()

    def get__a__log_prob(self, state):
        x = self.net(state)
        a_mean = self.net__mean(x)  # NOTICE! it needs a_mean.tanh()
        a_std_log = self.net__std_log(x).clamp(self.log_std_min, self.log_std_max)
        a_std = a_std_log.exp()

        """add noise to action, stochastic policy"""
        # a_noise = torch.normal(a_mean, a_std, requires_grad=True) # Cannot use it. Should use the below
        noise = torch.randn_like(a_mean, requires_grad=True, device=self.device)
        a_noise = a_mean + a_std * noise
        # the above is not same as below, because it needs gradient, so it cannot write in this way
        # a_noise = torch.normal(a_mean, a_std, requires_grad=True)

        '''compute log_prob according to mean and std of action (stochastic policy)'''
        # from torch.distributions.normal import Normal
        # log_prob_noise = Normal(a_mean, a_std).log_prob(a_noise)
        # same as:
        # a_delta = a_noise - a_mean).pow(2) /(2* a_std.pow(2)
        # log_prob_noise = -a_delta - a_std.log() - np.log(np.sqrt(2 * np.pi))
        # same as:
        # self.constant_log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))
        a_delta = ((a_noise - a_mean) / a_std).pow(2) * 0.5
        log_prob_noise = -(a_delta + a_std_log + self.constant_log_sqrt_2pi)

        a_noise_tanh = a_noise.tanh()
        # log_prob = log_prob_noise - (1 - a_noise_tanh.pow(2) + epsilon).log() # epsilon = 1e-6
        # same as:
        log_prob = log_prob_noise - (-a_noise_tanh.pow(2) + 1.000001).log()
        return a_noise_tanh, log_prob.sum(1, keepdim=True)  # todo


class ActorPPO(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim):
        super(ActorPPO, self).__init__()

        self.net__mean = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                       nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                       nn.Linear(mid_dim, action_dim), )
        self.net__std_log = nn.Parameter(torch.zeros(1, action_dim), requires_grad=True)
        self.constant_log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))

        layer_norm(self.net__mean[0], std=1.0)
        layer_norm(self.net__mean[2], std=1.0)
        layer_norm(self.net__mean[4], std=0.01)  # output layer for action

    def forward(self, s):
        a_mean = self.net__mean(s)
        return a_mean.tanh()

    def get__a__log_prob(self, state):
        a_mean = self.net__mean(state)
        a_log_std = self.net__std_log.expand_as(a_mean)
        a_std = torch.exp(a_log_std)
        a_noise = torch.normal(a_mean, a_std)

        a_delta = (a_noise - a_mean).pow(2) / (2 * a_std.pow(2))
        log_prob = -(a_delta + a_log_std + self.constant_log_sqrt_2pi)
        log_prob = log_prob.sum(1)
        return a_noise, log_prob

    def compute__log_prob(self, state, a_noise):
        a_mean = self.net__mean(state)
        a_log_std = self.net__std_log.expand_as(a_mean)
        a_std = torch.exp(a_log_std)

        a_delta = (a_noise - a_mean).pow(2) / (2 * a_std.pow(2))
        log_prob = -(a_delta + a_log_std + self.constant_log_sqrt_2pi)
        return log_prob.sum(1)


class ActorAdv(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim):
        super(ActorAdv, self).__init__()

        self.net = nn.Sequential(
            # nn.BatchNorm1d(state_dim),
            nn.Linear(state_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, mid_dim), nn.ReLU(), )
        self.net__mean = nn.Linear(mid_dim, action_dim)
        self.net__std_log = nn.Linear(mid_dim, action_dim)

        self.log_std_min = -20
        self.log_std_max = 2
        self.constant_log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        layer_norm(self.net[0], std=1.0)
        layer_norm(self.net[2], std=1.0)
        layer_norm(self.net__mean, std=0.01)  # output layer for action
        layer_norm(self.net__std_log, std=0.01)  # output layer for std_log

    def forward(self, s):
        x = self.net(s)
        a_mean = self.net__mean(x)
        return a_mean

    def get__a__log_prob(self, state):
        x = self.net(state)
        a_mean = self.net__mean(x)
        a_log_std = self.net__std_log(x).clamp(self.log_std_min, self.log_std_max)
        a_std = torch.exp(a_log_std)

        # a_noise = torch.normal(a_mean, a_std, requires_grad=True)
        noise = torch.randn_like(a_mean, requires_grad=True, device=self.device)
        a_noise = a_mean + a_std * noise

        a_delta = (a_noise - a_mean).pow(2) / (2 * a_std.pow(2))
        log_prob = -(a_delta + a_log_std + self.constant_log_sqrt_2pi)
        log_prob = log_prob.sum(1)
        return a_noise, log_prob

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
        super(Critic, self).__init__()

        self.net = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, 1), )

    def forward(self, s, a):
        x = torch.cat((s, a), dim=1)
        q = self.net(x)
        return q


class CriticTwin(nn.Module):  # TwinSAC <- TD3(TwinDDD) <- DoubleDQN <- Double Q-learning
    def __init__(self, state_dim, action_dim, mid_dim, use_dn, use_sn):
        super(CriticTwin, self).__init__()
        self.net1 = build_critic_network(state_dim, action_dim, mid_dim, use_dn, use_sn)
        self.net2 = build_critic_network(state_dim, action_dim, mid_dim, use_dn, use_sn)

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
    def __init__(self, state_dim, action_dim, mid_dim, use_dn, use_sn):  # todo
        super(CriticTwinShared, self).__init__()

        self.net = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                 DenseNet(mid_dim), )
        self.net1 = nn.utils.spectral_norm(nn.Linear(mid_dim * 4, 1))
        self.net2 = nn.utils.spectral_norm(nn.Linear(mid_dim * 4, 1))

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = self.net(x)
        q_value = self.net1(x)
        return q_value

    def get__q1_q2(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = self.net(x)
        q_value1 = self.net1(x)
        q_value2 = self.net2(x)
        return q_value1, q_value2


class CriticSN(nn.Module):  # SN: Spectral Normalization # 2020-05-05 fix bug
    def __init__(self, state_dim, action_dim, mid_dim, use_dense, use_sn):
        super(CriticSN, self).__init__()
        self.net = build_critic_network(state_dim, action_dim, mid_dim, use_dense, use_sn)

    def forward(self, s, a):
        x = torch.cat((s, a), dim=1)
        q = self.net(x)
        return q


class CriticAdv(nn.Module):  # 2020-05-05 fix bug
    def __init__(self, state_dim, mid_dim):
        super(CriticAdv, self).__init__()

        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, 1), )

        layer_norm(self.net[0], std=1.0)
        layer_norm(self.net[2], std=1.0)
        layer_norm(self.net[4], std=1.0)  # output layer for action

    def forward(self, s):
        q = self.net(s)
        return q


class QNet(nn.Module):  # class AgentQLearning
    def __init__(self, state_dim, action_dim, mid_dim):
        super(QNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, action_dim),
        )
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, s):
        q = self.net(s)
        return q


class QNetDL(nn.Module):  # class AgentQLearning
    def __init__(self, state_dim, action_dim, mid_dim, ):
        super(QNetDL, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, mid_dim), nn.ReLU(),
            DenseNet(mid_dim),
        )
        self.net_q = nn.utils.spectral_norm(nn.Linear(mid_dim * 4, action_dim))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, s, noise_std=0.0):
        x = self.net(s)
        if noise_std != 0.0:
            x += torch.randn_like(x, device=self.device) * noise_std
        q = self.net_q(x)
        return q


class QNetTwin(nn.Module):  # class AgentQLearning
    def __init__(self, state_dim, action_dim, mid_dim, ):
        super().__init__()  # same as: super(QNetworkDL, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, mid_dim), nn.ReLU(),
            DenseNet(mid_dim),
        )
        self.net_q1 = nn.Linear(mid_dim * 4, action_dim)
        self.net_q2 = nn.Linear(mid_dim * 4, action_dim)
        # self.net_q1 = nn.utils.spectral_norm(nn.Linear(mid_dim * 4, action_dim))
        # self.net_q2 = nn.utils.spectral_norm(nn.Linear(mid_dim * 4, action_dim))

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


"""utils"""


class DenseNet(nn.Module):
    def __init__(self, mid_dim):
        super(DenseNet, self).__init__()
        self.dense1 = nn.Sequential(
            nn.Linear(mid_dim * 1, mid_dim * 1),
            HardSwish(),
        )
        self.dense2 = nn.Sequential(
            nn.Linear(mid_dim * 2, mid_dim * 2),
            HardSwish(),
        )

        layer_norm(self.dense1[0], std=1.0)
        layer_norm(self.dense2[0], std=1.0)

        # self.dropout = nn.Dropout(p=0.1)

    def forward(self, x1):
        x2 = torch.cat((x1, self.dense1(x1)), dim=1)
        x3 = torch.cat((x2, self.dense2(x2)), dim=1)
        # self.dropout.p = rd.uniform(0.0, 0.1)
        # return self.dropout(x3)
        return x3


class HardSwish(nn.Module):
    def __init__(self):
        super(HardSwish, self).__init__()
        self.relu6 = nn.ReLU6()

    def forward(self, x):
        return self.relu6(x + 3.) / 6. * x


def layer_norm(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)


def build_actor_net_for_dpg(state_dim, action_dim, mid_dim, use_dn):
    # for deterministic policy gradient
    nn_list = list()
    nn_list.extend([nn.Linear(state_dim, mid_dim), nn.ReLU(), ])

    if use_dn:  # use DenseNet (replace all conv2d layers into linear layers)
        nn_list.extend([DenseNet(mid_dim),
                        nn.Linear(mid_dim * 4, 1), ])
    else:
        nn_list.extend([nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                        nn.Linear(mid_dim, action_dim), nn.Tanh(), ])

    # layer_norm(self.net[0], std=1.0)
    layer_norm(nn_list[-2], std=0.01)  # output layer for action

    net = nn.Sequential(*nn_list)
    return net


def build_actor_net_for_spg(state_dim, action_dim, mid_dim, use_dn):
    # for stochastic policy gradient
    nn_list = list()
    nn_list.extend([nn.Linear(state_dim, mid_dim), nn.ReLU(), ])

    if use_dn:  # use DenseNet (replace all conv2d layers into linear layers)
        nn_list.append(DenseNet(mid_dim), )
        layer_dim = mid_dim * 4
    else:
        nn_list.extend([nn.Linear(mid_dim, mid_dim), nn.ReLU(), ])
        layer_dim = mid_dim * 1

    net = nn.Sequential(*nn_list)
    net__mean = nn.Linear(layer_dim, action_dim)
    net__std_log = nn.Linear(layer_dim, action_dim)

    # layer_norm(self.net[0], std=1.0)
    layer_norm(net__mean, std=0.01)  # output layer for action

    return net, net__mean, net__std_log


def build_critic_network(state_dim, action_dim, mid_dim, use_dn, use_sn):
    nn_list = list()
    nn_list.extend([nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(), ])

    if use_dn:  # use DenseNet (replace all conv2d layers into linear layers)
        nn_list.extend([DenseNet(mid_dim),
                        nn.Linear(mid_dim * 4, 1), ])
    else:
        nn_list.extend([nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                        nn.Linear(mid_dim, 1), ])

    if use_sn:  # NOTICE: spectral normalization is conflict with soft target update.
        # output_layer = nn.utils.spectral_norm(nn.Linear(...)),
        nn_list[-1] = nn.utils.spectral_norm(nn_list[-1])

    # layer_norm(self.net[0], std=1.0)
    # layer_norm(self.net[-1], std=1.0)

    net = nn.Sequential(*nn_list)
    return net
