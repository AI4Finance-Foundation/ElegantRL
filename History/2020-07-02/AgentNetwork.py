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

I consider that Reinforcement Learning Algorithms before 2020 have not consciousness
They feel more like a Cerebellum (Little Brain) for Machines.

2020-06-06 Issay-0.3 check, plan to add DPG, SDG, discrete SAC
"""


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
    def __init__(self, state_dim, action_dim, mid_dim):
        super(CriticTwin, self).__init__()
        self.net1 = build_critic_network(state_dim, action_dim, mid_dim, use_dense=False, use_sn=False)
        self.net2 = build_critic_network(state_dim, action_dim, mid_dim, use_dense=False, use_sn=False)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        q_value = self.net1(x)
        return q_value

    def get__q1_q2(self, state, action):
        x = torch.cat((state, action), dim=1)
        q_value1 = self.net1(x)
        q_value2 = self.net2(x)
        return q_value1, q_value2


class ActorDL(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim, use_dense):
        super(ActorDL, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = build_actor_network(state_dim, action_dim, mid_dim, use_dense)

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


class CriticSN(nn.Module):  # 2020-05-05 fix bug
    def __init__(self, state_dim, action_dim, mid_dim, use_dense, use_sn):
        super(CriticSN, self).__init__()
        self.net = build_critic_network(state_dim, action_dim, mid_dim, use_dense, use_sn)

    def forward(self, s, a):
        x = torch.cat((s, a), dim=1)
        q = self.net(x)
        return q


class ActorSAC(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim):
        super(ActorSAC, self).__init__()
        self.log_std_min = -20
        self.log_std_max = 2
        self.constant_log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''network'''
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(), )
        self.net__mean = nn.Linear(mid_dim, action_dim)
        self.net__std_log = nn.Linear(mid_dim, action_dim)

    def forward(self, state, noise_std=0.0):  # in fact, noise_std is a boolean
        x = self.net(state)
        a_mean = self.net__mean(x)  # NOTICE! it needs a_mean.tanh()

        if noise_std != 0.0:
            a_std_log = self.net__std_log(x).clamp(self.log_std_min, self.log_std_max)
            a_std = a_std_log.exp()
            a_mean = torch.normal(a_mean, a_std)  # NOTICE! it needs a_mean.tanh()

        return a_mean.tanh()

    def get__a__log_prob(self, state):
        x = self.net(state)
        a_mean = self.net__mean(x)  # NOTICE! it needs a_mean.tanh()
        a_std_log = self.net__std_log(x).clamp(self.log_std_min, self.log_std_max)
        a_std = a_std_log.exp()

        """add noise to action, stochastic policy"""
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
        return a_noise_tanh, log_prob.sum(1, keepdim=True)


class ActorPPO(nn.Module):
    def __init__(self, action_dim, critic_dim, mid_dim):
        super(ActorPPO, self).__init__()

        self.net_action = nn.Sequential(
            nn.Linear(action_dim, mid_dim), HardSwish(),
            nn.Linear(mid_dim, mid_dim), HardSwish(),
            nn.Linear(mid_dim, critic_dim),
        )
        self.net__log_std = nn.Parameter(torch.zeros(1, critic_dim), requires_grad=True)

        # self.critic_fc = nn.Sequential(
        #     nn.Linear(action_dim, mid_dim), HardSwish(),
        #     nn.Linear(mid_dim, mid_dim), HardSwish(),
        #     nn.Linear(mid_dim, 1),
        # )

        self.constant_pi = np.log(np.sqrt(2 * np.pi))

        '''layer_norm'''
        layer_norm(self.net_action[0], std=1.0)
        layer_norm(self.net_action[2], std=1.0)
        layer_norm(self.net_action[4], std=0.01)  # output layer for action

    def forward(self, s):
        a_mean = self.net_action(s)
        return a_mean

    # def critic(self, s):
    #     q = self.critic_fc(s)
    #     return q

    def get__log_prob(self, a_mean, a_inp):  # for update_parameter
        a_log_std = self.net__log_std.expand_as(a_mean)
        a_std = a_log_std.exp()

        # log_prob = -(a_log_std + (a_inp - a_mean).pow(2) / (2 * a_std.pow(2)) + np.log(2 * np.pi) * 0.5)
        log_prob = -(a_log_std + self.constant_pi + (a_mean - a_inp) / a_std).pow(2) * 0.5
        log_prob = log_prob.sum(1)
        return log_prob

    def get__a__log_prob(self, a_mean):  # for select action
        a_log_std = self.net__log_std.expand_as(a_mean)
        a_std = torch.exp(a_log_std)

        a_noise = torch.normal(a_mean, a_std)

        # log_prob = -(a_log_std + (a_noise - a_mean).pow(2) / (2 * a_std.pow(2)) + np.log(2 * np.pi) * 0.5)
        log_prob = -(a_log_std + self.constant_pi + ((a_mean - a_noise) / a_std).pow(2) / 2)
        log_prob = log_prob.sum(1)
        return a_noise, log_prob


class CriticAdvantage(nn.Module):  # 2020-05-05 fix bug
    def __init__(self, state_dim, mid_dim):
        super(CriticAdvantage, self).__init__()

        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), HardSwish(),
                                 nn.Linear(mid_dim, mid_dim), HardSwish(),
                                 # nn.BatchNorm1d(mid_dim),
                                 nn.Linear(mid_dim, 1), )

        layer_norm(self.net[0], std=1.0)
        layer_norm(self.net[2], std=1.0)
        layer_norm(self.net[4], std=1.0)  # output layer for action

    def forward(self, s):
        q = self.net(s)
        return q


class QNetwork(nn.Module):  # class AgentQLearning
    def __init__(self, state_dim, action_dim, mid_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, mid_dim), nn.ReLU(),
            DenseNet(mid_dim),
            nn.utils.spectral_norm(nn.Linear(mid_dim * 4, action_dim)),
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, s, noise_std=0.0):
        q = self.net(s)
        if noise_std != 0.0:
            q += torch.randn_like(q, device=self.device) * noise_std
        return q


class ActorCritic(nn.Module):  # class AgentIntelAC
    def __init__(self, state_dim, action_dim, mid_dim, use_densenet):
        super(ActorCritic, self).__init__()
        self.enc_s = nn.Sequential(
            nn.Linear(state_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
        )
        self.enc_a = nn.Sequential(
            nn.Linear(action_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
        )

        if use_densenet:
            self.net = DenseNet(mid_dim)
            self.dec_a = nn.Sequential(
                nn.Linear(mid_dim * 4, mid_dim), HardSwish(),
                nn.Linear(mid_dim, action_dim), nn.Tanh(),
            )
            self.dec_q = nn.Sequential(
                nn.Linear(mid_dim * 4, mid_dim), HardSwish(),
                nn.utils.spectral_norm(nn.Linear(mid_dim, 1)),
            )
        else:
            self.net = LinearNet(mid_dim)
            self.dec_a = nn.Sequential(
                nn.Linear(mid_dim, mid_dim), HardSwish(),
                nn.Linear(mid_dim, action_dim), nn.Tanh(),
            )
            self.dec_q = nn.Sequential(
                nn.Linear(mid_dim, mid_dim), HardSwish(),
                nn.utils.spectral_norm(nn.Linear(mid_dim, 1)),
            )

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


"""utils"""


def layer_norm(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)


def build_actor_network(state_dim, action_dim, mid_dim, use_dense):
    nn_list = list()
    nn_list.extend([nn.Linear(state_dim, mid_dim), nn.ReLU(), ])

    if use_dense:  # use DenseNet (replace all conv2d layers into linear layers)
        nn_list.extend([DenseNet(mid_dim),
                        nn.Linear(mid_dim * 4, 1), ])
    else:
        nn_list.extend([nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                        nn.Linear(mid_dim, action_dim), nn.Tanh(), ])

    # layer_norm(self.net[0], std=1.0)
    layer_norm(nn_list[-2], std=0.01)  # output layer for action

    net = nn.Sequential(*nn_list)
    return net


def build_critic_network(state_dim, action_dim, mid_dim, use_dense, use_sn):
    nn_list = list()
    nn_list.extend([nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(), ])

    if use_dense:  # use DenseNet (replace all conv2d layers into linear layers)
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


class LinearNet(nn.Module):
    def __init__(self, mid_dim):
        super(LinearNet, self).__init__()
        self.dense1 = nn.Sequential(
            nn.Linear(mid_dim * 1, mid_dim * 1),
            HardSwish(),
        )
        self.dense2 = nn.Sequential(
            nn.Linear(mid_dim * 1, mid_dim * 1),
            HardSwish(),
        )

        layer_norm(self.dense1[0], std=1.0)
        layer_norm(self.dense2[0], std=1.0)

    def forward(self, x1):
        x2 = self.dense1(x1)
        x3 = self.dense2(x2)
        return x3


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
