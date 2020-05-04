import torch
import torch.nn as nn
import numpy as np
import numpy.random as rd

"""
2019-07-01 Zen4Jia1Hao2, GitHub: YonV1943 DL_RL_Zoo RL
2019-11-11 Issay-0.0 [Essay Consciousness]
2020-02-02 Issay-0.1 Deep Learning Techniques (spectral norm, DenseNet, etc.) 
2020-04-04 Issay-0.1 [An Essay of Consciousness by YonV1943], IntelAC
2020-04-20 Issay-0.2 SN_AC, IntelAC_UnitedLoss
2020-04-22 Issay-0.2 [Essay, LongDear's Cerebellum (Little Brain)]

I consider that Reinforcement Learning Algorithms before 2020 have not consciousness
They feel more like a Cerebellum (Little Brain) for Machines.
"""


def layer_norm(layer, std=1.0, bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim, use_densenet):
        super(Actor, self).__init__()

        if use_densenet:
            self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                     DenseNet(mid_dim),
                                     nn.Linear(mid_dim * 4, action_dim), nn.Tanh(), )
        else:
            self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                     ResNet(mid_dim),
                                     nn.Linear(mid_dim, action_dim), nn.Tanh(), )

        # layer_norm(self.net[0], std=1.0)
        layer_norm(self.net[-2], std=0.01)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim, use_densenet, use_spectral_norm):
        super(Critic, self).__init__()

        if use_densenet:
            self.net = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                     DenseNet(mid_dim),
                                     nn.Linear(mid_dim * 4, action_dim), )
        else:
            self.net = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                     ResNet(mid_dim),
                                     nn.Linear(mid_dim, action_dim), )

        if use_spectral_norm:  # NOTICE: spectral normalization is conflict with soft target update.
            # self.net[-1] = nn.utils.spectral_norm(nn.Linear(...)),
            self.net[-1] = nn.utils.spectral_norm(self.net[-1])

        # layer_norm(self.net[0], std=1.0)
        # layer_norm(self.net[-1], std=1.0)

    def forward(self, s, a):
        x = torch.cat((s, a), dim=1)
        q = self.net(x)
        return q


class CriticTwin(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim, use_densenet, use_spectral_norm):
        super(CriticTwin, self).__init__()

        net1__net2 = list()
        for _ in range(2):
            if use_densenet:
                net = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                    DenseNet(mid_dim),
                                    nn.Linear(mid_dim * 4, action_dim), )
            else:
                net = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                    ResNet(mid_dim),
                                    nn.Linear(mid_dim, action_dim), )

            if use_spectral_norm:  # NOTICE: spectral normalization is conflict with soft target update.
                # self.net[-1] = nn.utils.spectral_norm(nn.Linear(...)),
                net[-1] = nn.utils.spectral_norm(net[-1])

            net1__net2.append(net)

        self.net1, self.net2 = net1__net2

    def forward(self, s, a):
        x = torch.cat((s, a), dim=1)
        q1 = self.net1(x)
        return q1

    def get_q1_q2(self, s, a):
        x = torch.cat((s, a), dim=1)
        q1 = self.net1(x)
        q2 = self.net2(x)
        return q1, q2


class QNetwork(nn.Module):  # class AgentQLearning
    def __init__(self, state_dim, action_dim, mid_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, mid_dim), nn.ReLU(),
            DenseNet(mid_dim),
            nn.utils.spectral_norm(nn.Linear(mid_dim * 4, action_dim)),
        )
        self.d = nn.utils.spectral_norm(nn.Linear(mid_dim * 4, action_dim))
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
            self.net = ResNet(mid_dim)
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

    def next__q_a_fix_bug(self, s, s_next, noise_std):
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


class ActorCriticPPO(nn.Module):
    def __init__(self, action_dim, critic_dim, mid_dim, layer_norm=True):
        super(ActorCriticPPO, self).__init__()

        actor_fc1 = nn.Linear(action_dim, mid_dim)
        actor_fc2 = nn.Linear(mid_dim, mid_dim)
        actor_fc3 = nn.Linear(mid_dim, critic_dim)
        self.actor_fc = nn.Sequential(
            actor_fc1, HardSwish(),
            actor_fc2, HardSwish(),
            actor_fc3,
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, critic_dim), requires_grad=True)

        critic_fc1 = nn.Linear(action_dim, mid_dim)
        critic_fc2 = nn.Linear(mid_dim, mid_dim)
        critic_fc3 = nn.Linear(mid_dim, 1)
        self.critic_fc = nn.Sequential(
            critic_fc1, HardSwish(),
            critic_fc2, HardSwish(),
            critic_fc3,
        )

        if layer_norm:
            self.layer_norm(actor_fc1, std=1.0)
            self.layer_norm(actor_fc2, std=1.0)
            self.layer_norm(actor_fc3, std=0.01)

            self.layer_norm(critic_fc1, std=1.0)
            self.layer_norm(critic_fc2, std=1.0)
            self.layer_norm(critic_fc3, std=1.0)

    @staticmethod
    def layer_norm(layer, std=1.0, bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)

    def forward(self, s):
        a_mean = self.actor_fc(s)
        return a_mean

    def get__log_prob(self, s, a_inp):
        a_mean = self.actor_fc(s)
        a_log_std = self.actor_logstd.expand_as(a_mean)
        a_std = torch.exp(a_log_std)
        log_prob = -(a_log_std + (a_inp - a_mean).pow(2) / (2 * a_std.pow(2)) + np.log(2 * np.pi) * 0.5)
        log_prob = log_prob.sum(1)
        return log_prob

    def critic(self, s):
        q = self.critic_fc(s)
        return q

    def get__a__log_prob(self, a_mean):
        a_log_std = self.actor_logstd.expand_as(a_mean)
        a_std = torch.exp(a_log_std)
        a_noise = torch.normal(a_mean, a_std)

        log_prob = -(a_log_std + (a_noise - a_mean).pow(2) / (2 * a_std.pow(2)) + np.log(2 * np.pi) * 0.5)
        log_prob = log_prob.sum(1)
        return a_noise, log_prob


"""utils"""


class ResNet(nn.Module):
    def __init__(self, mid_dim):
        super(ResNet, self).__init__()
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
        x3 = self.dense2(x2) + x1
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

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x1):
        x2 = torch.cat((x1, self.dense1(x1)), dim=1)
        x3 = torch.cat((x2, self.dense2(x2)), dim=1)
        self.dropout.p = rd.uniform(0.0, 0.1)
        return self.dropout(x3)


class HardSwish(nn.Module):
    def __init__(self):
        super(HardSwish, self).__init__()
        self.relu6 = nn.ReLU6()

    def forward(self, x):
        return self.relu6(x + 3.) / 6. * x
