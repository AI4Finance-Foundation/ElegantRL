import torch
import torch.nn as nn
import numpy as np
import numpy.random as rd


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, mid_dim), nn.ReLU(),
            DenseNet(mid_dim),
            nn.Linear(mid_dim * 4, action_dim), nn.Tanh(),
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, s, noise_std=0.0):
        a = self.net(s)
        return a if noise_std == 0.0 else self.add_noise(a, noise_std)

    def add_noise(self, a, noise_std):  # 2020-03-03
        noise_normal = torch.randn_like(a, device=self.device) * noise_std
        a_temp = a + noise_normal
        mask = ((a_temp < -1.0) + (a_temp > 1.0)).type(torch.float32)  # 2019-12-30

        noise_uniform = torch.rand_like(a, device=self.device)
        a_noise = noise_uniform * mask + a_temp * (-mask + 1)
        return a_noise


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
            DenseNet(mid_dim),
            nn.utils.spectral_norm(nn.Linear(mid_dim * 4, 1)),
        )

    def forward(self, s, a):
        x = torch.cat((s, a), dim=1)
        q = self.net(x)
        return q


class QNetwork(nn.Module):
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


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim):
        super(ActorCritic, self).__init__()
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

        self.dec_a = nn.Sequential(
            nn.Linear(mid_dim * 4, mid_dim),
            HardSwish(),
            nn.Linear(mid_dim, action_dim),
            nn.Tanh(),
        )
        self.dec_q = nn.Sequential(
            nn.Linear(mid_dim * 4, mid_dim),
            HardSwish(),
            nn.utils.spectral_norm(nn.Linear(mid_dim, 1)),
        )

        self.net = DenseNet(mid_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add_noise(self, a, noise_std):  # 2020-03-03
        noise_normal = torch.randn_like(a, device=self.device) * noise_std
        a_temp = a + noise_normal
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

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x1):
        x2 = self.dense1(x1)
        x3 = self.dense2(x2)
        self.dropout.p = rd.uniform(0.0, 0.2)
        return self.dropout(x3)


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

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x1):
        x2 = torch.cat((x1, self.dense1(x1)), dim=1)
        x3 = torch.cat((x2, self.dense2(x2)), dim=1)
        self.dropout.p = rd.uniform(0.0, 0.2)
        return self.dropout(x3)


class HardSwish(nn.Module):
    def __init__(self):
        super(HardSwish, self).__init__()
        self.relu6 = nn.ReLU6()

    def forward(self, x):
        return self.relu6(x + 3.) / 6. * x


class OrnsteinUhlenbeckProcess(object):
    def __init__(self, size, theta=0.15, sigma=0.3, x0=0.0, dt=1e-2):
        """
        Source: https://github.com/slowbull/DDPG/blob/master/src/explorationnoise.py
        I think that:
        It makes Zero-mean Gaussian Noise more stable.
        It helps agent explore better in a inertial system.
        """
        self.theta = theta
        self.sigma = sigma
        self.x0 = x0
        self.dt = dt
        self.size = size

    def __call__(self):
        noise = self.sigma * np.sqrt(self.dt) * rd.normal(size=self.size)
        x = self.x0 - self.theta * self.x0 * self.dt + noise
        self.x0 = x  # update x0
        return x
