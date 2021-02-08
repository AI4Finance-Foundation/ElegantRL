import numpy as np
import torch
import torch.nn as nn


class ActorSAC(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()
        self.net__state = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                        nn.Linear(mid_dim, mid_dim), nn.ReLU(), )
        self.net_action = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                        nn.Linear(mid_dim, action_dim), )
        self.net__a_std = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                        nn.Linear(mid_dim, action_dim), )

        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))  # a constant
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, state):
        tmp = self.net__state(state)
        return self.net_action(tmp).tanh()  # action

    def get__a_noisy(self, state):
        t_tmp = self.net__state(state)
        a_avg = self.net_action(t_tmp)
        a_std = self.net__a_std(t_tmp).clamp(-16, 2).exp()
        return torch.normal(a_avg, a_std).tanh()  # re-parameterize

    def get__a__log_prob(self, s):
        t_tmp = self.net__state(s)
        a_avg = self.net_action(t_tmp)
        a_std_log = self.net__a_std(t_tmp).clamp(-16, 2)
        a_std = a_std_log.exp()

        noise = torch.randn_like(a_avg, requires_grad=True, device=self.device)
        a_tan = (a_avg + a_std * noise).tanh()  # action.tanh()
        log_prob = a_std_log + self.sqrt_2pi_log + noise.pow(2).__mul__(0.5) + (-a_tan.pow(2) + 1.000001).log()
        return a_tan, log_prob.sum(1, keepdim=True)


class ActorPPO(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, action_dim), )
        layer_norm(self.net[-1], std=0.1)  # output layer of action

        self.a_std_log = nn.Parameter(torch.zeros((1, action_dim)) - 0.5, requires_grad=True)  # trainable parameter
        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))  # a constant

    def forward(self, state):
        return self.net(state).tanh()  # action

    def get__a_noisy__noise(self, state):
        a_avg = self.net(state)
        a_std = self.a_std_log.exp()

        noise = torch.randn_like(a_avg)
        a_noisy = a_avg + noise * a_std
        return a_noisy, noise

    def compute__log_prob(self, state, a_noise):
        a_avg = self.net(state)
        a_std = self.a_std_log.exp()
        delta = ((a_avg - a_noise) / a_std).pow(2).__mul__(0.5)
        log_prob = -(self.a_std_log + self.sqrt_2pi_log + delta)
        return log_prob.sum(1)


class CriticTwin(nn.Module):  # shared parameter
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()
        self.net_sa = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, mid_dim), nn.ReLU(), )  # concat(state, action)
        self.net_q1 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                    nn.Linear(mid_dim, 1), )  # q1 value
        self.net_q2 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                    nn.Linear(mid_dim, 1), )  # q2 value

    def forward(self, state, action):
        tmp = self.net_sa(torch.cat((state, action), dim=1))
        return self.net_q1(tmp), self.net_q2(tmp)  # q1 value, q2 value


class CriticAdv(nn.Module):  # 2021-02-02
    def __init__(self, state_dim, mid_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, 1), )
        layer_norm(self.net[-1], std=1.0)  # output layer of action

    def forward(self, state):
        return self.net(state)  # q value


class NnnReshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.view((x.size(0),) + self.args)


def layer_norm(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
