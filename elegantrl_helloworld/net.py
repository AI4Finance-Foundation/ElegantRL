import torch
import torch.nn as nn
import numpy as np
import numpy.random as rd


class QNet(nn.Module):  # `nn.Module` is a PyTorch module for neural network
    def __init__(self, mid_dim, mid_layer_num, state_dim, action_dim):  # todo num_mid_layer
        super().__init__()
        self.net = build_fcn(mid_dim, mid_layer_num, inp_dim=state_dim, out_dim=action_dim)

        self.explore_rate = None
        self.action_dim = action_dim

    def forward(self, state):
        return self.net(state)  # Q values for multiple actions

    def get_action(self, state):  # todo return [int] the index of discrete action
        if rd.rand() > self.explore_rate:  # todo self.explore_rate < rd.rand():
            return self.net(state).argmax(dim=1, keepdim=True)
        else:
            action = torch.randint(self.action_dim, size=(state.shape[0], 1))
        return action


class Actor(nn.Module):  # todo inp --> input, out --> output
    def __init__(self, mid_dim, mid_layer_num, state_dim, action_dim):
        super().__init__()
        self.net = build_fcn(mid_dim, mid_layer_num, inp_dim=state_dim, out_dim=action_dim)

        self.explore_noise = None  # standard deviation of exploration action noise
        # todo explore_noise_std

    def forward(self, state):
        return self.net(state).tanh()  # action

    def get_action(self, state):  # for exploration
        action = self.net(state).tanh()
        noise = (torch.randn_like(action) * self.explore_noise).clamp(-0.5, 0.5)
        return (action + noise).clamp(-1.0, 1.0)


class Critic(nn.Module):
    def __init__(self, mid_dim, mid_layer_num, state_dim, action_dim):
        super().__init__()
        self.net = build_fcn(mid_dim, mid_layer_num, inp_dim=state_dim + action_dim, out_dim=1)

    def forward(self, state, action):
        return self.net(torch.cat((state, action), dim=1))  # Q value


class ActorPPO(nn.Module):
    def __init__(self, mid_dim, mid_layer_num, state_dim, action_dim):
        super().__init__()
        self.net = build_fcn(mid_dim, mid_layer_num, inp_dim=state_dim, out_dim=action_dim)

        # the logarithm (log) of standard deviation (std) of action, it is a trainable parameter
        self.a_std_log = nn.Parameter(torch.zeros((1, action_dim)) - 0.5, requires_grad=True)
        # todo action_std_log
        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))

    def forward(self, state):
        return self.net(state).tanh()  # action

    def get_action(self, state):
        a_avg = self.net(state)
        a_std = self.a_std_log.exp()
        # todo action_avg

        noise = torch.randn_like(a_avg)
        action = a_avg + noise * a_std
        return action, noise

    def get_old_logprob(self, _action, noise):
        delta = noise.pow(2) * 0.5
        return -(self.a_std_log + self.sqrt_2pi_log + delta).sum(1)  # old_logprob

    def get_logprob_entropy(self, state, action):
        a_avg = self.net(state)
        a_std = self.a_std_log.exp()

        delta = ((a_avg - action) / a_std).pow(2) * 0.5
        logprob = -(self.a_std_log + self.sqrt_2pi_log + delta).sum(1)  # new_logprob

        dist_entropy = (logprob.exp() * logprob).mean()  # policy entropy
        return logprob, dist_entropy  # todo entropy

    @staticmethod
    def get_a_to_e(action):  # convert action of network to action of environment
        return action.tanh()  # todo convert_action_for_env


class CriticPPO(nn.Module):
    def __init__(self, mid_dim, mid_layer_num, state_dim, _action_dim):
        super().__init__()
        self.net = build_fcn(mid_dim, mid_layer_num, inp_dim=state_dim, out_dim=1)

    def forward(self, state):
        return self.net(state)  # advantage value


def build_fcn(mid_dim, mid_layer_num, inp_dim, out_dim):  # fcn (Fully Connected Network)
    net_list = [nn.Linear(inp_dim, mid_dim), nn.ReLU(), ]
    for _ in range(mid_layer_num):
        net_list += [nn.Linear(mid_dim, mid_dim), nn.ReLU(), ]
    net_list += [nn.Linear(mid_dim, out_dim), ]
    return nn.Sequential(*net_list)
