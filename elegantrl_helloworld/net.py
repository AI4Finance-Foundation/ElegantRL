import torch
import torch.nn as nn
import numpy as np
import numpy.random as rd


class QNet(nn.Module):  # `nn.Module` is a PyTorch module for neural network
    def __init__(self, mid_dim, num_layer, state_dim, action_dim):
        super().__init__()
        self.net = build_mlp(mid_dim, num_layer, input_dim=state_dim, output_dim=action_dim)

        self.explore_rate = None
        self.action_dim = action_dim

    def forward(self, state):
        return self.net(state)  # Q values for multiple actions

    def get_action(self, state):  # return [int], which is the index of discrete action
        if self.explore_rate < rd.rand():
            return self.net(state).argmax(dim=1, keepdim=True)
        else:
            action = torch.randint(self.action_dim, size=(state.shape[0], 1))
        return action


class Actor(nn.Module):
    def __init__(self, mid_dim, num_layer, state_dim, action_dim):
        super().__init__()
        self.net = build_mlp(mid_dim, num_layer, input_dim=state_dim, output_dim=action_dim)

        self.explore_noise_std = None  # standard deviation of exploration action noise

    def forward(self, state):
        return self.net(state).tanh()  # action

    def get_action(self, state):  # for exploration
        action = self.net(state).tanh()
        noise = (torch.randn_like(action) * self.explore_noise_std).clamp(-0.5, 0.5)
        return (action + noise).clamp(-1.0, 1.0)


class Critic(nn.Module):
    def __init__(self, mid_dim, num_layer, state_dim, action_dim):
        super().__init__()
        self.net = build_mlp(mid_dim, num_layer, input_dim=state_dim + action_dim, output_dim=1)

    def forward(self, state, action):
        return self.net(torch.cat((state, action), dim=1))  # Q value



class ActorPPO(nn.Module):
    def __init__(self, mid_dim, num_layer, state_dim, action_dim):
        super().__init__()
        self.net = build_mlp(mid_dim, num_layer, input_dim=state_dim, output_dim=action_dim)

        # the logarithm (log) of standard deviation (std) of action, it is a trainable parameter
        self.a_std_log = nn.Parameter(torch.zeros((1, action_dim)) - 0.5, requires_grad=True)
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

    def get_old_logprob(self, _action, noise):  # noise = action - a_noise
        delta = noise.pow(2) * 0.5
        return -(self.a_std_log + self.sqrt_2pi_log + delta).sum(1)  # old_logprob

    def get_logprob_entropy(self, state, action):
        a_avg = self.net(state)
        a_std = self.a_std_log.exp()

        delta = ((a_avg - action) / a_std).pow(2) * 0.5
        logprob = -(self.a_std_log + self.sqrt_2pi_log + delta).sum(1)  # new_logprob

        entropy = (logprob.exp() * logprob).mean()  # the action distribution entropy of policy
        return logprob, entropy

    @staticmethod
    def convert_action_for_env(action):
        return action.tanh()


class CriticPPO(nn.Module):
    def __init__(self, mid_dim, num_layer, state_dim, _action_dim):
        super().__init__()
        self.net = build_mlp(mid_dim, num_layer, input_dim=state_dim, output_dim=1)

    def forward(self, state):
        return self.net(state)  # advantage value


def build_mlp(mid_dim, num_layer, input_dim, output_dim):  # MLP (MultiLayer Perceptron)
    net_list = [nn.Linear(input_dim, mid_dim), nn.ReLU(),
                nn.Linear(mid_dim, output_dim), ]
    assert num_layer >= 2
    for _ in range(num_layer - 2):
        net_list[1:1] = [nn.Linear(mid_dim, mid_dim), nn.ReLU(), ]
    return nn.Sequential(*net_list)
