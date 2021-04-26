import torch
import torch.nn as nn
import numpy as np

"""[ElegantRL](https://github.com/AI4Finance-LLC/ElegantRL)"""

'''Q Network'''


class QNet(nn.Module):  # nn.Module is a standard PyTorch Network
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, action_dim))

    def forward(self, state):
        return self.net(state)  # Q value

class UAQNet(nn.Module):  # nn.Module is a standard PyTorch Network
    def __init__(self, observation_space, n_outputs_1, n_outputs_2, width=100):

        super().__init__()

        n_inputs = observation_space

        #self.layer1 = torch.nn.Linear(n_inputs, width)

        self.output_1 = nn.Sequential(
                            torch.nn.Linear(n_inputs, width),
                            nn.ReLU(),
                            nn.Linear(width, width),
                            nn.ReLU(),
                            nn.Linear(width, width),
                            nn.ReLU(),
                            nn.Linear(width, n_outputs_1))

        self.output_2 = nn.Sequential(
                            torch.nn.Linear(n_inputs, width),
                            nn.ReLU(),
                            nn.Linear(width, width),
                            nn.ReLU(),
                            nn.Linear(width, width),
                            nn.ReLU(),
                            nn.Linear(width, n_outputs_2))

    def forward(self, obs):
        #out = F.relu(self.layer1(obs))
        return self.output_1(obs), self.output_2(obs)