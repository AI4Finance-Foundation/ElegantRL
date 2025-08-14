import torch
import numpy as np


from models.basecritic import AbstractCritic

class DenseCritic(torch.nn.Module, AbstractCritic):
    def __init__(self, m, n, t, lr=0.001):
        """
        max_input is the size of the maximum state size
        Let t be the max number of timesteps,
        maximum state/action size: (m + t - 1, n+1)
        """
        super(DenseCritic, self).__init__()

        self.model = torch.nn.Sequential(
            # input layer
            torch.nn.Linear((m + t - 1) * (n + 1), 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )
        self.m = m
        self.t = t
        self.n = n
        self.full_length = (m + t - 1) * (n + 1)

        # DEFINE THE OPTIMIZER
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
    def _compute_value_torch(self, state): # use batched version when possible
        Ab, _, _ = state
        x = Ab.flatten()
        padded_x = np.append(x, np.zeros(self.full_length - len(x)))

        score = self.model(torch.FloatTensor(padded_x)).flatten()
        return score
    def _compute_values_torch(self, states):
        batch = []
        for state in states:
            Ab, _, _ = state
            x = Ab.flatten()
            padded_x = np.append(x, np.zeros(self.full_length - len(x)))

            batch.append(padded_x)

        batch_torch = torch.FloatTensor(batch)
        scores = self.model(batch_torch).flatten()

        return scores

class NoCritic(AbstractCritic):
    def __init__(self):
        pass

    def _compute_values_torch(self, states):
        return torch.Tensor(np.zeros(len(states)))

    def compute_values(self, states):
        return np.zeros(len(states)) # baseline is zero

    def train(self, memory):
        pass



