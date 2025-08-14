import torch
import numpy as np

from models.basepolicy import AbstractPolicy

class RandomPolicy(torch.nn.Module, AbstractPolicy):
    def __init__(self):
        super(RandomPolicy, self).__init__()

    def _compute_prob_torch(self, state):
        Ab, c0, cuts = state
        probs = np.ones(len(cuts))/len(cuts)
        return torch.FloatTensor(probs)


    def train(self, memory):
        loss = 0
        return loss


class AttentionPolicy(torch.nn.Module, AbstractPolicy):
    def __init__(self, n, h, lr):
        """
        n: size of constraints and the b
        h: size of output
        """
        super(AttentionPolicy, self).__init__()
        self.model = torch.nn.Sequential(
            # input layer
            torch.nn.Linear(n+1, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, h)
        )

        # DEFINE THE OPTIMIZER
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # RECORD HYPER-PARAMS
        self.n = n
        self.h = h


    def _compute_prob_torch(self, state):
        Ab, _, cuts = state
        Ab = torch.nn.functional.normalize(torch.FloatTensor(np.array(Ab, dtype=np.float)), dim=1, p=1)
        cuts = torch.nn.functional.normalize(torch.FloatTensor(np.array(cuts, dtype=np.float)), dim=1, p=1)
        Ab = torch.FloatTensor(Ab).to(self.device())
        cuts = torch.FloatTensor(cuts).to(self.device())
        Ab_h = self.model(Ab)
        cuts_h = self.model(cuts)

        scores = torch.mean(torch.matmul(Ab_h, torch.transpose(cuts_h, 0, 1)), 0)
        prob = torch.nn.functional.softmax(scores, dim=0)
        return prob


class DoubleAttentionPolicy(torch.nn.Module, AbstractPolicy):
    """uses two MLP's to calculate the loss
    """
    def __init__(self, n, h, lr):
        """
        n: size of constraints and the b
        h: size of output
        """
        super(DoubleAttentionPolicy, self).__init__()
        self.model_1 = torch.nn.Sequential(
            # input layer
            torch.nn.Linear(n+1, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, h)
        )

        self.model_2 = torch.nn.Sequential(
            # input layer
            torch.nn.Linear(n + 1, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, h)
        )

        # DEFINE THE OPTIMIZER

        self.optimizer = torch.optim.Adam([
            {'params': self.model_1.parameters(), 'lr': lr},
            {'params': self.model_2.parameters(), 'lr': lr}
        ])

        # RECORD HYPER-PARAMS
        self.n = n
        self.h = h

    def _compute_prob_torch(self, state):
        Ab, _, cuts = state
        Ab = torch.nn.functional.normalize(torch.FloatTensor(np.array(Ab, dtype=np.float)), dim=1, p=1)
        cuts = torch.nn.functional.normalize(torch.FloatTensor(np.array(cuts, dtype=np.float)), dim=1, p=1)
        Ab = torch.FloatTensor(Ab).to(self.device())
        cuts = torch.FloatTensor(cuts).to(self.device())
        Ab_h = self.model_1(Ab)
        cuts_h = self.model_2(cuts)

        scores = torch.mean(torch.matmul(Ab_h, torch.transpose(cuts_h, 0, 1)), 0)
        prob = torch.nn.functional.softmax(scores, dim=0)
        return prob


class RNNPolicy(torch.nn.Module, AbstractPolicy):
    def __init__(self, n, lr):
        """
        num_dec_vars is n
        """
        super(RNNPolicy, self).__init__()
        self.model = RecurrentNetwork(n+1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _compute_prob_torch(self, state):
        Ab, _, cuts = state
        Ab = torch.nn.functional.normalize(torch.FloatTensor(np.array(Ab, dtype=np.float)), dim=1, p=1)
        cuts = torch.nn.functional.normalize(torch.FloatTensor(np.array(cuts, dtype=np.float)), dim=1, p=1)
        Ab = np.array(Ab)
        cuts = np.array(cuts)

        batch = []
        for cut in cuts:
            x = np.vstack([Ab, cut.flatten()])
            batch.append(x)
        batch_torch = torch.FloatTensor(batch).to(self.device())
        scores = self.model(batch_torch).flatten()

        return torch.nn.functional.softmax(scores, dim=-1)

class RecurrentNetwork(torch.nn.Module):
    def __init__(self, n):
        super(RecurrentNetwork, self).__init__()
        self.rnn = torch.nn.GRU(n, 64, num_layers=1,
                                batch_first=True)
        self.ffnn = torch.nn.Linear(64, 1)

    def forward(self, states):  # takes in batch of states of variable length
        # lens = (x != 0).sum(1)
        # p_embeds = rnn.pack_padded_sequence(embeds, lens, batch_first=True, enforce_sorted=False)
        _, hn = self.rnn(states)
        hns = hn.split(1, dim=0)
        last_hn = hns[-1]
        scores = self.ffnn(last_hn.squeeze(0))
        prob = torch.nn.functional.softmax(scores, dim=0)
        return prob

class DensePolicy(torch.nn.Module, AbstractPolicy):
    def __init__(self, m, n, t, lr):
        """
        max_input is the size of the maximum state + size of maximum action
        Let t be the max number of timesteps, ie the max number of cuts added
        maximum state/action size: (m + t - 1 + 1, n+1)
        """
        super(DensePolicy, self).__init__()
        self.model = torch.nn.Sequential(
            # input layer
            torch.nn.Linear((m + t) * (n + 1), 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )
        self.m = m
        self.t = t
        self.n = n
        self.full_length = (m + t) * (n + 1)

        # DEFINE THE OPTIMIZER
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _compute_prob_torch(self, state):
        Ab, c0, cuts = state
        batch = []
        for cut in cuts:
            x = np.append(Ab.flatten(), cut.flatten())
            padded_x = np.append(x, np.zeros(self.full_length - len(x))) # pad with zeros

            batch.append(padded_x)

        batch_torch = torch.FloatTensor(batch).to(self.device())
        scores = self.model(batch_torch).flatten()

        return torch.nn.functional.softmax(scores, dim=-1)

