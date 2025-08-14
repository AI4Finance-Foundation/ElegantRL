import torch
import numpy as np

class RNDNetwork(torch.nn.Module):
    def __init__(self, m, n, t, lr, rnd_filepath = None):
        """
        max_input is the size of the maximum state size
        Let t be the max number of timesteps,
        maximum state/action size: (m + t - 1, n+1)
        """
        super(RNDNetwork, self).__init__()

        self.m = m
        self.t = t
        self.n = n
        self.full_length = (m + t - 1) * (n + 1)

        self.target = torch.nn.Sequential(
            # input layer
            torch.nn.Linear(self.full_length, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )


        self.prediction = torch.nn.Sequential(
            # input layer
            torch.nn.Linear(self.full_length, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )

        # DEFINE THE OPTIMIZER
        self.optimizer = torch.optim.Adam(self.prediction.parameters(), lr=lr)
        if rnd_filepath != None:
            self.load(rnd_filepath)

    def forward(self):
        raise NotImplementedError

    def get_checkpoint(self):
        return self.target.state_dict(), self.prediction.state_dict()

    def load(self, target_filepath, pred_filepath):
        self.target.load_state_dict(torch.load(target_filepath))
        self.prediction.load_state_dict(torch.load(pred_filepath))

    def _compute_intrinsic_reward(self, next_state): # use batched version when possible
        Ab, _, _ = next_state
        x = Ab.flatten()
        padded_x = np.append(x, np.zeros(self.full_length - len(x)))

        target_score = self.target(torch.FloatTensor(padded_x)).flatten()
        pred_score = self.prediction(torch.FloatTensor(padded_x)).flatten()
        return (target_score - pred_score)**2

    def compute_intrinsic_reward(self, next_state):
        int_reward = self._compute_intrinsic_reward(next_state)
        return int_reward.cpu().data.numpy()[0]

    def _compute_loss(self, next_states):
        batch = []
        for state in next_states:
            Ab, _, _ = state
            x = Ab.flatten()
            padded_x = np.append(x, np.zeros(self.full_length - len(x)))

            batch.append(padded_x)

        batch_torch = torch.FloatTensor(batch)
        target_scores = self.target(batch_torch).flatten().detach()
        pred_scores = self.prediction(batch_torch).flatten()
        loss = torch.mean((target_scores - pred_scores) ** 2)
        assert loss.requires_grad == True
        return loss

    def train(self, memory):
        loss = self._compute_loss(memory.states)
        print(loss)

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        return loss.detach().cpu().data.numpy()


class NoRND(torch.nn.Module):


    def __init__(self):
        pass
    def compute_intrinsic_reward(self, next_state):
        return 0

    def get_checkpoint(self):
        return None, None
    def train(self, memory):
        pass
