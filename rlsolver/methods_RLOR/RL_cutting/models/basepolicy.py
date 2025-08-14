import torch
import numpy as np
from abc import ABC, abstractmethod


class AbstractPolicy(ABC):
    def __init__(self):
        # define a model and parameters
        pass

    def forward(self):
        raise NotImplementedError

    @abstractmethod
    def _compute_prob_torch(self, state):
        # given a state, returns a torch.nn.softmax of probs over the cuts
        pass

    def compute_prob(self, state):
        torch_prob = self._compute_prob_torch(state)
        return torch_prob.cpu().data.numpy()

    def _compute_loss(self, memory):
        loss = 0
        for i, state in enumerate(memory.states):
            adv = memory.advantages[i]
            action = int(memory.actions[i])
            prob = self._compute_prob_torch(state)

            k = len(prob)
            action_onehot = np.zeros(k)
            action_onehot[action] = 1
            action_onehot = torch.FloatTensor(action_onehot)

            prob_selected = torch.matmul(prob, action_onehot)
            loss += adv * torch.log(prob_selected + 1e-8)
        loss = -loss / len(memory.states)
        assert loss.requires_grad == True
        return loss

    def train(self, memory):
        loss = self._compute_loss(memory)
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

        return loss.detach().cpu().data.numpy()

    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
