import torch
import numpy as np
from abc import ABC, abstractmethod


class AbstractCritic(ABC):
    def __init__(self):
        pass

    def forward(self):
        raise NotImplementedError

    @abstractmethod
    def _compute_values_torch(self, states):
        # given an array of states, compute the estimated values
        pass

    def compute_values(self, states):
        torch_values = self._compute_values_torch(states)
        return torch_values.cpu().data.numpy()

    def _compute_loss(self, memory):
        targets = torch.FloatTensor(memory.values)
        v_preds = self._compute_values_torch(memory.states)

        loss = torch.mean((v_preds - targets) ** 2)
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
