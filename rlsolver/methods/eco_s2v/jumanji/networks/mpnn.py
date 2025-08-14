import torch
import torch.nn as nn

from rlsolver.methods.eco_s2v.src.networks.mpnn import MPNN


class MPNN_A2C(nn.Module):
    def __init__(self,
                 n_obs_in=7,
                 n_layers=3,
                 n_features=64,
                 tied_weights=False,
                 n_hid_readout=[], ):
        super().__init__()
        self.n_obs_in = n_obs_in
        self.n_layers = n_layers
        self.n_features = n_features
        self.tied_weights = tied_weights
        self.actor = MPNN(n_obs_in, n_layers, n_features, tied_weights, n_hid_readout)
        self.critic = MPNN(n_obs_in, n_layers, n_features, tied_weights, n_hid_readout)

    def forward(self, obs, use_tensor_core=False):
        logits = torch.softmax(self.actor(obs.clone(), use_tensor_core), dim=-1)
        value = self.critic(obs.clone(), use_tensor_core).mean(dim=-1)
        return logits, value
