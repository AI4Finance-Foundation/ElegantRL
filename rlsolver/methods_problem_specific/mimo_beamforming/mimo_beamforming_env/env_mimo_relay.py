import torch
import numpy as np
from rlsolver.problems.mimo_beamforming.mimo_beamforming_env.baseline_mmse import compute_mmse_beamformer_relay

class MIMORelayEnv():
    def __init__(self, K=2, N=2, M=2, P=10, noise_power=1, episode_length=6, num_env=4096, device=torch.device("cuda:0")):
        self.N = N # #antennas base stations
        self.M = M # #antennas relay
        self.K = K # #users
        self.P = P # power
        self.noise_power = noise_power
        self.device = device
        self.basis_vectors_G = torch.linalg.qr(torch.rand(2 * self.M * self.N, 2 * self.M * self.N, dtype=torch.float))[0].to(self.device)
        self.basis_vectors_H = torch.linalg.qr(torch.rand(2 * self.M * self.K, 2 * self.K * self.M, dtype=torch.float))[0].to(self.device)
        self.subspace_dim_H = 1
        self.subspace_dim_G = 1
        self.num_env = num_env
        self.episode_length = episode_length
        self.mat_F0 = torch.diag_embed(torch.ones(self.num_env, self.M, dtype=torch.cfloat)).to(self.device)
        
    def reset(self, if_test=False, test_H=None, test_G=None):
        if self.subspace_dim_H <= 2 * self.M * self.K:
            self.vec_H = self.generate_channel_batch(self.M, self.K, self.num_env, self.subspace_dim_H, self.basis_vectors_H).to(self.device)
        else:
            self.vec_H = torch.randn(self.num_env, 2 * self.M * self.K, dtype=torch.float).to(self.device)
        if self.subspace_dim_G <= 2 * self.M * self.N:
            self.vec_G = self.generate_channel_batch(self.M, self.N, self.num_env, self.subspace_dim_G, self.basis_vectors_G).to(self.device)
        else:
            self.vec_G = torch.randn(self.num_env, 2 * self.M * self.N, dtype=torch.float).to(self.device)
        self.mat_H = (self.vec_H[:, :self.K * self.M] + self.vec_H[:, self.K * self.M:] * 1.j).reshape(-1, self.M, self.K).to(self.device)
        self.mat_G = (self.vec_G[:, :self.M * self.N] + self.vec_G[:, self.M * self.N:] * 1.j).reshape(-1, self.M, self.N).to(self.device)
        if if_test:
                self.mat_H = test_H
                self.mat_G = test_G
        self.mat_F = self.mat_F0
        self.mat_HTFG = torch.bmm(torch.bmm(self.mat_H.transpose(-1, -2).conj(), self.mat_F), self.mat_G).to(self.device)
        self.mat_W = compute_mmse_beamformer_relay(self.mat_HTFG,  self.mat_H, self.mat_F)[0].to(self.device)
        self.num_steps = 0
        return (self.mat_HTFG, self.mat_F)

    def step(self, action):
        self.mat_F = action
        self.mat_HTFG = torch.bmm(torch.bmm(self.mat_H.transpose(-1, -2).conj(), self.mat_F), self.mat_G)
        self.mat_W, self.reward = compute_mmse_beamformer_relay(self.mat_HTFG, self.mat_H, self.mat_F, K=self.K, N=self.N, P=self.P, noise_power=self.noise_power, device=self.device)
        self.num_steps += 1
        self.mat_F = self.mat_F.detach()
        self.done = True if self.num_steps >= self.episode_length else False
        return (self.mat_HTFG, self.mat_F), self.reward, self.done

    def generate_channel_batch(self, dim1, dim2, batch_size, subspace_dim, basis_vectors):
        coordinates = torch.randn(batch_size, subspace_dim, 1).to(self.device)
        basis_vectors_batch = basis_vectors[:subspace_dim].T.repeat(batch_size, 1).reshape(-1, 2 * dim1 * dim2, subspace_dim).to(self.device)
        vec_channel = torch.bmm(basis_vectors_batch, coordinates).reshape(-1 ,2 * dim1 * dim2) * (( 2 * dim1 * dim2 / subspace_dim) ** 0.5)
        return  (dim1 * dim2) ** 0.5 * (vec_channel / vec_channel.norm(dim=1, keepdim = True))