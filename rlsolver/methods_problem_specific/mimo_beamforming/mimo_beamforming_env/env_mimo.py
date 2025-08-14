import torch
import numpy as np
from rlsolver.envs.mimo_beamforming.baseline_mmse import compute_mmse_beamformer

class MIMOEnv():
    def __init__(self, K=4, N=4, P=10, noise_power=1, episode_length=6, num_env=4, device=torch.device("cuda:0")):
        self.N = N  # #antennas
        self.K = K  # #users
        self.P = P  # Power
        self.noise_power = noise_power
        self.basis_vectors, _ = torch.linalg.qr(torch.rand(2 * self.K * self.N, 2 * self.K * self.N, dtype=torch.float))
        self.subspace_dim = 1
        self.num_env = num_env
        self.device = device
        self.episode_length = episode_length
        self.get_vec_reward = vmap(self.get_reward, in_dims = (0, 0), out_dims = (0))
        
    def reset(self, if_test=False, test_H=None):
        if self.subspace_dim <= 2 * self.K * self.N:
            self.vec_H = self.generate_channel_batch(self.N, self.K, self.num_env, self.subspace_dim, self.basis_vectors).to(self.device)
        else:
            self.vec_H = torch.randn(self.num_env, 2 * self.K * self.N, dtype=torch.cfloat).to(self.device)
        if if_test:
            self.mat_H = test_H
            self.P = torch.diag_embed(torch.ones(self.mat_H.shape[0], 1, device=self.device).repeat(1, self.K)).to(torch.cfloat)
        else:
            self.mat_H = (self.vec_H[:, :self.K * self.N] + self.vec_H[:, self.K * self.N:] * 1.j).reshape(-1, self.N, self.K).to(self.device)
            self.mat_H[:self.mat_H.shape[0] // 2] *= np.sqrt(10)
            self.mat_H[self.mat_H.shape[0] // 2:] *= np.sqrt(15)
            self.P = torch.diag_embed(torch.ones(self.mat_H.shape[0], 1, device=self.device).repeat(1, self.K)).to(torch.cfloat)
        self.mat_W, _ = compute_mmse_beamformer(self.mat_H, K=4, N=4, P=self.P, noise_power=1, device=self.device).to(self.device)
        self.num_steps = 0
        self.done = False
        return (self.mat_H, self.mat_W)

    def step(self, action):
        self.mat_W = action
        self.reward = self.get_vec_reward(self.mat_H, self.mat_W)
        self.num_steps += 1
        self.done = True if self.num_steps >= self.episode_length else False
        return (self.mat_H, self.mat_W), self.reward, self.done
    
    def generate_channel_batch(self, N, K, batch_size, subspace_dim, basis_vectors):
        coordinates = torch.randn(batch_size, subspace_dim, 1)
        basis_vectors_batch = basis_vectors[:subspace_dim].T.repeat(batch_size, 1).reshape(-1, 2 * K * N, subspace_dim)
        vec_channel = torch.bmm(basis_vectors_batch, coordinates).reshape(-1 ,2 * K * N) * (( 2 * K * N / subspace_dim) ** 0.5)
        return  (N * K) ** 0.5 * (vec_channel / vec_channel.norm(dim=1, keepdim = True))
    
    def get_reward(self, H, W):
        HW = torch.matmul(H, W.T)
        S = torch.abs(HW.diag()) ** 2
        I = torch.sum(torch.abs(HW)**2, dim=-1) - torch.abs(HW.diag()) ** 2
        N = 1
        SINR = S / (I + N)
        reward = torch.log2(1 + SINR).sum(dim=-1)
        return reward
