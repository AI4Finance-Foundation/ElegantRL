import torch
import functorch
import numpy as np

class MIMOEnv():
    def __init__(self, K=4, N=4, P=10, noise_power=1, episode_length=6, num_env=4, device=torch.device("cuda:0")):
        self.N = N  # #antennas
        self.K = K  # #users
        self.P = P  # Power
        self.noise_power = noise_power
        self.num_env = num_env
        self.device = device

        self.basis_vectors, _ = torch.linalg.qr(torch.rand(2 * self.K * self.N, 2 * self.K * self.N, dtype=torch.float))
        self.basis_vectors_tensor = self.basis_vectors.repeat(self.num_env, 1).reshape(self.num_env, 2 * self.K * self.N, 2 * self.K * self.N)
        self.subspace_dim = 2
        self.episode_length = episode_length
        self.generate_channel_tensor = functorch.vmap(self.generate_channel, in_dims=(0), randomness= 'different')
        self.step_tensor = functorch.vmap(self.step, in_dims=(0))

    def reset(self, if_test=False, test_H=None):
        if self.subspace_dim <= 2 * self.K * self.N:
            self.vec_H = self.generate_channel_tensor(self.basis_vectors_tensor[:, :self.subspace_dim]).to(self.device)
        else:
            self.vec_H = torch.randn(self.num_env, 2 * self.K * self.N, dtype=torch.cfloat).to(self.device)
        if if_test:
            self.mat_H = test_H
        else:
            self.mat_H = (self.vec_H[:, :self.K * self.N] + self.vec_H[:, self.K * self.N:] * 1.j).reshape(-1, self.N, self.K).to(self.device)
        self.mat_W = torch.randn(self.num_env, self.K, self.N, dtype=torch.cfloat).to(self.device)

        self.num_steps = 0
        self.done = False
        return (self.mat_H, self.mat_W)


    def step(self, action):
        self.mat_W = action
        HW = torch.matmul(self.mat_H, self.mat_W.transpose(-1, -2))
        S = torch.abs(torch.diagonal(HW))**2
        I = torch.sum(torch.abs(HW)**2) - torch.abs(torch.diagonal(HW))**2
        SINR = S/(I+self.noise_power)
        self.reward =  torch.log2(1+SINR).sum()
        self.num_steps += 1
        self.done = torch.as_tensor(1.0) if self.num_steps >= self.episode_length else torch.as_tensor(0.0)
        return (self.mat_H, self.mat_W), self.reward, self.done

    def generate_channel(self, basis_vectors):
        coordinates = torch.randn(self.subspace_dim, 1)
        basis_vectors = basis_vectors[:self.subspace_dim].T.reshape(2 * self.K * self.N, self.subspace_dim)
        vec_channel = torch.matmul(basis_vectors, coordinates).reshape(2 * self.K * self.N) * (( 2 * self.K * self.N / self.subspace_dim) ** 0.5)
        return  (self.N * self.K) ** 0.5 * (vec_channel / vec_channel.norm(dim=-1, keepdim = True))