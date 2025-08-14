import torch as th
import pickle as pkl
import numpy as np
from functorch import vmap
from baseline_mmse import compute_mmse_beamformer
class MIMOEnv():
    def __init__(self, K=4, N=4, P=10, noise_power=1, episode_length=6, num_env=4096, device=th.device("cuda:0"), reward_mode='sl', snr = 10):
        self.N = N  # #antennas
        self.K = K  # #users
        self.P = P  # Power
        self.noise_power = noise_power
        self.device = device
        self.basis_vectors, _ = th.linalg.qr(th.rand(2 * self.K * self.N, 2 * self.K * self.N, dtype=th.float, device=self.device))
        self.reward_mode = reward_mode
        if self.reward_mode =='rl':
            self.subspace_dim =  1# 2 * K * N
        else:
            self.subspace_dim =  2 * K * N
        self.num_env = num_env
        self.episode_length = episode_length
        self.get_vec_sum_rate = vmap(self.get_sum_rate, in_dims = (0, 0), out_dims = (0, 0))
        self.num_x = 1000
        self.epsilon = 1
        self.snr = snr
        self.test = False
        print(self.reward_mode)
        if self.reward_mode == 'empirical':
            self.get_vec_reward = vmap(self.get_reward_empirical, in_dims = (0, 0, None), out_dims = (0, 0))
        elif self.reward_mode == 'analytical':
            self.get_vec_reward = vmap(self.get_reward_analytical, in_dims = (0, 0, None), out_dims = (0, 0))
        elif self.reward_mode == 'supervised_mmse' or self.reward_mode == 'supervised_mmse_curriculum':
            self.get_vec_reward = vmap(self.get_reward_supervised_mmse, in_dims = (0, 0, 0), out_dims = (0, 0))
        with open(f"./K{self.K}N{self.N}Samples=100.pkl", 'rb') as f:
            self.test_H = th.as_tensor(pkl.load(f), dtype=th.cfloat, device=self.device)



    def reset(self, test=False, test_P = None, if_mmse = False):
        self.test = False
        if self.subspace_dim <= 2 * self.K * self.N:
            self.vec_H = self.generate_channel_batch(self.N, self.K, self.num_env, self.subspace_dim, self.basis_vectors)
        else:
            self.vec_H = th.randn(self.num_env, 2 * self.K * self.N, dtype=th.cfloat, device=self.device) / np.sqrt(2)
        if test:
            self.test = True
            self.mat_H = self.test_H * np.sqrt(test_P)
            # print(self.mat_H.shape)
        else:
            self.mat_H = (self.vec_H[:, :self.K * self.N] + self.vec_H[:, self.K * self.N:] * 1.j).reshape(-1, self.K, self.N)
            self.mat_H *= np.sqrt(self.P)
            # variable SNR
            # self.mat_H[:self.mat_H.shape[0] // 3] *= np.sqrt(10)
            # self.mat_H[self.mat_H.shape[0] // 3:2 * self.mat_H.shape[0] // 3] *= np.sqrt(10 ** 1.5)
            # self.mat_H[2 * self.mat_H.shape[0] // 3:] *= np.sqrt(10 ** 2)
            
        self.mat_H = self.mat_H
        if self.reward_mode != 'rl':
            self.mat_W = th.zeros_like(self.mat_H, device=self.device) # self.mat_H.conj().transpose(-1, -2)
        else:
            vec_W = th.randn((self.mat_H.shape[0], self.K* self.N), dtype=th.cfloat, device=self.device)
            vec_W = vec_W / th.norm(vec_W, dim=1, keepdim=True)
            self.mat_W = vec_W.reshape(-1, self.K, self.N)

        if if_mmse:
            self.mat_W_mmse, _= compute_mmse_beamformer(self.mat_H,  K=self.K, N=self.N, P=self.P, noise_power=1, device=self.device)
            self.mat_W = self.mat_W_mmse
        self.noise = th.randn(self.K, self.num_x, dtype=th.cfloat).to(self.device)
        if self.reward_mode == 'supervised_mmse' or self.reward_mode == 'supervised_mmse_curriculum':
            self.X, _= compute_mmse_beamformer(self.mat_H,  K=self.K, N=self.N, P=self.P, noise_power=1, device=self.device)

        else:
            self.X = th.randn(self.K, self.num_x, dtype=th.cfloat).to(self.device)
        HW = th.bmm(self.mat_H, self.mat_W.transpose(-1, -2))
        self.num_steps = 0
        self.done = False
        return (self.mat_H, self.mat_W, self.P, HW)
    def step(self, action ):
        sum_rate = 0
        if self.reward_mode != 'rl' and self.test != True:
            obj = 0
            obj, HW = self.get_vec_reward(self.mat_H, action, self.X)
            sum_rate, HW = self.get_vec_sum_rate(self.mat_H, action)
            if self.reward_mode == "supervised_mmse_curriculum":
                self.reward = (-obj) * (1 - self.epsilon) / obj.norm(keepdim=True) + sum_rate * self.epsilon / sum_rate.norm(keepdim=True)
            else:
                self.reward=  -obj

        else:

            sum_rate, HW = self.get_vec_sum_rate(self.mat_H, action)
            self.reward = sum_rate
            self.mat_W = action.detach()
        self.num_steps += 1
        self.done = True if self.num_steps >= self.episode_length else False
        return (self.mat_H, self.mat_W, self.P, HW.detach()), self.reward, self.done, sum_rate.detach()

    def generate_channel_batch(self, N, K, batch_size, subspace_dim, basis_vectors):
        coordinates = th.randn(batch_size, subspace_dim, 1, device=self.device)
        basis_vectors_batch = basis_vectors[:subspace_dim].T.repeat(batch_size, 1).reshape(-1, 2 * K * N, subspace_dim)
        vec_channel = th.bmm(basis_vectors_batch, coordinates).reshape(-1, 2 * K * N)
        return vec_channel / np.sqrt(2)
    def get_sum_rate(self, H, W):
        HW = th.matmul(H, W.T)
        S = th.abs(HW.diag()) ** 2
        I = th.sum(th.abs(HW)**2, dim=-1) - th.abs(HW.diag()) ** 2
        N = 1# / self.P_
        SINR = S / (I + N)
        reward = th.log2(1 + SINR).sum(dim=-1)
        return reward, HW

    def get_reward_analytical(self, H, W, X):
        HW = th.matmul(H, W.T)
        HW_T = HW.T.conj()
        return ((th.trace(th.matmul(HW,HW_T) - HW - HW_T).real + self.K * (1/self.snr)).mean()) / self.K, HW

    def get_reward_empirical(self, H, W, X):
        HW = th.matmul(H, W.T)
        HWX = th.matmul(HW / HW.norm(keepdim=True), self.X)
        # return ((((HWX + ((1/self.snr)**0.5)*self.noise) - X).norm(dim=1, keepdim=True)**2).mean().real) / 1000, HW
        return (((HWX - X).norm(dim=1, keepdim=True)**2).mean().real) / 1000, HW

    def get_reward_supervised_mmse(self, H, W, W_mmse):
        HW = th.matmul(H, W.T)
        return (th.abs(W - W_mmse)**2).mean(), HW
