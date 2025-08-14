import torch
import torch.nn as nn
import numpy as np
from functorch import vmap

class MIMOEnv():
    def __init__(self, K=4, N=4, P=10, noise_power=1, episode_length=6, num_env=4096, device=torch.device("cuda:0"), snr = 10):
        self.N = N  
        self.K = K  
        self.P = P  
        self.noise_power = noise_power
        self.device = device
        self.num_env = num_env
        self.episode_length = episode_length
        self.get_vec_sum_rate = vmap(self.get_sum_rate, in_dims = (0, 0), out_dims = (0, 0))
        self.epsilon = 1    
        self.snr = snr
        self.test = False

    def reset(self,):
        self.vec_H = self.generate_channel_batch(self.N, self.K, self.num_env, self.subspace_dim, self.basis_vectors)
        self.mat_H = (self.vec_H[:, :self.K * self.N] + self.vec_H[:, self.K * self.N:] * 1.j).reshape(-1, self.K, self.N)
        self.mat_H = self.mat_H 
        vec_W = torch.randn((self.mat_H.shape[0], self.K* self.K), dtype=torch.cfloat, device=self.device)
        vec_W = vec_W / torch.norm(vec_W, dim=1, keepdim=True)
        self.mat_W = vec_W.reshape(-1, self.K, self.N)
        HW = torch.bmm(self.mat_H, self.mat_W.transpose(-1, -2))
        self.num_steps = 0
        self.done = False
        return (self.mat_H, self.mat_W, self.P, HW)
    
    def step(self, action ):
        sum_rate = 0
        sum_rate, HW = self.get_vec_sum_rate(self.mat_H, action)
        self.reward = sum_rate
        self.mat_W = action.detach()
        self.num_steps += 1
        self.done = True if self.num_steps >= self.episode_length else False
        return (self.mat_H, self.mat_W, self.P, HW.detach()), self.reward, self.done, sum_rate.detach()

    def generate_channel_batch(self, N, K, batch_size):
        vec_channel = torch.randn(2 * batch_size * N * K, device=self.device)
        return vec_channel / np.sqrt(2)
    def get_sum_rate(self, H, W):
        HW = torch.matmul(H, W.T)
        S = torch.abs(HW.diag()) ** 2
        I = torch.sum(torch.abs(HW)**2, dim=-1) - torch.abs(HW.diag()) ** 2
        N = 1
        SINR = S / (I + N)
        reward = torch.log2(1 + SINR).sum(dim=-1)
        return reward, HW
class Policy_Net_MIMO(nn.Module):
    def __init__(self, mid_dim=1024, K=4, N=4, P=10, encode_dim=512, gnn_loop=4):
        super(Policy_Net_MIMO, self).__init__()
        self.encode_dim = encode_dim
        self.total_power = P
        self.K = K
        self.N = N
        self.state_dim = (6, K, N)
        self.action_dim = 2 * K * N
        self.loop = gnn_loop
        self.theta_0 = nn.Linear(self.K * 2, self.encode_dim)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mid_dim = mid_dim
        self.net = nn.Sequential(
        ConvNet(mid_dim, self.state_dim, mid_dim * 4), nn.ReLU(),
        nn.Linear(mid_dim * 4, mid_dim * 2), nn.ReLU(),
        nn.Linear(mid_dim * 2, mid_dim * 2), nn.ReLU(),
        nn.Linear(mid_dim * 2, mid_dim * 2), nn.ReLU(),
        nn.Linear(mid_dim * 2, mid_dim * 2), nn.ReLU(),
        nn.Linear(mid_dim * 2, mid_dim * 2), nn.Hardswish(),
        nn.Linear(mid_dim * 2, self.action_dim),
        )
        self.output_layer = nn.Sigmoid().to(self.device)

    def forward(self, state):
        mat_H, mat_W, _, mat_HW = state
        vec_H = torch.cat((mat_H.real.reshape(-1, self.K * self.N), mat_H.imag.reshape(-1, self.K * self.N)), 1)
        vec_W = torch.cat((mat_W.real.reshape(-1, self.K * self.N), mat_W.imag.reshape(-1, self.K * self.N)), 1)
        vec_HW = torch.cat((mat_HW.real.reshape(-1, self.K * self.N), mat_HW.imag.reshape(-1, self.K * self.N)), 1)
        net_input = torch.cat((vec_H, vec_W, vec_HW), 1).reshape(-1, 6, self.K * self.N)
        net_input = net_input.reshape(-1, 6, self.K, self.N)
        vec_W_new = (self.sigmoid(self.net(net_input)) - 0.5) * 2
        vec_W_new = vec_W_new / torch.norm(vec_W_new, dim=1, keepdim=True)
        mat_W_new =  (vec_W_new[:, :self.K * self.N] + vec_W_new[:, self.K * self.N:] * 1.j).reshape(-1, self.K, self.N)
        return mat_W_new

class DenseNet(nn.Module):
    def __init__(self, lay_dim):
        super().__init__()
        self.dense1 = nn.Sequential(nn.Linear(lay_dim * 1, lay_dim * 1), nn.Hardswish())
        self.dense2 = nn.Sequential(nn.Linear(lay_dim * 2, lay_dim * 2), nn.Hardswish())
        self.inp_dim = lay_dim
        self.out_dim = lay_dim

    def forward(self, x1):
        x2 = torch.cat((x1, self.dense1(x1)), dim=1)
        x3 = torch.cat((x2, self.dense2(x2)), dim=1)
        return x3

class ConvNet(nn.Module):
    def __init__(self, mid_dim, inp_dim, out_dim):
        super().__init__()
        i_c_dim, i_h_dim, i_w_dim = inp_dim
        mid_dim = 400
        self.cnn_h = nn.Sequential(
            nn.Conv2d(i_c_dim * 1, mid_dim * 2, (1, i_w_dim), bias=True), nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_dim * 2, mid_dim * 1, (1, 1), bias=True), nn.ReLU(inplace=True),)
        self.linear_h = nn.Linear(i_h_dim * mid_dim, out_dim)
        self.cnn_w = nn.Sequential(
            nn.Conv2d(i_c_dim * 1, mid_dim * 2, (i_h_dim, 1), bias=True), nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_dim * 2, mid_dim * 1, (1, 1), bias=True), nn.ReLU(inplace=True),)
        self.linear_w = nn.Linear(i_w_dim * mid_dim, out_dim)

    def forward(self, state):
        ch = self.cnn_h(state)
        xh = self.linear_h(ch.reshape(ch.shape[0], -1))
        cw = self.cnn_w(state)
        xw = self.linear_w(cw.reshape(cw.shape[0], -1))
        return xw + xh

def train_curriculum_learning(policy_net_mimo, optimizer, save_path, device, K=4, N=4, P=10, noise_power=1, num_epochs=100000000, num_env=512):
    env_mimo_relay = MIMOEnv(K=K, N=N, P=P, noise_power=noise_power, device=device, num_env=num_env, episode_length=1)
    for epoch in range(num_epochs):
        state = env_mimo_relay.reset()
        policy_net_mimo.previous = torch.randn(1, num_env, policy_net_mimo.mid_dim * 2, device=device)
        loss = 0
        sr = 0
        while(1):
            action = policy_net_mimo(state)
            next_state, reward, done, _ = env_mimo_relay.step(action)
            loss -= reward
            state = next_state
            if done:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                break
                
            
if __name__  == "__main__":
    N = 4   
    K = 4   
    SNR = 10 
    P = 20 ** (SNR / 10)
    
    mid_dim = 2048
    noise_power = 1
    learning_rate = 5e-5
    cwd = f"RL_N{N}K{K}SNR{SNR}"
    env_name = f"N{N}K{K}SNR{SNR}_mimo_beamforming"
    save_path = None
    
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy_net_mimo = Policy_Net_MIMO(mid_dim= mid_dim, K=K, N=N, P=P).to(device)
    optimizer = torch.optim.Adam(policy_net_mimo.parameters(), lr=learning_rate)
    
    train_curriculum_learning(policy_net_mimo, optimizer, K=K, N=N, save_path=save_path, device=device, P=P, noise_power=noise_power)
