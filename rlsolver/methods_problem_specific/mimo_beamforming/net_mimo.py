import torch
import numpy as np
import torch.nn as nn

class Policy_Net_MIMO(nn.Module):
    def __init__(self, mid_dim=256, K=4, N=4, total_power=10, encode_dim=512, gnn_loop=4):
        super(Policy_Net_MIMO, self).__init__()
        self.encode_dim = encode_dim
        self.total_power = total_power
        self.K = K
        self.N = N
        self.state_dim = (6,K,N)
        self.action_dim = 2 * K * N
        self.loop = gnn_loop
        self.theta_0 = nn.Linear(self.K * 2, self.encode_dim)
        self.if_gnn = False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.sigmoid = nn.Sigmoid()
        self.net = nn.Sequential(
            BiConvNet(mid_dim, self.state_dim, mid_dim * 4), nn.ReLU(),
            nn.Linear(mid_dim * 4, mid_dim * 2), nn.ReLU(),
            nn.Linear(mid_dim * 2, mid_dim * 4),
            nn.Linear(mid_dim * 4, mid_dim * 2), nn.Hardswish(),
            nn.Linear(mid_dim * 2, self.action_dim),
        )
        
        if self.if_gnn:
            self.gnn_weight = nn.ModuleList([ nn.Linear(self.K * 2, self.encode_dim), 
                                        nn.Linear(self.K * 2, self.encode_dim), 
                                        nn.Linear(self.encode_dim, self.encode_dim), 
                                        nn.Linear(self.encode_dim, self.encode_dim),
                                        nn.Linear(1, self.encode_dim),
                                        nn.Linear(2 * self.encode_dim, self.N * 2),
                                        nn.Linear(self.encode_dim, self.encode_dim), 
                                        nn.Linear(self.encode_dim, self.encode_dim), 
                                        nn.Linear(self.encode_dim, self.encode_dim),
                                        nn.Linear(self.encode_dim * 4, self.encode_dim)])
            self.mid = nn.ReLU()

    def forward(self, state, selected):
        mat_H, mat_W = state
        vec_H = torch.cat((mat_H.real.reshape(-1, self.K * self.N), mat_H.imag.reshape(-1, self.K * self.N)), 1)
        vec_W = torch.cat((mat_W.real.reshape(-1, self.K * self.N), mat_W.imag.reshape(-1, self.K * self.N)), 1)
        mat_HW = torch.bmm(mat_H, mat_W.transpose(1,2).conj())
        vec_HW = torch.cat((mat_HW.real.reshape(-1, self.K * self.N), mat_HW.imag.reshape(-1, self.K * self.N)), 1)
        mat_H
        mu = self.theta_0()

        for i in range(selected):
            pass
            for j in range(selected):
                pass
        return mat_W_new
class BiConvNet(nn.Module):
    def __init__(self, mid_dim, inp_dim, out_dim):
        super().__init__()
        i_c_dim, i_h_dim, i_w_dim = inp_dim 
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