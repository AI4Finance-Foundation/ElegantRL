import torch as th
import numpy as np
import torch.nn as nn

class Policy_Net_MIMO(nn.Module):
    def __init__(self, mid_dim=1024, K=4, N=4, P=10, encode_dim=512, gnn_loop=4):
        super(Policy_Net_MIMO, self).__init__()
        self.encode_dim = encode_dim
        self.total_power = P
        self.K = K
        self.N = N
        self.input_shape = 6
        self.state_dim = (self.input_shape, K, N)
        self.action_dim = 2 * K * N
        self.loop = gnn_loop
        self.theta_0 = nn.Linear(self.K * 2, self.encode_dim)
        self.if_gnn = False
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        self.sigmoid = nn.Sigmoid().to(self.device)
        self.mid_dim = mid_dim
        self.net = nn.Sequential(
        BiConvNet(mid_dim, self.state_dim, mid_dim * 4), nn.ReLU(),
        nn.Linear(mid_dim * 4, mid_dim * 2), nn.ReLU(),
        nn.Linear(mid_dim * 2, mid_dim * 1),
        DenseNet(mid_dim * 1), nn.ReLU(),
        nn.Linear(mid_dim * 4, mid_dim * 2), nn.Hardswish(),
        nn.Linear(mid_dim * 2, self.action_dim),
        )


    def forward(self, state):
        mat_H, mat_W, mat_P, mat_HW = state
        mat_H = mat_H / (1e-5 + mat_H.norm(dim=1, keepdim=True))
        mat_W = mat_W / (1e-5 + mat_W.norm(dim=1, keepdim=True))
        mat_HW = mat_HW / (1e-5 + mat_HW.norm(dim=1, keepdim=True))
        vec_H = th.cat((mat_H.real.reshape(-1, self.K * self.N), mat_H.imag.reshape(-1, self.K * self.N)), 1)
        vec_W = th.cat((mat_W.real.reshape(-1, self.K * self.N), mat_W.imag.reshape(-1, self.K * self.N)), 1)
        vec_HW = th.cat((mat_HW.real.reshape(-1, self.K * self.N), mat_HW.imag.reshape(-1, self.K * self.N)), 1)
        net_input = th.cat((vec_H, vec_W, vec_HW), 1).reshape(-1, self.input_shape, self.K * self.N)
        net_input = net_input.reshape(-1, self.input_shape, self.K, self.N)
        vec_W_new = (self.sigmoid(self.net(net_input)) - 0.5) * 2
        vec_W_new = vec_W_new / th.norm(vec_W_new, dim=1, keepdim=True)
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
        x2 = th.cat((x1, self.dense1(x1)), dim=1)
        x3 = th.cat((x2, self.dense2(x2)), dim=1)
        return x3

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

