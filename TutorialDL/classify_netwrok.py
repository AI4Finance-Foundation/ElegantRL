import numpy.random as rd

import torch.nn as nn
import torch.nn.functional as F


class Res1dNet(nn.Module):
    def __init__(self):
        super(Res1dNet, self).__init__()
        inp_dim = 28 ** 2
        mid_dim = int(2 ** 6 * 1.5)
        out_dim = 10

        self.dropout = nn.Dropout(p=0.25)
        self.flatten = nn_reshape(-1)

        self.dense0 = nn_linear_bn(inp_dim, mid_dim, bias=True)
        self.dense1 = nn_linear_bn(mid_dim, mid_dim, bias=False)
        self.dense2 = nn_linear_bn(mid_dim, mid_dim, bias=False)
        self.dense3 = nn_linear_bn(mid_dim, mid_dim, bias=False)
        self.dense4 = nn_linear_bn(mid_dim, mid_dim, bias=False)
        self.dense5 = nn_linear_bn(mid_dim, mid_dim, bias=False)

        self.dense_out = nn.Sequential(
            self.dropout,
            nn.Linear(mid_dim, out_dim, bias=False),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x0):
        x0 = self.flatten(x0)
        x0 = self.dense0(x0)

        x1 = self.dense1(x0)
        x0 = self.dense2(x1)
        x1 = self.dense3(x0) + x1
        x0 = self.dense4(x1)
        x0 = self.dense5(x0) + x1

        self.dropout.p = rd.uniform(0.125, 0.375)
        x0 = self.dense_out(x0)
        return x0


class Conv2dNet(nn.Module):  # Conv2d + GlobalPooling
    def __init__(self, inp_dim=1, mid_dim=32, out_dim=10):
        super(Conv2dNet, self).__init__()

        def idx_dim(i):
            return int(mid_dim * 1.6487 ** i)

        self.convs = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, padding=0, bias=True),
            nn.ReLU(),

            nn_conv2d_bn_avg2(32, 32, 3, 1, padding=0, bias=False),
            nn_conv2d_bn_avg2(32, 48, 3, 1, padding=0, bias=False),

            nn.Conv2d(48, 108, 5, 1, padding=0, bias=True),  # GlobalConv
            nn.BatchNorm2d(108),
            nn_hswish(),
        )

        self.flatten = nn_reshape(-1)
        self.dropout = nn.Dropout(p=0.25)
        self.dense_1 = nn.Sequential(
            nn.Linear(108, 108, bias=True),
            nn_hswish(),
        )
        self.dense_0 = nn.Sequential(
            self.dropout,
            nn.Linear(108, 10, bias=True),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        x = self.convs(x)

        x = self.flatten(x)
        x = self.dense_1(x)

        self.dropout.p = rd.uniform(0.125, 0.375)
        x = self.dense_0(x)
        return x


class SE2dNet(nn.Module):  # Squeeze-and-Excitation Network
    def __init__(self):
        super(SE2dNet, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, padding=0, bias=True),
            nn.ReLU(),
        )
        self.conv1 = nn_conv2d_bn_avg2(32, 32, 3, 1, padding=0, bias=False)
        self.conv1_se = nn_se_2d(32)
        self.conv2 = nn_conv2d_bn_avg2(32, 48, 3, 1, padding=0, bias=False)
        self.conv2_se = nn_se_2d(48)

        self.conv3 = nn.Sequential(
            nn.Conv2d(48, 108, 5, 1, padding=0, bias=False),
            nn_hswish(),
        )

        self.flatten = nn_reshape(-1)
        self.dropout = nn.Dropout(p=0.25)

        self.dense0 = nn.Sequential(
            nn_reshape(-1),
            nn.Linear(108, 108, bias=True),
            nn.BatchNorm1d(108),
            nn_hswish(),
        )
        self.dropout1 = nn.Dropout(p=0.25, inplace=True)
        self.dense1 = nn.Sequential(
            self.dropout1,
            nn.Linear(108, 10, bias=True),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x0):
        x0 = self.conv0(x0)
        x1 = self.conv1(x0)
        x1 *= self.conv1_se(x1)

        x2 = self.conv2(x1)
        x2 *= self.conv2_se(x2)
        x = self.conv3(x2)

        x = self.flatten(x)
        x = self.dense0(x)
        self.dropout1.p = rd.uniform(0.125, 0.375)
        x = self.dropout1(x)
        x = self.dense1(x)
        return x


'''network utils'''


def build_nn_module(func):
    class TorchModule(nn.Module):
        def __init__(self, *args):
            super(TorchModule, self).__init__()
            self.args = args

        def forward(self, x):
            return func(self, x)

    return TorchModule


@build_nn_module
def nn_hswish(_, x):
    return F.relu6(x + 3, inplace=True) / 6 * x


@build_nn_module
def nn_reshape(cls, x):
    shape = cls.args
    return x.view((x.size(0),) + shape)


def nn_linear_bn(inp_dim, out_dim, bias):
    return nn.Sequential(
        nn.utils.spectral_norm(nn.Linear(inp_dim, out_dim, bias=bias), n_power_iterations=2),
        # nn.Linear(inp_dim, out_dim, bias=bias),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(),
        # nn_hswish(),
    )


def nn_conv2d_bn_avg2(inp_dim, out_dim, kernel_size, stride, padding, bias):
    return nn.Sequential(
        nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=padding, bias=bias),
        nn.BatchNorm2d(out_dim),
        nn.AvgPool2d(2),
        nn.ReLU(),
    )


def nn_se_2d(inp_dim):
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn_reshape(-1),
        nn.Linear(inp_dim, inp_dim, bias=False),
        nn.ReLU(inplace=True),
        nn.Linear(inp_dim, inp_dim, bias=False),
        nn.Sigmoid(),
        nn_reshape(-1, 1, 1),
    )


'''abandoned'''

# def nn_se_1d(inp_dim):
#     return nn.Sequential(
#         nn.Linear(inp_dim, inp_dim, bias=False),
#         nn.ReLU(inplace=True),
#         nn.Linear(inp_dim, inp_dim, bias=False),
#         nn.Sigmoid(),
#     )

#
# class Dense1dNet(nn.Module):
#     def __init__(self):
#         super(Dense1dNet, self).__init__()
#         inp_dim = 28 ** 2
#         mid_dim = int(2 ** 6)
#         out_dim = 10
#         self.flatten = nn_reshape(-1)
#         self.dropout = nn.Dropout(p=0.25)
#
#         self.dense00 = nn_linear_bn(inp_dim, mid_dim * 1, bias=True)
#         self.dense10 = nn_linear_bn(mid_dim * 1, mid_dim * 1, bias=False)
#         self.dense11 = nn_linear_bn(mid_dim * 1, mid_dim * 1, bias=False)
#         self.dense20 = nn_linear_bn(mid_dim * 2, mid_dim * 2, bias=False)
#         self.dense21 = nn_linear_bn(mid_dim * 2, mid_dim * 2, bias=False)
#         self.dense30 = nn_linear_bn(mid_dim * 4, mid_dim * 4, bias=True)
#
#         self.dense_out = nn.Sequential(
#             self.dropout,
#             nn.Linear(mid_dim * 4, out_dim, bias=False),
#             nn.LogSoftmax(dim=1),
#         )
#
#     def forward(self, x00):
#         x00 = self.flatten(x00)
#         x10 = self.dense00(x00)
#
#         x11 = self.dense10(x10)
#         x11 = self.dense11(x11) + x10
#         x20 = torch.cat((x10, x11), dim=1)
#
#         x21 = self.dense20(x20)
#         x21 = self.dense21(x21) + x20
#         x30 = torch.cat((x20, x21), dim=1)
#
#         x30 = self.dense30(x30)
#         self.dropout.p = rd.uniform(0.125, 0.375)
#         x_o = self.dense_out(x30)
#         return x_o
#
#
# class SE1dNet(nn.Module):
#     def __init__(self):
#         super(SE1dNet, self).__init__()
#         self.dropout = nn.Dropout(p=0.25)
#         self.flatten = nn_reshape(-1)
#
#         self.dense0 = nn_linear_bn(28 ** 2, 96, bias=True)
#         self.se0 = nn_se_1d(96)
#         self.dense1 = nn_linear_bn(96, 192, bias=False)
#         self.se1 = nn_se_1d(192)
#
#         self.dense_out = nn.Sequential(
#             self.dropout,
#             nn.Linear(192, 10, bias=False),
#             nn.LogSoftmax(dim=1),
#         )
#
#     def forward(self, x):
#         x = self.flatten(x)
#
#         x = self.dense0(x)
#         x = x * self.se0(x)
#         x = self.dense1(x)
#         x = x * self.se1(x)
#
#         self.dropout.p = rd.uniform(0.125, 0.375)
#         x = self.dense_out(x)
#         return x
