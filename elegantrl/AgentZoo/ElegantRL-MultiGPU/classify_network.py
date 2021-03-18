import numpy.random as rd

import torch
import torch.nn as nn

'''input 1D'''


class Res1dNet(nn.Module):
    def __init__(self):
        super(Res1dNet, self).__init__()
        inp_dim = 28 ** 2
        mid_dim = int(2 ** 6 * 1.5)
        out_dim = 10

        self.dropout = nn.Dropout(p=0.25)
        self.flatten = NnnReshape((-1, inp_dim))

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


class FCNet(nn.Module):
    def __init__(self, inp_dim=3, mid_dim=128, glb_size=6):
        super().__init__()
        inp_dim = 32 ** 2 * inp_dim
        mid_dim = 2 ** 8
        out_dim = 10

        self.dropout = nn.Dropout(p=0.25)
        self.dense_out = nn.Sequential(
            NnnReshape((-1, inp_dim)),
            nn.Linear(inp_dim, mid_dim, bias=False),
            self.dropout,
            nn.Linear(mid_dim, out_dim, bias=False),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        self.dropout.p = rd.uniform(0.125, 0.375)
        return self.dense_out(x)


class Dense1dNet(nn.Module):
    def __init__(self):
        super(Dense1dNet, self).__init__()
        inp_dim = 28 ** 2
        mid_dim = int(2 ** 6)
        out_dim = 10
        self.flatten = NnnReshape((-1, inp_dim))
        self.dropout = nn.Dropout(p=0.25)

        self.dense00 = nn_linear_bn(inp_dim, mid_dim * 1, bias=True)
        self.dense10 = nn_linear_bn(mid_dim * 1, mid_dim * 1, bias=False)
        self.dense11 = nn_linear_bn(mid_dim * 1, mid_dim * 1, bias=False)
        self.dense20 = nn_linear_bn(mid_dim * 2, mid_dim * 2, bias=False)
        self.dense21 = nn_linear_bn(mid_dim * 2, mid_dim * 2, bias=False)
        self.dense30 = nn_linear_bn(mid_dim * 4, mid_dim * 4, bias=True)

        self.dense_out = nn.Sequential(
            self.dropout,
            nn.Linear(mid_dim * 4, out_dim, bias=False),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x00):
        x00 = self.flatten(x00)
        x10 = self.dense00(x00)

        x11 = self.dense10(x10)
        x11 = self.dense11(x11) + x10
        x20 = torch.cat((x10, x11), dim=1)

        x21 = self.dense20(x20)
        x21 = self.dense21(x21) + x20
        x30 = torch.cat((x20, x21), dim=1)

        x30 = self.dense30(x30)
        self.dropout.p = rd.uniform(0.125, 0.375)
        x_o = self.dense_out(x30)
        return x_o


class SE1dNet(nn.Module):
    def __init__(self):
        super(SE1dNet, self).__init__()
        inp_dim = 28 ** 2
        self.dropout = nn.Dropout(p=0.25)
        self.flatten = NnnReshape((-1, inp_dim))

        self.dense0 = nn_linear_bn(28 ** 2, 96, bias=True)
        self.se0 = nn_se_1d(96)
        self.dense1 = nn_linear_bn(96, 192, bias=False)
        self.se1 = nn_se_1d(192)

        self.dense_out = nn.Sequential(
            self.dropout,
            nn.Linear(192, 10, bias=False),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        x = self.flatten(x)

        x = self.dense0(x)
        x = x * self.se0(x)
        x = self.dense1(x)
        x = x * self.se1(x)

        self.dropout.p = rd.uniform(0.125, 0.375)
        x = self.dense_out(x)
        return x


'''input 2D'''


class ConvNet(nn.Module):  # Conv2d
    def __init__(self, mid_dim=128, img_shape=(32, 32, 3)):
        super().__init__()
        assert img_shape[0] == img_shape[1]
        global_size = int(((img_shape[0] - 2 - 2) / 2 - 2) / 2)
        inp_dim = img_shape[2]

        self.dropout = nn.Dropout(p=0.25)
        self.net = nn.Sequential(
            nn.Conv2d(inp_dim, 32, 3, 1, padding=0, bias=True),
            nn.ReLU(),

            nn_conv2d_avg2(32, 32, 3, 1, padding=0, bias=True),
            nn_conv2d_avg2(32, 48, 3, 1, padding=0, bias=True),

            # nn.BatchNorm2d(48),
            nn.Conv2d(48, mid_dim, global_size, 1, padding=0, bias=True),
            nn.Hardswish(),

            NnnReshape((-1, mid_dim)),
            # nn.BatchNorm1d(mid_dim),
            nn.Linear(mid_dim, mid_dim, bias=True),
            nn.Hardswish(),

            # nn.BatchNorm1d(mid_dim),
            self.dropout,
            nn.Linear(mid_dim, 10, bias=True),
        )

    def forward(self, x):
        self.dropout.p = rd.uniform(0.125, 0.375)
        return self.net(x)


class ConvNetBatchNorm(nn.Module):
    def __init__(self, mid_dim=128, img_shape=(32, 32, 3)):
        super().__init__()
        assert img_shape[0] == img_shape[1]
        global_size = int(((img_shape[0] - 2 - 2) / 2 - 2) / 2)
        inp_dim = img_shape[2]

        self.dropout = nn.Dropout(p=0.25)
        self.net = nn.Sequential(
            nn.Conv2d(inp_dim, 32, 3, 1, padding=0, bias=True),
            nn.ReLU(),

            nn_conv2d_bn_avg2(32, 32, 3, 1, padding=0, bias=True),
            nn_conv2d_bn_avg2(32, 48, 3, 1, padding=0, bias=True),

            nn.BatchNorm2d(48),
            nn.Conv2d(48, mid_dim, global_size, 1, padding=0, bias=True),
            nn.Hardswish(),

            NnnReshape((-1, mid_dim)),
            nn.BatchNorm1d(mid_dim),
            nn.Linear(mid_dim, mid_dim, bias=True),
            nn.Hardswish(),

            nn.BatchNorm1d(mid_dim),
            self.dropout,
            nn.Linear(mid_dim, 10, bias=True),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        self.dropout.p = rd.uniform(0.125, 0.375)
        return self.net(x)


class SENet(nn.Module):  # Squeeze-and-Excitation Network
    def __init__(self, mid_dim=128, img_shape=(32, 32, 3)):
        super().__init__()
        assert img_shape[0] == img_shape[1]
        global_size = int(((img_shape[0] - 2 - 2) / 2 - 2) / 2)
        inp_dim = img_shape[2]

        self.conv0 = nn.Sequential(
            nn.Conv2d(inp_dim, 32, 3, 1, padding=0, bias=True),
            nn.ReLU(),
        )
        self.conv1 = nn_conv2d_avg2(32, 32, 3, 1, padding=0, bias=False)
        self.conv1_se = nn_se_2d(32)
        self.conv2 = nn_conv2d_avg2(32, 48, 3, 1, padding=0, bias=False)
        self.conv2_se = nn_se_2d(48)

        self.conv3 = nn.Sequential(
            # nn.BatchNorm2d(48),
            nn.Conv2d(48, mid_dim, global_size, 1, padding=0, bias=False),
            nn.Hardswish(),
        )

        self.dropout = nn.Dropout(p=0.25)
        self.dense0 = nn.Sequential(
            NnnReshape((-1, mid_dim)),
            # nn.BatchNorm1d(mid_dim),
            nn.Linear(mid_dim, mid_dim, bias=True),
            nn.Hardswish(),

            # nn.BatchNorm1d(mid_dim),
            self.dropout,
            nn.Linear(mid_dim, 10, bias=True),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x0):
        x0 = self.conv0(x0)
        x1 = self.conv1(x0)
        x1 *= self.conv1_se(x1)

        x2 = self.conv2(x1)
        x2 *= self.conv2_se(x2)

        x = self.conv3(x2)

        self.dropout.p = rd.uniform(0.125, 0.375)
        x = self.dense0(x)
        return x


class SENetBatchNorm(nn.Module):  # Squeeze-and-Excitation Network
    def __init__(self, mid_dim=128, img_shape=(32, 32, 3)):
        super().__init__()
        assert img_shape[0] == img_shape[1]
        global_size = int(((img_shape[0] - 2 - 2) / 2 - 2) / 2)
        inp_dim = img_shape[2]

        self.conv0 = nn.Sequential(
            nn.Conv2d(inp_dim, 32, 3, 1, padding=0, bias=True),
            nn.ReLU(),
        )
        self.conv1 = nn_conv2d_bn_avg2(32, 32, 3, 1, padding=0, bias=False)
        self.conv1_se = nn_se_2d(32)
        self.conv2 = nn_conv2d_bn_avg2(32, 48, 3, 1, padding=0, bias=False)
        self.conv2_se = nn_se_2d(48)

        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(48),
            nn.Conv2d(48, mid_dim, global_size, 1, padding=0, bias=False),
            nn.Hardswish(),
        )

        self.dropout = nn.Dropout(p=0.25)
        self.dense0 = nn.Sequential(
            NnnReshape((-1, mid_dim)),
            nn.BatchNorm1d(mid_dim),
            nn.Linear(mid_dim, mid_dim, bias=True),
            nn.Hardswish(),

            nn.BatchNorm1d(mid_dim),
            self.dropout,
            nn.Linear(mid_dim, 10, bias=True),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x0):
        x0 = self.conv0(x0)
        x1 = self.conv1(x0)
        x1 *= self.conv1_se(x1)

        x2 = self.conv2(x1)
        x2 *= self.conv2_se(x2)

        x = self.conv3(x2)

        self.dropout.p = rd.uniform(0.125, 0.375)
        x = self.dense0(x)
        return x


'''network utils'''


class NnnReshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)


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
        nn.BatchNorm2d(inp_dim),
        nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=padding, bias=bias),
        nn.AvgPool2d(2),
        nn.ReLU(),
    )


def nn_conv2d_avg2(inp_dim, out_dim, kernel_size, stride, padding, bias):
    return nn.Sequential(
        nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=padding, bias=bias),
        nn.AvgPool2d(2),
        nn.ReLU(),
    )


def nn_se_1d(inp_dim):
    return nn.Sequential(
        nn.Linear(inp_dim, inp_dim, bias=False),
        nn.ReLU(inplace=True),
        nn.Linear(inp_dim, inp_dim, bias=False),
        nn.Sigmoid(),
    )


def nn_se_2d(inp_dim, ):
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        NnnReshape((-1, inp_dim)),
        nn.Linear(inp_dim, inp_dim, bias=False),  # should bias=False
        nn.ReLU(inplace=True),
        nn.Linear(inp_dim, inp_dim, bias=False),  # should bias=False
        nn.Sigmoid(),
        NnnReshape((-1, inp_dim, 1, 1)),
    )
