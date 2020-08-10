import numpy.random as rd

import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, inp_dim, mid_dim, out_dim):
        super(Generator, self).__init__()

        def idx_dim(i):
            return int(mid_dim * 1.6487 ** i)

        def nn_convtp_sn_bn(i, kernel_size, stride, padding, bias):
            return nn.Sequential(
                # nn.utils.spectral_norm(
                #     nn.ConvTranspose2d(idx_dim(i), idx_dim(i - 1), kernel_size, stride, padding=padding, bias=bias),
                #     n_power_iterations=1,
                # ),
                nn.ConvTranspose2d(idx_dim(i), idx_dim(i - 1), kernel_size, stride, padding=padding, bias=bias),
                nn.BatchNorm2d(idx_dim(i - 1)),
                nn.ReLU(),
            )

        self.linear_blocks = nn.Sequential(
            nn_linear_bn(inp_dim, inp_dim, bias=True),
            nn_linear_bn(inp_dim, idx_dim(3) * (4 ** 2), bias=False),
        )
        self.flatten4 = nn_reshape(idx_dim(3), 4, 4)

        self.conv_blocks = nn.Sequential(
            nn_convtp_sn_bn(3, 4, 2, padding=1, bias=False),
            nn_convtp_sn_bn(2, 4, 2, padding=1, bias=False),
            nn_convtp_sn_bn(1, 4, 2, padding=1, bias=False),

            nn.Conv2d(idx_dim(0), out_dim, 1, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x0):
        x0 = self.linear_blocks(x0)
        x0 = self.flatten4(x0)
        x0 = self.conv_blocks(x0)
        return x0


class Discriminator(nn.Module):  # Conv2d + GlobalPooling
    def __init__(self, inp_dim, mid_dim, out_dim):
        super(Discriminator, self).__init__()

        def idx_dim(i):
            return int(mid_dim * 1.6487 ** i)

        def nn_conv2d_bn_avg2(i, kernel_size, stride, padding, bias):
            return nn.Sequential(
                nn.utils.spectral_norm(
                    nn.Conv2d(idx_dim(i), idx_dim(i + 1), kernel_size, stride, padding=padding, bias=bias),
                    n_power_iterations=1,
                ),
                # nn.Conv2d(idx_dim(i), idx_dim(i + 1), kernel_size, stride, padding=padding, bias=bias),
                nn.BatchNorm2d(idx_dim(i + 1)),
                nn.AvgPool2d(2),
                nn.ReLU(),
            )

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(inp_dim, mid_dim, 3, 1, padding=1, bias=True),
            nn.ReLU(),

            nn_conv2d_bn_avg2(0, 3, 1, padding=1, bias=False),
            nn_conv2d_bn_avg2(1, 3, 1, padding=1, bias=False),
            nn_conv2d_bn_avg2(2, 3, 1, padding=1, bias=False),
            nn.Conv2d(idx_dim(3), idx_dim(4), 4, 1, padding=0, bias=False),  # GlobalConv
            nn.BatchNorm2d(idx_dim(4)),
            nn_hswish(),
        )

        self.flatten = nn_reshape(-1)
        self.dropout_0 = nn.Dropout(p=0.25)

        self.dense_1 = nn.Sequential(
            nn.Linear(idx_dim(4), idx_dim(4), bias=False),
            nn_hswish(),
            self.dropout_0,
        )
        self.dense_0 = nn.Sequential(
            nn.Linear(idx_dim(4), out_dim, bias=True),
            # nn.Tanh(),
        )

    def forward(self, x):
        x = self.conv_blocks(x)

        x = self.flatten(x)
        self.dropout_0.p = rd.uniform(0.125, 0.375)
        x = self.dense_1(x)
        x = self.dense_0(x)
        return x


# def weights_init_normal(m):
#     classname = m.__class__.__name__
#     if classname.find("Conv") != -1:
#         torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif classname.find("BatchNorm2d") != -1:
#         torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
#         torch.nn.init.constant_(m.bias.data, 0.0)
# # Initialize weights
# generator.apply(weights_init_normal)
# discriminator.apply(weights_init_normal)

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


if __name__ == '__main__':
    gene = Generator(128, 24, 3)

    noise = torch.rand(2, 128, dtype=torch.float32)
    image = gene(noise)
    print(image.size())

    disc = Discriminator(3, 24, 1)
    real_disc = disc(image)
    print(real_disc.size())
