import numpy as np
import torch
import torch.nn as nn

"""ElegantRL AgentNet.py GitHub: YonV1943 
Issay, Easy Essay, 谐音: 意识
plan to fake Multiple GPU: torch.nn.DataParallel(net_block) in single server
plan to true Multiple GPU: distributed computing DRL using socket between multiple servers.

notice: 
(self, state_dim, action_dim, mid_dim) --> (self, mid_dim, state_dim, action_dim)
CriticTwin  def get_q1q2
"""

'''Actor'''


class Actor(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, action_dim), )

    def forward(self, state):
        return self.net(state).tanh()  # action

    def get__a_noisy(self, state, a_std):  # action_std
        action = self.net(state).tanh()
        return (action + torch.randn_like(action) * a_std).clamp(-1.0, 1.0)


class ActorSAC(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net__state = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                        nn.Linear(mid_dim, mid_dim), nn.ReLU(), )
        self.net_action = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                        nn.Linear(mid_dim, action_dim), )
        self.net__a_std = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                        nn.Linear(mid_dim, action_dim), )

        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))  # a constant
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, state):
        tmp = self.net__state(state)
        return self.net_action(tmp).tanh()  # action

    def get__a_noisy(self, state):
        t_tmp = self.net__state(state)
        a_avg = self.net_action(t_tmp)
        a_std = self.net__a_std(t_tmp).clamp(-16, 2).exp()
        return torch.normal(a_avg, a_std).tanh()  # re-parameterize

    def get__a__log_prob(self, s):
        t_tmp = self.net__state(s)
        a_avg = self.net_action(t_tmp)
        a_std_log = self.net__a_std(t_tmp).clamp(-16, 2)
        a_std = a_std_log.exp()

        noise = torch.randn_like(a_avg, requires_grad=True, device=self.device)
        a_tan = (a_avg + a_std * noise).tanh()  # action.tanh()
        log_prob = a_std_log + self.sqrt_2pi_log + noise.pow(2).__mul__(0.5) + (-a_tan.pow(2) + 1.000001).log()
        return a_tan, log_prob.sum(1, keepdim=True)


class ActorPPO(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, action_dim), )
        layer_norm(self.net[-1], std=0.1)  # output layer of action

        self.a_std_log = nn.Parameter(torch.zeros((1, action_dim)) - 0.5, requires_grad=True)  # trainable parameter
        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))  # a constant

    def forward(self, state):
        return self.net(state).tanh()  # action

    def get__a_noisy__noise(self, state):
        a_avg = self.net(state)
        a_std = self.a_std_log.exp()

        noise = torch.randn_like(a_avg)
        a_noisy = a_avg + noise * a_std
        return a_noisy, noise

    def compute__log_prob(self, state, a_noise):
        a_avg = self.net(state)
        a_std = self.a_std_log.exp()
        delta = ((a_avg - a_noise) / a_std).pow(2).__mul__(0.5)
        log_prob = -(self.a_std_log + self.sqrt_2pi_log + delta)
        return log_prob.sum(1)


'''Critic'''


class Critic(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, 1), )

    def forward(self, state, action):
        return self.net(torch.cat((state, action), dim=1))  # q value


class CriticTwin(nn.Module):  # shared parameter
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net_sa = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, mid_dim), nn.ReLU(), )  # concat(state, action)
        self.net_q1 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                    nn.Linear(mid_dim, 1), )  # q1 value
        self.net_q2 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                    nn.Linear(mid_dim, 1), )  # q2 value

    def forward(self, state, action):
        tmp = self.net_sa(torch.cat((state, action), dim=1))
        return self.net_q1(tmp)  # q1 value

    def get_q1q2(self, state, action):
        tmp = self.net_sa(torch.cat((state, action), dim=1))
        return self.net_q1(tmp), self.net_q2(tmp)  # q1 value, q2 value


class CriticAdv(nn.Module):
    def __init__(self, mid_dim, state_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, 1), )
        layer_norm(self.net[-1], std=1.0)  # output layer of action

    def forward(self, state):
        return self.net(state)  # q value


'''Q Network'''


class QNet(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, action_dim), )

    def forward(self, state):
        return self.net(state)  # q value


class QNetTwin(nn.Module):  # shared parameter
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net__s = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, mid_dim), nn.ReLU(), )  # state
        self.net_q1 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, action_dim), )  # q1 value
        self.net_q2 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, action_dim), )  # q2 value

    def forward(self, state):
        tmp = self.net__s(state)
        return self.net_q1(tmp)  # q1 value

    def get__q1_q2(self, state):
        tmp = self.net__s(state)
        return self.net_q1(tmp), self.net_q2(tmp)  # q1 q2 value


class QNetDuel(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net__state = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                        nn.Linear(mid_dim, mid_dim), nn.ReLU(), )
        self.net__value = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                        nn.Linear(mid_dim, 1), )  # q value
        self.net__adv_v = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                        nn.Linear(mid_dim, action_dim), )  # advantage function value

    def forward(self, state):
        t_tmp = self.net__state(state)
        q_val = self.net__value(t_tmp)  # q value
        q_adv = self.net__adv_v(t_tmp)  # advantage function value
        return q_val + q_adv - q_adv.mean(dim=1, keepdim=True)  # dueling q value


class QNetDuelTwin(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net__state = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                        nn.Linear(mid_dim, mid_dim), nn.ReLU(), )

        self.net_val1 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                      nn.Linear(mid_dim, 1), )
        self.net_val2 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                      nn.Linear(mid_dim, 1), )
        self.net_adv1 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                      nn.Linear(mid_dim, action_dim), )
        self.net_adv2 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                      nn.Linear(mid_dim, action_dim), )

    def forward(self, state):
        t_tmp = self.net__state(state)
        q_val = self.net_val1(t_tmp)
        q_adv = self.net_adv1(t_tmp)
        return q_val + q_adv - q_adv.mean(dim=1, keepdim=True)  # single dueling q value

    def get_q1q2(self, state):
        tmp = self.net__state(state)

        val1 = self.net_val1(tmp)
        adv1 = self.net_adv1(tmp)
        q1 = val1 + adv1 - adv1.mean(dim=1, keepdim=True)

        val2 = self.net_val2(tmp)
        adv2 = self.net_adv2(tmp)
        q2 = val2 + adv2 - adv2.mean(dim=1, keepdim=True)
        return q1, q2


'''Integrated network (parameter sharing)'''


class InterSPG(nn.Module):  # plan to update __init__
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.log_std_min = -20
        self.log_std_max = 2
        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.enc_s = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                   nn.Linear(mid_dim, mid_dim), )  # state
        self.enc_a = nn.Sequential(nn.Linear(action_dim, mid_dim), nn.ReLU(),
                                   nn.Linear(mid_dim, mid_dim), )  # action without nn.Tanh()

        self.net = DenseNet(mid_dim)
        net_dim = self.net.out_dim

        self.dec_am = nn.Sequential(nn.Linear(net_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, action_dim), )  # action_mean
        self.dec_ad = nn.Sequential(nn.Linear(net_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, action_dim), )  # action_std_log (d means standard deviation)
        self.dec_q1 = nn.Sequential(nn.Linear(net_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, 1), )  # q_value1
        self.dec_q2 = nn.Sequential(nn.Linear(net_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, 1), )  # q_value2

    def forward(self, state):
        x = self.enc_s(state)
        x = self.net(x)
        return self.dec_am(x).tanh()  # action

    def get__noise_action(self, state):
        s_ = self.enc_s(state)
        a_ = self.net(s_)
        a_avg = self.dec_am(a_)  # NOTICE! it is a_avg without tensor.tanh()

        a_std_log = self.dec_ad(a_).clamp(self.log_std_min, self.log_std_max)
        a_std = a_std_log.exp()
        return torch.normal(a_avg, a_std).tanh()  # action

    def get__a__log_prob(self, state):  # actor
        s_ = self.enc_s(state)
        a_ = self.net(s_)

        """add noise to action, stochastic policy"""
        a_avg = self.dec_am(a_)  # NOTICE! it is action without .tanh()
        a_std_log = self.dec_ad(a_).clamp(self.log_std_min, self.log_std_max)
        a_std = a_std_log.exp()

        noise = torch.randn_like(a_avg, requires_grad=True)
        a_tan = (a_avg + a_std * noise).tanh()
        log_prob = a_std_log + self.sqrt_2pi_log + noise.pow(2).__mul__(0.5) + (-a_tan.pow(2) + 1.000001).log()
        return a_tan, log_prob.sum(1, keepdim=True)

    def get__q__log_prob(self, state):
        s_ = self.enc_s(state)
        a_ = self.net(s_)

        a_avg = self.dec_am(a_)  # NOTICE! it is action without .tanh()
        a_std_log = self.dec_ad(a_).clamp(self.log_std_min, self.log_std_max)
        a_std = a_std_log.exp()

        noise = torch.randn_like(a_avg, requires_grad=True)
        a_tan = (a_avg + a_std * noise).tanh()
        log_prob = a_std_log + self.sqrt_2pi_log + noise.pow(2).__mul__(0.5) + (-a_tan.pow(2) + 1.000001).log()

        a_ = self.enc_a(a_tan)
        q_ = self.net(s_ + a_)
        q = torch.min(self.dec_q1(q_), self.dec_q2(q_))
        return q, log_prob

    def get_q1q2(self, s, a):  # critic
        q_ = self.net(self.enc_s(s) + self.enc_a(a))
        return self.dec_q1(q_), self.dec_q2(q_)


class InterPPO(nn.Module):  # plan to update __init__
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()

        def idx_dim(i):
            return int(8 * 1.5 ** i)

        out_dim = mid_dim
        self.enc_s = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                   nn.Linear(mid_dim, mid_dim), )

        self.dec__a = nn.Sequential(nn.Linear(out_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, action_dim), )

        self.dec_q1 = nn.Sequential(nn.Linear(out_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, 1), )
        self.dec_q2 = nn.Sequential(nn.Linear(out_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, 1), )

        self.a_std_log = nn.Parameter(torch.zeros(1, action_dim) - 0.5, requires_grad=True)
        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))
        layer_norm(self.dec__a[-1], std=0.01)
        layer_norm(self.dec_q1[-1], std=0.01)
        layer_norm(self.dec_q2[-1], std=0.01)

    def forward(self, s):
        s_ = self.enc_s(s)
        return self.dec__a(s_).tanh()

    def get__a_noise__noise(self, state):
        s_ = self.enc_s(state)
        a_avg = self.dec__a(s_)

        noise = torch.randn_like(a_avg)
        a_noise = a_avg + noise * self.a_std_log.exp()
        return a_noise, noise

    def get__q__log_prob(self, state, noise):
        s_ = self.enc_s(state)

        q = torch.min(self.dec_q1(s_), self.dec_q2(s_))
        log_prob = -(self.a_std_log + self.sqrt_2pi_log + noise.pow(2).__mul__(0.5))
        return q, log_prob.sum(1)

    def get__q1q2__log_prob(self, state, action):
        s_ = self.enc_s(state)

        q1 = self.dec_q1(s_)
        q2 = self.dec_q2(s_)

        a_avg = self.dec__a(s_)
        a_std = self.a_std_log.exp()
        delta = ((a_avg - action) / a_std).pow(2).__mul__(0.5)
        log_prob = -(self.a_std_log + self.sqrt_2pi_log + delta)
        return q1, q2, log_prob.sum(1)


"""Private Utils"""


class NnnReshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.view((x.size(0),) + self.args)


class DenseNet(nn.Module):  # plan to hyper-param: layer_number
    def __init__(self, mid_dim):
        super().__init__()
        assert (mid_dim / (2 ** 3)) % 1 == 0

        def id2dim(i):
            return int((3 / 2) ** i * mid_dim)

        self.dense1 = nn.Sequential(nn.Linear(id2dim(0), id2dim(0) // 2), nn.ReLU(), )
        self.dense2 = nn.Sequential(nn.Linear(id2dim(1), id2dim(1) // 2), nn.Hardswish(), )
        self.out_dim = id2dim(2)

        layer_norm(self.dense1[0], std=1.0)
        layer_norm(self.dense2[0], std=1.0)

    def forward(self, x1):
        x2 = torch.cat((x1, self.dense1(x1)), dim=1)
        x3 = torch.cat((x2, self.dense2(x2)), dim=1)
        return x3


def layer_norm(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)


"""backup (unnecessary part)"""


def demo__con2d_state_encoder(state_dim=(4, 96, 96), action_dim=3, mid_dim=2 ** 7):
    # env_name='CarRacing-Fix', state_dim=(frame_num, height, width)

    def idx_dim(i):
        return int(16 * 1.5 ** i)

    net = nn.Sequential(NnnReshape(*state_dim),  # -> [batch_size, frame_num, 96, 96]
                        nn.Conv2d(state_dim[0], idx_dim(0), 4, 2, bias=True), nn.LeakyReLU(),
                        nn.Conv2d(idx_dim(0), idx_dim(1), 3, 2, bias=False), nn.ReLU(),
                        nn.Conv2d(idx_dim(1), idx_dim(2), 3, 2, bias=False), nn.ReLU(),
                        nn.Conv2d(idx_dim(2), idx_dim(3), 3, 2, bias=True), nn.ReLU(),
                        nn.Conv2d(idx_dim(3), idx_dim(4), 3, 1, bias=True), nn.ReLU(),
                        nn.Conv2d(idx_dim(4), idx_dim(5), 3, 1, bias=True), nn.ReLU(),
                        NnnReshape(-1),
                        nn.Linear(idx_dim(5), mid_dim), nn.ReLU(),
                        nn.Linear(mid_dim, action_dim), )

    inp_dim = int(np.prod(state_dim))
    batch_size = 2

    inp = torch.ones((batch_size, inp_dim), dtype=torch.float32)
    out = net(inp)
    print('inp.shape:', inp.shape)
    print('out.shape:', out.shape)


def demo__what_is_log_prob():
    batch_size = 2
    action_dim = 3
    sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))

    '''compute the action distribution (action average and std)'''
    a_avg = torch.tensor((batch_size, action_dim), )  # a_avg = net__a(state)
    a_std_log = torch.tensor((batch_size, action_dim), ).clamp(-16, 2)  # a_avg = net__d(state)
    a_std = a_std_log.exp()

    """add noise to action in stochastic policy"""
    a_noise = a_avg + a_std * torch.randn_like(a_avg, requires_grad=True, device=self.device)
    # Can only use above code instead of below, because the tensor need gradients here.
    # a_noise = torch.normal(a_avg, a_std, requires_grad=True)

    '''compute log_prob according to mean and std of action (stochastic policy)'''
    a_delta = ((a_avg - a_noise) / a_std).pow(2) * 0.5
    # self.constant_log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))
    log_prob_noise = a_delta + a_std_log + sqrt_2pi_log

    # same as below:
    # from torch.distributions.normal import Normal
    # log_prob_noise = Normal(a_avg, a_std).log_prob(a_noise)
    # same as below:
    # a_delta = (a_avg - a_noise).pow(2) /(2* a_std.pow(2)
    # log_prob_noise = -a_delta - a_std.log() - np.log(np.sqrt(2 * np.pi))

    a_tan = a_noise.tanh()
    log_prob = log_prob_noise + (-a_tan.pow(2) + 1.000001).log()

    # same as below:
    # epsilon = 1e-6
    # log_prob = log_prob_noise - (1 - a_tan.pow(2) + epsilon).log()
    return a_tan, log_prob.sum(1)


def demo__normalization():
    """Normalization
    After doing many experiments, I think that don't use BatchNorm or SpectralNorm in RL directly.
    See more in [
    强化学习需要批归一化(Batch Norm) 或归一化吗？
    Does Reinforcement Learning need BatchNorm or other Normalization?
    ](https://zhuanlan.zhihu.com/p/210761985)
    
    nn.BatchNorm1d()  # BatchNorm is conflict with RL
    nn.utils.spectral_norm(net)  # SpectralNorm is conflict with soft target update
    """
    pass
