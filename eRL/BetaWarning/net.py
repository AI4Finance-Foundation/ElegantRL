import torch
import torch.nn as nn
import numpy as np

'''Q Network'''


class QNet(nn.Module):  # nn.Module is a standard PyTorch Network
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, action_dim))

    def forward(self, state):
        return self.net(state)  # q value


class QNetDuel(nn.Module):  # Dueling DQN
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net__state = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                        nn.Linear(mid_dim, mid_dim), nn.ReLU())
        self.net_val = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                     nn.Linear(mid_dim, 1))  # q value
        self.net_adv = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                     nn.Linear(mid_dim, action_dim))  # advantage function value 1

    def forward(self, state):
        t_tmp = self.net__state(state)
        q_val = self.net_val(t_tmp)
        q_adv = self.net_adv(t_tmp)
        return q_val + q_adv - q_adv.mean(dim=1, keepdim=True)  # dueling q value


class QNetTwin(nn.Module):  # Double DQN
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net__s = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, mid_dim), nn.ReLU())  # state
        self.net_q1 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, action_dim))  # q1 value
        self.net_q2 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, action_dim))  # q2 value

    def forward(self, state):
        tmp = self.net__s(state)
        return self.net_q1(tmp)  # single q value

    def get__q1_q2(self, state):
        tmp = self.net__s(state)
        q1 = self.net_q1(tmp)
        q2 = self.net_q2(tmp)
        return q1, q2  # twin q value


class QNetTwinDuel(nn.Module):  # D3QN: Dueling Double DQN
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net__state = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                        nn.Linear(mid_dim, mid_dim), nn.ReLU())
        self.net_val1 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                      nn.Linear(mid_dim, 1))  # q1 value
        self.net_val2 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                      nn.Linear(mid_dim, 1))  # q2 value
        self.net_adv1 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                      nn.Linear(mid_dim, action_dim))  # advantage function value 1
        self.net_adv2 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                      nn.Linear(mid_dim, action_dim))  # advantage function value 1

    def forward(self, state):
        t_tmp = self.net__state(state)
        q_val = self.net_val1(t_tmp)
        q_adv = self.net_adv1(t_tmp)
        return q_val + q_adv - q_adv.mean(dim=1, keepdim=True)  # single dueling q value

    def get__q1_q2(self, state):
        tmp = self.net__state(state)

        val1 = self.net_val1(tmp)
        adv1 = self.net_adv1(tmp)
        q1 = val1 + adv1 - adv1.mean(dim=1, keepdim=True)

        val2 = self.net_val2(tmp)
        adv2 = self.net_adv2(tmp)
        q2 = val2 + adv2 - adv2.mean(dim=1, keepdim=True)
        return q1, q2


'''Policy Network (Actor)'''


class Actor(nn.Module):  # DPG: Deterministic Policy Gradient
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, action_dim))

    def forward(self, state):
        return self.net(state).tanh()  # action

    def get_action(self, state, action_std):
        action = self.net(state).tanh()
        noise = (torch.randn_like(action) * action_std).clamp(-0.5, 0.5)
        return (action + noise).clamp(-1.0, 1.0)


class ActorPPO(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        if isinstance(state_dim, int):
            self.net = nn.Sequential(
                nn.Linear(state_dim, mid_dim), nn.ReLU(),
                nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                nn.Linear(mid_dim, action_dim), )
        else:
            def set_dim(i):
                return int(12 * 1.5 ** i)

            self.net = nn.Sequential(
                NnnReshape(*state_dim),  # -> [batch_size, 4, 96, 96]
                nn.Conv2d(state_dim[0], set_dim(0), 4, 2, bias=True), nn.LeakyReLU(),
                nn.Conv2d(set_dim(0), set_dim(1), 3, 2, bias=False), nn.ReLU(),
                nn.Conv2d(set_dim(1), set_dim(2), 3, 2, bias=False), nn.ReLU(),
                nn.Conv2d(set_dim(2), set_dim(3), 3, 2, bias=True), nn.ReLU(),
                nn.Conv2d(set_dim(3), set_dim(4), 3, 1, bias=True), nn.ReLU(),
                nn.Conv2d(set_dim(4), set_dim(5), 3, 1, bias=True), nn.ReLU(),
                NnnReshape(-1),
                nn.Linear(set_dim(5), mid_dim), nn.ReLU(),
                nn.Linear(mid_dim, action_dim), )
        self.a_std_log = nn.Parameter(torch.zeros((1, action_dim)) - 0.5, requires_grad=True)  # trainable parameter
        self.sqrt_2pi_log = 0.9189385332046727  # =np.log(np.sqrt(2 * np.pi))

        layer_norm(self.net[-1], std=0.1)  # output layer for action

    def forward(self, state):
        return self.net(state).tanh()  # action

    def get__action_noise(self, state):
        a_avg = self.net(state)
        a_std = self.a_std_log.exp()

        noise = torch.randn_like(a_avg)
        action = a_avg + noise * a_std
        return action, noise

    def compute__log_prob(self, state, action):
        a_avg = self.net(state)
        a_std = self.a_std_log.exp()
        delta = ((a_avg - action) / a_std).pow(2).__mul__(0.5)  # __mul__(0.5) is * 0.5
        log_prob = -(self.a_std_log + self.sqrt_2pi_log + delta)
        return log_prob.sum(1)


class ActorSAC(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim, if_use_dn=False):
        super().__init__()
        if if_use_dn:  # use a DenseNet (DenseNet has both shallow and deep linear layer)
            nn_dense_net = DenseNet(mid_dim)
            self.net__state = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                            nn_dense_net, )
            lay_dim = nn_dense_net.out_dim
        else:  # use a simple network. Deeper network does not mean better performance in RL.
            lay_dim = mid_dim
            self.net__state = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                            nn.Linear(mid_dim, lay_dim), nn.ReLU(),
                                            nn.Linear(mid_dim, lay_dim), nn.Hardswish(), )
        self.net__a_avg = nn.Linear(lay_dim, action_dim)  # the average of action
        self.net__a_std = nn.Linear(lay_dim, action_dim)  # the log_std of action

        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))
        layer_norm(self.net__a_avg, std=0.01)  # net[-1] is output layer for action, it is no necessary.

    def forward(self, state):
        tmp = self.net__state(state)
        return self.net__a_avg(tmp).tanh()  # action

    def get_action(self, state):
        t_tmp = self.net__state(state)
        a_avg = self.net__a_avg(t_tmp)  # NOTICE! it is a_avg without .tanh()
        a_std = self.net__a_std(t_tmp).clamp(-20, 2).exp()
        return torch.normal(a_avg, a_std).tanh()  # re-parameterize

    def get__action__log_prob(self, state):
        t_tmp = self.net__state(state)
        a_avg = self.net__a_avg(t_tmp)  # NOTICE! it needs a_avg.tanh()
        a_std_log = self.net__a_std(t_tmp).clamp(-20, 2)
        a_std = a_std_log.exp()

        """add noise to action in stochastic policy"""
        noise = torch.randn_like(a_avg, requires_grad=True)
        a_tan = (a_avg + a_std * noise).tanh()  # action.tanh()
        # Can only use above code instead of below, because the tensor need gradients here.
        # a_noise = torch.normal(a_avg, a_std, requires_grad=True)

        '''compute log_prob according to mean and std of action (stochastic policy)'''
        # self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))
        log_prob = a_std_log + self.sqrt_2pi_log + noise.pow(2).__mul__(0.5)  # noise.pow(2) * 0.5
        # same as below:
        # from torch.distributions.normal import Normal
        # log_prob_noise = Normal(a_avg, a_std).log_prob(a_noise)
        # log_prob = log_prob_noise + (-a_noise_tanh.pow(2) + 1.000001).log()
        # same as below:
        # a_delta = (a_avg - a_noise).pow(2) /(2*a_std.pow(2))
        # log_prob_noise = -a_delta - a_std.log() - np.log(np.sqrt(2 * np.pi))
        # log_prob = log_prob_noise + (-a_noise_tanh.pow(2) + 1.000001).log()

        log_prob = log_prob + (-a_tan.pow(2) + 1.000001).log()  # fix log_prob using the derivative of action.tanh()
        # same as below:
        # epsilon = 1e-6
        # log_prob = log_prob_noise - (1 - a_noise_tanh.pow(2) + epsilon).log()
        return a_tan, log_prob.sum(1, keepdim=True)


'''Value Network (Critic)'''


class Critic(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, 1))

    def forward(self, state, action):
        return self.net(torch.cat((state, action), dim=1))  # q value


class CriticAdv(nn.Module):
    def __init__(self, state_dim, mid_dim):
        super().__init__()
        if isinstance(state_dim, int):
            self.net = nn.Sequential(
                nn.Linear(state_dim, mid_dim), nn.ReLU(),
                nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                nn.Linear(mid_dim, 1)
            )
        else:
            def set_dim(i):
                return int(12 * 1.5 ** i)

            self.net = nn.Sequential(
                NnnReshape(*state_dim),  # -> [batch_size, 4, 96, 96]
                nn.Conv2d(state_dim[0], set_dim(0), 4, 2, bias=True), nn.LeakyReLU(),
                nn.Conv2d(set_dim(0), set_dim(1), 3, 2, bias=False), nn.ReLU(),
                nn.Conv2d(set_dim(1), set_dim(2), 3, 2, bias=False), nn.ReLU(),
                nn.Conv2d(set_dim(2), set_dim(3), 3, 2, bias=True), nn.ReLU(),
                nn.Conv2d(set_dim(3), set_dim(4), 3, 1, bias=True), nn.ReLU(),
                nn.Conv2d(set_dim(4), set_dim(5), 3, 1, bias=True), nn.ReLU(),
                NnnReshape(-1),
                nn.Linear(set_dim(5), mid_dim), nn.ReLU(),
                nn.Linear(mid_dim, 1),
            )

        layer_norm(self.net[-1], std=0.5)  # output layer for q value

    def forward(self, state):
        return self.net(state)  # q value


class CriticTwin(nn.Module):  # 2020-06-18
    def __init__(self, mid_dim, state_dim, action_dim, if_use_dn=False):
        super().__init__()

        if if_use_dn:  # use DenseNet (DenseNet has both shallow and deep linear layer)
            nn_dense = DenseNet(mid_dim)
            lay_dim = nn_dense.out_dim
            self.net_sa = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                        nn_dense, )  # state-action value function
        else:  # use a simple network for actor. Deeper network does not mean better performance in RL.
            lay_dim = mid_dim
            self.net_sa = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                        nn.Linear(mid_dim, lay_dim), nn.ReLU(),
                                        nn.Linear(mid_dim, lay_dim), nn.Hardswish())

        self.net_q1 = nn.Linear(lay_dim, 1)
        self.net_q2 = nn.Linear(lay_dim, 1)
        layer_norm(self.net_q1, std=0.1)
        layer_norm(self.net_q2, std=0.1)

        '''Not need to use both SpectralNorm and TwinCritic
        I choose TwinCritc instead of SpectralNorm, 
        because SpectralNorm is conflict with soft target update,

        if is_spectral_norm:
            self.net1[1] = nn.utils.spectral_norm(self.dec_q1[1])
            self.net2[1] = nn.utils.spectral_norm(self.dec_q2[1])
        '''

    def forward(self, state, action):
        tmp = self.net_sa(torch.cat((state, action), dim=1))
        return self.net_q1(tmp)

    def get__q1_q2(self, state, action):
        tmp = self.net_sa(torch.cat((state, action), dim=1))
        return self.net_q1(tmp), self.net_q2(tmp)  # q1, q2 value


'''Integrated Network (Parameter sharing)'''


class InterDPG(nn.Module):  # DPG means deterministic policy gradient
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()
        self.enc_s = nn.Sequential(
            nn.Linear(state_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
        )
        self.enc_a = nn.Sequential(
            nn.Linear(action_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
        )

        self.net = DenseNet(mid_dim)
        net_out_dim = self.net.out_dim

        self.dec_a = nn.Sequential(nn.Linear(net_out_dim, mid_dim), nn.Hardswish(),
                                   nn.Linear(mid_dim, action_dim), nn.Tanh(), )
        self.dec_q = nn.Sequential(nn.Linear(net_out_dim, mid_dim), nn.Hardswish(),
                                   nn.utils.spectral_norm(nn.Linear(mid_dim, 1)), )

    @staticmethod
    def add_noise(a, noise_std):  # 2020-03-03
        a_temp = torch.normal(a, noise_std)
        # mask = ((a_temp < -1.0) + (a_temp > 1.0)).type(torch.float32)  # 2019-12-30
        mask = torch.tensor((a_temp < -1.0) + (a_temp > 1.0), dtype=torch.float32).cuda()

        noise_uniform = torch.rand_like(a)
        a_noise = noise_uniform * mask + a_temp * (-mask + 1)
        return a_noise

    def forward(self, s, noise_std=0.0):  # actor
        s_ = self.enc_s(s)
        a_ = self.net(s_)
        a = self.dec_a(a_)
        return a if noise_std == 0.0 else self.add_noise(a, noise_std)

    def critic(self, s, a):
        s_ = self.enc_s(s)
        a_ = self.enc_a(a)
        q_ = self.net(s_ + a_)
        q = self.dec_q(q_)
        return q

    def next__q_a(self, s, s_next, noise_std):
        s_ = self.enc_s(s)
        a_ = self.net(s_)
        a = self.dec_a(a_)

        '''q_target (without noise)'''
        a_ = self.enc_a(a)
        s_next_ = self.enc_s(s_next)
        q_target0_ = self.net(s_next_ + a_)
        q_target0 = self.dec_q(q_target0_)

        '''q_target (with noise)'''
        a_noise = self.add_noise(a, noise_std)
        a_noise_ = self.enc_a(a_noise)
        q_target1_ = self.net(s_next_ + a_noise_)
        q_target1 = self.dec_q(q_target1_)

        q_target = (q_target0 + q_target1) * 0.5
        return q_target, a


class InterSPG(nn.Module):  # SPG means stochastic policy gradient
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.log_sqrt_2pi_sum = np.log(np.sqrt(2 * np.pi)) * action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.enc_s = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                   nn.Linear(mid_dim, mid_dim), )  # state
        self.enc_a = nn.Sequential(nn.Linear(action_dim, mid_dim), nn.ReLU(),
                                   nn.Linear(mid_dim, mid_dim), )  # action without nn.Tanh()

        self.net = DenseNet(mid_dim)
        net_out_dim = self.net.out_dim

        self.dec_a = nn.Sequential(nn.Linear(net_out_dim, mid_dim), nn.ReLU(),
                                   nn.Linear(mid_dim, action_dim), )  # action_mean
        self.dec_d = nn.Sequential(nn.Linear(net_out_dim, mid_dim), nn.ReLU(),
                                   nn.Linear(mid_dim, action_dim), )  # action_std_log (d means standard deviation)
        self.dec_q1 = nn.Sequential(nn.Linear(net_out_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, 1), )  # q1 value
        self.dec_q2 = nn.Sequential(nn.Linear(net_out_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, 1), )  # q2 value

        layer_norm(self.dec_a[-1], std=0.5)
        layer_norm(self.dec_d[-1], std=0.1)
        layer_norm(self.dec_q1[-1], std=0.5)
        layer_norm(self.dec_q2[-1], std=0.5)

    def forward(self, s):
        x = self.enc_s(s)
        x = self.net(x)
        a_avg = self.dec_a(x)
        return a_avg.tanh()

    def get__noise_action(self, s):
        s_ = self.enc_s(s)
        a_ = self.net(s_)
        a_avg = self.dec_a(a_)  # NOTICE! it is a_avg without tensor.tanh()

        a_std_log = self.dec_d(a_).clamp(-20, 2)
        a_std = a_std_log.exp()

        action = torch.normal(a_avg, a_std)  # NOTICE! it is action without .tanh()
        return action.tanh()

    def get__a__log_prob(self, state):  # actor
        s_ = self.enc_s(state)
        a_ = self.net(s_)

        """add noise to action, stochastic policy"""
        a_avg = self.dec_a(a_)  # NOTICE! it is action without .tanh()
        a_std_log = self.dec_d(a_).clamp(-20, 2)
        a_std = a_std_log.exp()

        noise = torch.randn_like(a_avg, requires_grad=True)
        a_noise = a_avg + a_std * noise

        a_noise_tanh = a_noise.tanh()
        fix_term = (-a_noise_tanh.pow(2) + 1.00001).log()
        log_prob = (noise.pow(2) / 2 + a_std_log + fix_term).sum(1, keepdim=True) + self.log_sqrt_2pi_sum
        return a_noise_tanh, log_prob

    def get__q__log_prob(self, state):
        s_ = self.enc_s(state)
        a_ = self.net(s_)

        """add noise to action, stochastic policy"""
        a_avg = self.dec_a(a_)  # NOTICE! it is action without .tanh()
        a_std_log = self.dec_d(a_).clamp(-20, 2)
        a_std = a_std_log.exp()

        noise = torch.randn_like(a_avg, requires_grad=True)
        a_noise = a_avg + a_std * noise

        a_noise_tanh = a_noise.tanh()
        fix_term = (-a_noise_tanh.pow(2) + 1.00001).log()
        log_prob = (noise.pow(2) / 2 + a_std_log + fix_term).sum(1, keepdim=True) + self.log_sqrt_2pi_sum

        '''get q'''
        a_ = self.enc_a(a_noise_tanh)
        q_ = self.net(s_ + a_)
        q = torch.min(self.dec_q1(q_), self.dec_q2(q_))
        return q, log_prob

    def get__q1_q2(self, s, a):  # critic
        s_ = self.enc_s(s)
        a_ = self.enc_a(a)
        q_ = self.net(s_ + a_)
        q1 = self.dec_q1(q_)
        q2 = self.dec_q2(q_)
        return q1, q2


class InterPPO(nn.Module):  # Pixel-level state version
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()

        def set_dim(i):
            return int(12 * 1.5 ** i)

        if isinstance(state_dim, int):
            self.enc_s = nn.Sequential(  # the only difference.
                nn.Linear(state_dim, mid_dim), nn.ReLU(),
                nn.Linear(mid_dim, mid_dim), )
        else:
            self.enc_s = nn.Sequential(
                NnnReshape(*state_dim),  # -> [batch_size, 4, 96, 96]
                nn.Conv2d(state_dim[0], set_dim(0), 4, 2, bias=True), nn.LeakyReLU(),
                nn.Conv2d(set_dim(0), set_dim(1), 3, 2, bias=False), nn.ReLU(),
                nn.Conv2d(set_dim(1), set_dim(2), 3, 2, bias=False), nn.ReLU(),
                nn.Conv2d(set_dim(2), set_dim(3), 3, 2, bias=True), nn.ReLU(),
                nn.Conv2d(set_dim(3), set_dim(4), 3, 1, bias=True), nn.ReLU(),
                nn.Conv2d(set_dim(4), set_dim(5), 3, 1, bias=True), nn.ReLU(),
                NnnReshape(-1),
                nn.Linear(set_dim(5), mid_dim), nn.ReLU(),
                nn.Linear(mid_dim, mid_dim), )
        out_dim = mid_dim

        self.dec_a = nn.Sequential(
            nn.Linear(out_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, action_dim),
        )
        self.a_std_log = nn.Parameter(torch.zeros(1, action_dim) - 0.5, requires_grad=True)

        self.dec_q1 = nn.Sequential(
            nn.Linear(out_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, 1),
        )
        self.dec_q2 = nn.Sequential(
            nn.Linear(out_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, 1),
        )

        layer_norm(self.dec_a[-1], std=0.01)
        layer_norm(self.dec_q1[-1], std=0.01)
        layer_norm(self.dec_q2[-1], std=0.01)

        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))

    def forward(self, s):
        s_ = self.enc_s(s)
        a_avg = self.dec_a(s_)
        return a_avg.tanh()

    def get__a_noise__noise(self, state):
        s_ = self.enc_s(state)
        a_avg = self.dec_a(s_)
        a_std = self.a_std_log.exp()

        # a_noise = torch.normal(a_avg, a_std) # same as below
        noise = torch.randn_like(a_avg)
        a_noise = a_avg + noise * a_std
        return a_noise, noise

    def get__q__log_prob(self, state, noise):
        s_ = self.enc_s(state)

        q = torch.min(self.dec_q1(s_), self.dec_q2(s_))
        log_prob = -(noise.pow(2) / 2 + self.a_std_log + self.sqrt_2pi_log).sum(1)
        return q, log_prob

    def get__q1_q2__log_prob(self, state, action):
        s_ = self.enc_s(state)

        q1 = self.dec_q1(s_)
        q2 = self.dec_q2(s_)

        a_avg = self.dec_a(s_)
        a_std = self.a_std_log.exp()
        log_prob = -(((a_avg - action) / a_std).pow(2) / 2 + self.a_std_log + self.sqrt_2pi_log).sum(1)
        return q1, q2, log_prob


"""utils"""


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

        def set_dim(i):
            return int((3 / 2) ** i * mid_dim)

        self.dense1 = nn.Sequential(nn.Linear(set_dim(0), set_dim(0) // 2), nn.ReLU(), )
        self.dense2 = nn.Sequential(nn.Linear(set_dim(1), set_dim(1) // 2), nn.Hardswish(), )
        self.out_dim = set_dim(2)

        layer_norm(self.dense1[0], std=1.0)
        layer_norm(self.dense2[0], std=1.0)

    def forward(self, x1):
        x2 = torch.cat((x1, self.dense1(x1)), dim=1)
        x3 = torch.cat((x2, self.dense2(x2)), dim=1)
        return x3


def layer_norm(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)


def demo__conv2d_state():
    state_dim = (4, 96, 96)
    batch_size = 3

    def set_dim(i):
        return int(12 * 1.5 ** i)

    mid_dim = 128
    net = nn.Sequential(
        NnnReshape(*state_dim),  # -> [batch_size, 4, 96, 96]
        nn.Conv2d(state_dim[0], set_dim(0), 4, 2, bias=True), nn.LeakyReLU(),
        nn.Conv2d(set_dim(0), set_dim(1), 3, 2, bias=False), nn.ReLU(),
        nn.Conv2d(set_dim(1), set_dim(2), 3, 2, bias=False), nn.ReLU(),
        nn.Conv2d(set_dim(2), set_dim(3), 3, 2, bias=True), nn.ReLU(),
        nn.Conv2d(set_dim(3), set_dim(4), 3, 1, bias=True), nn.ReLU(),
        nn.Conv2d(set_dim(4), set_dim(5), 3, 1, bias=True), nn.ReLU(),
        NnnReshape(-1),
        nn.Linear(set_dim(5), mid_dim), nn.ReLU(),
    )

    inp_shape = list(state_dim)
    inp_shape.insert(0, batch_size)
    inp = torch.ones(inp_shape, dtype=torch.float32)
    inp = inp.view(3, -1)
    print(inp.shape)
    out = net(inp)
    print(out.shape)
    exit()
