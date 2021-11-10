import numpy as np
import torch
import torch.nn as nn

"""[ElegantRL.2021.09.01](https://github.com/AI4Finance-Foundation/ElegantRL)"""

'''Q Network'''


class QNet(nn.Module):  # nn.Module is a standard PyTorch Network
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 nn.Linear(mid_dim, action_dim))

    def forward(self, state):
        return self.net(state)  # Q value


class QNetDuel(nn.Module):  # Dueling DQN
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net_state = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                       nn.Linear(mid_dim, mid_dim), nn.ReLU())
        self.net_adv = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                     nn.Linear(mid_dim, 1))  # advantage function value 1
        self.net_val = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                     nn.Linear(mid_dim, action_dim))  # Q value

    def forward(self, state):
        t_tmp = self.net_state(state)  # tensor of encoded state
        q_adv = self.net_adv(t_tmp)
        q_val = self.net_val(t_tmp)
        return q_adv + q_val - q_val.mean(dim=1, keepdim=True)  # dueling Q value


class QNetTwin(nn.Module):  # Double DQN
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net_state = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                       nn.Linear(mid_dim, mid_dim), nn.ReLU())
        self.net_q1 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                    nn.Linear(mid_dim, action_dim))  # q1 value
        self.net_q2 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                    nn.Linear(mid_dim, action_dim))  # q2 value

    def forward(self, state):
        tmp = self.net_state(state)
        return self.net_q1(tmp)  # one Q value

    def get_q1_q2(self, state):
        tmp = self.net_state(state)
        return self.net_q1(tmp), self.net_q2(tmp)  # two Q values


class QNetTwinDuel(nn.Module):  # D3QN: Dueling Double DQN
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net_state = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                       nn.Linear(mid_dim, mid_dim), nn.ReLU())
        self.net_adv1 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                      nn.Linear(mid_dim, 1))  # q1 value
        self.net_adv2 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                      nn.Linear(mid_dim, 1))  # q2 value
        self.net_val1 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                      nn.Linear(mid_dim, action_dim))  # advantage function value 1
        self.net_val2 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                      nn.Linear(mid_dim, action_dim))  # advantage function value 1

    def forward(self, state):
        t_tmp = self.net_state(state)
        q_adv = self.net_adv1(t_tmp)
        q_val = self.net_val1(t_tmp)
        return q_adv + q_val - q_val.mean(dim=1, keepdim=True)  # one dueling Q value

    def get_q1_q2(self, state):
        tmp = self.net_state(state)

        adv1 = self.net_adv1(tmp)
        val1 = self.net_val1(tmp)
        q1 = adv1 + val1 - val1.mean(dim=1, keepdim=True)

        adv2 = self.net_adv2(tmp)
        val2 = self.net_val2(tmp)
        q2 = adv2 + val2 - val2.mean(dim=1, keepdim=True)
        return q1, q2  # two dueling Q values


'''Policy Network (Actor)'''


class Actor(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 nn.Linear(mid_dim, action_dim))

    def forward(self, state):
        return self.net(state).tanh()  # action.tanh()

    def get_action(self, state, action_std):
        action = self.net(state).tanh()
        noise = (torch.randn_like(action) * action_std).clamp(-0.5, 0.5)
        return (action + noise).clamp(-1.0, 1.0)


class ActorSAC(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim, if_use_dn=False):
        super().__init__()
        if if_use_dn:
            nn_middle = DenseNet(mid_dim // 2)
            inp_dim = nn_middle.inp_dim
            out_dim = nn_middle.out_dim
        else:
            nn_middle = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                      nn.Linear(mid_dim, mid_dim), nn.ReLU(), )
            inp_dim = mid_dim
            out_dim = mid_dim

        self.net_state = nn.Sequential(nn.Linear(state_dim, inp_dim), nn.ReLU(),
                                       nn_middle, )
        self.net_a_avg = nn.Sequential(nn.Linear(out_dim, mid_dim), nn.Hardswish(),
                                       nn.Linear(mid_dim, action_dim))  # the average of action
        self.net_a_std = nn.Sequential(nn.Linear(out_dim, mid_dim), nn.Hardswish(),
                                       nn.Linear(mid_dim, action_dim))  # the log_std of action
        self.log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))

    def forward(self, state):
        tmp = self.net_state(state)
        return self.net_a_avg(tmp).tanh()  # action

    def get_action(self, state):
        t_tmp = self.net_state(state)
        a_avg = self.net_a_avg(t_tmp)  # NOTICE! it is a_avg without .tanh()
        a_std = self.net_a_std(t_tmp).clamp(-20, 2).exp()
        return torch.normal(a_avg, a_std).tanh()  # re-parameterize

    def get_action_logprob(self, state):
        t_tmp = self.net_state(state)
        a_avg = self.net_a_avg(t_tmp)  # NOTICE! it needs a_avg.tanh()
        a_std_log = self.net_a_std(t_tmp).clamp(-20, 2)
        a_std = a_std_log.exp()

        """add noise to action in stochastic policy"""
        noise = torch.randn_like(a_avg, requires_grad=True)
        a_tan = (a_avg + a_std * noise).tanh()  # action.tanh()
        # Can only use above code instead of below, because the tensor need gradients here.
        # a_noise = torch.normal(a_avg, a_std, requires_grad=True)

        '''compute log_prob according to mean and std of action (stochastic policy)'''
        # self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))
        log_prob = a_std_log + self.log_sqrt_2pi + noise.pow(2).__mul__(0.5)  # noise.pow(2) * 0.5
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

    # def log_abs_det_jacobian(self, x, y):
    #     # https://github.com/denisyarats/pytorch_sac/blob/81c5b536d3a1c5616b2531e446450df412a064fb/agent/actor.py#L37
    #     # ↑ MIT License， Thanks for https://www.zhihu.com/people/Z_WXCY 2ez4U
    #     # We use a formula that is more numerically stable, see details in the following link
    #     # https://pytorch.org/docs/stable/_modules/torch/distributions/transforms.html#TanhTransform
    #     # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f
    #     return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class ActorPPO(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        if isinstance(state_dim, int):
            nn_middle = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                      nn.Linear(mid_dim, mid_dim), nn.ReLU(), )
        else:
            nn_middle = ConvNet(inp_dim=state_dim[2], out_dim=mid_dim, image_size=state_dim[0])

        self.net = nn.Sequential(nn_middle,
                                 nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 nn.Linear(mid_dim, action_dim), )
        layer_norm(self.net[-1], std=0.1)  # output layer for action

        # the logarithm (log) of standard deviation (std) of action, it is a trainable parameter
        self.a_std_log = nn.Parameter(torch.zeros((1, action_dim)) - 0.5, requires_grad=True)
        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))

    def forward(self, state):
        return self.net(state).tanh()  # action.tanh()

    def get_action(self, state):
        a_avg = self.net(state)
        a_std = self.a_std_log.exp()

        noise = torch.randn_like(a_avg)
        action = a_avg + noise * a_std
        return action, noise

    def get_logprob_entropy(self, state, action):
        a_avg = self.net(state)
        a_std = self.a_std_log.exp()

        delta = ((a_avg - action) / a_std).pow(2) * 0.5
        logprob = -(self.a_std_log + self.sqrt_2pi_log + delta).sum(1)  # new_logprob

        dist_entropy = (logprob.exp() * logprob).mean()  # policy entropy
        return logprob, dist_entropy

    def get_old_logprob(self, _action, noise):  # noise = action - a_noise
        delta = noise.pow(2) * 0.5
        return -(self.a_std_log + self.sqrt_2pi_log + delta).sum(1)  # old_logprob


class ActorDiscretePPO(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        if isinstance(state_dim, int):
            nn_middle = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(), )
        else:
            nn_middle = ConvNet(inp_dim=state_dim[2], out_dim=mid_dim, image_size=state_dim[0])

        self.net = nn.Sequential(nn_middle,
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 nn.Linear(mid_dim, action_dim), )
        layer_norm(self.net[-1], std=0.1)  # output layer for action

        self.action_dim = action_dim
        self.soft_max = nn.Softmax(dim=-1)
        self.Categorical = torch.distributions.Categorical

    def forward(self, state):
        return self.net(state)  # action_prob without softmax

    def get_action(self, state):
        a_prob = self.soft_max(self.net(state))
        # dist = Categorical(a_prob)
        # a_int = dist.sample()
        a_int = torch.multinomial(a_prob, num_samples=1, replacement=True)[:, 0]
        # samples_2d = torch.multinomial(a_prob, num_samples=1, replacement=True)
        # samples_2d.shape == (batch_size, num_samples)
        return a_int, a_prob

    def get_logprob_entropy(self, state, a_int):
        a_prob = self.soft_max(self.net(state))
        dist = self.Categorical(a_prob)
        return dist.log_prob(a_int), dist.entropy().mean()

    def get_old_logprob(self, a_int, a_prob):
        dist = self.Categorical(a_prob)
        return dist.log_prob(a_int)


class ActorBiConv(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            BiConvNet(mid_dim, state_dim, mid_dim * 4),
            nn.Linear(mid_dim * 4, mid_dim * 1), nn.ReLU(),
            DenseNet(mid_dim * 1), nn.ReLU(),
            nn.Linear(mid_dim * 4, action_dim),
        )
        layer_norm(self.net[-1], std=0.1)  # output layer for action

    def forward(self, state):
        action = self.net(state)
        return action * torch.pow((action ** 2).sum(), -0.5)  # action / sqrt(L2_norm(action))

    def get_action(self, state, action_std):
        action = self.net(state)
        action = action + (torch.randn_like(action) * action_std)
        return action * torch.pow((action ** 2).sum(), -0.5)  # action / sqrt(L2_norm(action))


'''Value Network (Critic)'''


class Critic(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 nn.Linear(mid_dim, 1))

    def forward(self, state, action):
        return self.net(torch.cat((state, action), dim=1))  # Q value


class CriticTwin(nn.Module):  # shared parameter
    def __init__(self, mid_dim, state_dim, action_dim, if_use_dn=False):
        super().__init__()
        if if_use_dn:
            nn_middle = DenseNet(mid_dim // 2)
            inp_dim = nn_middle.inp_dim
            out_dim = nn_middle.out_dim
        else:
            nn_middle = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                      nn.Linear(mid_dim, mid_dim), nn.ReLU(), )
            inp_dim = mid_dim
            out_dim = mid_dim

        self.net_sa = nn.Sequential(nn.Linear(state_dim + action_dim, inp_dim), nn.ReLU(),
                                    nn_middle, )  # concat(state, action)
        self.net_q1 = nn.Sequential(nn.Linear(out_dim, mid_dim), nn.Hardswish(),
                                    nn.Linear(mid_dim, 1))  # q1 value
        self.net_q2 = nn.Sequential(nn.Linear(out_dim, mid_dim), nn.Hardswish(),
                                    nn.Linear(mid_dim, 1))  # q2 value

    def forward(self, state, action):
        tmp = self.net_sa(torch.cat((state, action), dim=1))
        return self.net_q1(tmp)  # one Q value

    def get_q1_q2(self, state, action):
        tmp = self.net_sa(torch.cat((state, action), dim=1))
        return self.net_q1(tmp), self.net_q2(tmp)  # two Q values


class CriticPPO(nn.Module):
    def __init__(self, mid_dim, state_dim, _action_dim):
        super().__init__()
        if isinstance(state_dim, int):
            nn_middle = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(), )
        else:
            nn_middle = ConvNet(inp_dim=state_dim[2], out_dim=mid_dim, image_size=state_dim[0])

        self.net = nn.Sequential(nn_middle,
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                 nn.Linear(mid_dim, 1), )
        layer_norm(self.net[-1], std=0.5)  # output layer for advantage value

    def forward(self, state):
        return self.net(state)  # advantage value


class CriticTwinPPO(nn.Module):
    def __init__(self, mid_dim, state_dim, _action_dim):
        super().__init__()
        if isinstance(state_dim, int):
            nn_middle = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                      nn.Linear(mid_dim, mid_dim), nn.ReLU(), )
        else:
            nn_middle = ConvNet(inp_dim=state_dim[2], out_dim=mid_dim, image_size=state_dim[0])

        self.net = nn.Sequential(nn_middle, )
        self.net_v1 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                    nn.Linear(mid_dim, 1))  # advantage value1
        self.net_v2 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                    nn.Linear(mid_dim, 1))  # advantage value2

        layer_norm(self.net[-1], std=0.5)  # output layer for advantage value

    def forward(self, state):
        tmp = self.net(state)
        return self.net_v1(tmp)  # one advantage value

    def get_q1_q2(self, state):
        tmp = self.net(state)
        return self.net_v1(tmp), self.net_v2(tmp)  # two advantage values


class CriticBiConv(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        i_c_dim, i_h_dim, i_w_dim = state_dim  # inp_for_cnn.shape == (N, C, H, W)
        assert action_dim == int(np.prod((i_c_dim, i_w_dim, i_h_dim)))
        action_dim = (i_c_dim, i_w_dim, i_h_dim)  # (2, bs_n, ur_n)

        self.cnn_s = nn.Sequential(
            BiConvNet(mid_dim, state_dim, mid_dim * 4),
            nn.Linear(mid_dim * 4, mid_dim * 2), nn.ReLU(inplace=True),
            nn.Linear(mid_dim * 2, mid_dim * 1),
        )
        self.cnn_a = nn.Sequential(
            NnReshape(*action_dim),
            BiConvNet(mid_dim, action_dim, mid_dim * 4),
            nn.Linear(mid_dim * 4, mid_dim * 2), nn.ReLU(inplace=True),
            nn.Linear(mid_dim * 2, mid_dim * 1),
        )

        self.out_net = nn.Sequential(
            nn.Linear(mid_dim * 1, mid_dim * 1), nn.Hardswish(),
            nn.Linear(mid_dim * 1, 1),
        )
        layer_norm(self.out_net[-1], std=0.1)  # output layer for action

    def forward(self, state, action):
        xs = self.cnn_s(state)
        xa = self.cnn_a(action)
        return self.out_net(xs + xa)  # Q value


'''Parameter sharing Network'''


class ShareDPG(nn.Module):  # DPG means deterministic policy gradient
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()
        nn_dense = DenseNet(mid_dim // 2)
        inp_dim = nn_dense.inp_dim
        out_dim = nn_dense.out_dim

        self.enc_s = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                   nn.Linear(mid_dim, inp_dim))
        self.enc_a = nn.Sequential(nn.Linear(action_dim, mid_dim), nn.ReLU(),
                                   nn.Linear(mid_dim, inp_dim))

        self.net = nn_dense

        self.dec_a = nn.Sequential(nn.Linear(out_dim, mid_dim), nn.Hardswish(),
                                   nn.Linear(mid_dim, action_dim), nn.Tanh())
        self.dec_q = nn.Sequential(nn.Linear(out_dim, mid_dim), nn.Hardswish(),
                                   nn.utils.spectral_norm(nn.Linear(mid_dim, 1)))

    @staticmethod
    def add_noise(a, noise_std):
        a_temp = torch.normal(a, noise_std)

        mask = torch.lt(a_temp, -1) + torch.gt(a_temp, 1)  # mask = (a_temp < -1.0) + (a_temp > 1.0)
        mask = torch.tensor(mask, dtype=torch.float32).cuda()

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

    def next_q_action(self, s, s_next, noise_std):
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


class ShareSPG(nn.Module):  # SPG means stochastic policy gradient
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.log_sqrt_2pi_sum = np.log(np.sqrt(2 * np.pi)) * action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        nn_dense = DenseNet(mid_dim // 2)
        inp_dim = nn_dense.inp_dim
        out_dim = nn_dense.out_dim

        self.enc_s = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                   nn.Linear(mid_dim, inp_dim), )  # state
        self.enc_a = nn.Sequential(nn.Linear(action_dim, mid_dim), nn.ReLU(),
                                   nn.Linear(mid_dim, inp_dim), )  # action without nn.Tanh()

        self.net = nn_dense

        self.dec_a = nn.Sequential(nn.Linear(out_dim, mid_dim // 2), nn.ReLU(),
                                   nn.Linear(mid_dim // 2, action_dim), )  # action_mean
        self.dec_d = nn.Sequential(nn.Linear(out_dim, mid_dim // 2), nn.ReLU(),
                                   nn.Linear(mid_dim // 2, action_dim), )  # action_std_log (d means standard deviation)
        self.dec_q1 = nn.Sequential(nn.Linear(out_dim, mid_dim // 2), nn.ReLU(),
                                    nn.Linear(mid_dim // 2, 1), )  # q1 value
        self.dec_q2 = nn.Sequential(nn.Linear(out_dim, mid_dim // 2), nn.ReLU(),
                                    nn.Linear(mid_dim // 2, 1), )  # q2 value
        self.log_alpha = nn.Parameter(torch.zeros((1, action_dim)) - np.log(action_dim), requires_grad=True)

        layer_norm(self.dec_a[-1], std=0.5)
        layer_norm(self.dec_d[-1], std=0.1)
        layer_norm(self.dec_q1[-1], std=0.5)
        layer_norm(self.dec_q2[-1], std=0.5)

    def forward(self, s):
        x = self.enc_s(s)
        x = self.net(x)
        a_avg = self.dec_a(x)
        return a_avg.tanh()

    def get_action(self, s):
        s_ = self.enc_s(s)
        a_ = self.net(s_)
        a_avg = self.dec_a(a_)  # NOTICE! it is a_avg without tensor.tanh()

        a_std_log = self.dec_d(a_).clamp(-20, 2)
        a_std = a_std_log.exp()

        action = torch.normal(a_avg, a_std)  # NOTICE! it is action without .tanh()
        return action.tanh()

    def get_action_logprob(self, state):  # actor
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
        logprob = (noise.pow(2) / 2 + a_std_log + fix_term).sum(1, keepdim=True) + self.log_sqrt_2pi_sum
        return a_noise_tanh, logprob

    def get_q_logprob(self, state):
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
        logprob = (noise.pow(2) / 2 + a_std_log + fix_term).sum(1, keepdim=True) + self.log_sqrt_2pi_sum

        '''get q'''
        a_ = self.enc_a(a_noise_tanh)
        q_ = self.net(s_ + a_)
        q = torch.min(self.dec_q1(q_), self.dec_q2(q_))
        return q, logprob

    def get_q1_q2(self, s, a):  # critic
        s_ = self.enc_s(s)
        a_ = self.enc_a(a)
        q_ = self.net(s_ + a_)
        q1 = self.dec_q1(q_)
        q2 = self.dec_q2(q_)
        return q1, q2


class SharePPO(nn.Module):  # Pixel-level state version
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()
        if isinstance(state_dim, int):
            self.enc_s = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                       nn.Linear(mid_dim, mid_dim))  # the only difference.
        else:
            self.enc_s = ConvNet(inp_dim=state_dim[2], out_dim=mid_dim, image_size=state_dim[0])
        out_dim = mid_dim

        self.dec_a = nn.Sequential(nn.Linear(out_dim, mid_dim), nn.ReLU(),
                                   nn.Linear(mid_dim, action_dim))
        self.a_std_log = nn.Parameter(torch.zeros(1, action_dim) - 0.5, requires_grad=True)

        self.dec_q1 = nn.Sequential(nn.Linear(out_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, 1))
        self.dec_q2 = nn.Sequential(nn.Linear(out_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, 1))

        layer_norm(self.dec_a[-1], std=0.01)
        layer_norm(self.dec_q1[-1], std=0.01)
        layer_norm(self.dec_q2[-1], std=0.01)

        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))

    def forward(self, s):
        s_ = self.enc_s(s)
        a_avg = self.dec_a(s_)
        return a_avg.tanh()

    def get_action_noise(self, state):
        s_ = self.enc_s(state)
        a_avg = self.dec_a(s_)
        a_std = self.a_std_log.exp()

        # a_noise = torch.normal(a_avg, a_std) # same as below
        noise = torch.randn_like(a_avg)
        a_noise = a_avg + noise * a_std
        return a_noise, noise

    def get_q_logprob(self, state, noise):
        s_ = self.enc_s(state)

        q = torch.min(self.dec_q1(s_), self.dec_q2(s_))
        logprob = -(noise.pow(2) / 2 + self.a_std_log + self.sqrt_2pi_log).sum(1)
        return q, logprob

    def get_q1_q2_logprob(self, state, action):
        s_ = self.enc_s(state)

        q1 = self.dec_q1(s_)
        q2 = self.dec_q2(s_)

        a_avg = self.dec_a(s_)
        a_std = self.a_std_log.exp()
        logprob = -(((a_avg - action) / a_std).pow(2) / 2 + self.a_std_log + self.sqrt_2pi_log).sum(1)
        return q1, q2, logprob


class ShareBiConv(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        i_c_dim, i_h_dim, i_w_dim = state_dim  # inp_for_cnn.shape == (N, C, H, W)
        assert action_dim == int(np.prod((i_c_dim, i_w_dim, i_h_dim)))

        state_tuple = (i_c_dim, i_h_dim, i_w_dim)
        self.enc_s = nn.Sequential(
            # NnReshape(*state_tuple),
            BiConvNet(mid_dim, state_tuple, mid_dim * 4),
        )
        action_tuple = (i_c_dim, i_w_dim, i_h_dim)
        self.enc_a = nn.Sequential(
            NnReshape(*action_tuple),
            BiConvNet(mid_dim, action_tuple, mid_dim * 4),
        )

        self.mid_n = nn.Sequential(
            nn.Linear(mid_dim * 4, mid_dim * 2), nn.ReLU(),
            nn.Linear(mid_dim * 2, mid_dim * 1), nn.ReLU(),
            DenseNet(mid_dim),
        )

        self.dec_a = nn.Sequential(
            nn.Linear(mid_dim * 4, mid_dim * 2), nn.Hardswish(),
            nn.Linear(mid_dim * 2, action_dim),
        )
        layer_norm(self.dec_a[-1], std=0.1)  # output layer for action
        self.dec_q = nn.Sequential(
            nn.Linear(mid_dim * 4, mid_dim * 2), nn.Hardswish(),
            nn.Linear(mid_dim * 2, 1),
        )
        layer_norm(self.dec_q[-1], std=0.1)  # output layer for action

    def forward(self, state):  # actor
        xs = self.enc_s(state)
        xn = self.mid_n(xs)
        action = self.dec_a(xn)
        return action * torch.pow((action ** 2).sum(), -0.5)  # action / sqrt(L2_norm(action))

    def critic(self, state, action):
        xs = self.enc_s(state)
        xa = self.enc_a(action)
        xn = self.mid_n(xs + xa)
        return self.dec_q(xn)  # Q value

    def get_action(self, state, action_std):  # actor, get noisy action
        xs = self.enc_s(state)
        xn = self.mid_n(xs)
        action = self.dec_a(xn)
        action = action + (torch.randn_like(action) * action_std)
        return action * torch.pow((action ** 2).sum(), -0.5)  # action / sqrt(L2_norm(action))


"""utils"""


class NnReshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.view((x.size(0),) + self.args)


class DenseNet(nn.Module):  # plan to hyper-param: layer_number
    def __init__(self, lay_dim):
        super().__init__()
        self.dense1 = nn.Sequential(nn.Linear(lay_dim * 1, lay_dim * 1), nn.Hardswish())
        self.dense2 = nn.Sequential(nn.Linear(lay_dim * 2, lay_dim * 2), nn.Hardswish())
        self.inp_dim = lay_dim
        self.out_dim = lay_dim * 4

    def forward(self, x1):  # x1.shape==(-1, lay_dim*1)
        x2 = torch.cat((x1, self.dense1(x1)), dim=1)
        x3 = torch.cat((x2, self.dense2(x2)), dim=1)
        return x3  # x2.shape==(-1, lay_dim*4)


class ConcatNet(nn.Module):  # concatenate
    def __init__(self, lay_dim):
        super().__init__()
        self.dense1 = nn.Sequential(nn.Linear(lay_dim, lay_dim), nn.ReLU(),
                                    nn.Linear(lay_dim, lay_dim), nn.Hardswish(), )
        self.dense2 = nn.Sequential(nn.Linear(lay_dim, lay_dim), nn.ReLU(),
                                    nn.Linear(lay_dim, lay_dim), nn.Hardswish(), )
        self.dense3 = nn.Sequential(nn.Linear(lay_dim, lay_dim), nn.ReLU(),
                                    nn.Linear(lay_dim, lay_dim), nn.Hardswish(), )
        self.dense4 = nn.Sequential(nn.Linear(lay_dim, lay_dim), nn.ReLU(),
                                    nn.Linear(lay_dim, lay_dim), nn.Hardswish(), )
        self.inp_dim = lay_dim
        self.out_dim = lay_dim * 4

    def forward(self, x0):
        x1 = self.dense1(x0)
        x2 = self.dense2(x0)
        x3 = self.dense3(x0)
        x4 = self.dense4(x0)
        return torch.cat((x1, x2, x3, x4), dim=1)


class ConvNet(nn.Module):  # pixel-level state encoder
    def __init__(self, inp_dim, out_dim, image_size=224):
        super().__init__()
        if image_size == 224:
            self.net = nn.Sequential(  # size==(batch_size, inp_dim, 224, 224)
                nn.Conv2d(inp_dim, 32, (5, 5), stride=(2, 2), bias=False), nn.ReLU(inplace=True),  # size=110
                nn.Conv2d(32, 48, (3, 3), stride=(2, 2)), nn.ReLU(inplace=True),  # size=54
                nn.Conv2d(48, 64, (3, 3), stride=(2, 2)), nn.ReLU(inplace=True),  # size=26
                nn.Conv2d(64, 96, (3, 3), stride=(2, 2)), nn.ReLU(inplace=True),  # size=12
                nn.Conv2d(96, 128, (3, 3), stride=(2, 2)), nn.ReLU(inplace=True),  # size=5
                nn.Conv2d(128, 192, (5, 5), stride=(1, 1)), nn.ReLU(inplace=True),  # size=1
                NnReshape(-1),  # size (batch_size, 1024, 1, 1) ==> (batch_size, 1024)
                nn.Linear(192, out_dim),  # size==(batch_size, out_dim)
            )
        elif image_size == 112:
            self.net = nn.Sequential(  # size==(batch_size, inp_dim, 112, 112)
                nn.Conv2d(inp_dim, 32, (5, 5), stride=(2, 2), bias=False), nn.ReLU(inplace=True),  # size=54
                nn.Conv2d(32, 48, (3, 3), stride=(2, 2)), nn.ReLU(inplace=True),  # size=26
                nn.Conv2d(48, 64, (3, 3), stride=(2, 2)), nn.ReLU(inplace=True),  # size=12
                nn.Conv2d(64, 96, (3, 3), stride=(2, 2)), nn.ReLU(inplace=True),  # size=5
                nn.Conv2d(96, 128, (5, 5), stride=(1, 1)), nn.ReLU(inplace=True),  # size=1
                NnReshape(-1),  # size (batch_size, 1024, 1, 1) ==> (batch_size, 1024)
                nn.Linear(128, out_dim),  # size==(batch_size, out_dim)
            )
        else:
            assert image_size in {224, 112}

    def forward(self, x):
        # assert x.shape == (batch_size, inp_dim, image_size, image_size)
        x = x.permute(0, 3, 1, 2)
        x = x / 128.0 - 1.0
        return self.net(x)

    # @staticmethod
    # def check():
    #     inp_dim = 3
    #     out_dim = 32
    #     batch_size = 2
    #     image_size = [224, 112][1]
    #     # from elegantrl.net import Conv2dNet
    #     net = Conv2dNet(inp_dim, out_dim, image_size)
    #
    #     x = torch.ones((batch_size, image_size, image_size, inp_dim), dtype=torch.uint8) * 255
    #     print(x.shape)
    #     y = net(x)
    #     print(y.shape)


class BiConvNet(nn.Module):
    def __init__(self, mid_dim, inp_dim, out_dim):
        super().__init__()
        i_c_dim, i_h_dim, i_w_dim = inp_dim  # inp_for_cnn.shape == (N, C, H, W)

        self.cnn_h = nn.Sequential(
            nn.Conv2d(i_c_dim * 1, mid_dim * 2, (1, i_w_dim), bias=True), nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_dim * 2, mid_dim * 1, (1, 1), bias=True), nn.ReLU(inplace=True),
            NnReshape(-1),  # shape=(-1, i_h_dim * mid_dim)
            nn.Linear(i_h_dim * mid_dim, out_dim),
        )
        self.cnn_w = nn.Sequential(
            nn.Conv2d(i_c_dim * 1, mid_dim * 2, (i_h_dim, 1), bias=True), nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_dim * 2, mid_dim * 1, (1, 1), bias=True), nn.ReLU(inplace=True),
            NnReshape(-1),  # shape=(-1, i_w_dim * mid_dim)
            nn.Linear(i_w_dim * mid_dim, out_dim),
        )

    def forward(self, state):
        xh = self.cnn_h(state)
        xw = self.cnn_w(state)
        return xw + xh


def layer_norm(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)


class ActorSimplify:
    def __init__(self, gpu_id, actor_net):
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and gpu_id >= 0) else 'cpu')
        self.actor_net = actor_net.to(self.device)

    def get_action(self, state: np.ndarray) -> np.ndarray:
        states = torch.as_tensor(state[np.newaxis], dtype=torch.float32, device=self.device)
        action = self.actor_net(states)[0]
        return action.detach().cpu().numpy()


"""check"""


def check_actor_network():
    batch_size = 3
    mid_dim = 2 ** 8
    state_dim = 24
    action_dim = 4

    if_check_actor_net = 0
    if if_check_actor_net:
        net = ActorPPO(mid_dim, state_dim, action_dim)

        inp = torch.ones((batch_size, state_dim), dtype=torch.float32)
        out = net(inp)
        print(out.shape)

    if_check_pixel_level = 1
    if if_check_pixel_level:
        img_channel = 6
        img_size = 112  # {112, 224}

        mid_dim = 2 ** 8
        state_dim = (img_size, img_size, img_channel)
        action_dim = 4

        net = ActorPPO(mid_dim, state_dim, action_dim)

        inp = torch.ones((batch_size, img_size, img_size, img_channel), dtype=torch.int8)
        out = net(inp)
        print(out.shape)


def check_network():
    from envs.DownLink import DownLinkEnv
    env = DownLinkEnv()

    gpu_id = 1
    mid_dim = 128
    state_dim = env.state_dim
    action_dim = env.action_dim

    if_check_actor = 0
    if if_check_actor:
        net = ActorBiConv(mid_dim, state_dim, action_dim)
        act = ActorSimplify(gpu_id, net)

        state = env.reset()

        action = env.get_action_mmse(state)
        reward = env.get_reward(action)
        print(f"| mmse            : {reward:8.3f}")

        action = env.get_action_mmse(state)
        action = env.get_action_norm_power(action)
        reward = env.get_reward(action)
        print(f"| mmse (max Power): {reward:8.3f}")

        action = np.ones(env.action_dim)
        reward = env.get_reward(action)
        print(f"| ones (max Power): {reward:8.3f}")

        action = env.get_action_norm_power(action=None)  # random.normal action
        reward = env.get_reward(action)
        print(f"| rand (max Power): {reward:8.3f}")

        action = act.get_action(state)
        action = env.get_action_norm_power(action)
        reward = env.get_reward(action)
        print(f"| net  (max Power): {reward:8.3f}")

        # state, reward, done, _ = env.step(action)
        # print(reward)

    if_check_critic = 1
    if if_check_critic:
        batch_size = 3
        device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and gpu_id >= 0) else 'cpu')
        critic_net = CriticBiConv(mid_dim, state_dim, action_dim).to(device)

        ten_state = torch.randn(size=(batch_size, *state_dim), dtype=torch.float32, device=device)
        ten_action = torch.randn(size=(batch_size, action_dim), dtype=torch.float32, device=device)

        q_value = critic_net(ten_state, ten_action)
        print(q_value)
