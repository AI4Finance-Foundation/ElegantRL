from AgentRun import *
from AgentZoo import *
from AgentNet import *

"""
search 'todo CarRacing-v0' to see the change
CarRacing-v0 GAE 
3    4.99e+05    870.19 |  827.74     20.49 |  245.43     -0.08      0.12

offline conv
try merge train__online_policy and train__offline_policy 
"""


def test_conv2d():
    state_dim = (2, 96, 96)

    def idx_dim(i):
        return int(16 * 1.6487 ** i)

    mid_dim = 128
    net = nn.Sequential(
        NnnReshape(*state_dim),  # -> [?, 2, 96, 96]
        nn.Conv2d(state_dim[0], idx_dim(0), 3, 2, bias=True),  # todo CarRacing-v0
        nn.Conv2d(idx_dim(0), idx_dim(1), 4, 2, bias=False), nn.ReLU(),
        nn.Conv2d(idx_dim(1), idx_dim(2), 3, 2, bias=False), nn.ReLU(),
        nn.Conv2d(idx_dim(2), idx_dim(1), 3, 1, bias=True), nn.ReLU(),
        NnnReshape(-1),  # [?, 26, 8, 8] -> [?, 1664]
        nn.Linear(1664, mid_dim), nn.ReLU(),
        DenseNet(mid_dim),  # nn.Linear(mid_dim, mid_dim), nn.ReLU(),
    )

    inp = torch.ones((3, 2, 96, 96), dtype=torch.float32)
    inp = inp.view(3, -1)
    print(inp.shape)
    out = net(inp)
    print(out.shape)
    exit()


class ActorGAE(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()

        def idx_dim(i):
            return int(16 * 1.6487 ** i)

        self.net = nn.Sequential(
            NnnReshape(*state_dim),  # -> [?, 2, 96, 96]
            nn.Conv2d(state_dim[0], idx_dim(0), 3, 2, bias=True),  # todo CarRacing-v0
            nn.Conv2d(idx_dim(0), idx_dim(1), 4, 2, bias=False), nn.ReLU(),
            nn.Conv2d(idx_dim(1), idx_dim(2), 3, 2, bias=False), nn.ReLU(),
            nn.Conv2d(idx_dim(2), idx_dim(1), 3, 1, bias=True), nn.ReLU(),
            NnnReshape(-1),  # [?, 26, 8, 8] -> [?, 1664]
            nn.Linear(1664, mid_dim), nn.ReLU(),
            DenseNet(mid_dim),  # nn.Linear(mid_dim, mid_dim), nn.ReLU(),
        )

        self.net__mean = nn.Linear(mid_dim * 4, action_dim)
        self.net__std_log = nn.Linear(mid_dim * 4, action_dim)

        self.log_std_min = -20
        self.log_std_max = 2
        self.constant_log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # layer_norm(self.net[2], std=1.0)
        layer_norm(self.net__mean, std=0.01)  # output layer for action
        layer_norm(self.net__std_log, std=0.01)  # output layer for std_log

    def forward(self, s):
        x = self.net(s)
        a_mean = self.net__mean(x)
        return a_mean

    def get__a__log_prob(self, state):
        x = self.net(state)
        a_mean = self.net__mean(x)
        a_log_std = self.net__std_log(x).clamp(self.log_std_min, self.log_std_max)
        a_std = torch.exp(a_log_std)

        a_noise = a_mean + a_std * torch.randn_like(a_mean, requires_grad=True, device=self.device)

        a_delta = (a_noise - a_mean).pow(2) / (2 * a_std.pow(2))
        log_prob = -(a_delta + a_log_std + self.constant_log_sqrt_2pi)
        return a_noise, log_prob.sum(1)

    def compute__log_prob(self, state, a_noise):
        x = self.net(state)
        a_mean = self.net__mean(x)
        a_log_std = self.net__std_log(x).clamp(self.log_std_min, self.log_std_max)
        a_std = torch.exp(a_log_std)

        a_delta = (a_noise - a_mean).pow(2) / (2 * a_std.pow(2))
        log_prob = -(a_delta + a_log_std + self.constant_log_sqrt_2pi)
        log_prob = log_prob.sum(1)
        return log_prob


class CriticAdvTwin(nn.Module):
    def __init__(self, state_dim, mid_dim):
        super().__init__()

        def idx_dim(i):
            return int(16 * 1.6487 ** i)

        self.net = nn.Sequential(
            NnnReshape(*state_dim),  # -> [?, 2, 96, 96]
            nn.Conv2d(state_dim[0], idx_dim(0), 3, 2, bias=True),  # todo CarRacing-v0
            nn.Conv2d(idx_dim(0), idx_dim(1), 4, 2, bias=False), nn.ReLU(),
            nn.Conv2d(idx_dim(1), idx_dim(2), 3, 2, bias=False), nn.ReLU(),
            nn.Conv2d(idx_dim(2), idx_dim(1), 3, 1, bias=True), nn.ReLU(),
            NnnReshape(-1),  # [?, 26, 8, 8] -> [?, 1664]
            nn.Linear(1664, mid_dim), nn.ReLU(),
            DenseNet(mid_dim),  # nn.Linear(mid_dim, mid_dim), nn.ReLU(),
        )

        self.net_q1 = nn.Linear(mid_dim * 4, 1)
        self.net_q2 = nn.Linear(mid_dim * 4, 1)

        # layer_norm(self.net[2], std=1.0)
        layer_norm(self.net_q1, std=0.1)  # output layer for q value
        layer_norm(self.net_q2, std=0.1)  # output layer for q value

    def forward(self, s):
        x = self.net(s)
        q1 = self.net_q1(x)
        q2 = self.net_q2(x)
        return q1, q2


class ActorSAC(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim, use_dn):
        super().__init__()

        def idx_dim(i):
            return int(16 * 1.6487 ** i)

        if use_dn:  # use DenseNet (DenseNet has both shallow and deep linear layer)
            self.net__mid = nn.Sequential(
                NnnReshape(*state_dim),  # -> [?, 2, 96, 96]
                nn.Conv2d(state_dim[0], idx_dim(0), 3, 2, bias=True),  # todo CarRacing-v0
                nn.Conv2d(idx_dim(0), idx_dim(1), 4, 2, bias=False), nn.ReLU(),
                nn.Conv2d(idx_dim(1), idx_dim(2), 3, 2, bias=False), nn.ReLU(),
                nn.Conv2d(idx_dim(2), idx_dim(1), 3, 1, bias=True), nn.ReLU(),
                NnnReshape(-1),  # [?, 26, 8, 8] -> [?, 1664]
                nn.Linear(1664, mid_dim), nn.ReLU(),
                DenseNet(mid_dim),
            )
            lay_dim = mid_dim * 4  # the output layer dim of DenseNet is 'mid_dim * 4'
        else:  # use a simple network for actor. Deeper network does not mean better performance in RL.
            self.net__mid = nn.Sequential(
                nn.Linear(state_dim, mid_dim), nn.ReLU(),
                nn.Linear(mid_dim, mid_dim),
            )
            lay_dim = mid_dim

        self.net__mean = nn.Linear(lay_dim, action_dim)
        self.net__std_log = nn.Linear(lay_dim, action_dim)

        layer_norm(self.net__mean, std=0.01)  # net[-1] is output layer for action, it is no necessary.

        self.log_std_min = -20
        self.log_std_max = 2
        self.constant_log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, state, noise_std=0.0):  # in fact, noise_std is a boolean
        x = self.net__mid(state)
        a_mean = self.net__mean(x)  # NOTICE! it is a_mean without .tanh()

        if noise_std != 0.0:
            a_std_log = self.net__std_log(x).clamp(self.log_std_min, self.log_std_max)
            a_std = a_std_log.exp()
            a_mean = torch.normal(a_mean, a_std)  # NOTICE! it needs .tanh()

        return a_mean.tanh()

    def get__a__log_prob(self, state):
        x = self.net__mid(state)
        a_mean = self.net__mean(x)  # NOTICE! it needs a_mean.tanh()
        a_std_log = self.net__std_log(x).clamp(self.log_std_min, self.log_std_max)
        a_std = a_std_log.exp()

        """add noise to action in stochastic policy"""
        a_noise = a_mean + a_std * torch.randn_like(a_mean, requires_grad=True, device=self.device)
        # Can only use above code instead of below, because the tensor need gradients here.
        # a_noise = torch.normal(a_mean, a_std, requires_grad=True)

        '''compute log_prob according to mean and std of action (stochastic policy)'''
        a_delta = ((a_noise - a_mean) / a_std).pow(2) * 0.5
        # self.constant_log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))
        log_prob_noise = -(a_delta + a_std_log + self.constant_log_sqrt_2pi)

        # same as below:
        # from torch.distributions.normal import Normal
        # log_prob_noise = Normal(a_mean, a_std).log_prob(a_noise)
        # same as below:
        # a_delta = a_noise - a_mean).pow(2) /(2* a_std.pow(2)
        # log_prob_noise = -a_delta - a_std.log() - np.log(np.sqrt(2 * np.pi))

        a_noise_tanh = a_noise.tanh()
        log_prob = log_prob_noise - (-a_noise_tanh.pow(2) + 1.000001).log()

        # same as below:
        # epsilon = 1e-6
        # log_prob = log_prob_noise - (1 - a_noise_tanh.pow(2) + epsilon).log()
        return a_noise_tanh, log_prob.sum(1, keepdim=True)


class CriticTwinShared(nn.Module):  # 2020-06-18
    def __init__(self, state_dim, action_dim, mid_dim, use_dn):
        super().__init__()

        def idx_dim(i):
            return int(16 * 1.6487 ** i)

        self.enc_s = nn.Sequential(
            NnnReshape(*state_dim),  # -> [?, 2, 96, 96]
            nn.Conv2d(state_dim[0], idx_dim(0), 3, 2, bias=True),  # todo CarRacing-v0
            nn.Conv2d(idx_dim(0), idx_dim(1), 4, 2, bias=False), nn.ReLU(),
            nn.Conv2d(idx_dim(1), idx_dim(2), 3, 2, bias=False), nn.ReLU(),
            nn.Conv2d(idx_dim(2), idx_dim(1), 3, 1, bias=True), nn.ReLU(),
            NnnReshape(-1),  # [?, 26, 8, 8] -> [?, 1664]
            nn.Linear(1664, mid_dim), nn.ReLU(),
        )

        if use_dn:  # use DenseNet (DenseNet has both shallow and deep linear layer)
            self.net__mid = nn.Sequential(
                nn.Linear(mid_dim + action_dim, mid_dim), nn.ReLU(),
                DenseNet(mid_dim),
            )
            lay_dim = mid_dim * 4  # the output layer dim of DenseNet is 'mid_dim * 4'
        else:  # use a simple network for actor. Deeper network does not mean better performance in RL.
            self.net__mid = nn.Sequential(
                nn.Linear(mid_dim + action_dim, mid_dim), nn.ReLU(),
                nn.Linear(mid_dim, mid_dim),
            )
            lay_dim = mid_dim
        self.net__q1 = nn.Linear(lay_dim, 1)
        self.net__q2 = nn.Linear(lay_dim, 1)
        '''Not need to use both SpectralNorm and TwinCritic
        I choose TwinCritc instead of SpectralNorm, 
        because SpectralNorm is conflict with soft target update,

        if is_spectral_norm:
            self.net1[1] = nn.utils.spectral_norm(self.dec_q1[1])
            self.net2[1] = nn.utils.spectral_norm(self.dec_q2[1])
        '''

    def forward(self, state, action):
        state = self.enc_s(state)
        x = torch.cat((state, action), dim=1)
        x = self.net_mid(x)
        q_value = self.net__q1(x)
        return q_value

    def get__q1_q2(self, state, action):
        state = self.enc_s(state)
        x = torch.cat((state, action), dim=1)
        x = self.net__mid(x)
        q_value1 = self.net__q1(x)
        q_value2 = self.net__q2(x)
        return q_value1, q_value2


class InterSPG(nn.Module):  # class AgentIntelAC for SAC (SPG means stochastic policy gradient)
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()
        self.log_std_min = -20
        self.log_std_max = 2
        self.constant_log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def idx_dim(i):
            return int(16 * 1.6487 ** i)

        # encoder
        self.enc_s = nn.Sequential(
            NnnReshape(*state_dim),  # -> [?, 2, 96, 96]
            nn.Conv2d(state_dim[0], idx_dim(0), 3, 2, bias=True),  # todo CarRacing-v0
            nn.Conv2d(idx_dim(0), idx_dim(1), 4, 2, bias=False), nn.ReLU(),
            nn.Conv2d(idx_dim(1), idx_dim(2), 3, 2, bias=False), nn.ReLU(),
            nn.Conv2d(idx_dim(2), idx_dim(1), 3, 1, bias=True), nn.ReLU(),
            NnnReshape(-1),  # [?, 26, 8, 8] -> [?, 1664]
            nn.Linear(1664, mid_dim),
        )  # state
        self.enc_a = nn.Sequential(
            nn.Linear(action_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
        )  # action without nn.Tanh()

        self.net = DenseNet2(mid_dim)  # todo
        net_out_dim = self.net.out_dim

        # decoder
        self.dec_a = nn.Sequential(
            nn.Linear(net_out_dim, mid_dim), HardSwish(),
            nn.Linear(mid_dim, action_dim),
        )  # action_mean
        self.dec_d = nn.Sequential(
            nn.Linear(net_out_dim, mid_dim), HardSwish(),
            nn.Linear(mid_dim, action_dim),
        )  # action_std_log (d means standard dev.)
        self.dec_q1 = nn.Sequential(
            nn.Linear(net_out_dim, mid_dim), HardSwish(),
            nn.Linear(mid_dim, 1),
        )  # q_value1 SharedTwinCritic
        self.dec_q2 = nn.Sequential(
            nn.Linear(net_out_dim, mid_dim), HardSwish(),
            nn.Linear(mid_dim, 1),
        )  # q_value2 SharedTwinCritic

        layer_norm(self.dec_a[-1], std=0.01)  # net[-1] is output layer for action, it is no necessary.

        '''Not need to use both SpectralNorm and TwinCritic
        I choose TwinCritc instead of SpectralNorm, 
        because SpectralNorm is conflict with soft target update,

        if is_spectral_norm:
            self.dec_q1[1] = nn.utils.spectral_norm(self.dec_q1[1])
            self.dec_q2[1] = nn.utils.spectral_norm(self.dec_q2[1])
        '''

    def forward(self, s, noise_std=0.0):  # actor, in fact, noise_std is a boolean
        s_ = self.enc_s(s)
        a_ = self.net(s_)
        a_mean = self.dec_a(a_)  # NOTICE! it is a_mean without tensor.tanh()

        if noise_std != 0.0:
            a_std_log = self.dec_d(a_).clamp(self.log_std_min, self.log_std_max)
            a_std = a_std_log.exp()
            a_mean = torch.normal(a_mean, a_std)  # NOTICE! it is a_mean without .tanh()
        return a_mean.tanh()

    def get__a__log_prob(self, state):  # actor
        s_ = self.enc_s(state)
        a_ = self.net(s_)
        a_mean = self.dec_a(a_)  # NOTICE! it is a_mean without .tanh()
        a_std_log = self.dec_d(a_).clamp(self.log_std_min, self.log_std_max)
        a_std = a_std_log.exp()

        """add noise to action, stochastic policy"""
        # a_noise = torch.normal(a_mean, a_std, requires_grad=True)
        # the above is not same as below, because it needs gradient
        a_noise = a_mean + a_std * torch.randn_like(a_mean, requires_grad=True, device=self.device)

        '''compute log_prob according to mean and std of action (stochastic policy)'''
        # a_delta = a_noise - a_mean).pow(2) /(2* a_std.pow(2)
        # log_prob_noise = -a_delta - a_std.log() - np.log(np.sqrt(2 * np.pi))
        # same as:
        a_delta = ((a_noise - a_mean) / a_std).pow(2) * 0.5
        log_prob_noise = -(a_delta + a_std_log + self.constant_log_sqrt_2pi)

        a_noise_tanh = a_noise.tanh()
        # log_prob = log_prob_noise - (1 - a_noise_tanh.pow(2) + epsilon).log() # epsilon = 1e-6
        # same as:
        log_prob = log_prob_noise - (-a_noise_tanh.pow(2) + 1.000001).log()
        return a_noise_tanh, log_prob.sum(1, keepdim=True)

    def get__a__std(self, state):
        s_ = self.enc_s(state)
        a_ = self.net(s_)
        a_mean = self.dec_a(a_)  # NOTICE! it is a_mean without .tanh()
        a_std_log = self.dec_d(a_).clamp(self.log_std_min, self.log_std_max)
        return a_mean.tanh(), a_std_log

    def get__a__avg_std_noise_prob(self, state):  # actor
        s_ = self.enc_s(state)
        a_ = self.net(s_)
        a_mean = self.dec_a(a_)  # NOTICE! it is a_mean without .tanh()
        a_std_log = self.dec_d(a_).clamp(self.log_std_min, self.log_std_max)
        a_std = a_std_log.exp()

        """add noise to action, stochastic policy"""
        # a_noise = torch.normal(a_mean, a_std, requires_grad=True)
        # the above is not same as below, because it needs gradient
        noise = torch.randn_like(a_mean, requires_grad=True, device=self.device)
        a_noise = a_mean + a_std * noise

        '''compute log_prob according to mean and std of action (stochastic policy)'''
        # a_delta = a_noise - a_mean).pow(2) /(2* a_std.pow(2)
        # log_prob_noise = -a_delta - a_std.log() - np.log(np.sqrt(2 * np.pi))
        # same as:
        a_delta = ((a_noise - a_mean) / a_std).pow(2) * 0.5
        log_prob_noise = -(a_delta + a_std_log + self.constant_log_sqrt_2pi)

        a_noise_tanh = a_noise.tanh()
        # log_prob = log_prob_noise - (1 - a_noise_tanh.pow(2) + epsilon).log() # epsilon = 1e-6
        # same as:
        log_prob = log_prob_noise - (-a_noise_tanh.pow(2) + 1.000001).log()
        return a_mean.tanh(), a_std_log, a_noise_tanh, log_prob.sum(1, keepdim=True)

    def get__q1_q2(self, s, a):  # critic
        s_ = self.enc_s(s)
        a_ = self.enc_a(a)
        q_ = self.net(s_ + a_)
        q1 = self.dec_q1(q_)
        q2 = self.dec_q2(q_)
        return q1, q2


"""keep the same"""


class AgentGAE(AgentPPO):
    def __init__(self, state_dim, action_dim, net_dim):
        super(AgentPPO, self).__init__()
        self.learning_rate = 2e-4  # learning rate of actor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''network'''
        self.act = ActorGAE(state_dim, action_dim, net_dim).to(self.device)
        self.act.train()
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate, )  # betas=(0.5, 0.99))

        self.cri = CriticAdvTwin(state_dim, net_dim).to(self.device)
        self.cri.train()
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate, )  # betas=(0.5, 0.99))
        # cannot use actor target network
        # not need to use critic target network

        self.criterion = nn.SmoothL1Loss()

    def update_buffer(self, env, buffer, max_step, max_action, reward_scale, gamma):
        # collect tuple (reward, mask, state, action, log_prob, )
        # PPO is an online policy RL algorithm.
        # buffer = BufferTupleOnline(max_memo)

        rewards = list()
        steps = list()

        step_counter = 0
        while step_counter < buffer.max_memo:
            state = env.reset()

            reward_sum = 0
            step_sum = 0

            for step_sum in range(max_step):
                actions, log_probs = self.select_actions((state,), explore_noise=True)
                action = actions[0]
                log_prob = log_probs[0]

                next_state, reward, done, _ = env.step(action * max_action)
                reward_sum += reward

                mask = 0.0 if done else gamma

                reward_ = reward * reward_scale
                buffer.push(reward_, mask, state, action, log_prob, )

                if done:
                    break

                state = next_state

            rewards.append(reward_sum)
            steps.append(step_sum)

            step_counter += step_sum
        return rewards, steps

    def update_parameters(self, buffer, _max_step, batch_size, repeat_times):
        """Differences between AgentGAE and AgentPPO are:
        1. In AgentGAE, critic use TwinCritic. In AgentPPO, critic use a single critic.
        2. In AgentGAE, log_std is output by actor. In AgentPPO, log_std is just a trainable tensor.
        """

        self.act.train()
        self.cri.train()
        clip = 0.25  # ratio.clamp(1 - clip, 1 + clip)
        lambda_adv = 0.98  # why 0.98? cannot seem to use 0.99
        lambda_entropy = 0.01  # could be 0.02
        # repeat_times = 8 could be 2**2 ~ 2**4

        loss_a_sum = 0.0  # just for print
        loss_c_sum = 0.0  # just for print

        '''the batch for training'''
        max_memo = len(buffer)  # assert max_memo == _max_step
        all_batch = buffer.sample()
        all_reward, all_mask, all_state, all_action, all_log_prob = [
            torch.tensor(ary, dtype=torch.float32, device=self.device)
            for ary in (all_batch.reward, all_batch.mask, all_batch.state, all_batch.action, all_batch.log_prob,)
        ]

        # with torch.no_grad():
        # all__new_v = self.cri(all_state).detach_()  # all new value
        # all__new_v = torch.add(*self.cri(all_state)).detach_() * 0.5  # TwinCritic # OOM

        with torch.no_grad():
            b_size = 128
            b__len = all_state.size()[0]
            all__new_v = [torch.add(*self.cri(all_state[i:i + b_size])) * 0.5
                          for i in range(0, b__len, b_size)]
            all__new_v = torch.cat(all__new_v, dim=0)

        '''compute old_v (old policy value), adv_v (advantage value) 
        refer: Generalization Advantage Estimate. ICLR 2016. 
        https://arxiv.org/pdf/1506.02438.pdf
        '''
        all__delta = torch.empty(max_memo, dtype=torch.float32, device=self.device)  # delta of q value
        all__old_v = torch.empty(max_memo, dtype=torch.float32, device=self.device)  # old policy value
        all__adv_v = torch.empty(max_memo, dtype=torch.float32, device=self.device)  # advantage value

        prev_old_v = 0  # old q value
        prev_new_v = 0  # new q value
        prev_adv_v = 0  # advantage q value
        for i in range(max_memo - 1, -1, -1):
            all__delta[i] = all_reward[i] + all_mask[i] * prev_new_v - all__new_v[i]
            all__old_v[i] = all_reward[i] + all_mask[i] * prev_old_v
            all__adv_v[i] = all__delta[i] + all_mask[i] * prev_adv_v * lambda_adv

            prev_old_v = all__old_v[i]
            prev_new_v = all__new_v[i]
            prev_adv_v = all__adv_v[i]

        all__adv_v = (all__adv_v - all__adv_v.mean()) / (all__adv_v.std() + 1e-6)  # advantage_norm:

        '''mini batch sample'''
        sample_times = int(repeat_times * max_memo / batch_size)
        for _ in range(sample_times):
            '''random sample'''
            # indices = rd.choice(max_memo, batch_size, replace=True)  # False)
            indices = rd.randint(max_memo, size=batch_size)

            state = all_state[indices]
            action = all_action[indices]
            advantage = all__adv_v[indices]
            old_value = all__old_v[indices].unsqueeze(1)
            old_log_prob = all_log_prob[indices]

            """Adaptive KL Penalty Coefficient
            loss_KLPEN = surrogate_obj + value_obj * lambda_value + entropy_obj * lambda_entropy
            loss_KLPEN = (value_obj * lambda_value) + (surrogate_obj + entropy_obj * lambda_entropy)
            loss_KLPEN = (critic_loss) + (actor_loss)
            """

            '''critic_loss'''
            new_log_prob = self.act.compute__log_prob(state, action)
            new_value1, new_value2 = self.cri(state)  # TwinCritic
            # new_log_prob, new_value1, new_value2 = self.act_target.compute__log_prob(state, action)

            critic_loss = (self.criterion(new_value1, old_value) +
                           self.criterion(new_value2, old_value)) / (old_value.std() * 2 + 1e-6)
            loss_c_sum += critic_loss.item() * 0.5  # just for print
            self.cri_optimizer.zero_grad()
            critic_loss.backward()
            self.cri_optimizer.step()

            '''actor_loss'''
            # surrogate objective of TRPO
            ratio = (new_log_prob - old_log_prob).exp()
            surrogate_obj0 = advantage * ratio
            surrogate_obj1 = advantage * ratio.clamp(1 - clip, 1 + clip)
            surrogate_obj = -torch.min(surrogate_obj0, surrogate_obj1).mean()
            # policy entropy
            loss_entropy = (new_log_prob.exp() * new_log_prob).mean()

            actor_loss = surrogate_obj + loss_entropy * lambda_entropy
            loss_a_sum += actor_loss.item()  # just for print
            self.act_optimizer.zero_grad()
            actor_loss.backward()
            self.act_optimizer.step()

        buffer.storage_list = list()  # online policy
        loss_a_avg = loss_a_sum / sample_times
        loss_c_avg = loss_c_sum / sample_times
        return loss_a_avg, loss_c_avg


class AgentDeepSAC(AgentBasicAC):
    def __init__(self, state_dim, action_dim, net_dim):
        super(AgentBasicAC, self).__init__()
        use_dn = True  # and use hard target update
        self.learning_rate = 2e-4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''network'''
        actor_dim = net_dim
        self.act = ActorSAC(state_dim, action_dim, actor_dim, use_dn).to(self.device)
        self.act.train()
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate)

        self.act_target = ActorSAC(state_dim, action_dim, net_dim, use_dn).to(self.device)
        self.act_target.eval()
        self.act_target.load_state_dict(self.act.state_dict())

        critic_dim = int(net_dim * 1.25)
        self.cri = CriticTwinShared(state_dim, action_dim, critic_dim, use_dn).to(self.device)
        self.cri.train()
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate)

        self.cri_target = CriticTwinShared(state_dim, action_dim, critic_dim, use_dn).to(self.device)
        self.cri_target.eval()
        self.cri_target.load_state_dict(self.cri.state_dict())

        self.criterion = nn.SmoothL1Loss()

        '''training record'''
        self.state = None  # env.reset()
        self.reward_sum = 0.0
        self.step = 0
        self.update_counter = 0

        '''extension: auto-alpha for maximum entropy'''
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optimizer = torch.optim.Adam((self.log_alpha,), lr=self.learning_rate)
        self.target_entropy = -np.log(1.0 / action_dim) * 0.98
        '''extension: auto learning rate of actor'''
        self.trust_rho = TrustRho()

        '''constant'''
        self.explore_rate = 1.0  # explore rate when update_buffer(), 1.0 is better than 0.5
        self.explore_noise = True  # stochastic policy choose noise_std by itself.
        self.update_freq = 2 ** 7  # delay update frequency, for hard target update

    def update_parameters(self, buffer, max_step, batch_size, repeat_times):
        update_freq = self.update_freq * repeat_times  # delay update frequency, for soft target update
        self.act.train()

        loss_a_sum = 0.0
        loss_c_sum = 0.0
        rho = self.trust_rho()

        k = 1.0 + buffer.now_len / buffer.max_len
        batch_size_ = int(batch_size * k)
        update_times = int(max_step * k)

        for i in range(update_times * repeat_times):
            with torch.no_grad():
                reward, mask, state, action, next_s = buffer.random_sample(batch_size_, self.device)

                next_a_noise, next_log_prob = self.act_target.get__a__log_prob(next_s)
                next_q_target = torch.add(*self.cri_target.get__q1_q2(next_s, next_a_noise)) * 0.5  # CriticTwin
                next_q_target = next_q_target - next_log_prob * self.alpha  # SAC, alpha
                q_target = reward + mask * next_q_target
            '''critic_loss'''
            q1_value, q2_value = self.cri.get__q1_q2(state, action)  # CriticTwin
            critic_loss = self.criterion(q1_value, q_target) + self.criterion(q2_value, q_target)
            loss_c_tmp = critic_loss.item() * 0.5  # CriticTwin
            loss_c_sum += loss_c_tmp
            self.trust_rho.append_loss_c(loss_c_tmp)

            self.cri_optimizer.zero_grad()
            critic_loss.backward()
            self.cri_optimizer.step()

            '''actor_loss'''
            if i % repeat_times == 0 and rho > 2 ** -8:  # (self.rho>0.001) ~= (self.critic_loss<2.6)
                # stochastic policy
                actions_noise, log_prob = self.act.get__a__log_prob(state)  # policy gradient
                # auto alpha
                alpha_loss = -(self.log_alpha * (log_prob - self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

                # policy gradient
                self.alpha = self.log_alpha.exp()
                # q_eval_pg = self.cri(state, actions_noise)  # policy gradient
                q_eval_pg = torch.min(*self.cri.get__q1_q2(state, actions_noise))  # policy gradient, stable but slower

                actor_loss = (-q_eval_pg + log_prob * self.alpha).mean()  # policy gradient
                loss_a_sum += actor_loss.item()

                self.act_optimizer.zero_grad()
                actor_loss.backward()
                self.act_optimizer.step()

            """target update"""
            soft_target_update(self.act_target, self.act)  # soft target update
            soft_target_update(self.cri_target, self.cri)  # soft target update

            self.update_counter += 1
            if self.update_counter >= update_freq:
                self.update_counter = 0

                rho = self.trust_rho.update_rho()
                self.act_optimizer.param_groups[0]['lr'] = self.learning_rate * rho

        loss_a_avg = loss_a_sum / update_times
        loss_c_avg = loss_c_sum / (update_times * repeat_times)
        return loss_a_avg, loss_c_avg


class AgentInterSAC(AgentBasicAC):  # Integrated Soft Actor-Critic Methods
    def __init__(self, state_dim, action_dim, net_dim):
        super(AgentBasicAC, self).__init__()
        self.learning_rate = 2e-4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''network'''
        actor_dim = net_dim
        self.act = InterSPG(state_dim, action_dim, actor_dim).to(self.device)
        self.act.train()

        # critic_dim = int(net_dim * 1.25)
        # self.cri = ActorCriticSPG(state_dim, action_dim, critic_dim, use_dn).to(self.device)
        # self.cri.train()
        self.cri = self.act

        # self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate)
        # self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate)
        para_list = list(self.act.parameters())  # + list(self.cri.parameters())
        self.act_optimizer = torch.optim.Adam(para_list, lr=self.learning_rate)

        self.act_target = InterSPG(state_dim, action_dim, net_dim).to(self.device)
        self.act_target.eval()
        self.act_target.load_state_dict(self.act.state_dict())

        # self.cri_target = ActorCriticSPG(state_dim, action_dim, critic_dim, use_dn).to(self.device)
        # self.cri_target.eval()
        # self.cri_target.load_state_dict(self.cri.state_dict())

        self.criterion = nn.SmoothL1Loss()

        '''training record'''
        self.state = None  # env.reset()
        self.reward_sum = 0.0
        self.step = 0
        self.update_counter = 0

        '''extension: auto-alpha for maximum entropy'''
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optimizer = torch.optim.Adam((self.log_alpha,), lr=self.learning_rate)
        self.target_entropy = np.log(1.0 / action_dim)

        # self.auto_k = torch.ones(1, requires_grad=True, device=self.device)
        # self.auto_k_optimizer = torch.optim.Adam((self.auto_k,), lr=0.05)
        # self.target_rho = 0.9

        '''extension: auto learning rate of actor'''
        self.trust_rho = TrustRho()

        '''constant'''
        self.explore_rate = 1.0  # explore rate when update_buffer(), 1.0 is better than 0.5
        self.explore_noise = True  # stochastic policy choose noise_std by itself.
        self.update_freq = 2 ** 7  # delay update frequency, for hard target update

    def update_parameters(self, buffer, max_step, batch_size, repeat_times):
        update_freq = self.update_freq
        self.act.train()

        loss_a_sum = 0.0
        loss_c_sum = 0.0
        rho = self.trust_rho()

        k = 1.0 + buffer.now_len / buffer.max_len
        batch_size_ = int(batch_size * k)
        update_times = int(max_step * k)
        # k_loss = self.auto_k * (rho - self.target_rho) + torch.div(1, self.auto_k) + 0.5 * self.auto_k
        # self.auto_k_optimizer.zero_grad()
        # k_loss.backward()
        # self.auto_k_optimizer.step()
        # batch_size_ = int(batch_size * (1.0 + buffer.now_len / buffer.max_len))
        # update_times = int(max_step * self.auto_k.item())

        for i in range(update_times):
            with torch.no_grad():
                reward, mask, state, action, next_s = buffer.random_sample(batch_size_ + 1, self.device)

                next_a_noise, next_log_prob = self.act_target.get__a__log_prob(next_s)
                next_q_target1, next_q_target2 = self.act_target.get__q1_q2(next_s, next_a_noise)
                next_q_target = (next_q_target1 + next_q_target2) * 0.5
                next_q_target = next_q_target - next_log_prob * self.alpha  # SAC, alpha
                q_target = reward + mask * next_q_target
            '''critic_loss'''
            q1_value, q2_value = self.cri.get__q1_q2(state, action)  # CriticTwin
            critic_loss = self.criterion(q1_value, q_target) + self.criterion(q2_value, q_target)
            loss_c_tmp = critic_loss.item() * 0.5  # CriticTwin
            loss_c_sum += loss_c_tmp
            self.trust_rho.append_loss_c(loss_c_tmp)

            '''stochastic policy'''
            a_mean1, a_std_log_1, a_noise, log_prob = self.act.get__a__avg_std_noise_prob(state)  # policy gradient

            '''auto alpha'''
            alpha_loss = -(self.log_alpha * (log_prob - self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            '''action correction term'''
            a_mean2, a_std_log_2 = self.act_target.get__a__std(state)
            actor_term = self.criterion(a_mean1, a_mean2) + self.criterion(a_std_log_1, a_std_log_2)

            '''actor_loss'''
            if rho > 2 ** -8:  # (self.rho>0.001) ~= (self.critic_loss<2.6)
                self.alpha = self.log_alpha.exp()
                q_eval_pg = torch.min(*self.act_target.get__q1_q2(state, a_noise))  # policy gradient

                actor_loss = (-q_eval_pg + log_prob * self.alpha).mean()  # policy gradient
                loss_a_sum += actor_loss.item()
            else:
                actor_loss = 0

            united_loss = critic_loss + actor_term * (1 - rho) + actor_loss * rho

            self.act_optimizer.zero_grad()
            united_loss.backward()
            self.act_optimizer.step()

            """target update"""
            soft_target_update(self.act_target, self.act, tau=2 ** -8)

            self.update_counter += 1
            if self.update_counter >= update_freq:
                self.update_counter = 0
                # self.act_target.load_state_dict(self.act.state_dict())  # hard target update
                rho = self.trust_rho.update_rho()

        loss_a_avg = loss_a_sum / update_times
        loss_c_avg = loss_c_sum / update_times
        return loss_a_avg, loss_c_avg


"""run"""


def test_car_racing():
    env_name = 'CarRacing-v0'
    env, state_dim, action_dim, max_action, target_reward, is_discrete = build_gym_env(env_name, is_print=True)

    state = env.reset()
    import cv2
    action = np.array((0, 1.0, -1.0))
    for i in range(321):
        # action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        # env.render
        show = state.reshape((2, 96, 96))
        show = ((show[0] + 1.0) * 128).astype(np.uint8)
        cv2.imshow('', show)
        cv2.waitKey(1)
        if done:
            break
        # env.render()


def run__car_racing(gpu_id=None):
    print('pixel-level state')

    """run offline policy"""
    # args = Arguments(rl_agent=AgentInterSAC, gpu_id=gpu_id) # todo beta2
    args = Arguments(rl_agent=AgentDeepSAC, gpu_id=gpu_id)  # todo beta3
    args.env_name = "CarRacing-v0"
    args.max_total_step = int(1e6 * 4)
    args.batch_size = 2 ** 9  # beta1  #  2 ** 8 beta2
    args.max_memo = 2 ** 19
    args.eva_size = 4
    args.reward_scale = 2 ** -1
    args.init_for_training()
    train_agent(**vars(args))
    # build_for_mp(args)  # plan to update it, but complex

    """run online policy"""
    # args = Arguments(rl_agent=AgentGAE, gpu_id=gpu_id) # todo beta1
    # args.env_name = "CarRacing-v0"
    # args.max_memo = 2 ** 11
    # args.batch_size = 2 ** 8
    # args.repeat_times = 2 ** 4
    # args.net_dim = 2 ** 8
    # args.gamma = 0.99
    # args.random_seed = 1942
    # args.max_total_step = int(1e6 * 4)
    # args.max_step = int(2000)
    # args.eva_size = 4
    # args.reward_scale = 2 ** -1
    # args.init_for_training()
    # train_agent(**vars(args))


if __name__ == '__main__':
    # test_conv2d()
    # run0826()
    run__car_racing()
