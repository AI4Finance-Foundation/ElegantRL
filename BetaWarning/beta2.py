from AgentRun import *
from AgentZoo import *
from AgentNet import *

"""
search 'todo CarRacing-v0' to see the change
CarRacing-v0 GAE 
1482 |           691.11   691.11   119.00 |   -0.11   0.17 |6.29e+05

offline conv
"""


class NnnReshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.view((x.size(0),) + self.args)


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


class AgentPixelInterSAC(AgentBasicAC):  # Integrated Soft Actor-Critic Methods
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


def train_offline_policy(
        rl_agent, net_dim, batch_size, repeat_times, gamma, reward_scale, cwd,
        env_name, max_step, max_memo, max_total_step,
        eva_size, gpu_id, show_gap, **_kwargs):  # 2020-06-01
    env = gym.make(env_name)
    state_dim, action_dim, max_action, target_reward, is_discrete = get_env_info(env, is_print=False)
    if env_name == 'CarRacing-v0':
        env = fix_car_racing_v0(env)
        state_dim = (2, state_dim[0], state_dim[1])

    '''init: agent, buffer, recorder'''
    agent = rl_agent(state_dim, action_dim, net_dim)  # training agent
    agent.state = env.reset()
    buffer = BufferArray(max_memo, state_dim, 1 if is_discrete else action_dim)  # experiment replay buffer

    recorder = Recorder0825()
    if env_name == 'CarRacing-v0':
        recorder.eva_size1 = 2
    act = agent.act

    '''loop'''
    with torch.no_grad():  # update replay buffer
        rewards, steps = initial_exploration(env, buffer, max_step, max_action, reward_scale, gamma, action_dim)
    recorder.update__record_explore(steps, rewards, loss_a=0, loss_c=0)

    is_training = True
    while is_training:
        '''update replay buffer by interact with environment'''
        with torch.no_grad():  # for saving the GPU buffer
            rewards, steps = agent.update_buffer(
                env, buffer, max_step, max_action, reward_scale, gamma)

        '''update network parameters by random sampling buffer for gradient descent'''
        buffer.init_before_sample()
        loss_a, loss_c = agent.update_parameters(
            buffer, max_step, batch_size, repeat_times)

        '''saves the agent with max reward'''
        with torch.no_grad():  # for saving the GPU buffer
            recorder.update__record_explore(steps, rewards, loss_a, loss_c)

            is_saved = recorder.update__record_evaluate(
                env, act, max_step, max_action, eva_size, agent.device)
            recorder.save_act(cwd, act, gpu_id) if is_saved else None

            is_solved = recorder.check_is_solved(target_reward, gpu_id, show_gap)
        '''break loop rules'''
        if is_solved or recorder.total_step > max_total_step or os.path.exists(f'{cwd}/stop.mark'):
            is_training = False

    recorder.save_npy__plot_png(cwd)


def run__offline_policy(gpu_id=None):
    args = Arguments(rl_agent=AgentPixelInterSAC, gpu_id=gpu_id)

    args.env_name = "CarRacing-v0"  # todo CarRacing-v0

    args.max_total_step = int(1e6 * 4)
    args.batch_size = 2 ** 8
    args.max_memo = 2 ** 19
    args.eva_size = 4  # todo CarRacing-v0
    args.reward_scale = 2 ** -1
    args.init_for_training()
    train_offline_policy(**vars(args))


def fix_car_racing_v0(env):  # todo CarRacing-v0
    env.old_step = env.step
    """
    comment 'car_racing.py' line 233-234: print('Track generation ...
    comment 'car_racing.py' line 308-309: print("retry to generate track ...
    """

    def decorator_step(env_step):
        def new_env_step(action):
            state3, reward, done, info = env_step(action)
            state = state3[:, :, 1]  # show green
            state[86:, :24] = 0  # shield id
            state[86:, 24:36] = state3[86:, 24:36, 2]  # show red
            state[86:, 72:] = state3[86:, 72:, 0]  # show blue
            state = state.astype(np.float32) / 128.0 - 0.5
            if state.mean() > 0.95:  # fix CarRacing-v0 bug
                reward -= 10.0
                done = True

            # state2 = np.stack((env.prev_state, state))
            state2 = np.stack((env.prev_state, state)).flatten()  # todo pixel flatten
            env.prev_state = state
            return state2, reward, done, info

        return new_env_step

    env.step = decorator_step(env.step)

    def decorator_reset(env_reset):
        def new_env_reset():
            env_reset()
            action = np.zeros(3)
            for _ in range(16):
                env.old_step(action)
            env.prev_state = env.old_step(action)[0][:, :, 1]
            env.prev_state = env.prev_state.astype(np.float32) / 128.0 - 0.5
            return env.step(action)[0]  # state

        return new_env_reset

    env.reset = decorator_reset(env.reset)
    return env


if __name__ == '__main__':
    # test_conv2d()
    run__offline_policy()
    # run__online_policy()
