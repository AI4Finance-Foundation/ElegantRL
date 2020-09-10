from AgentRun import *
from AgentNet import *
from AgentZoo import *

"""
New version InterSAC 2020-09-09 
update  AgentInterSAC (solve log_alpha trouble)
update InterSPG (solve negative log_prob trouble)

loss C can't decrease, I try and move auto-alpha into loss_C > epi (fail?)
        self.alpha = torch.tensor((0.5,), requires_grad=True, device=self.device)  # todo
"""


class InterSPG(nn.Module):  # class AgentIntelAC for SAC (SPG means stochastic policy gradient)
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()
        self.log_std_min = -20
        self.log_std_max = 2
        self.constant_log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # encoder
        self.enc_s = nn.Sequential(
            nn.Linear(state_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
        )  # state
        self.enc_a = nn.Sequential(
            nn.Linear(action_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
        )  # action without nn.Tanh()

        self.net = DenseNet2(mid_dim)  # todo
        net_out_dim = self.net.out_dim

        # decoder
        self.dec_a = nn.Sequential(
            nn.Linear(net_out_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, action_dim),
        )  # action_mean
        self.dec_d = nn.Sequential(
            nn.Linear(net_out_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, action_dim),
        )  # action_std_log (d means standard dev.)
        self.dec_q1 = nn.Sequential(
            nn.Linear(net_out_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, 1),
        )  # q_value1 SharedTwinCritic
        self.dec_q2 = nn.Sequential(
            nn.Linear(net_out_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, 1),
        )  # q_value2 SharedTwinCritic

        layer_norm(self.dec_a[-1], std=0.01)  # net[-1] is output layer for action, it is no necessary.
        layer_norm(self.dec_q1[-1], std=0.1)
        layer_norm(self.dec_q2[-1], std=0.1)

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
        log_prob_noise = a_delta + a_std_log + self.constant_log_sqrt_2pi

        a_noise_tanh = a_noise.tanh()
        # log_prob = log_prob_noise - (1 - a_noise_tanh.pow(2) + epsilon).log() # epsilon = 1e-6
        # same as:
        log_prob = log_prob_noise + (-a_noise_tanh.pow(2) + 1.000001).log()
        return a_noise_tanh, log_prob.sum(1, keepdim=True)  # todo neg_log_prob

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
        log_prob_noise = a_delta + a_std_log + self.constant_log_sqrt_2pi

        a_noise_tanh = a_noise.tanh()
        # log_prob = log_prob_noise - (1 - a_noise_tanh.pow(2) + epsilon).log() # epsilon = 1e-6
        # same as:
        log_prob = log_prob_noise + (-a_noise_tanh.pow(2) + 1.000001).log()  # todo neg_log_prob
        return a_mean.tanh(), a_std_log, a_noise_tanh, log_prob.sum(1, keepdim=True)

    def get__q1_q2(self, s, a):  # critic
        s_ = self.enc_s(s)
        a_ = self.enc_a(a)
        q_ = self.net(s_ + a_)
        q1 = self.dec_q1(q_)
        q2 = self.dec_q2(q_)
        return q1, q2


class AgentInterSAC(AgentBasicAC):  # Integrated Soft Actor-Critic Methods
    def __init__(self, state_dim, action_dim, net_dim):
        super(AgentBasicAC, self).__init__()
        self.learning_rate = 2e-4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''network'''
        actor_dim = net_dim
        self.act = InterSPG(state_dim, action_dim, actor_dim).to(self.device)
        self.act.train()

        self.cri = self.act

        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate)

        self.act_target = InterSPG(state_dim, action_dim, net_dim).to(self.device)
        self.act_target.eval()
        self.act_target.load_state_dict(self.act.state_dict())

        self.criterion = nn.SmoothL1Loss()

        '''training record'''
        self.state = None  # env.reset()
        self.reward_sum = 0.0
        self.step = 0
        self.update_counter = 0

        '''extension: auto-alpha for maximum entropy'''
        self.alpha = torch.tensor((0.5,), requires_grad=True, device=self.device)  # todo
        self.alpha_optimizer = torch.optim.Adam((self.alpha,), lr=self.learning_rate)
        self.target_entropy = np.log(action_dim) * 0.5  # todo

        '''extension: auto learning rate of actor'''
        self.trust_rho = TrustRho0909()

        '''constant'''
        self.explore_rate = 1.0  # explore rate when update_buffer(), 1.0 is better than 0.5
        self.explore_noise = True  # stochastic policy choose noise_std by itself.

    def update_parameters(self, buffer, max_step, batch_size, repeat_times):
        self.act.train()

        loss_a_list = list()
        loss_c_list = list()

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
                reward, mask, state, action, next_s = buffer.random_sample(batch_size_, self.device)

                next_a_noise, next_log_prob = self.act_target.get__a__log_prob(next_s)
                next_q_target = torch.min(*self.act_target.get__q1_q2(next_s, next_a_noise))  # twin critic
                q_target = reward + mask * (next_q_target + next_log_prob * self.alpha)  # policy entropy
            '''critic_loss'''
            q1_value, q2_value = self.cri.get__q1_q2(state, action)  # CriticTwin
            critic_loss = self.criterion(q1_value, q_target) + self.criterion(q2_value, q_target)
            loss_c_tmp = critic_loss.item() * 0.5  # CriticTwin
            loss_c_list.append(loss_c_tmp)
            rho = self.trust_rho.update_rho(loss_c_tmp)

            '''stochastic policy'''
            a1_mean, a1_log_std, a_noise, log_prob = self.act.get__a__avg_std_noise_prob(state)  # policy gradient

            '''action correction term'''
            a2_mean, a2_log_std = self.act_target.get__a__std(state)
            actor_term = self.criterion(a1_mean, a2_mean) + self.criterion(a1_log_std, a2_log_std)

            if rho > 2 ** -8:  # (self.rho>0.001) ~= (self.critic_loss<2.6)
                '''auto alpha'''
                # alpha_loss = -(self.log_alpha * (log_prob - self.target_entropy).detach()).mean()
                alpha_loss = (self.alpha.log() * (log_prob - self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

                '''actor_loss'''
                q_eval_pg = torch.min(*self.act_target.get__q1_q2(state, a_noise))  # policy gradient
                actor_loss = -(q_eval_pg + log_prob * self.alpha).mean()  # policy gradient
                loss_a_list.append(actor_loss.item())
            else:
                actor_loss = 0

            united_loss = critic_loss + actor_term * (1 - rho) + actor_loss * rho
            self.act_optimizer.zero_grad()
            united_loss.backward()
            self.act_optimizer.step()

            soft_target_update(self.act_target, self.act, tau=2 ** -8)

        loss_a_avg = (sum(loss_a_list) / len(loss_a_list)) if len(loss_a_list) > 0 else 0.0
        loss_c_avg = sum(loss_c_list) / len(loss_c_list)
        return loss_a_avg, loss_c_avg


def mp__update_params(args, q_i_buf, q_o_buf, q_i_eva, q_o_eva):  # update network parameters using replay buffer
    class_agent = args.rl_agent
    max_memo = args.max_memo
    net_dim = args.net_dim
    max_step = args.max_step
    max_total_step = args.max_total_step
    batch_size = args.batch_size
    repeat_times = args.repeat_times
    cwd = args.cwd
    del args

    state_dim, action_dim = q_o_buf.get()  # q_o_buf 1.
    agent = class_agent(state_dim, action_dim, net_dim)

    from copy import deepcopy
    act_cpu = deepcopy(agent.act).to(torch.device("cpu"))
    act_cpu.eval()
    [setattr(param, 'requires_grad', False) for param in act_cpu.parameters()]
    q_i_buf.put(act_cpu)  # q_i_buf 1.
    # q_i_buf.put(act_cpu)  # q_i_buf 2. # warning
    q_i_eva.put(act_cpu)  # q_i_eva 1.

    buffer = BufferArrayGPU(max_memo, state_dim, action_dim)  # experiment replay buffer

    '''initial_exploration'''
    buffer_array, reward_list, step_list = q_o_buf.get()  # q_o_buf 2.
    reward_avg = np.average(reward_list)
    step_sum = sum(step_list)
    buffer.extend_memo(buffer_array)

    q_i_eva.put((act_cpu, reward_avg, step_sum, 0, 0))  # q_i_eva 1.

    total_step = step_sum
    is_training = True
    # is_solved = False
    while is_training:
        buffer_array, reward_list, step_list = q_o_buf.get()  # q_o_buf n.
        reward_avg = np.average(reward_list)
        step_sum = sum(step_list)
        total_step += step_sum
        buffer.extend_memo(buffer_array)

        buffer.init_before_sample()
        loss_a_avg, loss_c_avg = agent.update_parameters(buffer, max_step, batch_size, repeat_times)

        act_cpu.load_state_dict(agent.act.state_dict())
        q_i_buf.put(act_cpu)  # q_i_buf n.
        q_i_eva.put((act_cpu, reward_avg, step_sum, loss_a_avg, loss_c_avg))  # q_i_eva n.

        if q_o_eva.qsize() > 0:
            is_solved = q_o_eva.get()  # q_o_eva n.
        '''break loop rules'''
        # if is_solved: # todo
        #     is_training = False
        if total_step > max_total_step or os.path.exists(f'{cwd}/stop.mark'):
            is_training = False

    q_i_buf.put('stop')
    q_i_eva.put('stop')
    while q_i_buf.qsize() > 0 or q_i_eva.qsize() > 0:
        time.sleep(1)
    time.sleep(4)
    # print('; quit: params')


def build_for_mp(args):
    import multiprocessing as mp
    q_i_buf = mp.Queue(maxsize=8)  # buffer I
    q_o_buf = mp.Queue(maxsize=8)  # buffer O
    q_i_eva = mp.Queue(maxsize=8)  # evaluate I
    q_o_eva = mp.Queue(maxsize=8)  # evaluate O
    process = [mp.Process(target=mp__update_params, args=(args, q_i_buf, q_o_buf, q_i_eva, q_o_eva)),
               mp.Process(target=mp__update_buffer, args=(args, q_i_buf, q_o_buf,)),
               mp.Process(target=mp_evaluate_agent, args=(args, q_i_eva, q_o_eva)), ]
    [p.start() for p in process]
    [p.join() for p in process]
    print('\n')


def run_continuous_action(gpu_id=None):
    # import AgentZoo as Zoo
    args = Arguments(rl_agent=AgentInterSAC, gpu_id=gpu_id)

    # args.env_name = "Pendulum-v0"  # It is easy to reach target score -200.0 (-100 is harder)
    # args.max_total_step = int(1e4 * 4)
    # args.reward_scale = 2 ** -2
    # args.init_for_training()
    # build_for_mp(args)  # train_offline_policy(**vars(args))
    # # exit()
    #
    # args.env_name = "LunarLanderContinuous-v2"
    # args.max_total_step = int(1e5 * 4)
    # args.init_for_training()
    # build_for_mp(args)  # train_offline_policy(**vars(args))
    # # exit()

    # args.env_name = "BipedalWalker-v3"
    # args.random_seed = 1945
    # args.max_total_step = int(2e5 * 4)
    # args.init_for_training()
    # build_for_mp(args)  # train_offline_policy(**vars(args))
    # # exit()

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)
    args.env_name = "AntBulletEnv-v0"
    args.max_total_step = int(5e5 * 4)
    args.max_epoch = 2 ** 13
    args.max_memo = 2 ** 20
    args.max_step = 2 ** 10
    args.batch_size = 2 ** 9
    args.reward_scale = 2 ** -2
    args.eva_size = 2 ** 3  # for Recorder
    args.show_gap = 2 ** 8  # for Recorder
    args.init_for_training()
    build_for_mp(args)  # train_offline_policy(**vars(args))

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)
    args.env_name = "MinitaurBulletEnv-v0"
    args.max_total_step = int(1e6 * 2)
    args.max_epoch = 2 ** 13
    args.max_memo = 2 ** 20  # todo
    args.max_step = 2 ** 11  # todo 10
    args.net_dim = 2 ** 8
    args.batch_size = 2 ** 9
    args.reward_scale = 2 ** 4
    args.eva_size = 2 ** 3  # for Recorder
    args.show_gap = 2 ** 8  # for Recorder
    args.init_for_training(cpu_threads=4)
    # train_agent(**vars(args))
    build_for_mp(args)


run_continuous_action()
