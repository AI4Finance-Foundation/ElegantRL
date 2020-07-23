from AgentRun import *
from AgentZoo import *
from AgentNet import *

"""
beta2 ArgumentsBeta
beta1 cancel SN, soft update
beta0 # todo # self.act_optimizer.param_groups[0]['lr'] = self.learning_rate * rho

"""


class InterSPG(nn.Module):  # class AgentIntelAC for SAC (SPG means stochastic policy gradient)
    def __init__(self, state_dim, action_dim, mid_dim):  # plan todo use_dn
        super(InterSPG, self).__init__()
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
        )  # action (without nn.Tanh())

        self.net = DenseNet(mid_dim)
        net_out_dim = mid_dim * 4

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
            # nn.utils.spectral_norm(nn.Linear(mid_dim, 1)), # todo
        )  # q_value1 SharedTwinCritic
        self.dec_q2 = nn.Sequential(
            nn.Linear(net_out_dim, mid_dim), HardSwish(),
            nn.Linear(mid_dim, 1),
            # nn.utils.spectral_norm(nn.Linear(mid_dim, 1)), # todo
        )  # q_value2 SharedTwinCritic

    def forward(self, s, noise_std=0.0):  # actor, in fact, noise_std is a boolean
        s_ = self.enc_s(s)
        a_ = self.net(s_)
        a_mean = self.dec_a(a_)  # NOTICE! it is a_mean without .tanh()

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
        return a_noise_tanh, log_prob.sum(1, keepdim=True)

    def get__a__std(self, state):
        s_ = self.enc_s(state)
        a_ = self.net(s_)
        a_mean = self.dec_a(a_)  # NOTICE! it is a_mean without .tanh()
        a_std_log = self.dec_d(a_).clamp(self.log_std_min, self.log_std_max)
        a_std = a_std_log.exp()

        return a_mean.tanh(), a_std

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
        return a_mean.tanh(), a_std, a_noise_tanh, log_prob.sum(1, keepdim=True)

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
        self.target_entropy = np.log(1.0 / action_dim) * 0.98
        '''extension: auto learning rate of actor'''
        self.trust_rho = TrustRho()

        '''constant'''
        self.explore_rate = 1.0  # explore rate when update_buffer(), 1.0 is better than 0.5
        self.explore_noise = True  # stochastic policy choose noise_std by itself.
        self.update_freq = 2 ** 7 # todo # delay update frequency, for hard target update

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
                next_q_target = torch.min(*self.act_target.get__q1_q2(next_s, next_a_noise))  # CriticTwin
                next_q_target = next_q_target - next_log_prob * self.alpha  # SAC, alpha
                q_target = reward + mask * next_q_target
            '''critic_loss'''
            q1_value, q2_value = self.cri.get__q1_q2(state, action)  # CriticTwin
            critic_loss = self.criterion(q1_value, q_target) + self.criterion(q2_value, q_target)
            loss_c_tmp = critic_loss.item() * 0.5  # CriticTwin
            loss_c_sum += loss_c_tmp
            self.trust_rho.append_loss_c(loss_c_tmp)

            '''actor correction term'''
            a_mean2, a_std2 = self.act_target.get__a__std(state)

            '''actor_loss'''
            if i % repeat_times == 0 and rho > 0.001:  # (self.rho>0.001) ~= (self.critic_loss<2.6)
                '''stochastic policy'''
                a_mean1, a_std1, a_noise, log_prob = self.act.get__a__avg_std_noise_prob(state)  # policy gradient

                '''auto alpha'''
                alpha_loss = -(self.log_alpha * (log_prob - self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

                '''policy gradient'''
                self.alpha = self.log_alpha.exp()
                q_eval_pg = torch.min(*self.act_target.get__q1_q2(state, a_noise))

                actor_loss = (-q_eval_pg + log_prob * self.alpha).mean()  # policy gradient
                loss_a_sum += actor_loss.item()

                actor_term = self.criterion(a_mean1, a_mean2) + self.criterion(a_std1, a_std2)
                united_loss = critic_loss + actor_term * (1 - rho) + actor_loss * (rho * 0.5)
            else:
                a_mean1, a_std1 = self.act.get__a__std(state)

                actor_term = self.criterion(a_mean1, a_mean2) + self.criterion(a_std1, a_std2)
                united_loss = critic_loss + actor_term * (1 - rho)

            self.act_optimizer.zero_grad()
            united_loss.backward()
            self.act_optimizer.step()

            """target update"""
            soft_target_update(self.act_target, self.act)  # soft target update

            self.update_counter += 1
            if self.update_counter >= update_freq:
                self.update_counter = 0
                # self.act_target.load_state_dict(self.act.state_dict())  # hard target update

                rho = self.trust_rho.update_rho()

        loss_a_avg = loss_a_sum / update_times
        loss_c_avg = loss_c_sum / (update_times * repeat_times)
        return loss_a_avg, loss_c_avg


def run__mp(gpu_id=None, cwd='MP__InterSAC'):
    import multiprocessing as mp
    q_i_buf = mp.Queue(maxsize=8)  # buffer I
    q_o_buf = mp.Queue(maxsize=8)  # buffer O
    q_i_eva = mp.Queue(maxsize=8)  # evaluate I
    q_o_eva = mp.Queue(maxsize=8)  # evaluate O

    def build_for_mp():
        process = [mp.Process(target=mp__update_params, args=(args, q_i_buf, q_o_buf, q_i_eva, q_o_eva)),
                   mp.Process(target=mp__update_buffer, args=(args, q_i_buf, q_o_buf,)),
                   mp.Process(target=mp_evaluate_agent, args=(args, q_i_eva, q_o_eva)), ]
        [p.start() for p in process]
        [p.join() for p in process]
        [p.close() for p in process]

    # args = ArgumentsBeta(AgentInterSAC, gpu_id, cwd, env_name="LunarLanderContinuous-v2")
    # build_for_mp()

    args = ArgumentsBeta(AgentInterSAC, gpu_id, cwd, env_name="BipedalWalker-v3")
    build_for_mp()

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)
    args = ArgumentsBeta(AgentInterSAC, gpu_id, cwd, env_name="AntBulletEnv-v0")
    args.max_epoch = 2 ** 13
    args.max_memo = 2 ** 20
    args.max_step = 2 ** 10
    args.net_dim = 2 ** 8
    args.batch_size = 2 ** 9
    args.reward_scale = 2 ** -2
    args.eva_size = 2 ** 5  # for Recorder
    args.show_gap = 2 ** 8  # for Recorder
    build_for_mp()
    #
    # import pybullet_envs  # for python-bullet-gym
    # dir(pybullet_envs)
    # args.env_name = "MinitaurBulletEnv-v0"
    # args.cwd = f'./{cwd}/{args.env_name}_{gpu_id}'
    # args.max_epoch = 2 ** 13
    # args.max_memo = 2 ** 20
    # args.net_dim = 2 ** 8
    # args.max_step = 2 ** 10
    # args.batch_size = 2 ** 9
    # args.reward_scale = 2 ** 3
    # args.is_remove = True
    # args.eva_size = 2 ** 5  # for Recorder
    # args.show_gap = 2 ** 8  # for Recorder
    # build_for_mp()


if __name__ == '__main__':
    run__mp()
