from AgentRun import *
from AgentZoo import *
from AgentNet import *

"""
Conv2D dim 8, leaklyReLU
"""


def _layer_norm(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)


class InterPPO(nn.Module):  # Pixel-level state version
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()

        def idx_dim(i):
            return int(16 + 8 * i)

        # nn_dense = DenseNet(mid_dim)
        # out_dim = nn_dense.out_dim
        out_dim = mid_dim
        self.enc_s = nn.Sequential(  # the only difference.
            nn.Linear(state_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            # nn_dense,
        ) if isinstance(state_dim, int) else nn.Sequential(
            NnnReshape(*state_dim),  # -> [batch_size, 4, 96, 96]
            nn.Conv2d(state_dim[0], idx_dim(0), 4, 2, bias=True), nn.LeakyReLU(),
            nn.Conv2d(idx_dim(0), idx_dim(1), 3, 2, bias=False), nn.ReLU(),
            nn.Conv2d(idx_dim(1), idx_dim(2), 3, 2, bias=False), nn.ReLU(),
            nn.Conv2d(idx_dim(2), idx_dim(3), 3, 2, bias=True), nn.ReLU(),
            nn.Conv2d(idx_dim(3), idx_dim(4), 3, 1, bias=True), nn.ReLU(),
            nn.Conv2d(idx_dim(4), idx_dim(5), 3, 1, bias=True), nn.ReLU(),
            NnnReshape(-1),
            nn.Linear(idx_dim(5), mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
        )

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

        _layer_norm(self.dec_a[-1], std=0.01)
        _layer_norm(self.dec_q1[-1], std=0.01)
        _layer_norm(self.dec_q2[-1], std=0.01)

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


class AgentInterPPO(AgentPPO):
    def __init__(self, state_dim, action_dim, net_dim):
        super(AgentPPO, self).__init__()
        self.learning_rate = 1e-4  # learning rate of actor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''network'''
        self.act = InterPPO(state_dim, action_dim, net_dim).to(self.device)
        self.act.train()
        self.act_optimizer = torch.optim.Adam([
            {'params': self.act.enc_s.parameters(), 'lr': self.learning_rate * 0.8},  # more stable
            # {'params': self.act.net.parameters(), 'lr': self.learning_rate},
            {'params': self.act.dec_a.parameters(), },
            {'params': self.act.a_std_log, },
            {'params': self.act.dec_q1.parameters(), },
            {'params': self.act.dec_q2.parameters(), },
        ], lr=self.learning_rate)

        self.criterion = nn.SmoothL1Loss()
        self.action_dim = action_dim

        '''extension: reliable lambda for auto-learning-rate'''
        self.avg_loss_c = (-np.log(0.5)) ** 0.5

    def update_policy(self, buffer, _max_step, batch_size, repeat_times):
        buffer.update__now_len__before_sample()

        clip = 0.25  # ratio.clamp(1 - clip, 1 + clip)
        lambda_adv = 0.98  # why 0.98? cannot use 0.99
        lambda_entropy = 0.01  # could be 0.02
        # repeat_times = 8 could be 2**3 ~ 2**5

        '''the batch for training'''
        max_memo = buffer.now_len
        all_reward, all_mask, all_state, all_action, all_noise = buffer.all_sample()

        b_size = 2 ** 10
        with torch.no_grad():
            all__new_v = list()
            all_log_prob = list()
            for i in range(0, all_state.size()[0], b_size):
                new_v, log_prob = self.act.get__q__log_prob(all_state[i:i + b_size], all_noise[i:i + b_size])
                all__new_v.append(new_v)
                all_log_prob.append(log_prob)

            all__new_v = torch.cat(all__new_v, dim=0)
            all_log_prob = torch.cat(all_log_prob, dim=0)

        '''compute old_v (old policy value), adv_v (advantage value) 
        refer: GAE. ICLR 2016. Generalization Advantage Estimate. 
        https://arxiv.org/pdf/1506.02438.pdf'''
        all__delta = torch.empty(max_memo, dtype=torch.float32, device=self.device)
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
        all__adv_v = (all__adv_v - all__adv_v.mean()) / (all__adv_v.std() + 1e-5)
        # Q_value_norm is necessary. Because actor_obj = surrogate_obj + loss_entropy * lambda_entropy.

        '''mini batch sample'''
        all__old_v = all__old_v.unsqueeze(1)
        sample_times = int(repeat_times * max_memo / batch_size)

        for _ in range(sample_times):
            '''random sample'''
            indices = rd.randint(max_memo, size=batch_size)

            state = all_state[indices]
            advantage = all__adv_v[indices]
            old_value = all__old_v[indices]
            action = all_action[indices]
            old_log_prob = all_log_prob[indices]

            q1_value, q2_value, new_log_prob = self.act.get__q1_q2__log_prob(state, action)

            """critic_obj"""
            critic_obj = (self.criterion(q1_value, old_value) +
                          self.criterion(q2_value, old_value)) / (old_value.std() + 1e-5)
            '''auto reliable lambda'''
            self.avg_loss_c = 0.995 * self.avg_loss_c + 0.005 * critic_obj.item() / 2  # soft update, twin critics
            lamb = np.exp(-self.avg_loss_c ** 2)

            """actor_obj"""
            # surrogate objective of TRPO
            ratio = torch.exp(new_log_prob - old_log_prob)
            surrogate_obj0 = advantage * ratio
            surrogate_obj1 = advantage * ratio.clamp(1 - clip, 1 + clip)
            surrogate_obj = -torch.min(surrogate_obj0, surrogate_obj1).mean()
            # policy entropy
            loss_entropy = (torch.exp(new_log_prob) * new_log_prob).mean()

            actor_obj = surrogate_obj + loss_entropy * lambda_entropy

            '''united_loss'''
            united_loss = critic_obj + actor_obj * lamb
            self.act_optimizer.zero_grad()
            united_loss.backward()
            self.act_optimizer.step()

        buffer.empty_memories__before_explore()
        return self.act.a_std_log.mean().item(), self.avg_loss_c


def train__pixel_level_state2d__car_racing():
    # from AgentZoo import AgentPPO
    '''
    700, 20e5, 31550, 1577(per 1e5)
    '''

    '''DEMO 4: Fix gym Box2D env CarRacing-v0 (pixel-level 2D-state, continuous action) using PPO'''
    import gym  # gym of OpenAI is not necessary for ElegantRL (even RL)
    gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'
    env = gym.make('CarRacing-v0')
    env = fix_car_racing_env(env)
    env.target_reward = 900  # todo

    args = Arguments(rl_agent=AgentInterPPO, env=env, gpu_id=None)
    args.if_break_early = True
    args.eval_times2 = 1
    args.eval_times2 = 3  # CarRacing Env is so slow. The GPU-util is low while training CarRacing.
    args.rollout_workers_num = 8  # (num, step, time) (8, 1e5, 1360) (4, 1e4, 1860)

    args.break_step = int(2e6 * 8)  # (1e5) 2e5 4e5 (8e5) used time (7,000s) 10ks 30ks (60ks)
    # Sometimes bad luck (5%), it reach 300 score in 5e5 steps and don't increase.
    # You just need to change the random seed and retrain.
    args.reward_scale = 2 ** -2  # (-1) 50 ~ 700 ~ 900 (1001)
    args.max_memo = 2 ** 11
    args.batch_size = 2 ** 7
    args.repeat_times = 2 ** 4
    args.net_dim = 2 ** 7
    args.max_step = 2 ** 10
    args.show_gap = 2 ** 8  # for Recorder
    args.init_for_training()
    train_agent_mp(args)  # train_agent(args)
    exit()


if __name__ == '__main__':
    train__pixel_level_state2d__car_racing()
