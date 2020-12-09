from AgentRun import *
from AgentZoo import *
from AgentNet import *

"""
ModPPO use_dn=True
beta0 BW 256
beta1 BW 128
ceta0 Mini 256
ceta1 Mini 128

ModPPO StdPPO
beta2
"""


class ActorPPO(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim, use_dn=False):
        super().__init__()

        def idx_dim(i):
            return int(8 * 1.5 ** i)

        # if isinstance(state_dim, int):
        if use_dn:
            nn_dense_net = DenseNet(mid_dim)
            lay_dim = nn_dense_net.out_dim
        else:
            nn_dense_net = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU())
            lay_dim = mid_dim

        self.net = nn.Sequential(
            nn.Linear(state_dim, mid_dim), nn.ReLU(),
            nn_dense_net,
        )
        self.net__a = nn.Sequential(nn.Linear(lay_dim, mid_dim), HardSwish(),
                                    nn.Linear(mid_dim, action_dim), )
        self.net__d = nn.Sequential(nn.Linear(lay_dim, mid_dim), HardSwish(),  # todo
                                    nn.Linear(mid_dim, action_dim), )
        # else:
        #     self.net = nn.Sequential(
        #         NnnReshape(*state_dim),  # -> [batch_size, 4, 96, 96]
        #         nn.Conv2d(state_dim[0], idx_dim(0), 4, 2, bias=True), nn.LeakyReLU(),
        #         nn.Conv2d(idx_dim(0), idx_dim(1), 3, 2, bias=False), nn.ReLU(),
        #         nn.Conv2d(idx_dim(1), idx_dim(2), 3, 2, bias=False), nn.ReLU(),
        #         nn.Conv2d(idx_dim(2), idx_dim(3), 3, 2, bias=True), nn.ReLU(),
        #         nn.Conv2d(idx_dim(3), idx_dim(4), 3, 1, bias=True), nn.ReLU(),
        #         nn.Conv2d(idx_dim(4), idx_dim(5), 3, 1, bias=True), nn.ReLU(),
        #         NnnReshape(-1),
        #         nn.Linear(idx_dim(5), mid_dim), nn.ReLU(),
        #         nn.Linear(mid_dim, action_dim),
        #     )

        # self.a_std_log = nn.Parameter(torch.zeros(1, action_dim) - 0.5, requires_grad=True) #todo
        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))

        # layer_norm(self.net__mean[0], std=1.0)
        # layer_norm(self.net__mean[2], std=1.0)
        # layer_norm(self.net[-1], std=0.01)  # output layer for action
        layer_norm(self.net__a[-1], std=0.01)  # output layer for action
        layer_norm(self.net__d[-1], std=0.01, bias_const=-0.5)  # output layer for action

    def forward(self, s):
        x = self.net(s)
        a = self.net__a(x)  # todo
        return a.tanh()

    def get__a_noise__noise(self, state):
        x = self.net(state)  # todo
        a_avg = self.net__a(x)
        a_std_log = self.net__d(x)
        a_std = a_std_log.exp()

        noise = torch.randn_like(a_avg)
        a_noise = a_avg + noise * a_std
        return a_noise, noise

    def get__log_prob(self, state, noise): # todo
        x = self.net(state)
        a_std_log = self.net__d(x)

        a_delta = noise.pow(2) / 2
        log_prob = -(a_delta + (a_std_log + self.sqrt_2pi_log))
        return log_prob.sum(1)

    def compute__log_prob(self, state, a_noise):
        x = self.net(state)  # todo
        a_avg = self.net__a(x)
        a_std_log = self.net__d(x)
        a_std = a_std_log.exp()

        a_delta = ((a_avg - a_noise) / a_std).pow(2) / 2
        log_prob = -(a_delta + (a_std_log + self.sqrt_2pi_log))
        return log_prob.sum(1)


class AgentModPPO(AgentPPO):
    def __init__(self, state_dim, action_dim, net_dim):
        super(AgentPPO, self).__init__()
        self.learning_rate = 1e-4  # learning rate of actor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_dn = True

        '''network'''
        self.act = ActorPPO(state_dim, action_dim, net_dim, use_dn).to(self.device)
        self.act.train()
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate, )  # betas=(0.5, 0.99))

        self.cri = CriticAdv(state_dim, net_dim, use_dn).to(self.device)
        self.cri.train()
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate, )  # betas=(0.5, 0.99))

        self.criterion = nn.SmoothL1Loss()

        '''extension: reliable lambda for auto-learning-rate'''
        self.avg_loss_c = (-np.log(0.5)) ** 0.5

    def update_policy(self, buffer, _max_step, batch_size, repeat_times):
        self.act.train()
        self.cri.train()
        clip = 0.25  # ratio.clamp(1 - clip, 1 + clip)
        lambda_adv = 0.98  # why 0.98? cannot use 0.99
        lambda_entropy = 0.01  # could be 0.02
        # repeat_times = 8 could be 2**3 ~ 2**5

        actor_loss = critic_loss = None  # just for print return

        '''the batch for training'''
        max_memo = buffer.now_len
        all_reward, all_mask, all_state, all_action, all_noise = buffer.all_sample()

        all__new_v = list()
        all_log_prob = list()
        with torch.no_grad():
            b_size = 1024
            for i in range(0, all_state.size()[0], b_size):
                state = all_state[i:i + b_size]
                new_v = self.cri(state)
                all__new_v.append(new_v)

                log_prob = self.act.get__log_prob(state, all_noise[i:i + b_size]) # todo
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

        all__adv_v = (all__adv_v - all__adv_v.mean()) / (all__adv_v.std() + 1e-5)  # advantage_norm:

        '''mini batch sample'''
        sample_times = int(repeat_times * max_memo / batch_size)
        for _ in range(sample_times):
            '''random sample'''
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
            new_log_prob = self.act.compute__log_prob(state, action)  # it is actor_loss
            new_value = self.cri(state)

            critic_loss = (self.criterion(new_value, old_value)) / (old_value.std() + 1e-5)
            '''auto reliable lambda'''
            self.avg_loss_c = 0.995 * self.avg_loss_c + 0.005 * critic_loss.item()  # soft update, twin critics
            lamb = np.exp(-self.avg_loss_c ** 2)

            self.cri_optimizer.zero_grad()
            critic_loss.backward()
            self.cri_optimizer.step()

            '''actor_loss'''
            # surrogate objective of TRPO
            ratio = torch.exp(new_log_prob - old_log_prob)
            surrogate_obj0 = advantage * ratio
            surrogate_obj1 = advantage * ratio.clamp(1 - clip, 1 + clip)
            surrogate_obj = -torch.min(surrogate_obj0, surrogate_obj1).mean()
            loss_entropy = (torch.exp(new_log_prob) * new_log_prob).mean()  # policy entropy

            actor_loss = (surrogate_obj + loss_entropy * lambda_entropy) * lamb
            # self.act_optimizer.param_groups[0]['lr'] = self.learning_rate * lamb
            self.act_optimizer.zero_grad()
            actor_loss.backward()
            self.act_optimizer.step()

        self.act.eval()
        self.cri.eval()
        return actor_loss.item(), critic_loss.item()


def run__on_policy():
    args = Arguments(rl_agent=None, env_name=None, gpu_id=None)
    args.rl_agent = AgentModPPO
    args.random_seed += 12
    args.if_break_early = False

    args.net_dim = 2 ** 8
    args.max_memo = 2 ** 12
    args.batch_size = 2 ** 9
    args.repeat_times = 2 ** 4
    args.reward_scale = 2 ** 0  # unimportant hyper-parameter in PPO which do normalization on Q value
    args.gamma = 0.99  # important hyper-parameter, related to episode steps

    # args.env_name = "BipedalWalker-v3"
    # args.break_step = int(8e5 * 8)  # (6e5) 8e5 (6e6), UsedTimes: (800s) 1500s (8000s)
    # args.reward_scale = 2 ** 0  # (-150) -90 ~ 300 (324)
    # args.gamma = 0.95  # important hyper-parameter, related to episode steps
    # args.init_for_training()
    # train_agent_mp(args)  # train_agent(args)
    # exit()

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)
    args.env_name = "MinitaurBulletEnv-v0"  # PPO is the best, I don't know why.
    args.break_step = int(5e5 * 8)  # (PPO 3e5) 5e5
    args.reward_scale = 2 ** 4  # (-2) 0 ~ 16 (PPO 34)
    args.gamma = 0.95  # important hyper-parameter, related to episode steps
    args.net_dim = 2 ** 8
    args.max_memo = 2 ** 11
    args.batch_size = 2 ** 9
    args.repeat_times = 2 ** 4
    args.init_for_training()
    train_agent_mp(args)
    exit()


run__on_policy()
