from AgentRun import *
from AgentNet import *
from AgentZoo import *

"""
ISAC1101 Minitaur
beta0 batch_size = (2 ** 8), net_dim = int(2 ** 8)
ceta1 batch_size = (2 ** 8), net_dim = int(2 ** 8), max_step = 2 ** 12
ceta0 batch_size = int(2 ** 8 * 1.5), args.net_dim = int(2 ** 8 * 1.5)
"""


class InterGAE(nn.Module):  # 2020-10-10
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()
        self.log_std_min = -20
        self.log_std_max = 2
        self.constant_log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # encoder
        self.enc_s = nn.Sequential(
            nn.Linear(state_dim, mid_dim), nn.ReLU(),
        )  # state

        '''use densenet'''
        self.net = DenseNet(mid_dim)
        net_out_dim = self.net.out_dim
        # '''not use densenet'''
        # self.net = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(), )
        # net_out_dim = mid_dim

        '''todo two layer'''
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
        '''todo one layer'''
        # self.dec_a = nn.Sequential(
        #     nn.Linear(net_out_dim, action_dim),
        # )  # action_mean
        # self.dec_d = nn.Sequential(
        #     nn.Linear(net_out_dim, action_dim),
        # )  # action_std_log (d means standard dev.)
        #
        # self.dec_q1 = nn.Sequential(
        #     nn.Linear(net_out_dim, 1),
        # )  # q_value1 SharedTwinCritic
        # self.dec_q2 = nn.Sequential(
        #     nn.Linear(net_out_dim, 1),
        # )  # q_value2 SharedTwinCritic

        # layer_norm(self.net[0], std=1.0)
        layer_norm(self.dec_a[-1], std=0.5)  # output layer for action
        layer_norm(self.dec_d[-1], std=0.1, bias_const=-0.5)  # output layer for std_log
        layer_norm(self.dec_q1[-1], std=0.1)  # output layer for q value
        layer_norm(self.dec_q2[-1], std=0.5)  # output layer for q value

    def forward(self, s):
        x = self.enc_s(s)
        x = self.net(x)
        a_mean = self.dec_a(x)
        return a_mean.tanh()

    def get__a_avg_std(self, state):
        x = self.enc_s(state)
        x = self.net(x)
        a_mean = self.dec_a(x)
        a_log_std = self.dec_d(x).clamp(self.log_std_min, self.log_std_max)
        a_std = torch.exp(a_log_std)
        return a_mean, a_std

    def get__q__log_prob1(self, state, noise):
        x = self.enc_s(state)
        x = self.net(x)
        q = torch.min(self.dec_q1(x), self.dec_q2(x))

        a_log_std = self.dec_d(x).clamp(self.log_std_min, self.log_std_max)
        log_prob = -(noise.pow(2) / 2 + a_log_std + self.constant_log_sqrt_2pi)
        return q, log_prob.sum(1)

    def get__q12__log_prob2(self, state, a_noise):
        x = self.enc_s(state)
        x = self.net(x)

        q1 = self.dec_q1(x)
        q2 = self.dec_q2(x)

        a_avg = self.dec_a(x)
        a_log_std = self.dec_d(x).clamp(self.log_std_min, self.log_std_max)
        a_std = torch.exp(a_log_std)

        log_prob = -(((a_avg - a_noise) / a_std).pow(2) / 2 + a_log_std + self.constant_log_sqrt_2pi)
        return q1, q2, log_prob.sum(1)


class AgentInterOffPPO:
    def __init__(self, state_dim, action_dim, net_dim):
        self.learning_rate = 1e-4  # learning rate of actor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''network'''
        self.act = InterGAE(state_dim, action_dim, net_dim).to(self.device)
        self.act.train()
        self.act_optimizer = torch.optim.Adam([
            {'params': self.act.enc_s.parameters(), 'lr': self.learning_rate},  # more stable
            {'params': self.act.net.parameters(), 'lr': self.learning_rate},
            {'params': self.act.dec_a.parameters(), },
            {'params': self.act.dec_d.parameters(), },
            {'params': self.act.dec_q1.parameters(), },
            {'params': self.act.dec_q2.parameters(), 'lr': self.learning_rate},
        ], lr=self.learning_rate * 1.25)  # todo same lr

        self.criterion = nn.SmoothL1Loss()
        self.action_dim = action_dim

        '''extension: reliable lambda for auto-learning-rate'''
        self.avg_loss_c = (-np.log(0.5)) ** 0.5

    def update_buffer(self, env, buffer, max_step, reward_scale, gamma):
        rewards = list()
        steps = list()

        # a_std = np.exp(self.act.a_log_std.cpu().data.numpy()[0])

        step_counter = 0
        max_memo = buffer.max_len - max_step
        while step_counter < max_memo:
            reward_sum = 0
            step_sum = 0

            state = env.reset()
            for step_sum in range(max_step):
                states = torch.tensor((state,), dtype=torch.float32, device=self.device)
                a_avg, a_std = [t.cpu().data.numpy()[0]
                                for t in self.act.get__a_avg_std(states)]  # todo target
                noise = rd.randn(self.action_dim)  # pure_noise
                action = a_avg + noise * a_std

                next_state, reward, done, _ = env.step(np.tanh(action))
                reward_sum += reward

                mask = 0.0 if done else gamma

                reward_ = reward * reward_scale
                buffer.append_memo((reward_, mask, state, action, noise))

                if done:
                    break

                state = next_state

            rewards.append(reward_sum)
            steps.append(step_sum)

            step_counter += step_sum
        return rewards, steps

    def update_policy(self, buffer, _max_step, batch_size, repeat_times):
        buffer.update_pointer_before_sample()

        self.act.train()
        clip = 0.25  # ratio.clamp(1 - clip, 1 + clip)
        lambda_adv = 0.98  # why 0.98? cannot use 0.99
        lambda_entropy = 0.01  # could be 0.02
        # repeat_times = 8 could be 2**3 ~ 2**5

        _a_log_std = critic_loss = None  # just for print

        '''the batch for training'''
        max_memo = buffer.now_len
        all_reward, all_mask, all_state, all_action, all_noise = buffer.all_sample(self.device)
        buffer.empty_memories_before_explore()  # todo notice! online-policy empties buffer

        '''compute all__new_q, all_log_prob'''
        b_size = 2 ** 10
        with torch.no_grad():
            all__new_q = list()
            all_log_prob = list()
            for i in range(0, all_state.size()[0], b_size):
                new_v, log_prob = self.act.get__q__log_prob1(
                    all_state[i:i + b_size], all_noise[i:i + b_size])  # todo target
                all__new_q.append(new_v)
                all_log_prob.append(log_prob)

            all__new_q = torch.cat(all__new_q, dim=0)
            all_log_prob = torch.cat(all_log_prob, dim=0)

        '''compute old_v (old policy value), adv_v (advantage value) 
        refer: GAE. ICLR 2016. Generalization Advantage Estimate. 
        paper: https://arxiv.org/pdf/1506.02438.pdf'''
        all__old_q = torch.empty(max_memo, dtype=torch.float32, device=self.device)  # old policy value
        all__adv_q = torch.empty(max_memo, dtype=torch.float32, device=self.device)  # advantage value
        all__delta = torch.empty(max_memo, dtype=torch.float32, device=self.device)
        prev_old_q = 0  # old q value
        prev_new_q = 0  # new q value
        prev_adv_q = 0  # advantage q value
        for i in range(max_memo - 1, -1, -1):
            all__delta[i] = all_reward[i] + all_mask[i] * prev_new_q - all__new_q[i]
            all__old_q[i] = all_reward[i] + all_mask[i] * prev_old_q
            all__adv_q[i] = all__delta[i] + all_mask[i] * prev_adv_q * lambda_adv
            prev_old_q = all__old_q[i]
            prev_new_q = all__new_q[i]
            prev_adv_q = all__adv_q[i]

        '''mini batch sample'''
        all__old_q = all__old_q.unsqueeze(1)
        sample_times = int(repeat_times * max_memo / batch_size)
        for _ in range(sample_times):
            '''random sample'''
            indices = rd.randint(max_memo, size=batch_size)

            state = all_state[indices]
            adv_q = all__adv_q[indices]
            old_q = all__old_q[indices]
            action = all_action[indices]
            old_log_prob = all_log_prob[indices]

            new_q1, new_q2, new_log_prob = self.act.get__q12__log_prob2(state, action)

            '''critic_loss'''
            critic_loss = self.criterion(new_q1, old_q) + self.criterion(new_q2, old_q)
            # auto reliable lambda
            self.avg_loss_c = 0.995 * self.avg_loss_c + 0.005 * critic_loss.item() / 2  # soft update, twin critics
            lamb = np.exp(-self.avg_loss_c ** 2)

            '''actor_loss'''
            # surrogate objective of TRPO
            ratio = torch.exp(new_log_prob - old_log_prob)
            surrogate_obj0 = adv_q * ratio
            surrogate_obj1 = adv_q * ratio.clamp(1 - clip, 1 + clip)
            surrogate_obj = -torch.min(surrogate_obj0, surrogate_obj1).mean()
            loss_entropy = (torch.exp(new_log_prob) * new_log_prob).mean()  # KL divergence

            actor_loss = surrogate_obj + loss_entropy * lambda_entropy

            united_loss = critic_loss + actor_loss * lamb
            self.act_optimizer.zero_grad()
            united_loss.backward()
            self.act_optimizer.step()

        self.act.eval()
        # return actor_loss.item(), critic_loss.item()
        return log_prob.mean().item(), critic_loss.item()

    def save_or_load_model(self, cwd, if_save):  # 2020-05-20
        act_save_path = '{}/actor.pth'.format(cwd)

        def load_torch_file(network, save_path):
            network_dict = torch.load(save_path, map_location=lambda storage, loc: storage)
            network.load_state_dict(network_dict)

        if if_save:
            torch.save(self.act.state_dict(), act_save_path)
            # print("Saved act and cri:", mod_dir)
        elif os.path.exists(act_save_path):
            load_torch_file(self.act, act_save_path)
        else:
            print("FileNotFound when load_model: {}".format(cwd))


def run_continuous_action_off_ppo(gpu_id=None):
    args = Arguments()
    args.rl_agent = AgentInterOffPPO
    args.gpu_id = gpu_id
    args.if_break_early = False
    args.if_remove_history = True

    args.random_seed += 1264
    # args.show_gap = 2 ** 6

    # args.env_name = "Pendulum-v0"  # It is easy to reach target score -200.0 (-100 is harder)
    # args.break_step = int(5e5 * 8)  # 1e4 means the average total training step of InterSAC to reach target_reward
    # args.reward_scale = 2 ** -2  # (-1800) -1000 ~ -200 (-50)
    # args.max_memo = 2 ** 11
    # args.batch_size = 2 ** 8
    # args.net_dim = 2 ** 7
    # args.repeat_times = 2 ** 4  # 4
    # args.init_for_training()
    # train_agent(**vars(args))
    # exit()

    args.env_name = "BipedalWalker-v3"
    args.break_step = int(4e6 * 4)
    args.reward_scale = 2 ** -1
    args.net_dim = 2 ** 8
    args.max_memo = 2 ** 11  # 12
    args.batch_size = 2 ** 9
    args.repeat_times = 2 ** 3
    args.init_for_training()
    train_agent_mp(args)  # train_agent(**vars(args))
    exit()

    # args.env_name = "LunarLanderContinuous-v2"
    # args.break_step = int(1e5 * 8)
    # args.reward_scale = 2 ** -3
    # args.net_dim = 2 ** 8
    # args.max_memo = 2 ** 11  # 12
    # args.batch_size = 2 ** 9
    # args.repeat_times = 2 ** 3  # 4
    # args.init_for_training()
    # train_agent_mp(args)  # train_agent(**vars(args))
    # exit()

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)
    args.env_name = "MinitaurBulletEnv-v0"  # PPO is the best, I don't know why.
    args.break_step = int(5e5 * 8)  # (PPO 3e5) 5e5
    args.reward_scale = 2 ** 2  # (-2) 0 ~ 16 (PPO 34)
    args.net_dim = 2 ** 8
    args.max_memo = 2 ** 11
    args.batch_size = 2 ** 9
    args.repeat_times = 2 ** 4
    args.init_for_training()
    train_agent_mp(args)  # train_agent(**vars(args))
    exit()


run_continuous_action_off_ppo()
