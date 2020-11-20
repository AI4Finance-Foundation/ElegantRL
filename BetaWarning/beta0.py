from AgentRun import *
from AgentNet import *
from AgentZoo import *

"""
CarRacing
beta3 OffPPO frame 3
ID      Step   TargetR |    avgR      stdR |    ExpR  UsedTime  ########
3   1.93e+05    900.00 |  939.68    113.69 |  227.59      9112  ########
beta3 check
beta0 check


InterOffPPO 
ceta2 ceta3 ceta4
beta2 CarRacing
"""


class InterGAE(nn.Module):  # 2020-10-10
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()
        self.log_std_min = -20
        self.log_std_max = 2
        self.constant_log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def idx_dim(i):
            return int(8 * 1.5 ** i)

        # encoder
        self.enc_s = nn.Sequential(
            NnnReshape(*state_dim),  # -> [batch_size, 4, 96, 96]
            nn.Conv2d(state_dim[0], idx_dim(0), 4, 2, bias=True), nn.LeakyReLU(),
            nn.Conv2d(idx_dim(0), idx_dim(1), 3, 2, bias=False), nn.ReLU(),
            nn.Conv2d(idx_dim(1), idx_dim(2), 3, 2, bias=False), nn.ReLU(),
            nn.Conv2d(idx_dim(2), idx_dim(3), 3, 2, bias=True), nn.ReLU(),
            nn.Conv2d(idx_dim(3), idx_dim(4), 3, 1, bias=True), nn.ReLU(),
            nn.Conv2d(idx_dim(4), idx_dim(5), 3, 1, bias=True), nn.ReLU(),
            NnnReshape(-1),
            nn.Linear(idx_dim(5), mid_dim), nn.ReLU(),
            # nn.Linear(state_dim, mid_dim), nn.ReLU(),
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
        layer_norm(self.dec_a[-1], std=0.01)  # output layer for action
        layer_norm(self.dec_d[-1], std=0.01)  # output layer for std_log
        layer_norm(self.dec_q1[-1], std=0.1)  # output layer for q value
        layer_norm(self.dec_q1[-1], std=0.1)  # output layer for q value

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

        # a_noise = a_mean + a_std * torch.randn_like(a_mean, requires_grad=True, device=self.device)
        #
        # a_delta = (a_noise - a_mean).pow(2) / (2 * a_std.pow(2))
        # log_prob = -(a_delta + a_log_std + self.constant_log_sqrt_2pi)

        # noise = torch.randn_like(a_mean, requires_grad=True, device=self.device)
        # a_noise = a_mean + a_std * noise
        # log_prob = -(noise.pow(2) / 2 + a_log_std + self.constant_log_sqrt_2pi)
        return a_mean, a_std

    def get__q__log_prob1(self, state, noise):
        x = self.enc_s(state)
        x = self.net(x)
        q = torch.min(self.dec_q1(x), self.dec_q2(x))

        a_log_std = self.dec_d(x).clamp(self.log_std_min, self.log_std_max)
        log_prob = -(noise.pow(2) / 2 + a_log_std + self.constant_log_sqrt_2pi)
        return q, log_prob.sum(1)

    def compute__log_prob(self, state, a_noise):
        x = self.enc_s(state)
        x = self.net(x)
        a_mean = self.dec_a(x)
        a_log_std = self.dec_d(x).clamp(self.log_std_min, self.log_std_max)
        a_std = torch.exp(a_log_std)

        a_delta = (a_noise - a_mean).pow(2) / (2 * a_std.pow(2))
        log_prob = -(a_delta + a_log_std + self.constant_log_sqrt_2pi)
        log_prob = log_prob.sum(1)
        return log_prob

    def get__q1_q2(self, s):
        x = self.enc_s(s)
        x = self.net(x)
        q1 = self.dec_q1(x)
        q2 = self.dec_q2(x)
        return q1, q2


class AgentInterOffPPO:
    def __init__(self, state_dim, action_dim, net_dim):
        self.learning_rate = 1e-4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''network'''
        self.act = InterGAE(state_dim, action_dim, net_dim).to(self.device)
        self.act.train()
        # self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate, )  # betas=(0.5, 0.99))
        self.act_optimizer = torch.optim.Adam([
            {'params': self.act.enc_s.parameters(), 'lr': self.learning_rate},  # more stable
            {'params': self.act.net.parameters(), 'lr': self.learning_rate},
            {'params': self.act.dec_a.parameters(), },
            {'params': self.act.dec_d.parameters(), },
            {'params': self.act.dec_q1.parameters(), },
            {'params': self.act.dec_q2.parameters(), },
        ], lr=self.learning_rate * 1.5)  # todo 2020-11-18

        self.criterion = nn.SmoothL1Loss()
        self.action_dim = action_dim

    def update_buffer(self, env, buffer, max_step, reward_scale, gamma):
        # buffer.storage_list = list()  # PPO is an online policy RL algorithm.
        rewards = list()
        steps = list()

        step_counter = 0
        max_memo = buffer.max_len - max_step
        while step_counter < max_memo:
            reward_sum = 0
            step_sum = 0

            state = env.reset()
            for step_sum in range(max_step):
                states = torch.tensor((state,), dtype=torch.float32, device=self.device)
                a_avg, a_std = [t.cpu().data.numpy()[0] for t in self.act.get__a_avg_std(states)]
                noise = rd.randn(self.action_dim)
                action = a_avg + a_std * noise

                next_state, reward, done, _ = env.step(np.tanh(action))
                reward_sum += reward

                mask = 0.0 if done else gamma

                reward_ = reward * reward_scale
                # buffer.push(reward_, mask, state, action, noise, )
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
        # self.cri.train()
        clip = 0.25  # ratio.clamp(1 - clip, 1 + clip)
        lambda_adv = 0.98  # why 0.98? cannot seem to use 0.99
        lambda_entropy = 0.01  # could be 0.02
        # repeat_times = 8 could be 2**2 ~ 2**4

        # loss_a_sum = 0.0  # just for print
        loss_c_sum = 0.0  # just for print

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
                    all_state[i:i + b_size], all_noise[i:i + b_size])
                all__new_q.append(new_v)
                all_log_prob.append(log_prob)

            all__new_q = torch.cat(all__new_q, dim=0)
            all_log_prob = torch.cat(all_log_prob, dim=0)

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
            all__delta[i] = all_reward[i] + all_mask[i] * prev_new_v - all__new_q[i]
            all__old_v[i] = all_reward[i] + all_mask[i] * prev_old_v
            all__adv_v[i] = all__delta[i] + all_mask[i] * prev_adv_v * lambda_adv

            prev_old_v = all__old_v[i]
            prev_new_v = all__new_q[i]
            prev_adv_v = all__adv_v[i]

        all__adv_v = (all__adv_v - all__adv_v.mean()) / (all__adv_v.std() + 1e-5)  # advantage_norm:

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
            new_value1, new_value2 = self.act.get__q1_q2(state)  # TwinCritic
            # new_log_prob, new_value1, new_value2 = self.act_target.compute__log_prob(state, action)

            critic_loss = (self.criterion(new_value1, old_value) +
                           self.criterion(new_value2, old_value)) / (old_value.std() * 2 + 1e-5)
            loss_c_sum += critic_loss.item()  # just for print
            # self.cri_optimizer.zero_grad()
            # critic_loss.backward()
            # self.cri_optimizer.step()

            '''actor_loss'''
            # surrogate objective of TRPO
            ratio = (new_log_prob - old_log_prob).exp()
            surrogate_obj0 = advantage * ratio
            surrogate_obj1 = advantage * ratio.clamp(1 - clip, 1 + clip)
            surrogate_obj = -torch.min(surrogate_obj0, surrogate_obj1).mean()
            # policy entropy
            loss_entropy = (new_log_prob.exp() * new_log_prob).mean()

            actor_loss = surrogate_obj + loss_entropy * lambda_entropy
            # loss_a_sum += actor_loss.item()  # just for print

            united_loss = actor_loss + critic_loss
            self.act_optimizer.zero_grad()
            united_loss.backward()
            self.act_optimizer.step()

        # loss_a_avg = loss_a_sum / sample_times
        loss_c_avg = loss_c_sum / sample_times
        return all_log_prob.mean().item(), loss_c_avg

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


class ActorPPO(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()

        def idx_dim(i):
            return int(8 * 1.5 ** i)

        self.net__mean = nn.Sequential(
            NnnReshape(*state_dim),  # -> [batch_size, 4, 96, 96]
            nn.Conv2d(state_dim[0], idx_dim(0), 4, 2, bias=True), nn.LeakyReLU(),
            nn.Conv2d(idx_dim(0), idx_dim(1), 3, 2, bias=False), nn.ReLU(),
            nn.Conv2d(idx_dim(1), idx_dim(2), 3, 2, bias=False), nn.ReLU(),
            nn.Conv2d(idx_dim(2), idx_dim(3), 3, 2, bias=True), nn.ReLU(),
            nn.Conv2d(idx_dim(3), idx_dim(4), 3, 1, bias=True), nn.ReLU(),
            nn.Conv2d(idx_dim(4), idx_dim(5), 3, 1, bias=True), nn.ReLU(),
            NnnReshape(-1),
            nn.Linear(idx_dim(5), mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, action_dim),
        )
        self.net__std_log = nn.Parameter(torch.zeros(1, action_dim) - 0.5, requires_grad=True)
        self.constant_log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))

        # layer_norm(self.net__mean[0], std=1.0)
        # layer_norm(self.net__mean[2], std=1.0)
        layer_norm(self.net__mean[-1], std=0.01)  # output layer for action

    def forward(self, s):
        a_mean = self.net__mean(s)
        return a_mean.tanh()

    def get__a__log_prob(self, state):  # todo plan to cancel
        a_mean = self.net__mean(state)
        a_log_std = self.net__std_log.expand_as(a_mean)
        a_std = torch.exp(a_log_std)
        a_noise = torch.normal(a_mean, a_std)

        a_delta = (a_noise - a_mean).pow(2) / (2 * a_std.pow(2))
        log_prob = -(a_delta + a_log_std + self.constant_log_sqrt_2pi)
        log_prob = log_prob.sum(1)
        return a_noise, log_prob

    def compute__log_prob(self, state, a_noise):  # todo may plan to cancel
        a_mean = self.net__mean(state)
        a_log_std = self.net__std_log.expand_as(a_mean)
        a_std = torch.exp(a_log_std)

        a_delta = (a_noise - a_mean).pow(2) / (2 * a_std.pow(2))
        log_prob = -(a_delta + a_log_std + self.constant_log_sqrt_2pi)
        return log_prob.sum(1)


class CriticAdv(nn.Module):  # 2020-05-05 fix bug
    def __init__(self, state_dim, mid_dim):
        super().__init__()

        def idx_dim(i):
            return int(8 * 1.5 ** i)

        self.net = nn.Sequential(
            NnnReshape(*state_dim),  # -> [batch_size, 4, 96, 96]
            nn.Conv2d(state_dim[0], idx_dim(0), 4, 2, bias=True), nn.LeakyReLU(),
            nn.Conv2d(idx_dim(0), idx_dim(1), 3, 2, bias=False), nn.ReLU(),
            nn.Conv2d(idx_dim(1), idx_dim(2), 3, 2, bias=False), nn.ReLU(),
            nn.Conv2d(idx_dim(2), idx_dim(3), 3, 2, bias=True), nn.ReLU(),
            nn.Conv2d(idx_dim(3), idx_dim(4), 3, 1, bias=True), nn.ReLU(),
            nn.Conv2d(idx_dim(4), idx_dim(5), 3, 1, bias=True), nn.ReLU(),
            NnnReshape(-1),
            nn.Linear(idx_dim(5), mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, 1),
        )

        # layer_norm(self.net[0], std=1.0)
        # layer_norm(self.net[2], std=1.0)
        layer_norm(self.net[-1], std=1.0)  # output layer for action

    def forward(self, s):
        q = self.net(s)
        return q


class AgentOffPPO:
    def __init__(self, state_dim, action_dim, net_dim):
        self.learning_rate = 1e-4  # learning rate of actor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''network'''
        self.act = ActorPPO(state_dim, action_dim, net_dim).to(self.device)
        self.act.train()
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate, )  # betas=(0.5, 0.99))
        # self.act_optimizer = torch.optim.SGD(self.act.parameters(), momentum=0.9, lr=self.learning_rate, )

        self.cri = CriticAdv(state_dim, net_dim).to(self.device)
        self.cri.train()
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate, )  # betas=(0.5, 0.99))
        # self.cri_optimizer = torch.optim.SGD(self.cri.parameters(), momentum=0.9, lr=self.learning_rate, )

        self.criterion = nn.SmoothL1Loss()

    def update_buffer(self, env, buffer, max_step, reward_scale, gamma):
        rewards = list()
        steps = list()

        noise_std = np.exp(self.act.net__std_log.cpu().data.numpy()[0])
        noise_dim = noise_std.shape[0]
        # assert noise_std.shape = (action_dim, )

        step_counter = 0
        max_memo = buffer.max_len - max_step
        while step_counter < max_memo:
            reward_sum = 0
            step_sum = 0

            state = env.reset()
            for step_sum in range(max_step):
                states = torch.tensor((state,), dtype=torch.float32, device=self.device)
                a_mean = self.act.net__mean(states).cpu().data.numpy()[0]  # todo fix bug
                noise = rd.randn(noise_dim)
                action = a_mean + noise * noise_std  # todo pure_noise

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
        self.cri.train()
        clip = 0.25  # ratio.clamp(1 - clip, 1 + clip)
        lambda_adv = 0.98  # why 0.98? cannot use 0.99
        lambda_entropy = 0.01  # could be 0.02
        # repeat_times = 8 could be 2**3 ~ 2**5

        actor_loss = critic_loss = None  # just for print

        '''the batch for training'''
        max_memo = buffer.now_len

        all_reward, all_mask, all_state, all_action, all_noise = buffer.all_sample(self.device)

        b_size = 2 ** 10
        with torch.no_grad():
            a_log_std = self.act.net__std_log

            all__new_v = torch.cat([self.cri(all_state[i:i + b_size])
                                    for i in range(0, all_state.size()[0], b_size)], dim=0)
            all_log_prob = torch.cat([-(all_noise[i:i + b_size].pow(2) / 2 + a_log_std).sum(1)
                                      for i in range(0, all_state.size()[0], b_size)], dim=0)

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
        all__adv_v = (all__adv_v - all__adv_v.mean()) / (all__adv_v.std() + 1e-5)  # todo cancel value_norm
        # Q_value_norm is necessary. Because actor_loss = surrogate_obj + loss_entropy * lambda_entropy.

        '''mini batch sample'''
        all_old_value_std = all__old_v.std() + 1e-5
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

            '''critic_loss'''
            new_value = self.cri(state)
            critic_loss = self.criterion(new_value, old_value)

            self.cri_optimizer.zero_grad()
            (critic_loss / all_old_value_std).backward()
            self.cri_optimizer.step()

            '''actor_loss'''
            a_mean = self.act.net__mean(state)  # todo fix bug
            a_log_std = self.act.net__std_log.expand_as(a_mean)
            a_std = a_log_std.exp()
            new_log_prob = -(((a_mean - action) / a_std).pow(2) / 2 + a_log_std).sum(1)

            # surrogate objective of TRPO
            ratio = torch.exp(new_log_prob - old_log_prob)
            surrogate_obj0 = advantage * ratio
            surrogate_obj1 = advantage * ratio.clamp(1 - clip, 1 + clip)
            surrogate_obj = -torch.min(surrogate_obj0, surrogate_obj1).mean()
            # policy entropy
            loss_entropy = (torch.exp(new_log_prob) * new_log_prob).mean()

            actor_loss = surrogate_obj + loss_entropy * lambda_entropy
            self.act_optimizer.zero_grad()
            actor_loss.backward()
            self.act_optimizer.step()

        self.act.eval()
        self.cri.eval()
        buffer.empty_memories_before_explore()
        # return actor_loss.item(), critic_loss.item()
        return self.act.net__std_log.mean().item(), critic_loss.item()  # todo

    def save_or_load_model(self, cwd, if_save):  # 2020-05-20
        act_save_path = '{}/actor.pth'.format(cwd)
        cri_save_path = '{}/critic.pth'.format(cwd)
        has_cri = 'cri' in dir(self)

        def load_torch_file(network, save_path):
            network_dict = torch.load(save_path, map_location=lambda storage, loc: storage)
            network.load_state_dict(network_dict)

        if if_save:
            torch.save(self.act.state_dict(), act_save_path)
            torch.save(self.cri.state_dict(), cri_save_path) if has_cri else None
            # print("Saved act and cri:", mod_dir)
        elif os.path.exists(act_save_path):
            load_torch_file(self.act, act_save_path)
            load_torch_file(self.cri, cri_save_path) if has_cri else None
        else:
            print("FileNotFound when load_model: {}".format(cwd))


def mp__update_params(args, q_i_eva, q_o_eva):  # update network parameters using replay buffer
    rl_agent = args.rl_agent
    max_memo = args.max_memo
    net_dim = args.net_dim
    max_step = args.max_step
    max_total_step = args.break_step
    batch_size = args.batch_size
    repeat_times = args.repeat_times
    cwd = args.cwd
    env_name = args.env_name
    reward_scale = args.reward_scale
    if_stop = args.if_break_early
    gamma = args.gamma
    del args

    env, state_dim, action_dim, target_reward, if_discrete = build_gym_env(env_name, if_print=False)

    '''build agent'''
    agent = rl_agent(state_dim, action_dim, net_dim)  # training agent
    agent.state = env.reset()

    '''send agent to q_i_eva'''
    from copy import deepcopy
    act_cpu = deepcopy(agent.act).to(torch.device("cpu"))
    act_cpu.eval()
    [setattr(param, 'requires_grad', False) for param in act_cpu.parameters()]
    q_i_eva.put(act_cpu)  # q_i_eva 1.

    '''build replay buffer'''
    total_step = 0
    if_ppo = bool(rl_agent.__name__ in {'AgentOffPPO', 'AgentInterOffPPO'})
    buffer_max_memo = max_memo + max_step if if_ppo else max_memo
    buffer = BufferArrayGPU(buffer_max_memo, state_dim, action_dim, if_ppo)  # experiment replay buffer
    if if_ppo:
        with torch.no_grad():
            reward_avg = get_episode_reward(env, act_cpu, max_step, torch.device("cpu"), if_discrete)
    else:
        '''initial exploration'''
        with torch.no_grad():  # update replay buffer
            rewards, steps = initial_exploration(env, buffer, max_step, if_discrete, reward_scale, gamma, action_dim)
        reward_avg = np.average(rewards)
        step_sum = sum(steps)

        '''pre training and hard update before training loop'''
        buffer.update_pointer_before_sample()
        agent.update_policy(buffer, max_step, batch_size, repeat_times)
        agent.act_target.load_state_dict(agent.act.state_dict())

        q_i_eva.put((act_cpu, reward_avg, step_sum, 0, 0))  # q_i_eva n.
        total_step += step_sum

    '''training loop'''
    if_train = True
    if_solve = False
    while if_train:
        '''update replay buffer by interact with environment'''
        with torch.no_grad():  # speed up running
            rewards, steps = agent.update_buffer(env, buffer, max_step, reward_scale, gamma)
        reward_avg = np.average(rewards) if len(rewards) else reward_avg
        step_sum = sum(steps)
        total_step += step_sum

        '''update network parameters by random sampling buffer for gradient descent'''
        buffer.update_pointer_before_sample()
        loss_a_avg, loss_c_avg = agent.update_policy(buffer, max_step, batch_size, repeat_times)

        '''saves the agent with max reward'''
        act_cpu.load_state_dict(agent.act.state_dict())
        q_i_eva.put((act_cpu, reward_avg, step_sum, loss_a_avg, loss_c_avg))  # q_i_eva n.

        if q_o_eva.qsize() > 0:
            if_solve = q_o_eva.get()  # q_o_eva n.

        '''break loop rules'''
        if_train = not ((if_stop and if_solve)
                        or total_step > max_total_step
                        or os.path.exists(f'{cwd}/stop'))

    env, state_dim, action_dim, target_reward, if_discrete = build_gym_env(env_name, if_print=False)
    buffer.print_state_norm(env.neg_state_avg, env.div_state_std)

    q_i_eva.put('stop')
    while q_i_eva.qsize() > 0:
        time.sleep(1)
    time.sleep(4)
    # print('; quit: params')


def mp_evaluate_agent(args, q_i_eva, q_o_eva):  # evaluate agent and get its total reward of an episode
    env_name = args.env_name
    cwd = args.cwd
    gpu_id = args.gpu_id
    max_step = args.max_memo
    show_gap = args.show_gap
    eval_size1 = args.eval_times1
    eval_size2 = args.eval_times2
    del args

    env, state_dim, action_dim, target_reward, if_discrete = build_gym_env(env_name, if_print=True)

    '''build evaluated only actor'''
    act = q_i_eva.get()  # q_i_eva 1, act == act.to(device_cpu), requires_grad=False

    torch.set_num_threads(4)
    device = torch.device('cpu')
    recorder = Recorder(eval_size1, eval_size2)
    recorder.update__record_evaluate(env, act, max_step, device, if_discrete)

    is_training = True
    with torch.no_grad():  # for saving the GPU buffer
        while is_training:
            is_saved = recorder.update__record_evaluate(env, act, max_step, device, if_discrete)
            recorder.save_act(cwd, act, gpu_id) if is_saved else None

            is_solved = recorder.check_is_solved(target_reward, gpu_id, show_gap, cwd)
            q_o_eva.put(is_solved)  # q_o_eva n.

            '''update actor'''
            while q_i_eva.qsize() == 0:  # wait until q_i_eva has item
                time.sleep(1)
            while q_i_eva.qsize():  # get the latest actor
                q_i_eva_get = q_i_eva.get()  # q_i_eva n.
                if q_i_eva_get == 'stop':
                    is_training = False
                    break
                act, exp_r_avg, exp_s_sum, loss_a_avg, loss_c_avg = q_i_eva_get
                recorder.update__record_explore(exp_s_sum, exp_r_avg, loss_a_avg, loss_c_avg)

    recorder.save_npy__draw_plot(cwd)

    while q_o_eva.qsize() > 0:
        q_o_eva.get()
    while q_i_eva.qsize() > 0:
        q_i_eva.get()
    # print('; quit: evaluate')


def train_agent_mp(args):  # 2020-1111
    import multiprocessing as mp
    q_i_eva = mp.Queue(maxsize=8)  # evaluate I
    q_o_eva = mp.Queue(maxsize=8)  # evaluate O
    process = [mp.Process(target=mp__update_params, args=(args, q_i_eva, q_o_eva)),
               mp.Process(target=mp_evaluate_agent, args=(args, q_i_eva, q_o_eva)), ]
    [p.start() for p in process]
    [p.join() for p in process]
    print('\n')


def run__car_racing(gpu_id=None, random_seed=432):
    print('pixel-level state')

    args = Arguments(rl_agent=AgentOffPPO, gpu_id=gpu_id)
    args.env_name = "CarRacing-v0"
    args.random_seed = 1943 + random_seed
    args.break_step = int(2e6 * 1)
    args.max_memo = 2 ** 12
    args.batch_size = 2 ** 7
    args.repeat_times = 2 ** 4
    args.net_dim = 2 ** 7
    args.max_step = 2 ** 10
    args.eval_times2 = 1
    args.eval_times2 = 2
    args.reward_scale = 2 ** -3
    args.show_gap = 2 ** 8  # for Recorder
    args.init_for_training()
    # train_agent(**vars(args))
    train_agent_mp(args)  # train_agent(**vars(args))


run__car_racing(random_seed=6353)
