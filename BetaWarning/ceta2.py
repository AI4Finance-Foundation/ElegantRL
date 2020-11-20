from AgentRun import *
from AgentNet import *
from AgentZoo import *


class InterPPO(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()

        self.enc_s = nn.Sequential(
            nn.Linear(state_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, mid_dim), nn.ReLU(),
        )

        self.dec_a = nn.Sequential(
            nn.Linear(mid_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, action_dim),
        )
        self.a_std_log = nn.Parameter(torch.zeros(1, action_dim) - 0.5, requires_grad=True)

        self.dec_q1 = nn.Sequential(
            nn.Linear(mid_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, 1),
        )
        self.dec_q2 = nn.Sequential(
            nn.Linear(mid_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, 1),
        )

        layer_norm(self.enc_s[0], std=1.0)
        layer_norm(self.enc_s[2], std=1.0)
        layer_norm(self.dec_a[-1], std=0.01)
        layer_norm(self.dec_q1[-1], std=0.01)
        layer_norm(self.dec_q2[-1], std=0.01)

        self.constant_log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))

    def forward(self, s):
        s_ = self.enc_s(s)
        a_avg = self.dec_a(s_)
        return a_avg.tanh()

    def get__a_avg(self, s):
        s_ = self.enc_s(s)
        a_avg = self.dec_a(s_)
        return a_avg


    def get__q__log_prob(self, state, noise):
        s_ = self.enc_s(state)
        q = torch.min(self.dec_q1(s_), self.dec_q2(s_))

        log_prob = -(noise.pow(2) / 2 + self.a_std_log).sum(1)
        return q, log_prob

    def get__q1_q2__log_prob(self, state, action):
        s_ = self.enc_s(state)

        q1 = self.dec_q1(s_)
        q2 = self.dec_q2(s_)

        a_avg = self.dec_a(s_)  # todo fix bug
        a_log_std = self.a_std_log.expand_as(a_avg)
        a_std = a_log_std.exp()
        log_prob = -(((a_avg - action) / a_std).pow(2) / 2 + self.a_std_log).sum(1)
        return q1, q2, log_prob


class AgentInterOffPPO:
    def __init__(self, state_dim, action_dim, net_dim):
        self.learning_rate = 1e-4  # learning rate of actor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''network'''
        self.act = InterPPO(state_dim, action_dim, net_dim).to(self.device)
        self.act.train()
        self.act_optimizer = torch.optim.Adam([
            {'params': self.act.enc_s.parameters(), 'lr': self.learning_rate},  # more stable
            # {'params': self.act.net.parameters(), 'lr': self.learning_rate},
            {'params': self.act.dec_a.parameters(), },
            # {'params': self.act.dec_d.parameters(), },
            {'params': self.act.a_std_log, },
            {'params': self.act.dec_q1.parameters(), },
            {'params': self.act.dec_q2.parameters(), },
        ], lr=self.learning_rate * 1.5)

        self.criterion = nn.SmoothL1Loss()

    def update_buffer(self, env, buffer, max_step, reward_scale, gamma):
        rewards = list()
        steps = list()

        a_std = np.exp(self.act.a_std_log.cpu().data.numpy()[0])
        noise_dim = a_std.shape[0]
        # assert noise_std.shape = (action_dim, )

        step_counter = 0
        max_memo = buffer.max_len - max_step
        while step_counter < max_memo:
            reward_sum = 0
            step_sum = 0

            state = env.reset()
            for step_sum in range(max_step):
                states = torch.tensor((state,), dtype=torch.float32, device=self.device)
                a_avg = self.act.get__a_avg(states).cpu().data.numpy()[0]  # todo fix bug
                noise = rd.randn(noise_dim)
                action = a_avg + noise * a_std  # todo pure_noise

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

        _actor_loss = critic_loss = None  # just for print

        '''the batch for training'''
        max_memo = buffer.now_len
        all_reward, all_mask, all_state, all_action, all_noise = buffer.all_sample(self.device)

        b_size = 2 ** 10
        with torch.no_grad():
            all__new_v = list()
            all_log_prob = list()
            for i in range(0, all_state.size()[0], b_size):
                new_v, log_prob = self.act.get__q__log_prob(
                    all_state[i:i + b_size], all_noise[i:i + b_size])
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

            new_value1, new_value2, new_log_prob = self.act.get__q1_q2__log_prob(state, action)

            '''critic_loss'''
            critic_loss = self.criterion(new_value1, old_value) + self.criterion(new_value2, old_value)

            '''actor_loss'''
            # surrogate objective of TRPO
            ratio = torch.exp(new_log_prob - old_log_prob)
            surrogate_obj0 = advantage * ratio
            surrogate_obj1 = advantage * ratio.clamp(1 - clip, 1 + clip)
            surrogate_obj = -torch.min(surrogate_obj0, surrogate_obj1).mean()
            # policy entropy
            loss_entropy = (torch.exp(new_log_prob) * new_log_prob).mean()

            actor_loss = surrogate_obj + loss_entropy * lambda_entropy

            '''united_loss'''
            united_loss = critic_loss / all_old_value_std + actor_loss
            self.act_optimizer.zero_grad()
            united_loss.backward()
            self.act_optimizer.step()

        self.act.eval()
        buffer.empty_memories_before_explore()
        # return actor_loss.item(), critic_loss.item()
        return self.act.a_std_log.mean().item(), critic_loss.item()  # todo

    def save_or_load_model(self, cwd, if_save):  # 2020-05-20
        act_save_path = '{}/actor.pth'.format(cwd)
        # cri_save_path = '{}/critic.pth'.format(cwd)
        # has_cri = 'cri' in dir(self)

        def load_torch_file(network, save_path):
            network_dict = torch.load(save_path, map_location=lambda storage, loc: storage)
            network.load_state_dict(network_dict)

        if if_save:
            torch.save(self.act.state_dict(), act_save_path)
            # torch.save(self.cri.state_dict(), cri_save_path) if has_cri else None
            # print("Saved act and cri:", mod_dir)
        elif os.path.exists(act_save_path):
            load_torch_file(self.act, act_save_path)
            # load_torch_file(self.cri, cri_save_path) if has_cri else None
        else:
            print("FileNotFound when load_model: {}".format(cwd))


def run_continuous_action_off_ppo(gpu_id=None):
    args = Arguments()
    args.rl_agent = AgentInterOffPPO
    args.gpu_id = gpu_id
    args.if_break_early = False
    args.if_remove_history = True

    args.random_seed += 126
    args.eval_times1 = 3
    args.eval_times2 = 9
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
    args.net_dim = 2 ** 7
    args.max_memo = 2 ** 11  # 12
    args.batch_size = 2 ** 9
    args.repeat_times = 2 ** 3
    args.init_for_training()
    # train_agent(**vars(args))
    train_agent_mp(args)  # train_agent(**vars(args))
    exit()

    args.env_name = "LunarLanderContinuous-v2"
    args.break_step = int(1e5 * 8)
    args.reward_scale = 2 ** -3
    args.net_dim = 2 ** 8
    args.max_memo = 2 ** 11  # 12
    args.batch_size = 2 ** 9
    args.repeat_times = 2 ** 3  # 4
    args.init_for_training()
    train_agent_mp(args)  # train_agent(**vars(args))
    exit()

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
    # train_agent_mp(args)  # train_agent(**vars(args))
    train_agent(**vars(args))
    exit()


run_continuous_action_off_ppo()
