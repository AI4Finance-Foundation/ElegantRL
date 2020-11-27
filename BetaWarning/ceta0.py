from AgentRun import *
from AgentNet import *
from AgentZoo import *


class ActorSAC(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim, use_dn):
        super().__init__()

        if use_dn:  # use DenseNet (DenseNet has both shallow and deep linear layer)
            nn_dense_net = DenseNet(mid_dim)
            self.net__mid = nn.Sequential(
                nn.Linear(state_dim, mid_dim), nn.ReLU(),
                nn_dense_net,
            )
            lay_dim = nn_dense_net.out_dim
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

    def forward(self, state):  # in fact, noise_std is a boolean
        x = self.net__mid(state)
        a_mean = self.net__mean(x)  # NOTICE! it is a_mean without .tanh()
        return a_mean.tanh()

    def get__noise_action(self, s):
        x = self.net__mid(s)
        a_mean = self.net__mean(x)  # NOTICE! it is a_mean without .tanh()

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
        log_prob_noise = a_delta + a_std_log + self.constant_log_sqrt_2pi

        # same as below:
        # from torch.distributions.normal import Normal
        # log_prob_noise = Normal(a_mean, a_std).log_prob(a_noise)
        # same as below:
        # a_delta = a_noise - a_mean).pow(2) /(2* a_std.pow(2)
        # log_prob_noise = -a_delta - a_std.log() - np.log(np.sqrt(2 * np.pi))

        a_noise_tanh = a_noise.tanh()
        log_prob = log_prob_noise + (-a_noise_tanh.pow(2) + 1.000001).log()

        # same as below:
        # epsilon = 1e-6
        # log_prob = log_prob_noise - (1 - a_noise_tanh.pow(2) + epsilon).log()
        return a_noise_tanh, log_prob.sum(1, keepdim=True)


class CriticTwin(nn.Module):  # TwinSAC <- TD3(TwinDDD) <- DoubleDQN <- Double Q-learning
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()

        def build_critic_network():
            net = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                nn.Linear(mid_dim, 1), )
            layer_norm(net[-1], std=0.01)  # It is no necessary.
            return net

        self.net1 = build_critic_network()
        self.net2 = build_critic_network()

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        q_value = self.net1(x)
        return q_value

    def get__q1_q2(self, state, action):
        x = torch.cat((state, action), dim=1)
        q_value1 = self.net1(x)
        q_value2 = self.net2(x)
        return q_value1, q_value2


class CriticTwinShared(nn.Module):  # 2020-06-18
    def __init__(self, state_dim, action_dim, mid_dim, use_dn):
        super().__init__()

        nn_dense = DenseNet(mid_dim)
        if use_dn:  # use DenseNet (DenseNet has both shallow and deep linear layer)
            self.net__mid = nn.Sequential(
                nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                nn_dense,
            )
            lay_dim = nn_dense.out_dim
        else:  # use a simple network for actor. Deeper network does not mean better performance in RL.
            self.net__mid = nn.Sequential(
                nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                nn.Linear(mid_dim, mid_dim),
            )
            lay_dim = mid_dim

        self.net__q1 = nn.Linear(lay_dim, 1)
        self.net__q2 = nn.Linear(lay_dim, 1)
        layer_norm(self.net__q1, std=0.1)
        layer_norm(self.net__q2, std=0.1)

        '''Not need to use both SpectralNorm and TwinCritic
        I choose TwinCritc instead of SpectralNorm, 
        because SpectralNorm is conflict with soft target update,

        if is_spectral_norm:
            self.net1[1] = nn.utils.spectral_norm(self.dec_q1[1])
            self.net2[1] = nn.utils.spectral_norm(self.dec_q2[1])
        '''

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = self.net_mid(x)
        q_value = self.net__q1(x)
        return q_value

    def get__q1_q2(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = self.net__mid(x)
        q_value1 = self.net__q1(x)
        q_value2 = self.net__q2(x)
        return q_value1, q_value2


class AgentBasicAC:  # DEMO (basic Actor-Critic Methods, it is a DDPG without OU-Process)
    def __init__(self, state_dim, action_dim, net_dim):
        self.learning_rate = 1e-4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''network'''
        actor_dim = net_dim
        self.act = Actor(state_dim, action_dim, actor_dim).to(self.device)
        self.act.train()
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate)

        self.act_target = Actor(state_dim, action_dim, actor_dim).to(self.device)
        self.act_target.eval()
        self.act_target.load_state_dict(self.act.state_dict())

        critic_dim = int(net_dim * 1.25)
        self.cri = Critic(state_dim, action_dim, critic_dim).to(self.device)
        self.cri.train()
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate)

        self.cri_target = Critic(state_dim, action_dim, critic_dim).to(self.device)
        self.cri_target.eval()
        self.cri_target.load_state_dict(self.cri.state_dict())

        self.criterion = nn.SmoothL1Loss()

        '''training record'''
        self.state = None  # env.reset()
        self.reward_sum = 0.0
        self.step = 0
        self.update_counter = 0  # delay update counter

        '''constant'''
        self.explore_noise = 0.05  # standard deviation of explore noise
        self.policy_noise = 0.1  # standard deviation of policy noise
        self.update_freq = 1  # set as 1 or 2 for soft target update

    def update_buffer(self, env, buffer, max_step, reward_scale, gamma):
        explore_noise = self.explore_noise  # standard deviation of explore noise
        self.act.eval()

        rewards = list()
        steps = list()
        for _ in range(max_step):
            '''inactive with environment'''
            action = self.select_actions((self.state,), explore_noise)[0]
            next_state, reward, done, _ = env.step(action)

            self.reward_sum += reward
            self.step += 1

            '''update replay buffer'''
            # reward_ = reward * reward_scale
            # mask = 0.0 if done else gamma
            # buffer.append_memo((reward_, mask, self.state, action, next_state))
            reward_mask = np.array((reward * reward_scale, 0.0 if done else gamma), dtype=np.float32)
            buffer.append_memo((reward_mask, self.state, action, next_state))

            self.state = next_state
            if done:
                rewards.append(self.reward_sum)
                self.reward_sum = 0.0

                steps.append(self.step)
                self.step = 0

                self.state = env.reset()
        return rewards, steps

    def update_policy(self, buffer, max_step, batch_size, repeat_times):
        policy_noise = self.policy_noise  # standard deviation of policy noise
        update_freq = self.update_freq  # delay update frequency, for soft target update
        self.act.train()

        loss_a_sum = 0.0
        loss_c_sum = 0.0

        k = 1.0 + buffer.now_len / buffer.max_len
        batch_size_ = int(batch_size * k)
        update_times = int(max_step * k)

        for i in range(update_times * repeat_times):
            with torch.no_grad():
                reward, mask, state, action, next_state = buffer.random_sample(batch_size_, self.device)

                next_action = self.act_target(next_state, policy_noise)
                q_target = self.cri_target(next_state, next_action)
                q_target = reward + mask * q_target

            '''critic_loss'''
            q_eval = self.cri(state, action)
            critic_loss = self.criterion(q_eval, q_target)
            loss_c_sum += critic_loss.item()

            self.cri_optimizer.zero_grad()
            critic_loss.backward()
            self.cri_optimizer.step()

            '''actor_loss'''
            if i % repeat_times == 0:
                action_pg = self.act(state)  # policy gradient
                actor_loss = -self.cri(state, action_pg).mean()  # policy gradient
                loss_a_sum += actor_loss.item()

                self.act_optimizer.zero_grad()
                actor_loss.backward()
                self.act_optimizer.step()

            '''soft target update'''
            self.update_counter += 1
            if self.update_counter >= update_freq:
                self.update_counter = 0
                soft_target_update(self.act_target, self.act)  # soft target update
                soft_target_update(self.cri_target, self.cri)  # soft target update

        loss_a_avg = loss_a_sum / update_times
        loss_c_avg = loss_c_sum / (update_times * repeat_times)
        return loss_a_avg, loss_c_avg

    def select_actions(self, states, explore_noise=0.0):  # CPU array to GPU tensor to CPU array
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = self.act(states, explore_noise)  # tensor
        return actions.cpu().data.numpy()  # array

    def save_or_load_model(self, cwd, if_save):  # 2020-07-07
        act_save_path = '{}/actor.pth'.format(cwd)
        cri_save_path = '{}/critic.pth'.format(cwd)
        has_act = 'act' in dir(self)
        has_cri = 'cri' in dir(self)

        def load_torch_file(network, save_path):
            network_dict = torch.load(save_path, map_location=lambda storage, loc: storage)
            network.load_state_dict(network_dict)

        if if_save:
            torch.save(self.act.state_dict(), act_save_path) if has_act else None
            torch.save(self.cri.state_dict(), cri_save_path) if has_cri else None
            # print("Saved act and cri:", mod_dir)
        elif os.path.exists(act_save_path):
            load_torch_file(self.act, act_save_path) if has_act else None
            load_torch_file(self.cri, cri_save_path) if has_cri else None
        else:
            print("FileNotFound when load_model: {}".format(cwd))


class AgentSAC(AgentBaseAC):
    def __init__(self, state_dim, action_dim, net_dim):
        super(AgentBaseAC, self).__init__()
        self.learning_rate = 1e-4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''network'''
        actor_dim = net_dim
        self.act = ActorSAC(state_dim, action_dim, actor_dim, use_dn=False).to(self.device)
        self.act.train()
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate)
        # SAC uses target update network for critic only. Not for actor

        critic_dim = int(net_dim * 1.25)
        self.cri = CriticTwin(state_dim, action_dim, critic_dim).to(self.device)
        self.cri.train()
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate)

        self.cri_target = CriticTwin(state_dim, action_dim, critic_dim).to(self.device)
        self.cri_target.eval()
        self.cri_target.load_state_dict(self.cri.state_dict())

        self.criterion = nn.MSELoss()

        '''training record'''
        self.state = None  # env.reset()
        self.reward_sum = 0.0
        self.step = 0
        self.update_counter = 0

        '''extension: auto-alpha for maximum entropy'''
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optimizer = torch.optim.Adam((self.log_alpha,), lr=self.learning_rate)
        self.target_entropy = np.log(action_dim)

        '''constant'''
        self.explore_noise = True  # stochastic policy choose noise_std by itself.
        self.update_freq = 1  # delay update frequency, for soft target update

    def select_action(self, states):
        action = self.act.get__noise_action(states)
        return action

    def update_policy(self, buffer, max_step, batch_size, repeat_times):
        loss_a_sum = 0.0
        loss_c_sum = 0.0

        k = 1.0 + buffer.now_len / buffer.max_len
        batch_size_ = int(batch_size * k)
        update_times = int(max_step * k)

        for i in range(update_times * repeat_times):
            with torch.no_grad():
                reward, mask, state, action, next_s = buffer.random_sample(batch_size_)

                next_a_noise, next_log_prob = self.act.get__a__log_prob(next_s)
                next_q_target = torch.min(*self.cri_target.get__q1_q2(next_s, next_a_noise))  # CriticTwin
                next_q_target = next_q_target + next_log_prob * self.alpha  # SAC, alpha
                q_target = reward + mask * next_q_target
            '''critic_loss'''
            q1_value, q2_value = self.cri.get__q1_q2(state, action)  # CriticTwin
            critic_loss = self.criterion(q1_value, q_target) + self.criterion(q2_value, q_target)
            loss_c_sum += critic_loss.item() * 0.5  # CriticTwin

            self.cri_optimizer.zero_grad()
            critic_loss.backward()
            self.cri_optimizer.step()

            '''actor_loss'''
            if i % repeat_times == 0:
                # stochastic policy
                actions_noise, log_prob = self.act.get__a__log_prob(state)  # policy gradient
                # auto alpha
                alpha_loss = (self.log_alpha * (log_prob - self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

                # policy gradient
                self.alpha = self.log_alpha.exp()
                # q_eval_pg = self.cri(state, actions_noise)  # policy gradient
                q_eval_pg = torch.min(*self.cri.get__q1_q2(state, actions_noise))  # policy gradient, stable but slower
                actor_loss = -(q_eval_pg + log_prob * self.alpha).mean()  # policy gradient
                loss_a_sum += actor_loss.item()

                self.act_optimizer.zero_grad()
                actor_loss.backward()
                self.act_optimizer.step()

            """target update"""
            soft_target_update(self.cri_target, self.cri)  # soft target update

        loss_a_avg = loss_a_sum / update_times
        loss_c_avg = loss_c_sum / (update_times * repeat_times)
        return loss_a_avg, loss_c_avg


class AgentModSAC(AgentBasicAC):
    def __init__(self, state_dim, action_dim, net_dim):
        super(AgentBasicAC, self).__init__()
        use_dn = True  # and use hard target update
        self.learning_rate = 1e-4
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
        self.target_entropy = np.log(action_dim + 1) * 0.5
        self.log_alpha = torch.tensor((-self.target_entropy * np.e,), requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam((self.log_alpha,), lr=self.learning_rate)
        self.trust_rho = TrustRho()

        '''constant'''
        self.explore_noise = True  # stochastic policy choose noise_std by itself.

    def update_policy(self, buffer, max_step, batch_size, repeat_times):
        self.act.train()

        loss_a_sum = 0.0
        loss_c_sum = 0.0

        alpha = self.log_alpha.exp().detach()

        k = 1.0 + buffer.now_len / buffer.max_len
        batch_size_ = int(batch_size * k)
        update_times_c = int(max_step * k)

        update_times_a = 0
        for i in range(1, update_times_c):
            with torch.no_grad():
                reward, mask, state, action, next_s = buffer.random_sample(batch_size_)

                next_a_noise, next_log_prob = self.act_target.get__a__log_prob(next_s)
                next_q_target = torch.min(*self.cri_target.get__q1_q2(next_s, next_a_noise))  # CriticTwin
                q_target = reward + mask * (next_q_target + next_log_prob * alpha)  # policy entropy
            '''critic_loss'''
            q1_value, q2_value = self.cri.get__q1_q2(state, action)  # CriticTwin
            critic_loss = self.criterion(q1_value, q_target) + self.criterion(q2_value, q_target)
            loss_c_tmp = critic_loss.item() * 0.5  # CriticTwin
            loss_c_sum += loss_c_tmp
            rho = self.trust_rho.update_rho(loss_c_tmp)

            self.cri_optimizer.zero_grad()
            critic_loss.backward()
            self.cri_optimizer.step()

            actions_noise, log_prob = self.act.get__a__log_prob(state)  # policy gradient

            '''auto temperature parameter (alpha)'''
            alpha_loss = (self.log_alpha * (log_prob - self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            with torch.no_grad():
                self.log_alpha[:] = self.log_alpha.clamp(-16, 1)
            alpha = self.log_alpha.exp().detach()

            '''actor_loss'''
            if update_times_a / i < rho + 0.5:
                update_times_a += 1

                q_eval_pg = torch.min(*self.cri.get__q1_q2(state, actions_noise))  # policy gradient, stable but slower

                actor_loss = -(q_eval_pg + log_prob * alpha).mean()  # policy gradient
                loss_a_sum += actor_loss.item()

                self.act_optimizer.param_groups[0]['lr'] = self.learning_rate * rho
                self.act_optimizer.zero_grad()
                actor_loss.backward()
                self.act_optimizer.step()

                """target update"""
                soft_target_update(self.act_target, self.act)  # soft target update
            """target update"""
            soft_target_update(self.cri_target, self.cri)  # soft target update

        loss_a_avg = (loss_a_sum / update_times_a) if update_times_a else 0.0
        loss_c_avg = loss_c_sum / update_times_c
        return loss_a_avg, loss_c_avg


def test__train_agent():
    args = Arguments(gpu_id=None)
    args.rl_agent = AgentSAC
    args.random_seed += 12

    # args.env_name = "LunarLanderContinuous-v2"
    # args.break_step = int(5e5 * 8)  # (2e4) 5e5, used time 1500s
    # args.reward_scale = 2 ** -3  # (-800) -200 ~ 200 (302)
    # args.init_for_training()
    # train_agent_mp(args)  # train_agent(**vars(args))
    # exit()

    args.env_name = "BipedalWalker-v3"
    args.break_step = int(2e5 * 8)  # (1e5) 2e5, used time 3500s
    args.reward_scale = 2 ** -1  # (-200) -140 ~ 300 (341)
    args.init_for_training()
    train_agent_mp(args)  # train_agent(**vars(args))
    exit()

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)

    args.env_name = "ReacherBulletEnv-v0"
    args.break_step = int(5e4 * 8)  # (4e4) 5e4
    args.reward_scale = 2 ** 0  # (-37) 0 ~ 18 (29) # todo wait update
    args.init_for_training()
    train_agent_mp(args)  # train_agent(**vars(args))
    exit()


test__train_agent()
