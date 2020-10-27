from AgentRun import *
from AgentNet import *
from AgentZoo import *

"""OffPPO
beta0 OffPPO, reset memo, cancel con_pi

ceta0 OffPPO, reset memo, cancel con_pi, Minitaur
ceta1 ceta0, SmallOffBuffer, 

ceta2 ceta0, cancel std
"""


class AgentInterSAC(AgentBasicAC):  # Integrated Soft Actor-Critic Methods
    def __init__(self, state_dim, action_dim, net_dim):
        super(AgentBasicAC, self).__init__()
        self.learning_rate = 1e-4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''network'''
        self.act = InterSPG(state_dim, action_dim, net_dim).to(self.device)
        self.act.train()
        # self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate)
        self.act_optimizer = torch.optim.Adam([
            {'params': self.act.enc_s.parameters(), 'lr': 1e-4},
            {'params': self.act.enc_a.parameters(), },
            {'params': self.act.net.parameters(), 'lr': 1e-4},
            {'params': self.act.dec_a.parameters(), },
            {'params': self.act.dec_d.parameters(), },
            {'params': self.act.dec_q1.parameters(), },
            {'params': self.act.dec_q2.parameters(), },
        ], lr=4e-4)

        self.act_target = InterSPG(state_dim, action_dim, net_dim).to(self.device)
        self.act_target.eval()
        self.act_target.load_state_dict(self.act.state_dict())

        self.act_anchor = InterSPG(state_dim, action_dim, net_dim).to(self.device)
        self.act_anchor.eval()
        self.act_anchor.load_state_dict(self.act.state_dict())

        self.criterion = nn.SmoothL1Loss()

        '''training record'''
        self.state = None  # env.reset()
        self.reward_sum = 0.0
        self.step = 0
        self.update_counter = 0

        '''extension: auto-alpha for maximum entropy'''
        self.target_entropy = np.log(action_dim)
        self.log_alpha = torch.tensor((-self.target_entropy,), requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam((self.log_alpha,), lr=self.learning_rate)

        '''extension: reliable lambda for auto-learning-rate'''
        self.avg_loss_c = (-np.log(0.5)) ** 0.5

        '''constant'''
        self.explore_noise = True  # stochastic policy choose noise_std by itself.

    def update_policy(self, buffer, max_step, batch_size, repeat_times):
        self.act.train()

        lamb = 0
        actor_loss = critic_loss = None
        a2_log_std = None

        k = 1.0 + buffer.now_len / buffer.max_len
        batch_size = int(batch_size * k)  # increase batch_size
        train_step = int(max_step * k)  # increase training_step

        alpha = self.log_alpha.exp().detach()  # auto temperature parameter

        update_a = 0
        for update_c in range(1, train_step):
            with torch.no_grad():
                reward, mask, state, action, next_s = buffer.random_sample(batch_size, self.device)

                next_a_noise, next_log_prob = self.act_target.get__a__log_prob(next_s)
                next_q_target = torch.min(*self.act_target.get__q1_q2(next_s, next_a_noise))  # twin critic
                q_target = reward + mask * (next_q_target + next_log_prob * alpha)  # # auto temperature parameter

            '''critic_loss'''
            q1_value, q2_value = self.act.get__q1_q2(state, action)  # CriticTwin
            critic_loss = self.criterion(q1_value, q_target) + self.criterion(q2_value, q_target)

            '''auto reliable lambda'''
            self.avg_loss_c = 0.995 * self.avg_loss_c + 0.005 * critic_loss.item() / 2  # soft update, twin critics
            lamb = np.exp(-self.avg_loss_c ** 2)

            '''stochastic policy'''
            a1_mean, a1_log_std, a_noise, log_prob = self.act.get__a__avg_std_noise_prob(state)  # policy gradient
            log_prob = log_prob.mean()

            '''auto temperature parameter: alpha'''
            alpha_loss = lamb * self.log_alpha * (log_prob - self.target_entropy).detach()  # todo sable
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            with torch.no_grad():
                self.log_alpha[:] = self.log_alpha.clamp(-16, 1)
                alpha = self.log_alpha.exp()  # .detach()

            '''action correction term'''
            with torch.no_grad():
                a2_mean, a2_log_std = self.act_anchor.get__a__std(state)
            actor_term = self.criterion(a1_mean, a2_mean) + self.criterion(a1_log_std, a2_log_std)

            if update_a / update_c > 1 / (2 - lamb):
                united_loss = critic_loss + actor_term * (1 - lamb)
            else:
                update_a += 1  # auto TTUR
                '''actor_loss'''
                q_eval_pg = torch.min(*self.act_target.get__q1_q2(state, a_noise)).mean()  # twin critics
                actor_loss = -(q_eval_pg + log_prob * alpha)  # policy gradient

                united_loss = critic_loss + actor_term * (1 - lamb) + actor_loss * lamb

            self.act_optimizer.zero_grad()
            united_loss.backward()
            self.act_optimizer.step()

            soft_target_update(self.act_target, self.act, tau=2 ** -8)
        soft_target_update(self.act_anchor, self.act, tau=lamb if lamb > 0.1 else 0.0)
        # return actor_loss.item(), critic_loss.item() / 2
        return a2_log_std.mean().item(), critic_loss.item() / 2


class BufferArray2:  # 2020-10-20
    def __init__(self, max_len, state_dim, action_dim, ):
        memo_dim = 1 + 1 + state_dim + action_dim + action_dim
        '''reward, mask, state, action, action_noise'''

        self.max_len = max_len
        self.memories = np.empty((max_len, memo_dim), dtype=np.float32)

        self.next_idx = 0
        self.is_full = False
        self.now_len = 0

        self.state_idx = 1 + 1 + state_dim  # reward_dim==1, done_dim==1
        self.action_idx = self.state_idx + action_dim

    def append_memo(self, memo_tuple):
        # memo_array == (reward, mask, state, action, next_state)
        self.memories[self.next_idx] = np.hstack(memo_tuple)
        self.next_idx = self.next_idx + 1
        if self.next_idx >= self.max_len:
            self.is_full = True
            self.next_idx = 0

    def extend_memo(self, memo_array):  # 2020-07-07
        # assert isinstance(memo_array, np.ndarray)
        size = memo_array.shape[0]
        next_idx = self.next_idx + size
        if next_idx < self.max_len:
            self.memories[self.next_idx:next_idx] = memo_array
        if next_idx >= self.max_len:
            if next_idx > self.max_len:
                self.memories[self.next_idx:self.max_len] = memo_array[:self.max_len - self.next_idx]
            self.is_full = True
            next_idx = next_idx - self.max_len
            self.memories[0:next_idx] = memo_array[-next_idx:]
        else:
            self.memories[self.next_idx:next_idx] = memo_array
        self.next_idx = next_idx

    def init_before_sample(self):
        self.now_len = self.max_len if self.is_full else self.next_idx

    def reset_memories(self):
        self.next_idx = 0
        self.is_full = False
        self.now_len = 0

    def random_sample(self, batch_size, device):
        indices = rd.randint(self.now_len, size=batch_size)
        memory = self.memories[indices]
        if device:
            memory = torch.tensor(memory, device=device)

        '''convert array into torch.tensor'''
        tensors = (
            memory[:, 0:1],  # rewards
            memory[:, 1:2],  # masks, mask == (1-float(done)) * gamma
            memory[:, 2:self.state_idx],  # states
            memory[:, self.state_idx:self.action_idx],  # actions
            memory[:, self.action_idx:],  # action_noise
        )
        return tensors

    def print_state_norm(self, neg_avg=None, div_std=None):  # non-essential
        max_sample_size = 2 ** 14
        if self.now_len > max_sample_size:
            indices = rd.randint(self.now_len, size=min(self.now_len, max_sample_size))
            memory_state = self.memories[indices, 2:self.state_idx]
        else:
            memory_state = self.memories[:, 2:self.state_idx]
        print_norm(memory_state, neg_avg, div_std)


class ActorPPO(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()

        self.net__mean = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                       nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                       nn.Linear(mid_dim, action_dim), )
        self.net__std_log = nn.Parameter(torch.zeros(1, action_dim) - 0.5, requires_grad=True)
        self.constant_log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))

        layer_norm(self.net__mean[0], std=1.0)
        layer_norm(self.net__mean[2], std=1.0)
        layer_norm(self.net__mean[4], std=0.01)  # output layer for action

    def forward(self, s):
        a_mean = self.net__mean(s)
        return a_mean.tanh()

    def get__a__log_prob(self, state):
        a_mean = self.net__mean(state)
        a_log_std = self.net__std_log.expand_as(a_mean)
        a_std = torch.exp(a_log_std)
        a_noise = torch.normal(a_mean, a_std)

        a_delta = (a_noise - a_mean).pow(2) / (2 * a_std.pow(2))
        log_prob = -(a_delta + a_log_std + self.constant_log_sqrt_2pi)
        log_prob = log_prob.sum(1)
        return a_noise, log_prob

    def compute__log_prob(self, state, a_noise):
        a_mean = self.net__mean(state)
        a_log_std = self.net__std_log.expand_as(a_mean)
        a_std = torch.exp(a_log_std)

        a_delta = (a_noise - a_mean).pow(2) / (2 * a_std.pow(2))
        log_prob = -(a_delta + a_log_std + self.constant_log_sqrt_2pi)
        return log_prob.sum(1)

    def compute__log_prob2(self, a_mean, a_noise):
        a_log_std = self.net__std_log.expand_as(a_mean)
        a_std = torch.exp(a_log_std)

        a_delta = (a_noise - a_mean).pow(2) / (2 * a_std.pow(2))
        log_prob = -(a_delta + a_log_std + self.constant_log_sqrt_2pi)
        return log_prob.sum(1)


def train_agent(
        rl_agent, env_name, gpu_id, cwd,
        net_dim, max_memo, max_step, batch_size, repeat_times, reward_scale, gamma,
        break_step, if_break_early, show_gap, eval_times1, eval_times2, **_kwargs):  # 2020-09-18
    env, state_dim, action_dim, target_reward, if_discrete = build_gym_env(env_name, if_print=False)
    rl_agent = AgentOffPPO
    '''init: agent, buffer, recorder'''
    recorder = Recorder(eval_size1=eval_times1, eval_size2=eval_times2)
    agent = rl_agent(state_dim, action_dim, net_dim)  # training agent
    agent.state = env.reset()

    # if_online_policy = bool(rl_agent.__name__ in {'AgentPPO', 'AgentGAE', 'AgentInterGAE', 'AgentDiscreteGAE'})
    # if if_online_policy:
    #     buffer = BufferTupleOnline(max_memo)
    # else:
    #     buffer = BufferArray(max_memo, state_dim, 1 if if_discrete else action_dim)
    #     with torch.no_grad():  # update replay buffer
    #         rewards, steps = initial_exploration(env, buffer, max_step, if_discrete, reward_scale, gamma, action_dim)
    #     recorder.update__record_explore(steps, rewards, loss_a=0, loss_c=0)
    #
    #     '''pre training and hard update before training loop'''
    #     buffer.init_before_sample()
    #     agent.update_policy(buffer, max_step, batch_size, repeat_times)
    #     agent.act_target.load_state_dict(agent.act.state_dict())
    buffer = BufferArray2(max_memo * 2, state_dim, action_dim)

    '''loop'''
    if_train = True
    while if_train:
        '''update replay buffer by interact with environment'''
        with torch.no_grad():  # speed up running
            rewards, steps = agent.update_buffer(env, buffer, max_step, reward_scale, gamma)

        '''update network parameters by random sampling buffer for gradient descent'''
        buffer.init_before_sample()
        loss_a, loss_c = agent.update_policy(buffer, max_step, batch_size, repeat_times)
        buffer.reset_memories()

        '''saves the agent with max reward'''
        with torch.no_grad():  # for saving the GPU buffer
            recorder.update__record_explore(steps, rewards, loss_a, loss_c)

            if_save = recorder.update__record_evaluate(env, agent.act, max_step, agent.device, if_discrete)
            recorder.save_act(cwd, agent.act, gpu_id) if if_save else None
            recorder.save_npy__plot_png(cwd)

            if_solve = recorder.check_is_solved(target_reward, gpu_id, show_gap)

        '''break loop rules'''
        if_train = not ((if_break_early and if_solve)
                        or recorder.total_step > break_step
                        or os.path.exists(f'{cwd}/stop'))

    recorder.save_npy__plot_png(cwd)
    buffer.print_state_norm(env.neg_state_avg, env.div_state_std)


class AgentOffPPO:
    def __init__(self, state_dim, action_dim, net_dim):
        self.learning_rate = 1e-4  # learning rate of actor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''network'''
        self.act = ActorPPO(state_dim, action_dim, net_dim).to(self.device)
        self.act.train()
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate, )  # betas=(0.5, 0.99))

        self.cri = CriticAdv(state_dim, net_dim).to(self.device)
        self.cri.train()
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate, )  # betas=(0.5, 0.99))

        self.criterion = nn.SmoothL1Loss()

    def update_buffer(self, env, buffer, max_step, reward_scale, gamma):
        rewards = list()
        steps = list()

        noise_std = np.exp(self.act.net__std_log.cpu().data.numpy()[0])
        # assert noise_std.shape = (action_dim, )

        step_counter = 0
        max_memo = buffer.max_len // 2  # todo notice
        while step_counter < max_memo:
            reward_sum = 0
            step_sum = 0

            state = env.reset()
            for step_sum in range(max_step):
                states = torch.tensor((state,), dtype=torch.float32, device=self.device)

                a_mean = self.act(states).cpu().data.numpy()[0]
                noise = rd.normal(scale=noise_std)
                action = a_mean + noise

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
        self.act.train()
        self.cri.train()
        clip = 0.25  # ratio.clamp(1 - clip, 1 + clip)
        lambda_adv = 0.98  # why 0.98? cannot use 0.99
        lambda_entropy = 0.01  # could be 0.02
        # repeat_times = 8 could be 2**3 ~ 2**5

        actor_loss = critic_loss = None  # just for print

        '''the batch for training'''
        max_memo = buffer.now_len
        all_reward, all_mask, all_state, all_action, all_noise = [
            torch.tensor(ary, device=self.device)
            for ary in (buffer.memories[:, 0:1],
                        buffer.memories[:, 1:2],
                        buffer.memories[:, 2:buffer.state_idx],
                        buffer.memories[:, buffer.state_idx:buffer.action_idx],
                        buffer.memories[:, buffer.action_idx:],)
        ]

        b_size = 2 ** 10
        with torch.no_grad():
            a_log_std = self.act.net__std_log
            # a_std = self.act.net__std_log.exp()

            all__new_v = list()
            all_log_prob = list()

            for i in range(0, all_state.size()[0], b_size):
                all__new_v.append(self.cri(all_state[i:i + b_size]))

                # a_delta = (all_noise[i:i + b_size] / (1.41421 * a_std)).pow(2)
                a_delta = (all_noise[i:i + b_size]).pow(2)  # todo cancel std
                log_prob = -(a_delta + a_log_std).sum(1)
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
            advantage = all__adv_v[indices]
            old_value = all__old_v[indices].unsqueeze(1)
            action = all_action[indices]
            old_log_prob = all_log_prob[indices]

            '''critic_loss'''
            # new_log_prob = self.act.compute__log_prob(state, action)
            a_mean = self.act(state)
            a_log_std = self.act.net__std_log.expand_as(a_mean)
            # a_std = a_log_std.exp()
            # a_delta = ((a_mean - action) / (1.41421 * a_std)).pow(2)
            a_delta = (a_mean - action).pow(2)  # todo cancel std
            new_log_prob = -(a_delta + a_log_std).sum(1)

            new_value = self.cri(state)

            critic_loss = (self.criterion(new_value, old_value)) / (old_value.std() + 1e-5)
            self.cri_optimizer.zero_grad()
            critic_loss.backward()
            self.cri_optimizer.step()

            '''actor_loss'''
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
        # return actor_loss.item(), critic_loss.item()# todo
        return self.act.net__std_log.mean().item(), critic_loss.item()  # todo

    def select_action(self, state):  # CPU array to GPU tensor to CPU array
        states = torch.tensor((state,), dtype=torch.float32, device=self.device)

        actions = self.act(states)
        action = actions.cpu().data.numpy()[0]
        return action  # not tanh()

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


def run_continuous_action(gpu_id=None):
    rl_agent = AgentInterSAC
    args = Arguments(rl_agent, gpu_id)
    args.if_break_early = False
    args.if_remove_history = True

    args.random_seed += 2

    # args.env_name = "LunarLanderContinuous-v2"
    # args.break_step = int(5e4 * 8)  # (2e4) 5e4
    # args.reward_scale = 2 ** -3  # (-800) -200 ~ 200 (302)
    # args.init_for_training(8)
    # train_agent_mp(args)  # train_agent_mix(**vars(args))
    # exit()
    #
    # args.env_name = "BipedalWalker-v3"
    # args.break_step = int(2e5 * 8)  # (1e5) 2e5
    # args.reward_scale = 2 ** -1  # (-200) -150 ~ 300 (342)
    # args.init_for_training(4)
    # train_agent_mp(args)  # train_agent_mix(**vars(args))
    # exit()
    #
    # import pybullet_envs  # for python-bullet-gym
    # dir(pybullet_envs)
    # args.env_name = "AntBulletEnv-v0"
    # args.break_step = int(1e6 * 8)  # (8e5) 10e5
    # args.reward_scale = 2 ** -3  # (-50) 0 ~ 2500 (3340)
    # args.max_step = 2 ** 10
    # args.batch_size = 2 ** 8
    # args.max_memo = 2 ** 20
    # args.eva_size = 2 ** 3  # for Recorder
    # args.show_gap = 2 ** 8  # for Recorder
    # args.init_for_training(8)
    # train_agent_mp(args)  # train_agent(**vars(args))
    # exit()

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)
    args.env_name = "MinitaurBulletEnv-v0"
    args.break_step = int(4e6 * 4)  # (2e6) 4e6
    args.reward_scale = 2 ** 5  # (-2) 0 ~ 16 (20)
    args.batch_size = 2 ** 8
    args.repeat_times = 2 ** 0
    args.max_memo = 2 ** 20
    args.net_dim = 2 ** 8
    args.eval_times2 = 2 ** 5  # for Recorder
    args.show_gap = 2 ** 9  # for Recorder
    args.init_for_training()
    train_agent_mp(args)  # train_agent(**vars(args))
    exit()


def run_continuous_action_off_ppo(gpu_id=None):
    rl_agent = AgentOffPPO
    args = Arguments(rl_agent, gpu_id)
    args.if_break_early = False
    args.if_remove_history = True

    args.random_seed += 2

    args.net_dim = 2 ** 8
    args.max_memo = 2 ** 11  # 12
    args.batch_size = 2 ** 9
    args.repeat_times = 2 ** 3  # 4

    # args.env_name = "LunarLanderContinuous-v2"
    # args.break_step = int(1e5 * 8)
    # args.reward_scale = 2 ** -1
    # args.init_for_training()
    # train_agent(**vars(args))
    # exit()

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)
    args.env_name = "MinitaurBulletEnv-v0"  # PPO is the best, I don't know why.
    args.break_step = int(5e5 * 8)  # (PPO 3e5) 5e5
    args.reward_scale = 2 ** 4  # (-2) 0 ~ 16 (PPO 34)
    args.net_dim = 2 ** 8
    args.max_memo = 2 ** 11  # todo
    args.batch_size = 2 ** 9
    args.repeat_times = 2 ** 4
    args.init_for_training()
    train_agent(**vars(args))
    exit()


run_continuous_action_off_ppo()
