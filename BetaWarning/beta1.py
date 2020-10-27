from AgentRun import *
from AgentNet import *
from AgentZoo import *



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
            a_std = self.act.net__std_log.exp()

            all__new_v = list()
            all_log_prob = list()

            for i in range(0, all_state.size()[0], b_size):
                all__new_v.append(self.cri(all_state[i:i + b_size]))

                a_delta = (all_noise[i:i + b_size] / (1.41421 * a_std)).pow(2)
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
            a_std = a_log_std.exp()
            a_delta = ((a_mean - action) / (1.41421 * a_std)).pow(2)
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

    def update_policy_imitate(self, buffer, act_target, batch_size, repeat_times):
        self.act.train()
        self.cri.train()
        # clip = 0.25  # ratio.clamp(1 - clip, 1 + clip)
        # lambda_adv = 0.98  # why 0.98? cannot use 0.99
        # lambda_entropy = 0.01  # could be 0.02
        # # repeat_times = 8 could be 2**3 ~ 2**5

        actor_term = critic_loss = None  # just for print

        '''the batch for training'''
        max_memo = len(buffer)
        all_batch = buffer.sample_all()
        all_reward, all_mask, all_state, all_action, all_log_prob = [
            torch.tensor(ary, dtype=torch.float32, device=self.device)
            for ary in (all_batch.reward, all_batch.mask, all_batch.state, all_batch.action, all_batch.log_prob,)
        ]

        # # all__new_v = self.cri(all_state).detach_()  # all new value
        # with torch.no_grad():
        #     b_size = 512
        #     all__new_v = torch.cat(
        #         [self.cri(all_state[i:i + b_size])
        #          for i in range(0, all_state.size()[0], b_size)], dim=0)

        '''compute old_v (old policy value), adv_v (advantage value) 
        refer: GAE. ICLR 2016. Generalization Advantage Estimate. 
        https://arxiv.org/pdf/1506.02438.pdf'''
        # all__delta = torch.empty(max_memo, dtype=torch.float32, device=self.device)
        all__old_v = torch.empty(max_memo, dtype=torch.float32, device=self.device)  # old policy value
        # all__adv_v = torch.empty(max_memo, dtype=torch.float32, device=self.device)  # advantage value

        prev_old_v = 0  # old q value
        # prev_new_v = 0  # new q value
        # prev_adv_v = 0  # advantage q value
        for i in range(max_memo - 1, -1, -1):
            # all__delta[i] = all_reward[i] + all_mask[i] * prev_new_v - all__new_v[i]
            all__old_v[i] = all_reward[i] + all_mask[i] * prev_old_v
            # all__adv_v[i] = all__delta[i] + all_mask[i] * prev_adv_v * lambda_adv

            prev_old_v = all__old_v[i]
            # prev_new_v = all__new_v[i]
            # prev_adv_v = all__adv_v[i]

        # all__adv_v = (all__adv_v - all__adv_v.mean()) / (all__adv_v.std() + 1e-5)  # advantage_norm:

        '''mini batch sample'''
        sample_times = int(repeat_times * max_memo / batch_size)
        for _ in range(sample_times):
            '''random sample'''
            # indices = rd.choice(max_memo, batch_size, replace=True)  # False)
            indices = rd.randint(max_memo, size=batch_size)

            state = all_state[indices]
            # action = all_action[indices]
            # advantage = all__adv_v[indices]
            old_value = all__old_v[indices].unsqueeze(1)
            # old_log_prob = all_log_prob[indices]

            """Adaptive KL Penalty Coefficient
            loss_KLPEN = surrogate_obj + value_obj * lambda_value + entropy_obj * lambda_entropy
            loss_KLPEN = (value_obj * lambda_value) + (surrogate_obj + entropy_obj * lambda_entropy)
            loss_KLPEN = (critic_loss) + (actor_loss)
            """

            '''critic_loss'''
            # new_log_prob = self.act.compute__log_prob(state, action)
            new_value = self.cri(state)

            # critic_loss = (self.criterion(new_value, old_value)).mean() / (old_value.std() + 1e-5)
            critic_loss = (self.criterion(new_value, old_value)) / (old_value.std() + 1e-5)
            self.cri_optimizer.zero_grad()
            critic_loss.backward()
            self.cri_optimizer.step()

            '''actor_term'''
            a_train = self.act(state)
            with torch.no_grad():
                a_target = act_target(state)
            actor_term = self.criterion(a_train, a_target)
            self.act_optimizer.zero_grad()
            actor_term.backward()
            self.act_optimizer.step()

            # '''actor_loss'''
            # # surrogate objective of TRPO
            # ratio = torch.exp(new_log_prob - old_log_prob)
            # surrogate_obj0 = advantage * ratio
            # surrogate_obj1 = advantage * ratio.clamp(1 - clip, 1 + clip)
            # # surrogate_obj = -torch.mean(torch.min(surrogate_obj0, surrogate_obj1)) # todo wait check
            # surrogate_obj = -torch.min(surrogate_obj0, surrogate_obj1).mean()
            # # policy entropy
            # loss_entropy = (torch.exp(new_log_prob) * new_log_prob).mean()  # todo wait check
            #
            # # actor_loss = (surrogate_obj + loss_entropy * lambda_entropy).mean()
            # actor_loss = surrogate_obj + loss_entropy * lambda_entropy  # todo wait check
            # self.act_optimizer.zero_grad()
            # actor_loss.backward()
            # self.act_optimizer.step()

        self.act.eval()
        self.cri.eval()
        return actor_term.item(), critic_loss.item()

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
    rl_agent = AgentOffPPO
    args = Arguments(rl_agent, gpu_id)
    args.if_break_early = False
    args.if_remove_history = True

    args.random_seed += 2

    args.net_dim = 2 ** 8
    args.max_memo = 2 ** 11  # 12
    args.batch_size = 2 ** 9
    args.repeat_times = 2 ** 3  # 4

    args.env_name = "LunarLanderContinuous-v2"
    args.break_step = int(1e5 * 8)
    args.reward_scale = 2 ** -1
    args.init_for_training()
    train_agent(**vars(args))
    exit()

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


run_continuous_action()
