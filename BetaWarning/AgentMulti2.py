from AgentRun import *
from AgentNet import *
from AgentZoo import *
from multiwalker_base import *

"""MultiWalker (don't delete 2020-09-10)
Test ISAC
multi reward
MA-ISAC
"""


class MInterSPG(nn.Module):  # class AgentIntelAC for SAC (SPG means stochastic policy gradient)
    def __init__(self, state_dims, action_dims, mid_dim, reward_dim):
        super().__init__()
        self.reward_dim = reward_dim

        self.log_std_min = -20
        self.log_std_max = 2
        self.constant_log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''encoders'''
        self.enc_s_l = [
            nn.Sequential(nn.Linear(inp_dim, mid_dim), nn.ReLU(),
                          nn.Linear(mid_dim, mid_dim), ).to(self.device)
            for inp_dim in state_dims
        ]  # encoders of state
        self.enc_a_l = [
            nn.Sequential(nn.Linear(inp_dim, mid_dim), nn.ReLU(),
                          nn.Linear(mid_dim, mid_dim), ).to(self.device)
            for inp_dim in action_dims
        ]  # encoders of action

        self.net = DenseNet2(mid_dim)
        net_out_dim = self.net.out_dim

        '''decoders'''
        self.dec_a_l = [
            nn.Sequential(nn.Linear(net_out_dim, mid_dim), nn.ReLU(),
                          nn.Linear(mid_dim, out_dim), ).to(self.device)
            for out_dim in action_dims
        ]  # decoder of action mean
        self.dec_d_l = [
            nn.Sequential(nn.Linear(net_out_dim, mid_dim), nn.ReLU(),
                          nn.Linear(mid_dim, out_dim), ).to(self.device)
            for out_dim in action_dims
        ]  # decoder of action LogStd (d means standard dev.)
        self.dec_q1 = nn.Sequential(
            nn.Linear(net_out_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, reward_dim),
        ).to(self.device)  # decoder of q1 (Q value for twin critics)
        self.dec_q2 = nn.Sequential(
            nn.Linear(net_out_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, reward_dim),
        ).to(self.device)  # decoder of q2 (Q value for twin critics)

        for i in range(reward_dim):
            layer_norm(self.dec_a_l[i][-1], std=0.01)  # net[-1] is output layer for action, it is no necessary.
        layer_norm(self.dec_q1[-1], std=0.1)
        layer_norm(self.dec_q2[-1], std=0.1)

    def forward(self, s_l, noise_std=0.0):  # actor, in fact, noise_std is a boolean
        x_ = torch.stack([self.enc_s_l[i](s_l[i]) for i in range(self.reward_dim)]).mean(dim=0)
        a_ = self.net(x_)

        a_mean_l = list()
        for i in range(self.reward_dim):
            a_mean = self.dec_a_l[i](a_)  # NOTICE! it is a_mean without tensor.tanh()
            if noise_std != 0.0:
                a_std_log = self.dec_d_l[i](a_).clamp(self.log_std_min, self.log_std_max)
                a_std = a_std_log.exp()
                a_mean = torch.normal(a_mean, a_std)  # NOTICE! it is a_mean without .tanh()

            a_mean_l.append(a_mean.tanh())
        return a_mean_l

    def get__a__log_prob(self, s_l):  # actor
        x_ = torch.stack([self.enc_s_l[i](s_l[i]) for i in range(self.reward_dim)]).mean(dim=0)
        a_ = self.net(x_)

        a_noise_tanh_l = list()
        log_prob_l = list()
        for i in range(self.reward_dim):
            a_mean = self.dec_a_l[i](a_)  # NOTICE! it is a_mean without .tanh()
            a_std_log = self.dec_d_l[i](a_).clamp(self.log_std_min, self.log_std_max)
            a_std = a_std_log.exp()

            """add noise to action, stochastic policy"""
            a_noise = a_mean + a_std * torch.randn_like(a_mean, requires_grad=True, device=self.device)

            '''compute log_prob according to mean and std of action (stochastic policy)'''
            a_delta = ((a_noise - a_mean) / a_std).pow(2) * 0.5
            log_prob_noise = a_delta + a_std_log + self.constant_log_sqrt_2pi

            a_noise_tanh = a_noise.tanh()
            log_prob = log_prob_noise + (-a_noise_tanh.pow(2) + 1.000001).log()

            a_noise_tanh_l.append(a_noise_tanh)
            log_prob_l.append(log_prob.sum(1, keepdim=True))
        return a_noise_tanh_l, log_prob_l

    def get__a__std(self, s_l):
        x_ = torch.stack([self.enc_s_l[i](s_l[i]) for i in range(self.reward_dim)]).mean(dim=0)
        a_ = self.net(x_)

        a_mean_l = list()
        a_std_log_l = list()
        for i in range(self.reward_dim):
            a_mean = self.dec_a_l[i](a_)  # NOTICE! it is a_mean without .tanh()
            a_std_log = self.dec_d_l[i](a_).clamp(self.log_std_min, self.log_std_max)

            a_mean_l.append(a_mean.tanh())
            a_std_log_l.append(a_std_log)
        return a_mean_l, a_std_log_l

    def get__a__avg_std_noise_prob(self, s_l):  # actor
        x_ = torch.stack([self.enc_s_l[i](s_l[i]) for i in range(self.reward_dim)]).mean(dim=0)
        a_ = self.net(x_)

        a_mean_tanh_l = list()
        a_std_log_l = list()
        a_noise_tanh_l = list()
        log_prob_l = list()
        for i in range(self.reward_dim):
            a_mean = self.dec_a_l[i](a_)  # NOTICE! it is a_mean without .tanh()
            a_std_log = self.dec_d_l[i](a_).clamp(self.log_std_min, self.log_std_max)
            a_std = a_std_log.exp()

            """add noise to action, stochastic policy"""
            noise = torch.randn_like(a_mean, requires_grad=True, device=self.device)
            a_noise = a_mean + a_std * noise

            '''compute log_prob according to mean and std of action (stochastic policy)'''
            a_delta = ((a_noise - a_mean) / a_std).pow(2) * 0.5
            log_prob_noise = a_delta + a_std_log + self.constant_log_sqrt_2pi

            a_noise_tanh = a_noise.tanh()
            log_prob = log_prob_noise + (-a_noise_tanh.pow(2) + 1.000001).log()  # todo neg_log_prob

            a_mean_tanh_l.append(a_mean.tanh())
            a_std_log_l.append(a_std_log)
            a_noise_tanh_l.append(a_noise_tanh)
            log_prob_l.append(log_prob.sum(1, keepdim=True))

        return a_mean_tanh_l, a_std_log_l, a_noise_tanh_l, log_prob_l

    def get__q1_q2(self, s_l, a_l):  # critic
        x_ = torch.stack([self.enc_s_l[i](s_l[i]) + self.enc_a_l[i](a_l[i])
                          for i in range(self.reward_dim)]).mean(dim=0)
        q_ = self.net(x_)
        q1 = self.dec_q1(q_)
        q2 = self.dec_q2(q_)
        return q1, q2

    def get__policy_gradient(self, s_l, a_l):  # critic
        s_l_ = [self.enc_s_l[i](s_l[i]) for i in range(self.reward_dim)]
        s_sum = torch.stack(s_l_).mean(dim=0)

        a_l_ = [self.enc_a_l[i](a_l[i]) for i in range(self.reward_dim)]

        pg_l = list()
        for i in range(self.reward_dim):
            a_sum = torch.stack([(a_l_[i] if j == i else a_l_[i].detach())
                                 for j in range(self.reward_dim)]).mean(dim=0)

            q_ = self.net(s_sum + a_sum)
            pg = torch.min(self.dec_q1(q_)[:, i], self.dec_q2(q_)[:, i])
            pg_l.append(pg)

        pg_sum = torch.stack(pg_l).mean()
        return pg_sum


class MAgentInterSAC(AgentBasicAC):  # Integrated Soft Actor-Critic Methods
    def __init__(self, state_dim, action_dim, net_dim, reward_dim):  # todo multi reward
        super(AgentBasicAC, self).__init__()
        self.learning_rate = 8e-5  # todo learning rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reward_dim = reward_dim

        '''network'''
        actor_dim = net_dim
        self.act = MInterSPG(state_dim, action_dim, actor_dim, reward_dim).to(self.device)  # todo multi reward
        self.act.train()

        self.cri = self.act

        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate)

        self.act_target = MInterSPG(state_dim, action_dim, actor_dim, reward_dim).to(self.device)  # todo multi reward
        self.act_target.eval()
        self.act_target.load_state_dict(self.act.state_dict())

        self.criterion1 = nn.SmoothL1Loss(reduction='none')
        self.criterion = nn.SmoothL1Loss(reduction='mean')

        '''training record'''
        self.state = None  # env.reset()
        self.reward_sum = 0.0
        self.step = 0
        self.update_counter = 0

        '''extension: auto-alpha for maximum entropy'''
        # self.alpha = torch.tensor((1.0,), requires_grad=True, device=self.device)  # I don't know why np.e ** 0.5
        # self.alpha_optimizer = torch.optim.Adam((self.alpha,), lr=self.learning_rate)
        # self.target_entropy = np.log(action_dim) * 0.5
        self.log_alpha = torch.tensor([0.0, ] * reward_dim, requires_grad=True, device=self.device)  # I don't know why
        self.alpha_optimizer = torch.optim.Adam((self.log_alpha,), lr=self.learning_rate)
        self.target_entropy = torch.tensor([np.log(dim) * 0.5 for dim in action_dim],
                                           requires_grad=False, device=self.device)

        '''extension: auto learning rate of actor'''
        self.trust_rho_l = [TrustRho0909() for _ in range(reward_dim)]

        '''constant'''
        self.explore_rate = 1.0  # explore rate when update_buffer(), 1.0 is better than 0.5
        self.explore_noise = True  # stochastic policy choose noise_std by itself.

    def update_buffer(self, env, buffer, max_step, max_action, reward_scale, gamma):  # todo MA
        self.act.eval()

        rewards = list()
        steps = list()
        for _ in range(max_step):
            '''inactive with environment'''
            action_l = self.select_actions(self.state, True)
            next_state_l, reward_l, done_l, _ = env.step(action_l)
            done = any(done_l)

            self.reward_sum += sum(reward_l)
            self.step += 1

            '''update replay buffer'''
            reward_ = [reward * reward_scale for reward in reward_l]
            mask = 0.0 if done else gamma
            buffer.add_memo((*reward_, mask, *self.state, *action_l, *next_state_l))

            self.state = next_state_l
            if done:
                rewards.append(self.reward_sum)
                self.reward_sum = 0.0

                steps.append(self.step)
                self.step = 0

                self.state = env.reset()
        return rewards, steps

    def select_actions(self, state_l, explore_noise=0.0):  # CPU array to GPU tensor to CPU array
        state_l = [torch.tensor((state,), dtype=torch.float32, device=self.device)
                   for state in state_l]
        action_l = self.act(state_l, explore_noise)  # tensor
        return [action.cpu().data.numpy()[0] for action in action_l]  # array

    def update_parameters(self, buffer, max_step, batch_size, repeat_times):
        self.act.train()

        loss_a_list = list()
        loss_c_list = list()

        alpha = self.log_alpha.exp().detach()
        k = 1.0 + buffer.now_len / buffer.max_len
        batch_size_ = int(batch_size * k)
        update_times = int(max_step * k)

        for i in range(update_times):
            with torch.no_grad():
                reward, mask, state_l, action_l, next_s_l = buffer.random_sample(batch_size_, self.device)

                next_a_noise_l, next_log_prob_l = self.act_target.get__a__log_prob(next_s_l)
                next_log_prob = torch.cat(next_log_prob_l, dim=1)

                next_q_target = torch.min(*self.act_target.get__q1_q2(next_s_l, next_a_noise_l))  # twin critic
                q_target = reward + mask * (next_q_target + next_log_prob * alpha)  # policy entropy

            '''critic_loss'''
            q1_value, q2_value = self.cri.get__q1_q2(state_l, action_l)  # CriticTwin
            critic_loss = (self.criterion1(q1_value, q_target).mean(dim=0) +
                           self.criterion1(q2_value, q_target).mean(dim=0))
            loss_c_tmp_l = critic_loss.data.cpu().numpy() * 0.5  # CriticTwin
            loss_c_list.append(loss_c_tmp_l.mean())
            rho_l = np.array([self.trust_rho_l[i].update_rho(loss_c_tmp_l[i])
                              for i in range(self.reward_dim)])

            '''stochastic policy'''
            a1_mean_l, a1_log_std_l, a_noise_l, log_prob_l = self.act.get__a__avg_std_noise_prob(
                state_l)  # policy gradient
            log_prob = torch.cat(log_prob_l, dim=1)

            '''action correction term'''
            a2_mean_l, a2_log_std_l = self.act_target.get__a__std(state_l)
            actor_term = torch.cat([
                (self.criterion(a1_mean_l[i], a2_mean_l[i]) +
                 self.criterion(a1_log_std_l[i], a2_log_std_l[i])).unsqueeze(0)
                for i in range(self.reward_dim)
            ], dim=0)

            '''auto alpha'''
            alpha_loss = (self.log_alpha * (log_prob - self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp().detach()

            rho_l[rho_l < 2 ** -8] = 0
            rho_l = torch.tensor(rho_l, dtype=torch.float32, device=self.device)
            # if rho > 2 ** -8:  # (self.rho>0.001) ~= (self.critic_loss<2.6)
            '''actor_loss'''
            # q_eval_pg = torch.min(*self.act_target.get__q1_q2(state_l, a_noise_l))  # policy gradient
            q_eval_pg = self.act_target.get__policy_gradient(state_l, a_noise_l)  # policy gradient
            actor_loss = -(q_eval_pg + log_prob * alpha).mean()  # policy gradient
            loss_a_list.append(actor_loss.item())
            # else:
            #     actor_loss = 0

            united_loss = (critic_loss + actor_term * (- rho_l + 1) + actor_loss * rho_l).sum()
            self.act_optimizer.zero_grad()
            united_loss.backward()
            self.act_optimizer.step()

            soft_target_update(self.act_target, self.act, tau=2 ** -8)

        loss_a_avg = (sum(loss_a_list) / len(loss_a_list)) if len(loss_a_list) > 0 else 0.0
        loss_c_avg = sum(loss_c_list) / len(loss_c_list)
        return loss_a_avg, loss_c_avg


def multi_to_single_walker_decorator_ma(env):
    def decorator_step(env_step):
        def new_env_step(action):
            action = action.clip(-1, 1)  # fix bug in class GymMultiWalkerEnv
            action_l = (action[0:4], action[4:8], action[8:12])
            state_l, reward_l, done_l, info = env_step(action_l)
            state = np.hstack(state_l)
            done = any(done_l)
            # reward = sum(reward_l)
            # return state, reward, done, info
            return state, reward_l, done, info  # todo multi reward

        return new_env_step

    env.step = decorator_step(env.step)

    def decorator_reset(env_reset):
        def new_env_reset():
            state_l = env_reset()
            state = np.hstack(state_l)
            return state

        return new_env_reset

    env.reset = decorator_reset(env.reset)
    return env


class BufferArrayMA:  # 2020-05-20
    def __init__(self, memo_max_len, state_dim_l, action_dim_l, reward_dim):
        dim_l = [reward_dim, 1] + list(state_dim_l) + list(action_dim_l) + list(state_dim_l)
        self.memories = np.empty((memo_max_len, sum(dim_l)), dtype=np.float32)

        self.idx_l = [0, ]
        for i in range(len(dim_l)):
            self.idx_l.append(self.idx_l[i] + dim_l[i])
        self.num_l = (2, 2 + reward_dim * 1, 2 + reward_dim * 2, 2 + reward_dim * 3)

        '''pointer'''
        self.next_idx = 0
        self.is_full = False
        self.max_len = memo_max_len
        self.now_len = self.max_len if self.is_full else self.next_idx

    def add_memo(self, memo_tuple):
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

    def random_sample(self, batch_size, device):
        indices = rd.randint(self.now_len, size=batch_size)
        memory = self.memories[indices]
        if device:
            memory = torch.tensor(memory, device=device)

        '''convert array into torch.tensor'''
        tensors = [memory[:, self.idx_l[i]:self.idx_l[i + 1]]
                   for i in range(len(self.idx_l) - 1)]
        tensors = (tensors[0], tensors[1],
                   tensors[self.num_l[0]:self.num_l[1]],
                   tensors[self.num_l[1]:self.num_l[2]],
                   tensors[self.num_l[2]:self.num_l[3]],
                   )
        return tensors


def initial_exploration_ma(env, memo, max_step, action_max, reward_scale, gamma, action_dim_l):
    state_l = env.reset()

    rewards = list()
    reward_sum = 0.0
    steps = list()
    step = 0

    if isinstance(action_max, int) and action_max == int(1):
        def random_uniform_policy_for_discrete_action():
            return [rd.randint(dim) for dim in action_dim_l]

        get_random_action = random_uniform_policy_for_discrete_action
    else:
        def random_uniform_policy_for_continuous_action():
            return [rd.uniform(-1, 1, size=dim) * action_max for dim in action_dim_l]

        get_random_action = random_uniform_policy_for_continuous_action

    global_step = 0
    while global_step < max_step:
        # action = np.tanh(rd.normal(0, 0.25, size=action_dim))  # zero-mean gauss exploration
        action_l = get_random_action()
        next_state_l, reward_l, done_l, _ = env.step(action_l)
        done = any(done_l)
        reward_sum += sum(reward_l)  # todo multi reward
        step += 1

        adjust_reward_l = [reward * reward_scale for reward in reward_l]
        mask = 0.0 if done else gamma
        memo.add_memo((*adjust_reward_l, mask, *state_l, *action_l, *next_state_l))

        state_l = next_state_l
        if done:
            rewards.append(reward_sum)
            steps.append(step)
            global_step += step

            state_l = env.reset()  # reset the environment
            reward_sum = 0.0
            step = 1

    memo.init_before_sample()
    return rewards, steps


def get_episode_reward_ma(env, act, max_step, max_action, device, is_discrete) -> float:
    # better to 'with torch.no_grad()'
    reward_item = 0.0

    state_l = env.reset()
    for _ in range(max_step):
        s_tensor = [torch.tensor((state,), dtype=torch.float32, device=device)
                    for state in state_l]

        action = [a_tensor.cpu().data.numpy()[0] * max_action
                  for a_tensor in act(s_tensor)]

        next_state_l, reward, done_l, _ = env.step(action)
        done = any(done_l)
        reward_item += sum(reward)  # todo multi reward

        if done:
            break
        state_l = next_state_l
    return reward_item


class Recorder:
    def __init__(self):
        self.eva_r_max = -np.inf
        self.total_step = 0
        self.record_exp = list()  # total_step, exp_r_avg, loss_a_avg, loss_c_avg
        self.record_eva = list()  # total_step, eva_r_avg, eva_r_std
        self.is_solved = False

        '''constant'''
        self.eva_size1 = 2

        '''print_reward'''
        self.used_time = None
        self.start_time = time.time()
        self.print_time = time.time()

        print(f"{'GPU':>3}  {'Step':>8}  {'MaxR':>8} |"
              f"{'avgR':>8}  {'stdR':>8} |"
              f"{'ExpR':>8}  {'LossA':>8}  {'LossC':>8}")

    def update__record_evaluate(self, env, act, max_step, max_action, eva_size, device, is_discrete):
        is_saved = False
        reward_list = [get_episode_reward_ma(env, act, max_step, max_action, device, is_discrete)
                       for _ in range(self.eva_size1)]

        eva_r_avg = np.average(reward_list)
        if eva_r_avg > self.eva_r_max:  # check 1
            reward_list.extend([get_episode_reward_ma(env, act, max_step, max_action, device, is_discrete)
                                for _ in range(eva_size - self.eva_size1)])
            eva_r_avg = np.average(reward_list)
            if eva_r_avg > self.eva_r_max:  # check final
                self.eva_r_max = eva_r_avg
                is_saved = True

        eva_r_std = np.std(reward_list)
        self.record_eva.append((self.total_step, eva_r_avg, eva_r_std))
        return is_saved

    def update__record_explore(self, exp_s_sum, exp_r_avg, loss_a, loss_c):
        if isinstance(exp_s_sum, int):
            exp_s_sum = (exp_s_sum,)
            exp_r_avg = (exp_r_avg,)
        for s, r in zip(exp_s_sum, exp_r_avg):
            self.total_step += s
            self.record_exp.append((self.total_step, r, loss_a, loss_c))

    def save_act(self, cwd, act, gpu_id):
        act_save_path = f'{cwd}/actor.pth'
        torch.save(act.state_dict(), act_save_path)
        print(f"{gpu_id:<3}  {self.total_step:8.2e}  {self.eva_r_max:8.2f} |")

    def check_is_solved(self, target_reward, gpu_id, show_gap):
        if self.eva_r_max > target_reward:
            self.is_solved = True
            if self.used_time is None:
                self.used_time = int(time.time() - self.start_time)
                print(f"{'GPU':>3}  {'Step':>8}  {'TargetR':>8} |"
                      f"{'avgR':>8}  {'stdR':>8} |"
                      f"{'ExpR':>8}  {'UsedTime':>8}  ########")

                total_step, eva_r_avg, eva_r_std = self.record_eva[-1]
                total_step, exp_r_avg, loss_a_avg, loss_c_avg = self.record_exp[-1]
                print(f"{gpu_id:<3}  {total_step:8.2e}  {target_reward:8.2f} |"
                      f"{eva_r_avg:8.2f}  {eva_r_std:8.2f} |"
                      f"{exp_r_avg:8.2f}  {self.used_time:>8}  ########")

        if time.time() - self.print_time > show_gap:
            self.print_time = time.time()

            total_step, eva_r_avg, eva_r_std = self.record_eva[-1]
            total_step, exp_r_avg, loss_a_avg, loss_c_avg = self.record_exp[-1]
            print(f"{gpu_id:<3}  {total_step:8.2e}  {self.eva_r_max:8.2f} |"
                  f"{eva_r_avg:8.2f}  {eva_r_std:8.2f} |"
                  f"{exp_r_avg:8.2f}  {loss_a_avg:8.2f}  {loss_c_avg:8.2f}")
        return self.is_solved

    def save_npy__plot_png(self, cwd):
        np.save('%s/record_explore.npy' % cwd, self.record_exp)
        np.save('%s/record_evaluate.npy' % cwd, self.record_eva)
        draw_plot_with_2npy(cwd, train_time=time.time() - self.start_time)

    def demo(self):
        pass


def train_agent(
        rl_agent, net_dim, batch_size, repeat_times, gamma, reward_scale, cwd,
        env_name, max_step, max_memo, max_total_step,
        eva_size, gpu_id, show_gap, **_kwargs):  # 2020-06-01
    assert env_name == 'MultiWalker'
    from multiwalker_base import MultiWalkerEnv
    env = MultiWalkerEnv()

    state_dim_l = [box.shape[0] for box in env.observation_space]
    action_dim_l = [box.shape[0] for box in env.action_space]
    max_action = 1.0
    target_reward = 300
    is_discrete = False
    reward_dim = env.num_agents

    '''init: agent, buffer, recorder'''
    recorder = Recorder()
    agent = rl_agent(state_dim_l, action_dim_l, net_dim, reward_dim)  # training agent
    agent.state = env.reset()

    buffer = BufferArrayMA(max_memo, state_dim_l, 1 if is_discrete else action_dim_l, reward_dim)
    with torch.no_grad():  # update replay buffer
        rewards, steps = initial_exploration_ma(env, buffer, max_step, max_action, reward_scale, gamma, action_dim_l)
    recorder.update__record_explore(steps, rewards, loss_a=0, loss_c=0)

    '''loop'''
    is_training = True
    while is_training:
        '''update replay buffer by interact with environment'''
        with torch.no_grad():  # for saving the GPU buffer
            rewards, steps = agent.update_buffer(
                env, buffer, max_step, max_action, reward_scale, gamma)

        '''update network parameters by random sampling buffer for gradient descent'''
        buffer.init_before_sample()
        loss_a, loss_c = agent.update_parameters(
            buffer, max_step, batch_size, repeat_times)

        '''saves the agent with max reward'''
        with torch.no_grad():  # for saving the GPU buffer
            recorder.update__record_explore(steps, rewards, loss_a, loss_c)

            is_saved = recorder.update__record_evaluate(
                env, agent.act, max_step, max_action, eva_size, agent.device, is_discrete)
            recorder.save_act(cwd, agent.act, gpu_id) if is_saved else None

            is_solved = recorder.check_is_solved(target_reward, gpu_id, show_gap)
        '''break loop rules'''
        if is_solved or recorder.total_step > max_total_step or os.path.exists(f'{cwd}/stop.mark'):
            is_training = False

    recorder.save_npy__plot_png(cwd)


def run_continuous_action(gpu_id=None):
    # import AgentZoo as Zoo
    args = Arguments(rl_agent=MAgentInterSAC, gpu_id=gpu_id)

    args.env_name = "MultiWalker"
    args.random_seed = 1945
    args.max_total_step = int(2e5 * 4)
    args.repeat_times = 1.5  # todo should be larger?
    args.init_for_training()
    train_agent(**vars(args))  # build_for_mp(args)  #
    # exit()


if __name__ == '__main__':
    run_continuous_action()
