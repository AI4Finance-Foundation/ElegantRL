from AgentRun import *
from AgentNet import *
from AgentZoo import *

"""     
beta11  MixSAC, Ant,  args.max_step = 2 ** 10 norm std + 1e-5


beta0   mix, BW LL,    ppo_need_enable > 3, print
beta3   mix, Minitaur, ppo_need_enable > 3, print
beta10  mix, LL BW,    ppo_need_enable > 4, print
beta12  mix, Ant,      ppo_need_enable > 4, print

beta14  mix, BWLL Ant,  fix bug
"""


class AgentMixSAC(AgentBasicAC):  # Integrated Soft Actor-Critic Methods
    def __init__(self, state_dim, action_dim, net_dim):
        super(AgentBasicAC, self).__init__()
        self.learning_rate = 1e-4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''network'''
        self.act = InterSPG(state_dim, action_dim, net_dim).to(self.device)
        self.act.train()
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate)

        self.act_target = InterSPG(state_dim, action_dim, net_dim).to(self.device)
        self.act_target.eval()
        self.act_target.load_state_dict(self.act.state_dict())

        self.cri = self.act

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
        self.target_entropy = np.log(action_dim + 1) * 0.5
        self.log_alpha = torch.tensor((-self.target_entropy * np.e,), requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam((self.log_alpha,), lr=self.learning_rate)

        '''extension: reliable lambda for auto-learning-rate'''
        self.avg_loss_c = (-np.log(0.5)) ** 0.5

        '''constant'''
        self.explore_noise = True  # stochastic policy choose noise_std by itself.

    def update_policy(self, buffer, max_step, batch_size, repeat_times):
        self.act.train()

        lamb = 0
        actor_loss = critic_loss = None

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
            q1_value, q2_value = self.cri.get__q1_q2(state, action)  # CriticTwin
            critic_loss = self.criterion(q1_value, q_target) + self.criterion(q2_value, q_target)

            '''auto reliable lambda'''
            self.avg_loss_c = 0.995 * self.avg_loss_c + 0.005 * critic_loss.item() / 2  # soft update, twin critics
            lamb = np.exp(-self.avg_loss_c ** 2)

            '''stochastic policy'''
            a1_mean, a1_log_std, a_noise, log_prob = self.act.get__a__avg_std_noise_prob(state)  # policy gradient
            log_prob = log_prob.mean()  # todo need check

            '''auto temperature parameter: alpha'''
            alpha_loss = self.log_alpha * (log_prob - self.target_entropy).detach()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            with torch.no_grad():
                self.log_alpha[:] = self.log_alpha.clamp(-16, 1)
            alpha = self.log_alpha.exp().detach()

            '''action correction term'''
            with torch.no_grad():  # todo need check
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
        return actor_loss.item(), critic_loss.item() / 2


class AgentMixPPO:
    def __init__(self, state_dim, action_dim, net_dim):
        self.learning_rate = 1e-4  # learning rate of actor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''network'''
        self.act = ActorPPO(state_dim, action_dim, net_dim).to(self.device)
        self.act.train()
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate, )  # betas=(0.5, 0.99))
        self.imt_optimizer = torch.optim.SGD(self.act.parameters(), lr=self.learning_rate,
                                             momentum=0.1)  # betas=(0.5, 0.99))

        self.cri = CriticAdv(state_dim, net_dim).to(self.device)
        self.cri.train()
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate, )  # betas=(0.5, 0.99))

        self.criterion = nn.SmoothL1Loss()

    def update_buffer(self, env, buffer, max_step, reward_scale, gamma):
        # collect tuple (reward, mask, state, action, log_prob, )
        buffer.storage_list = list()  # PPO is an online policy RL algorithm.
        # PPO (or GAE) should be an online policy.
        # Don't use Offline for PPO (or GAE). It won't speed up training but slower

        rewards = list()
        steps = list()

        step_counter = 0
        while step_counter < buffer.max_memo:
            state = env.reset()

            reward_sum = 0
            step_sum = 0

            for step_sum in range(max_step):
                actions, log_probs = self.select_actions((state,), explore_noise=True)
                action = actions[0]
                log_prob = log_probs[0]

                next_state, reward, done, _ = env.step(np.tanh(action))
                reward_sum += reward

                mask = 0.0 if done else gamma

                reward_ = reward * reward_scale
                buffer.push(reward_, mask, state, action, log_prob, )

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
        max_memo = len(buffer)
        all_batch = buffer.sample_all()
        all_reward, all_mask, all_state, all_action, all_log_prob = [
            torch.tensor(ary, dtype=torch.float32, device=self.device)
            for ary in (all_batch.reward, all_batch.mask, all_batch.state, all_batch.action, all_batch.log_prob,)
        ]

        # all__new_v = self.cri(all_state).detach_()  # all new value
        with torch.no_grad():
            b_size = 512
            all__new_v = torch.cat(
                [self.cri(all_state[i:i + b_size])
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
            loss_entropy = (torch.exp(new_log_prob) * new_log_prob).mean()  # policy entropy

            actor_loss = surrogate_obj + loss_entropy * lambda_entropy
            self.act_optimizer.zero_grad()
            actor_loss.backward()
            self.act_optimizer.step()

        self.act.eval()
        self.cri.eval()
        return actor_loss.item(), critic_loss.item()

    def update_policy_imitate_cri(self, buffer, batch_size, repeat_times):
        # self.act.train()
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
            critic_loss = (self.criterion(new_value, old_value)) / (old_value.std() + 1e-5)  # todo need check
            self.cri_optimizer.zero_grad()
            critic_loss.backward()
            self.cri_optimizer.step()

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

        # self.act.eval()
        self.cri.eval()
        return critic_loss.item()

    def update_policy_imitate_act(self, buffer, act_target, batch_size, repeat_time=16):
        train_step = int(buffer.now_len / batch_size * repeat_time)
        buffer_state = torch.tensor(
            buffer.memories[:, 2:buffer.state_idx], device=self.device)

        min_loss = 0.05
        actor_loss = min_loss
        for i in range(train_step):
            indices = rd.randint(buffer.now_len, size=batch_size + i)
            state = buffer_state[indices]

            a_train = self.act(state)
            with torch.no_grad():
                a_target = act_target(state)
            actor_term = self.criterion(a_train, a_target)
            actor_loss = 0.9 * actor_loss + 0.1 * actor_term.item()
            if actor_loss < min_loss:
                break

            self.imt_optimizer.zero_grad()
            actor_term.backward()
            self.imt_optimizer.step()

        return actor_loss

    def select_actions(self, states, explore_noise=0.0):  # CPU array to GPU tensor to CPU array
        states = torch.tensor(states, dtype=torch.float32, device=self.device)

        if explore_noise == 0.0:
            a_mean = self.act(states)
            a_mean = a_mean.cpu().data.numpy()
            return a_mean.tanh()
        else:
            a_noise, log_prob = self.act.get__a__log_prob(states)
            a_noise = a_noise.cpu().data.numpy()
            log_prob = log_prob.cpu().data.numpy()
            return a_noise, log_prob  # not tanh()

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


def train_agent_mix(
        rl_agent, env_name, gpu_id, cwd,
        net_dim, max_memo, max_step, batch_size, repeat_times, reward_scale, gamma,
        break_step, if_break_early, show_gap, eval_times1, eval_times2, **_kwargs):  # 2020-09-18
    env_off, state_dim, action_dim, target_reward, if_discrete = build_gym_env(env_name, if_print=False)

    '''init: agent, buffer, recorder'''
    id_off = gpu_id  # todo id
    recorder = Recorder(eval_size1=eval_times1, eval_size2=eval_times2)
    agent_off = AgentInterSAC(state_dim, action_dim, net_dim)  # training agent
    agent_off.state = env_off.reset()

    buffer_off = BufferArray(max_memo, state_dim, 1 if if_discrete else action_dim)
    with torch.no_grad():  # update replay buffer
        rewards, steps = initial_exploration(
            env_off, buffer_off, max_step, if_discrete, reward_scale, gamma, action_dim)
    recorder.update__record_explore(steps, rewards, loss_a=0, loss_c=0)

    '''pre training and hard update before training loop'''
    buffer_off.init_before_sample()
    agent_off.update_policy(buffer_off, max_step, batch_size, repeat_times)
    agent_off.act_target.load_state_dict(agent_off.act.state_dict())

    # todo online
    id_onn = -1  # todo id
    max_memo2 = max_step * 2
    net_dim2 = 2 ** 8
    env_onn, state_dim, action_dim, target_reward, if_discrete = build_gym_env(env_name, if_print=False)
    recorder2 = Recorder(eval_size1=eval_times1, eval_size2=eval_times2)
    agent_onn = AgentMixPPO(state_dim, action_dim, net_dim2)  # training agent
    agent_onn.state = env_onn.reset()
    buffer_onn = BufferTupleOnline(max_memo2)

    batch_size2 = 2 ** 9
    repeat_times2 = 2 ** 3

    # # todo best explored reward record
    from copy import deepcopy
    tmp_act = deepcopy(agent_off.act)
    exp_r_sac = exp_r_tmp = sum(rewards) / len(rewards)
    ppo_need_update = False
    ppo_need_enable = 0

    '''loop'''
    if_train = True
    while if_train:
        if exp_r_sac > exp_r_tmp:
            tmp_act = deepcopy(agent_off.act)
            exp_r_tmp = exp_r_sac
            ppo_need_update = True

        # todo online
        if exp_r_sac < exp_r_tmp:  # todo enable PPO
            ppo_need_enable += 1
        else:
            ppo_need_enable = 0

        if ppo_need_enable > 3:
            if ppo_need_update:
                buffer_off.init_before_sample()
                l_a = agent_onn.update_policy_imitate_act(buffer_off, tmp_act, batch_size * 4)

                with torch.no_grad():  # speed up running
                    rewards2, steps2 = agent_onn.update_buffer(env_onn, buffer_onn, max_memo2, reward_scale, gamma)
                    exp_r_tmp = 0.9 * exp_r_tmp + 0.1 * sum(rewards2) / len(rewards2)

                buffer_onn.init_before_sample()
                l_c = agent_onn.update_policy_imitate_cri(buffer_onn, batch_size2, repeat_times2)

                ppo_need_update = False
                print(f"\t\t{l_a:.3f}\t{l_c:.3f}\t{exp_r_sac:.3f}\t{exp_r_tmp:.3f}")
            else:
                with torch.no_grad():  # speed up running
                    rewards2, steps2 = agent_onn.update_buffer(env_onn, buffer_onn, max_memo2, reward_scale, gamma)
                    if len(rewards) > 0:
                        exp_r_tmp = 0.9 * exp_r_tmp + 0.1 * sum(rewards2) / len(rewards2)

            buffer_onn.init_before_sample()
            loss_a2, loss_c2 = agent_onn.update_policy(
                buffer_onn, max_step, batch_size2, repeat_times2)

            with torch.no_grad():
                recorder2.update__record_explore(steps2, rewards2, loss_a2, loss_c2)

                if_save2 = recorder2.update__record_evaluate(
                    env_onn, agent_onn.act, max_step, agent_off.device, if_discrete)
                # recorder2.save_act(cwd, agent_onn.act, id_onn) if if_save else None
                # recorder2.save_npy__plot_png(cwd)
                if_solve2 = recorder2.check_is_solved(target_reward, id_onn, show_gap)

                # todo onn to off
                buffer_ary = buffer_onn.convert_to_rmsas()
                buffer_off.extend_memo(buffer_ary)
            buffer_off.init_before_sample()
            loss_a, loss_c = agent_off.update_policy(
                buffer_off, max_memo2, batch_size, repeat_times)  # todo max_memo2
        else:
            pass  # SAC is better than tmp or PPO

        # todo offline
        with torch.no_grad():  # speed up running
            rewards, steps = agent_off.update_buffer(env_off, buffer_off, max_step, reward_scale, gamma)
            if len(rewards) > 0:
                exp_r_sac = 0.9 * exp_r_sac + 0.1 * sum(rewards) / len(rewards)

        buffer_off.init_before_sample()
        loss_a, loss_c = agent_off.update_policy(buffer_off, max_step, batch_size, repeat_times)

        with torch.no_grad():  # for saving the GPU buffer
            recorder.update__record_explore(steps, rewards, loss_a, loss_c)

            if_save = recorder.update__record_evaluate(env_off, agent_off.act, max_step, agent_off.device, if_discrete)
            recorder.save_act(cwd, agent_off.act, id_off) if if_save else None
            recorder.save_npy__plot_png(cwd)

            if_solve = recorder.check_is_solved(target_reward, id_off, show_gap)

        '''break loop rules'''
        if_train = not ((if_break_early and if_solve)
                        or recorder.total_step > break_step
                        or os.path.exists(f'{cwd}/stop'))
    recorder.save_npy__plot_png(cwd)
    buffer_off.print_state_norm(env_off.neg_state_avg, env_off.div_state_std)


def run_continuous_action(gpu_id=None):
    rl_agent = AgentInterSAC
    args = Arguments(rl_agent, gpu_id)
    args.if_break_early = False
    args.if_remove_history = True

    args.random_seed += 4
    args.max_step = 2 ** 10  # 12

    args.env_name = "BipedalWalker-v3"
    args.break_step = int(2e5 * 8)  # (1e5) 2e5
    args.reward_scale = 2 ** -1  # todo  # (-200) -150 ~ 300 (342)
    args.init_for_training(4)
    train_agent_mix(**vars(args))
    # exit()
    args.env_name = "LunarLanderContinuous-v2"
    args.break_step = int(5e4 * 16)  # (2e4) 5e4
    args.reward_scale = 2 ** -1  # (-800) -200 ~ 200 (302)
    args.init_for_training(4)
    train_agent_mix(**vars(args))
    # exit()

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)
    args.env_name = "AntBulletEnv-v0"
    args.break_step = int(1e6 * 8)  # (8e5) 10e5
    args.reward_scale = 2 ** -3  # (-50) 0 ~ 2500 (3340)
    args.max_step = 2 ** 10
    args.batch_size = 2 ** 8
    args.max_memo = 2 ** 20
    args.eva_size = 2 ** 3  # for Recorder
    args.show_gap = 2 ** 8  # for Recorder
    args.init_for_training(8)
    train_agent_mp(args)  # train_agent(**vars(args))
    exit()

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
    args.init_for_training(8)
    train_agent_mp(args)  # train_agent(**vars(args))
    exit()


run_continuous_action()
