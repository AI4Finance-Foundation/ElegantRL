from AgentRun import *
from AgentNet import *
from AgentZoo import *

"""
beta2
beta3 InterAdvPPO wait todo TwinCritic
beta0     args.repeat_times = 2 ** 4
beta1     target_network
beta0     DenseNet
"""


class InterAdv(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, mid_dim), nn.ReLU(),
            # nn.Linear(mid_dim, mid_dim), nn.ReLU(),
            DenseNet(mid_dim),
        )
        self.net__mean = nn.Linear(mid_dim * 4, action_dim)
        self.net__std_log = nn.Linear(mid_dim * 4, action_dim)
        self.net__q1 = nn.Linear(mid_dim * 4, 1)
        self.net__q2 = nn.Linear(mid_dim * 4, 1)

        self.log_std_min = -20
        self.log_std_max = 2
        self.constant_log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        layer_norm(self.net[0], std=1.0)
        layer_norm(self.net[2], std=1.0)
        layer_norm(self.net__mean, std=0.01)  # output layer for action
        layer_norm(self.net__std_log, std=0.01)  # output layer for std_log
        layer_norm(self.net__q1, std=1.0)  # output layer for q value
        layer_norm(self.net__q2, std=1.0)  # TwinCritic (DoubleDQN TD3)

    def forward(self, s):
        x = self.net(s)
        a_mean = self.net__mean(x)
        return a_mean

    def get__q_min(self, s):
        x = self.net(s)
        q1 = self.net__q1(x)
        q2 = self.net__q2(x)
        return torch.min(q1, q2)

    def get__a__log_prob(self, state):
        x = self.net(state)
        a_mean = self.net__mean(x)
        a_log_std = self.net__std_log(x).clamp(self.log_std_min, self.log_std_max)
        a_std = torch.exp(a_log_std)

        # a_noise = torch.normal(a_mean, a_std, requires_grad=True)
        noise = torch.randn_like(a_mean, requires_grad=True, device=self.device)
        a_noise = a_mean + a_std * noise

        a_delta = (a_noise - a_mean).pow(2) / (2 * a_std.pow(2))
        log_prob = -(a_delta + a_log_std + self.constant_log_sqrt_2pi).sum(1)
        return a_noise, log_prob

    def compute__log_prob(self, state, a_noise):
        x = self.net(state)
        a_mean = self.net__mean(x)
        a_log_std = self.net__std_log(x).clamp(self.log_std_min, self.log_std_max)
        a_std = torch.exp(a_log_std)

        a_delta = (a_noise - a_mean).pow(2) / (2 * a_std.pow(2))
        log_prob = -(a_delta + a_log_std + self.constant_log_sqrt_2pi)
        log_prob = log_prob.sum(1)

        q1 = self.net__q1(x)
        q2 = self.net__q2(x)
        return log_prob, q1, q2


class AgentAdvPPO(AgentPPO):
    def __init__(self, state_dim, action_dim, net_dim):
        super(AgentPPO, self).__init__()
        self.learning_rate = 2e-4  # learning rate of actor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''network'''
        self.act = ActCriAdv(state_dim, action_dim, net_dim).to(self.device)
        self.act.train()
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate, )  # betas=(0.5, 0.99))

        self.act_target = ActCriAdv(state_dim, action_dim, net_dim).to(self.device)
        self.act_target.eval()
        self.act_target.load_state_dict(self.act.state_dict())

        # self.cri = CriticAdv(state_dim, net_dim).to(self.device)
        # self.cri.train()
        # self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate, )  # betas=(0.5, 0.99))

        self.criterion = nn.SmoothL1Loss()

        self.update_freq = 2 ** 7  # delay update frequency, for hard target update
        self.update_counter = 0  # delay update counter

    def update_parameters_gae(self, buffer, batch_size, repeat_times):
        self.act.train()
        # self.cri.train()
        clip = 0.25  # ratio.clamp(1 - clip, 1 + clip)
        lambda_adv = 0.98  # Could be 0.97 or 0.98. Wait but why? Cannot use 0.99.
        lambda_entropy = 0.01  # could be 0.02. Set greater when speed up training in a easy task.
        # repeat_times = 2 ** 4 # could be 2 ** 3. Set smaller when speed up training in a easy task.

        loss_a_sum = 0.0  # just for print
        loss_c_sum = 0.0  # just for print

        '''the batch for training'''
        max_memo = len(buffer)
        all_batch = buffer.sample()
        all_reward, all_mask, all_state, all_action, all_log_prob = [
            torch.tensor(ary, dtype=torch.float32, device=self.device)
            for ary in (all_batch.reward, all_batch.mask, all_batch.state, all_batch.action, all_batch.log_prob,)
        ]
        # with torch.no_grad():
        # all__new_v = self.cri(all_state).detach_()
        all__new_v = self.act.get__q_min(all_state).detach_()

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
            all__delta[i] = all_reward[i] + all_mask[i] * prev_new_v - all__new_v[i]
            all__old_v[i] = all_reward[i] + all_mask[i] * prev_old_v
            all__adv_v[i] = all__delta[i] + all_mask[i] * prev_adv_v * lambda_adv

            prev_old_v = all__old_v[i]
            prev_new_v = all__new_v[i]
            prev_adv_v = all__adv_v[i]

        all__adv_v = (all__adv_v - all__adv_v.mean()) / (all__adv_v.std() + 1e-6)  # advantage_norm:

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
            # new_log_prob = self.act.compute__log_prob(state, action)
            # new_log_prob, new_value1, new_value2 = self.act.compute__log_prob(state, action)
            new_log_prob, new_value1, new_value2 = self.act_target.compute__log_prob(state, action)

            critic_loss = (self.criterion(new_value1, old_value) +
                           self.criterion(new_value2, old_value)) / (old_value.std() * 2 + 1e-6)
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
            loss_a_sum += actor_loss.item()  # just for print

            # self.act_optimizer.zero_grad()
            # actor_loss.backward()
            # self.act_optimizer.step()

            united_loss = critic_loss + actor_loss
            self.act_optimizer.zero_grad()
            united_loss.backward()
            self.act_optimizer.step()

            if self.update_counter >= self.update_freq:
                self.update_counter = 0
                # self.soft_target_update(self.act_target, self.act)  # soft target update
                self.act_target.load_state_dict(self.act.state_dict())

        loss_a_avg = loss_a_sum / sample_times
        loss_c_avg = loss_c_sum / sample_times
        return loss_a_avg, loss_c_avg

    def select_actions(self, states, explore_noise=0.0):  # CPU array to GPU tensor to CPU array
        states = torch.tensor(states, dtype=torch.float32, device=self.device)

        if explore_noise == 0.0:
            a_mean = self.act(states)
            a_mean = a_mean.cpu().data.numpy()
            return a_mean
        else:
            a_noise, log_prob = self.act.get__a__log_prob(states)
            a_noise = a_noise.cpu().data.numpy()
            log_prob = log_prob.cpu().data.numpy()
            return a_noise, log_prob


class AgentAdvDis:  # Advantage PPO for discrete action space
    def __init__(self, state_dim, action_dim, net_dim):
        self.learning_rate = 2e-4  # learning rate of actor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''network'''
        self.act = ActCriAdv(state_dim, action_dim, net_dim).to(self.device)
        self.act.train()
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate, )  # betas=(0.5, 0.99))

        self.act_target = ActCriAdv(state_dim, action_dim, net_dim).to(self.device)
        self.act_target.eval()
        self.act_target.load_state_dict(self.act.state_dict())

        # self.cri = CriticAdv(state_dim, net_dim).to(self.device)
        # self.cri.train()
        # self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate, )  # betas=(0.5, 0.99))

        self.criterion = nn.SmoothL1Loss()

        self.update_freq = 2 ** 7  # delay update frequency, for hard target update
        self.update_counter = 0  # delay update counter

    def update_buffer_ppo(self, env, max_step, max_memo, max_action, reward_scale, gamma):
        self.act.eval()

        # collect tuple (reward, mask, state, action, log_prob, )
        # PPO is an on policy RL algorithm.
        buffer = BufferTuplePPO()

        rewards = list()
        steps = list()

        step_counter = 0
        while step_counter < max_memo:
            state = env.reset()
            # state = running_state(state)  # if state_norm:
            reward_sum = 0
            step_sum = 0

            for step_sum in range(max_step):
                action, log_prob = self.select_actions((state,), explore_noise=True)
                action = action[0]
                log_prob = log_prob[0]

                next_state, reward, done, _ = env.step(action * max_action)
                reward_sum += reward

                # next_state = running_state(next_state)  # if state_norm:
                mask = 0.0 if done else gamma

                reward_ = reward * reward_scale
                buffer.push(reward_, mask, state, action, log_prob, )

                if done:
                    break

                state = next_state

            rewards.append(reward_sum)
            steps.append(step_sum)

            step_counter += step_sum
        return rewards, steps, buffer

    def update_parameters_gae(self, buffer, batch_size, repeat_times):
        self.act.train()
        # self.cri.train()
        clip = 0.25  # ratio.clamp(1 - clip, 1 + clip)
        lambda_adv = 0.98  # Could be 0.97 or 0.98. Wait but why? Cannot use 0.99.
        lambda_entropy = 0.01  # could be 0.02. Set greater when speed up training in a easy task.
        # repeat_times = 2 ** 4 # could be 2 ** 3. Set smaller when speed up training in a easy task.

        loss_a_sum = 0.0  # just for print
        loss_c_sum = 0.0  # just for print

        '''the batch for training'''
        max_memo = len(buffer)
        all_batch = buffer.sample()
        all_reward, all_mask, all_state, all_action, all_log_prob = [
            torch.tensor(ary, dtype=torch.float32, device=self.device)
            for ary in (all_batch.reward, all_batch.mask, all_batch.state, all_batch.action, all_batch.log_prob,)
        ]
        # with torch.no_grad():
        # all__new_v = self.cri(all_state).detach_()
        all__new_v = self.act.get__q_min(all_state).detach_()

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
            all__delta[i] = all_reward[i] + all_mask[i] * prev_new_v - all__new_v[i]
            all__old_v[i] = all_reward[i] + all_mask[i] * prev_old_v
            all__adv_v[i] = all__delta[i] + all_mask[i] * prev_adv_v * lambda_adv

            prev_old_v = all__old_v[i]
            prev_new_v = all__new_v[i]
            prev_adv_v = all__adv_v[i]

        all__adv_v = (all__adv_v - all__adv_v.mean()) / (all__adv_v.std() + 1e-6)  # advantage_norm:

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
            # new_log_prob = self.act.compute__log_prob(state, action)
            # new_log_prob, new_value1, new_value2 = self.act.compute__log_prob(state, action)
            new_log_prob, new_value1, new_value2 = self.act_target.compute__log_prob(state, action)

            critic_loss = (self.criterion(new_value1, old_value) +
                           self.criterion(new_value2, old_value)) / (old_value.std() * 2 + 1e-6)
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
            loss_a_sum += actor_loss.item()  # just for print

            # self.act_optimizer.zero_grad()
            # actor_loss.backward()
            # self.act_optimizer.step()

            united_loss = critic_loss + actor_loss
            self.act_optimizer.zero_grad()
            united_loss.backward()
            self.act_optimizer.step()

            if self.update_counter >= self.update_freq:
                self.update_counter = 0
                # self.soft_target_update(self.act_target, self.act)  # soft target update
                self.act_target.load_state_dict(self.act.state_dict())

        loss_a_avg = loss_a_sum / sample_times
        loss_c_avg = loss_c_sum / sample_times
        return loss_a_avg, loss_c_avg

    def select_actions(self, states, explore_noise=0.0):  # CPU array to GPU tensor to CPU array
        states = torch.tensor(states, dtype=torch.float32, device=self.device)

        if explore_noise == 0.0:
            a_mean = self.act(states)
            a_mean = a_mean.cpu().data.numpy()
            return a_mean
        else:
            a_noise, log_prob = self.act.get__a__log_prob(states)
            a_noise = a_noise.cpu().data.numpy()
            log_prob = log_prob.cpu().data.numpy()
            return a_noise, log_prob


def run__ppo(gpu_id, cwd):
    # from AgentZoo import AgentPPO
    from AgentZoo import AgentAdv
    args = Arguments(AgentAdv)

    args.gpu_id = gpu_id
    args.max_memo = 2 ** 12
    args.batch_size = 2 ** 9
    args.repeat_times = 2 ** 4
    args.net_dim = 2 ** 8
    args.gamma = 0.99



    # args.env_name = "LunarLanderContinuous-v2"
    # args.cwd = './{}/LL_{}'.format(cwd, gpu_id)
    # args.init_for_training()
    # while not train_agent__on_policy(**vars(args)):
    #     args.random_seed += 42
    #
    # args.env_name = "BipedalWalker-v3"
    # args.cwd = './{}/BW_{}'.format(cwd, gpu_id)
    # args.init_for_training()
    # while not train_agent__on_policy(**vars(args)):
    #     args.random_seed += 42


if __name__ == '__main__':
    # run__ppo(gpu_id=sys.argv[-1][-4], cwd='AC_InterAdv')
    run__multi_process(run__ppo, gpu_tuple=((0, 1), (2, 3))[0], cwd='AC_InterAdvD')
