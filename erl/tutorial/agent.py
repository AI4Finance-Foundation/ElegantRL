from copy import deepcopy
import torch
import numpy as np
import numpy.random as rd
from eRL.tutorial.net import QNet, QNetTwin
from eRL.tutorial.net import Actor, ActorSAC, ActorPPO
from eRL.tutorial.net import Critic, CriticAdv, CriticTwin


class AgentDQN:
    def __init__(self, net_dim, state_dim, action_dim, learning_rate=1e-4):
        self.explore_rate = 0.1  # the probability of choosing action randomly in epsilon-greedy
        self.action_dim = action_dim

        self.state = None  # set for self.update_buffer(), initialize self.state before training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.act = QNet(net_dim, state_dim, action_dim).to(self.device)
        self.act_target = deepcopy(self.act)

        self.criterion = torch.torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.act.parameters(), lr=learning_rate)

    def select_actions(self, states):  # for discrete action space
        if rd.rand() < self.explore_rate:  # epsilon-greedy
            a_int = rd.randint(self.action_dim, size=(len(states),))  # choosing action randomly
        else:
            states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
            actions = self.act(states)
            a_int = actions.argmax(dim=1).detach().cpu().numpy()
        return a_int

    def update_buffer(self, env, buffer, max_step, reward_scale, gamma):
        for _ in range(max_step):
            action = self.select_actions((self.state,))[0]
            next_s, reward, done, _ = env.step(action)

            other = (reward * reward_scale, 0.0 if done else gamma, action)  # action is an int
            buffer.append_memo(self.state, other)
            self.state = env.reset() if done else next_s
        return max_step

    def update_policy(self, buffer, max_step, batch_size, repeat_times):
        buffer.update__now_len__before_sample()

        next_q = obj_critic = None
        for _ in range(int(max_step * repeat_times)):
            with torch.no_grad():
                reward, mask, action, state, next_s = buffer.random_sample(batch_size)  # next_state
                next_q = self.act_target(next_s).max(dim=1, keepdim=True)[0]
                q_label = reward + mask * next_q
            q_eval = self.act(state).gather(1, action.type(torch.long))
            obj_critic = self.criterion(q_eval, q_label)

            self.optimizer.zero_grad()
            obj_critic.backward()
            self.optimizer.step()
            soft_target_update(self.act_target, self.act, tau=5e-3)
        return next_q.mean().item(), obj_critic.item()  #


class AgentDoubleDQN(AgentDQN):
    def __init__(self, net_dim, state_dim, action_dim, learning_rate=1e-4):
        super().__init__(net_dim, state_dim, action_dim, learning_rate)
        self.explore_rate = 0.25  # epsilon-greedy, the rate of choosing random action
        self.softmax = torch.nn.Softmax(dim=1)
        self.action_dim = action_dim

        self.act = QNetTwin(net_dim, state_dim, action_dim).to(self.device)
        self.act_target = deepcopy(self.act)

        self.criterion = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.act.parameters(), lr=learning_rate)

    def select_actions(self, states):  # for discrete action space
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = self.act(states)
        if rd.rand() < self.explore_rate:  # epsilon-greedy
            a_prob_l = self.softmax(actions).detach().cpu().numpy()  # choose action according to Q value
            a_int = [rd.choice(self.action_dim, p=a_prob) for a_prob in a_prob_l]
        else:
            a_int = actions.argmax(dim=1).detach().cpu().numpy()
        return a_int

    def update_policy(self, buffer, max_step, batch_size, repeat_times):
        buffer.update__now_len__before_sample()

        next_q = obj_critic = None
        for _ in range(int(max_step * repeat_times)):
            with torch.no_grad():
                reward, mask, action, state, next_s = buffer.random_sample(batch_size)
                next_q = self.act_target(next_s).max(dim=1, keepdim=True)[0]
                q_label = reward + mask * next_q
            action = action.type(torch.long)
            q_eval1, q_eval2 = [qs.gather(1, action) for qs in self.act.get__q1_q2(state)]
            obj_critic = self.criterion(q_eval1, q_label) + self.criterion(q_eval2, q_label)

            self.optimizer.zero_grad()
            obj_critic.backward()
            self.optimizer.step()
            soft_target_update(self.act_target, self.act)
        return next_q.mean().item(), obj_critic.item() / 2


class AgentBase:
    def __init__(self):
        self.state = None  # set for self.update_buffer(), initialize self.state before training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def select_actions(self, states):  # states = (state, ...)
        return (None,)  # -1 < action < +1

    def update_buffer(self, env, buffer, max_step, reward_scale, gamma):
        for _ in range(max_step):
            action = self.select_actions((self.state,))[0]
            next_s, reward, done, _ = env.step(action)
            other = (reward * reward_scale, 0.0 if done else gamma, *action)
            buffer.append_memo(self.state, other)
            self.state = env.reset() if done else next_s
        return max_step


class AgentDDPG(AgentBase):
    def __init__(self, net_dim, state_dim, action_dim, learning_rate=1e-4):
        super().__init__()
        self.explore_noise = 0.05  # explore noise of action

        self.act = Actor(net_dim, state_dim, action_dim).to(self.device)
        self.act_target = deepcopy(self.act)
        self.cri = Critic(net_dim, state_dim, action_dim).to(self.device)
        self.cri_target = deepcopy(self.cri)

        self.criterion = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam([{'params': self.act.parameters(), 'lr': learning_rate},
                                           {'params': self.cri.parameters(), 'lr': learning_rate}])

    def select_actions(self, states):  # states = (state, ...)
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = self.act(states)
        actions = (actions + torch.randn_like(actions) * self.explore_noise).clamp(-1, 1)
        return actions.detach().cpu().numpy()

    def update_policy(self, buffer, max_step, batch_size, repeat_times):
        buffer.update__now_len__before_sample()
        obj_critic = obj_actor = None  # just for print return
        for _ in range(int(max_step * repeat_times)):
            with torch.no_grad():
                reward, mask, action, state, next_s = buffer.random_sample(batch_size)
                next_q = self.cri_target(next_s, self.act_target(next_s))
                q_label = reward + mask * next_q
            q_value = self.cri(state, action)
            obj_critic = self.criterion(q_value, q_label)

            q_value_pg = self.act(state)  # policy gradient
            obj_actor = -self.cri_target(state, q_value_pg).mean()

            obj_united = obj_actor + obj_critic  # objective
            self.optimizer.zero_grad()
            obj_united.backward()
            self.optimizer.step()

            soft_target_update(self.cri_target, self.cri)
            soft_target_update(self.act_target, self.act)
        return obj_actor.item(), obj_critic.item()


class AgentTD3(AgentDDPG):
    def __init__(self, net_dim, state_dim, action_dim, learning_rate=1e-4):
        super().__init__(net_dim, state_dim, action_dim, learning_rate)
        self.explore_noise = 0.1  # standard deviation of explore noise
        self.policy_noise = 0.2  # standard deviation of policy noise
        self.update_freq = 2  # delay update frequency, for soft target update

        self.cri = CriticTwin(net_dim, state_dim, action_dim).to(self.device)
        self.cri_target = deepcopy(self.cri)

        self.optimizer = torch.optim.Adam([{'params': self.act.parameters(), 'lr': learning_rate},
                                           {'params': self.cri.parameters(), 'lr': learning_rate}])

    def update_policy(self, buffer, max_step, batch_size, repeat_times):
        buffer.update__now_len__before_sample()

        obj_critic = obj_actor = None
        for i in range(int(max_step * repeat_times)):
            with torch.no_grad():
                reward, mask, action, state, next_s = buffer.random_sample(batch_size)
                next_a = self.act_target.get_action(next_s, self.policy_noise)  # policy noise
                next_q = torch.min(*self.cri_target.get__q1_q2(next_s, next_a))  # twin critics
                q_label = reward + mask * next_q
            q1, q2 = self.cri.get__q1_q2(state, action)
            obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)  # twin critics

            q_value_pg = self.act(state)  # policy gradient
            obj_actor = -self.cri_target(state, q_value_pg).mean()

            obj_united = obj_actor + obj_critic  # objective
            self.optimizer.zero_grad()
            obj_united.backward()
            self.optimizer.step()

            if i % self.update_freq == 0:  # delay update
                soft_target_update(self.cri_target, self.cri)
                soft_target_update(self.act_target, self.act)
        return obj_actor.item(), obj_critic.item() / 2


class AgentSAC(AgentBase):
    def __init__(self, net_dim, state_dim, action_dim, learning_rate=1e-4):
        super().__init__()
        self.target_entropy = np.log(action_dim)
        self.alpha_log = torch.tensor((-np.log(action_dim) * np.e,), dtype=torch.float32,
                                      requires_grad=True, device=self.device)  # trainable parameter

        self.act = ActorSAC(net_dim, state_dim, action_dim).to(self.device)
        self.act_target = deepcopy(self.act)
        self.cri = CriticTwin(int(net_dim * 1.25), state_dim, action_dim, ).to(self.device)
        self.cri_target = deepcopy(self.cri)

        self.criterion = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam([{'params': self.act.parameters(), 'lr': learning_rate * 0.75},
                                           {'params': self.cri.parameters(), 'lr': learning_rate * 1.25},
                                           {'params': (self.alpha_log,), 'lr': learning_rate}])

    def select_actions(self, states):  # states = (state, ...)
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = self.act.get_action(states)
        return actions.detach().cpu().numpy()

    def update_policy(self, buffer, max_step, batch_size, repeat_times):
        buffer.update__now_len__before_sample()

        alpha = self.alpha_log.exp().detach()
        obj_actor = obj_critic = None
        for _ in range(int(max_step * repeat_times)):
            with torch.no_grad():
                reward, mask, action, state, next_s = buffer.random_sample(batch_size)
                next_a, next_log_prob = self.act_target.get__action__log_prob(next_s)
                next_q = torch.min(*self.cri_target.get__q1_q2(next_s, next_a))
                q_label = reward + mask * (next_q + next_log_prob * alpha)
            q1, q2 = self.cri.get__q1_q2(state, action)
            obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)

            action_pg, log_prob = self.act.get__action__log_prob(state)  # policy gradient
            obj_alpha = (self.alpha_log * (log_prob - self.target_entropy).detach()).mean()

            alpha = self.alpha_log.exp().detach()
            with torch.no_grad():
                self.alpha_log[:] = self.alpha_log.clamp(-16, 2)
            obj_actor = -(torch.min(*self.cri_target.get__q1_q2(state, action_pg)) + log_prob * alpha).mean()

            obj_united = obj_critic + obj_alpha + obj_actor
            self.optimizer.zero_grad()
            obj_united.backward()
            self.optimizer.step()

            soft_target_update(self.cri_target, self.cri)
            soft_target_update(self.act_target, self.act)
        # return obj_actor.item(), obj_critic.item()
        return alpha.item(), obj_critic.item()


class AgentPPO(AgentBase):
    def __init__(self, net_dim, state_dim, action_dim, learning_rate=1e-4):
        super().__init__()
        self.clip = 0.25  # ratio.clamp(1 - clip, 1 + clip)
        self.lambda_entropy = 0.01  # larger lambda_entropy means more exploration

        self.act = ActorPPO(net_dim, state_dim, action_dim).to(self.device)
        self.cri = CriticAdv(state_dim, net_dim).to(self.device)

        self.criterion = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam([{'params': self.act.parameters(), 'lr': learning_rate},
                                           {'params': self.cri.parameters(), 'lr': learning_rate}])

    def select_actions(self, states):  # states = (state, ...)
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        a_noise, noise = self.act.get__action_noise(states)
        return a_noise.detach().cpu().numpy(), noise.detach().cpu().numpy()

    def update_buffer(self, env, buffer, max_step, reward_scale, gamma):
        buffer.empty_memories__before_explore()

        step_counter = 0
        target_step = buffer.max_len - max_step
        while step_counter < target_step:
            state = env.reset()
            for _ in range(max_step):
                action, noise = self.select_actions((state,))
                action = action[0]
                noise = noise[0]

                next_state, reward, done, _ = env.step(np.tanh(action))
                step_counter += 1

                other = (reward * reward_scale, 0.0 if done else gamma, *action, *noise)
                buffer.append_memo(state, other)
                if done:
                    break
                state = next_state
        return step_counter

    def update_policy(self, buffer, _max_step, batch_size, repeat_times=8):
        buffer.update__now_len__before_sample()
        max_memo = buffer.now_len

        with torch.no_grad():  # Trajectory using reverse reward
            buf_reward, buf_mask, buf_action, buf_noise, buf_state = buffer.sample_for_ppo()

            bs = 2 ** 10  # set a smaller 'bs: batch size' when out of GPU memory.
            buf_value = torch.cat([self.cri(buf_state[i:i + bs]) for i in range(0, buf_state.size(0), bs)], dim=0)
            buf_log_prob = -(buf_noise.pow(2).__mul__(0.5) + self.act.a_std_log + self.act.sqrt_2pi_log).sum(1)

            buf_r_sum = torch.empty(max_memo, dtype=torch.float32, device=self.device)  # reward sum
            pre_r_sum = 0  # reward sum of previous step
            for i in range(max_memo - 1, -1, -1):
                buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
                pre_r_sum = buf_r_sum[i]
            buf_advantage = buf_r_sum - (buf_mask * buf_value).squeeze(1)
            buf_advantage = buf_advantage / (buf_advantage.std() + 1e-5)

            del buf_reward, buf_mask, buf_noise

        obj_actor = obj_critic = None
        for _ in range(int(repeat_times * max_memo / batch_size)):  # PPO: Surrogate objective of Trust Region
            indices = torch.randint(max_memo, size=(batch_size,), requires_grad=False, device=self.device)
            state = buf_state[indices]
            action = buf_action[indices]
            r_sum = buf_r_sum[indices]
            log_prob = buf_log_prob[indices]
            advantage = buf_advantage[indices]

            new_log_prob = self.act.compute__log_prob(state, action)  # it is obj_actor
            ratio = (new_log_prob - log_prob).exp()
            obj_surrogate1 = advantage * ratio
            obj_surrogate2 = advantage * ratio.clamp(1 - self.clip, 1 + self.clip)
            obj_surrogate = -torch.min(obj_surrogate1, obj_surrogate2).mean()
            obj_entropy = (new_log_prob.exp() * new_log_prob).mean()  # policy entropy
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy

            value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, r_sum)

            obj_united = obj_actor + obj_critic / (r_sum.std() + 1e-5)
            self.optimizer.zero_grad()
            obj_united.backward()
            self.optimizer.step()

        return obj_actor.item(), obj_critic.item()


class AgentGaePPO(AgentPPO):
    def __init__(self, net_dim, state_dim, action_dim, learning_rate=1e-4):
        super().__init__(net_dim, state_dim, action_dim, learning_rate)
        self.clip = 0.25  # ratio.clamp(1 - clip, 1 + clip)
        self.lambda_entropy = 0.01  # could be 0.02
        self.lambda_adv = 0.98  # could be 0.95~0.99

    def update_policy(self, buffer, _max_step, batch_size, repeat_times=8):
        buffer.update__now_len__before_sample()
        max_memo = buffer.now_len

        '''Trajectory using GAE (Generalized Advantage Estimation. ICLR.2016.)'''
        with torch.no_grad():
            buf_reward, buf_mask, buf_action, buf_noise, buf_state = buffer.sample_for_ppo()

            bs = 2 ** 10  # set a smaller 'bs: batch size' when out of GPU memory.
            buf_value = torch.cat([self.cri(buf_state[i:i + bs]) for i in range(0, buf_state.size(0), bs)], dim=0)
            buf_log_prob = -(buf_noise.pow(2).__mul__(0.5) + self.act.a_std_log + self.act.sqrt_2pi_log).sum(1)

            buf_r_sum = torch.empty(max_memo, dtype=torch.float32, device=self.device)  # old policy value
            buf_advantage = torch.empty(max_memo, dtype=torch.float32, device=self.device)  # advantage value

            pre_r_sum = 0  # reward sum of previous step
            pre_advantage = 0  # advantage value of previous step
            for i in range(max_memo - 1, -1, -1):
                buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
                pre_r_sum = buf_r_sum[i]

                buf_advantage[i] = buf_reward[i] + buf_mask[i] * pre_advantage - buf_value[i]
                pre_advantage = buf_value[i] + buf_advantage[i] * self.lambda_adv

            buf_advantage = buf_advantage / (buf_advantage.std() + 1e-5)
            del buf_reward, buf_mask, buf_noise

        '''PPO: Clipped Surrogate objective of Trust Region'''
        obj_critic = None
        for _ in range(int(repeat_times * max_memo / batch_size)):
            indices = torch.randint(max_memo, size=(batch_size,), device=self.device)

            state = buf_state[indices]
            action = buf_action[indices]
            advantage = buf_advantage[indices]
            old_value = buf_r_sum[indices]
            old_log_prob = buf_log_prob[indices]

            new_log_prob = self.act.compute__log_prob(state, action)  # it is obj_actor
            ratio = (new_log_prob - old_log_prob).exp()
            obj_surrogate1 = advantage * ratio
            obj_surrogate2 = advantage * ratio.clamp(1 - self.clip, 1 + self.clip)
            obj_surrogate = -torch.min(obj_surrogate1, obj_surrogate2).mean()
            obj_entropy = (new_log_prob.exp() * new_log_prob).mean()  # policy entropy
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy

            new_value = self.cri(state).squeeze(1)
            obj_critic = self.criterion(new_value, old_value)

            obj_united = obj_actor + obj_critic / (old_value.std() + 1e-5)
            self.optimizer.zero_grad()
            obj_united.backward()
            self.optimizer.step()

        return self.act.a_std_log.mean().item(), obj_critic.item()


def soft_target_update(target, current, tau=5e-3):
    for target_param, param in zip(target.parameters(), current.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
