import os
import sys
import time

import torch
import torch.nn as nn
import numpy as np
import numpy.random as rd

"""AgentNet"""


class ActorDPG(nn.Module):  # Deterministic Policy Gradient
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, action_dim), )

    def forward(self, state):
        return self.net(state).tanh()  # action

    def get__a_noisy(self, state, a_std):  # action_std
        action = self.net(state).tanh()
        return (action + torch.randn_like(action) * a_std).clamp(-1.0, 1.0)


class ActorSAC(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()
        self.net__state = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                        nn.Linear(mid_dim, mid_dim), nn.ReLU(), )
        self.net_action = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                        nn.Linear(mid_dim, action_dim), )
        self.net__a_std = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                        nn.Linear(mid_dim, action_dim), )

        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))  # a constant
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, state):
        tmp = self.net__state(state)  # temp tensor
        return self.net_action(tmp).tanh()  # action

    def get__a_noisy(self, state):
        t_tmp = self.net__state(state)
        a_avg = self.net_action(t_tmp)
        a_std = self.net__a_std(t_tmp).clamp(-16, 2).exp()
        return torch.normal(a_avg, a_std).tanh()  # re-parameterize

    def get__a__log_prob(self, s):
        t_tmp = self.net__state(s)
        a_avg = self.net_action(t_tmp)
        a_std_log = self.net__a_std(t_tmp).clamp(-16, 2)
        a_std = a_std_log.exp()

        noise = torch.randn_like(a_avg, requires_grad=True, device=self.device)
        a_tan = (a_avg + a_std * noise).tanh()  # action.tanh()
        log_prob = a_std_log + self.sqrt_2pi_log + noise.pow(2).__mul__(0.5) + (-a_tan.pow(2) + 1.000001).log()
        return a_tan, log_prob.sum(1, keepdim=True)


class ActorPPO(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, action_dim), )
        layer_norm(self.net[-1], std=0.1)  # output layer of action

        self.a_std_log = nn.Parameter(torch.zeros((1, action_dim)) - 0.5, requires_grad=True)  # trainable parameter
        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))  # a constant

    def forward(self, state):
        return self.net(state).tanh()  # action

    def get__a_noisy__noise(self, state):
        a_avg = self.net(state)
        a_std = self.a_std_log.exp()

        noise = torch.randn_like(a_avg)
        a_noisy = a_avg + noise * a_std
        return a_noisy, noise

    def compute__log_prob(self, state, a_noise):
        a_avg = self.net(state)
        a_std = self.a_std_log.exp()

        delta = ((a_avg - a_noise) / a_std).pow(2).__mul__(0.5)
        log_prob = -(self.a_std_log + self.sqrt_2pi_log + delta)
        return log_prob.sum(1)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()
        self.net__value = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                        nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                        nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                        nn.Linear(mid_dim, 1), )

    def forward(self, state, action):
        tmp = torch.cat((state, action), dim=1)
        return self.net__value(tmp)  # q value


class CriticTwin(nn.Module):  # shared parameter
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()
        self.net_sa = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, mid_dim), nn.ReLU(), )  # concat(state, action)
        self.net_q1 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                    nn.Linear(mid_dim, 1), )  # q1 value
        self.net_q2 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                    nn.Linear(mid_dim, 1), )  # q2 value

    def forward(self, state, action):
        tmp = self.net_sa(torch.cat((state, action), dim=1))
        return self.net_q1(tmp), self.net_q2(tmp)  # q1 value, q2 value


class CriticAdv(nn.Module):  # 2021-02-02
    def __init__(self, state_dim, mid_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, 1), )
        layer_norm(self.net[-1], std=1.0)  # output layer of action

    def forward(self, state):
        return self.net(state)  # q value


class QNet(nn.Module):  # class AgentQLearning
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, action_dim), )

    def forward(self, state):
        return self.net(state)  # q value


class QNetTwin(nn.Module):  # shared parameter
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()
        self.net__s = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, mid_dim), nn.ReLU(), )  # state
        self.net_q1 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, action_dim), )  # q1 value
        self.net_q2 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, action_dim), )  # q2 value

    def forward(self, state):
        tmp = self.net__s(state)
        return self.net_q1(tmp)  # q1 value

    def get__q1_q2(self, state):
        tmp = self.net__s(state)
        return self.net_q1(tmp), self.net_q2(tmp)  # q1 q2 value


class QNetDuel(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()
        self.net__state = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                        nn.Linear(mid_dim, mid_dim), nn.ReLU(), )
        self.net__value = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                        nn.Linear(mid_dim, 1), )  # q value
        self.net__adv_v = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                        nn.Linear(mid_dim, action_dim), )  # advantage function value

    def forward(self, state):
        t_tmp = self.net__state(state)
        q_val = self.net__value(t_tmp)  # q value
        q_adv = self.net__adv_v(t_tmp)  # advantage function value
        return q_val + q_adv - q_adv.mean(dim=1, keepdim=True)  # dueling q value


class QNetDuelTwin(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()
        self.net__state = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                        nn.Linear(mid_dim, mid_dim), nn.ReLU(), )

        self.net_val1 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                      nn.Linear(mid_dim, 1), )
        self.net_val2 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                      nn.Linear(mid_dim, 1), )
        self.net_adv1 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                      nn.Linear(mid_dim, action_dim), )
        self.net_adv2 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                      nn.Linear(mid_dim, action_dim), )

    def forward(self, state):
        t_tmp = self.net__state(state)
        q_val = self.net_val1(t_tmp)
        q_adv = self.net_adv1(t_tmp)
        return q_val + q_adv - q_adv.mean(dim=1, keepdim=True)  # single dueling q value

    def get__q1_q2(self, state):
        tmp = self.net__state(state)

        val1 = self.net_val1(tmp)
        adv1 = self.net_adv1(tmp)
        q1 = val1 + adv1 - adv1.mean(dim=1, keepdim=True)

        val2 = self.net_val2(tmp)
        adv2 = self.net_adv2(tmp)
        q2 = val2 + adv2 - adv2.mean(dim=1, keepdim=True)
        return q1, q2


class NnnReshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.view((x.size(0),) + self.args)


def layer_norm(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)


"""AgentZoo"""


class AgentBase:  # DDPG-style.2016 (Deep Deterministic Policy Gradient)
    def __init__(self, ):
        self.state = self.action = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.act = None
        self.cri = None
        self.criterion = None
        self.optimizer = None

        self.obj_a = 0.0
        self.obj_c = (-np.log(0.5)) ** 0.5

    def select_actions(self, states):  # states = (state, )
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = self.act(states)
        return actions.detach().cpu().numpy()

    def update_buffer(self, env, buffer, max_step, reward_scale, gamma):
        for _ in range(max_step):
            action = self.select_actions((self.state,))[0]
            next_state, reward, done, _ = env.step(action)
            buffer.append_memo((reward * reward_scale, 0.0 if done else gamma, *self.state, *action, *next_state))
            self.state = env.reset() if done else next_state
        return max_step

    def update_buffer__pipe(self, pipes, buffer, max_step):
        env_num = len(pipes)
        env_num2 = env_num // 2

        trajectories = [list() for _ in range(env_num)]
        for _ in range(max_step // env_num):
            for i_beg, i_end in ((0, env_num2), (env_num2, env_num)):
                for i in range(i_beg, i_end):
                    reward, mask, next_state = pipes[i].recv()
                    trajectories[i].append([reward, mask, *self.state[i], *self.action[i], *next_state])
                    self.state[i] = next_state

                self.action[i_beg:i_end] = self.select_actions(self.state[i_beg:i_end])
                for i in range(i_beg, i_end):
                    pipes[i].send(self.action[i])  # pipes action

        steps_sum = 0
        for trajectory in trajectories:
            steps_sum += len(trajectory)
            buffer.extend_memo(memo_tuple=trajectory)
        return steps_sum

    def update_policy(self, buffer, max_step, batch_size, repeat_times):
        buffer.update__now_len__before_sample()

        for i in range(int(max_step * repeat_times)):
            with torch.no_grad():
                buffer.random_sample(batch_size)

            self.optimizer.zero_grad()
            # united_obj.backward()
            self.optimizer.step()

        self.obj_a = 0.0
        self.obj_c = 0.5

    def save_or_load_model(self, cwd, if_save):
        for net, name in ((self.act, 'act'), (self.cri, 'cri')):
            if net is None:
                continue

            save_path = f'{cwd}/{name}.pth'
            if if_save:
                torch.save(net.state_dict(), save_path)
            elif os.path.exists(save_path):
                net = torch.load(save_path, map_location=lambda storage, loc: storage)
                net.load_state_dict(net)
                print(f"Loaded act and cri: {cwd}")
            else:
                print(f"FileNotFound when load_model: {cwd}")


class AgentTD3(AgentBase):  # TD3.2018 (Twin Delay DDPG)
    def __init__(self, state_dim, action_dim, net_dim, learning_rate=1e-4):
        AgentBase.__init__(self)
        self.explore_noise = 0.1  # standard deviation of explore noise
        self.policy__noise = 0.2  # standard deviation of policy noise
        self.update_freq = 2  # set as 2 or 4 for soft target update

        self.act = ActorDPG(state_dim, action_dim, net_dim).to(self.device)
        self.act_target = ActorDPG(state_dim, action_dim, net_dim).to(self.device)
        self.act_target.load_state_dict(self.act.state_dict())

        cri_dim = int(net_dim * 1.25)
        self.cri = CriticTwin(state_dim, action_dim, cri_dim).to(self.device)
        self.cri_target = CriticTwin(state_dim, action_dim, cri_dim).to(self.device)
        self.cri_target.load_state_dict(self.cri.state_dict())

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam([
            {'params': self.act.parameters(), 'lr': learning_rate},
            {'params': self.cri.parameters(), 'lr': learning_rate},
        ], lr=learning_rate)

    def select_actions(self, states):
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = self.act.get__a_noisy(states, self.explore_noise)
        return actions.detach().cpu().numpy()

    def update_policy(self, buffer, max_step, batch_size, repeat_times):
        buffer.update__now_len__before_sample()
        """Contribution of DDPG (Deep Deterministic Policy Gradient)
        1. Policy Gradient with Deep network: DQN + DPG -> DDPG
           Q_value = reward + gamma * next_Q_value
           Q-learning -> DQN (Deep Q-learning): (discrete state space Q-table -> continuous state space Q-net)
           DQN + DPG -> DDPG: (discrete action space Q-net -> continuous action space Policy Gradient)
        2. experiment replay buffer for stabilizing training
        3. soft target update for stabilizing training

        Contribution of TD3 (TDDD, Twin Delay DDPG)
        1. twin critics (DoubleDQN -> TwinCritic, good idea)
        2. policy noise ('Deterministic Policy Gradient + policy noise' looks like Stochastic PG)
        3. delay update (I think it is not very useful)
        """

        k = 1.0 + buffer.now_len / buffer.max_len
        batch_size_ = int(batch_size * k)
        update_times = int(max_step * k * repeat_times)

        for i in range(update_times):
            with torch.no_grad():
                reward, mask, state, action, next_s = buffer.random_sample(batch_size_, )

                next_action = self.act_target.get__a_noisy(next_s, self.policy__noise)
                q_label = reward + mask * self.cri_target(next_s, next_action)

            q1, q2 = self.cri(state, action)
            cri_obj = self.criterion(q1, q_label) + self.criterion(q2, q_label)
            self.obj_c = 0.995 * self.obj_c + 0.005 * cri_obj.item() / 2
            lamb = np.exp(-self.obj_c ** 2)

            act_obj = -self.cri_target(state, self.act(state)).mean()
            self.obj_a = 0.995 * self.obj_a + 0.005 * q_label.mean().item()

            united_obj = cri_obj + act_obj * lamb
            self.optimizer.zero_grad()
            united_obj.backward()
            self.optimizer.step()

            if i % self.update_freq == 0:
                soft_target_update(self.cri_target, self.cri)
                soft_target_update(self.act_target, self.act)


class AgentModSAC(AgentBase):
    def __init__(self, state_dim, action_dim, net_dim, learning_rate=1e-4):
        AgentBase.__init__(self)
        self.target_entropy = np.log(action_dim)
        self.alpha_log = torch.tensor((-np.log(action_dim) * np.e,), requires_grad=True,
                                      dtype=torch.float32, device=self.device)

        self.act = ActorSAC(state_dim, action_dim, net_dim).to(self.device)
        self.act_target = ActorSAC(state_dim, action_dim, net_dim).to(self.device)
        self.act_target.load_state_dict(self.act.state_dict())

        self.cri = CriticTwin(state_dim, action_dim, int(net_dim * 1.25)).to(self.device)
        self.cri_target = CriticTwin(state_dim, action_dim, int(net_dim * 1.25)).to(self.device)
        self.cri_target.load_state_dict(self.cri.state_dict())

        self.criterion = nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam([
            {'params': self.act.parameters(), 'lr': learning_rate},
            {'params': self.cri.parameters(), 'lr': learning_rate},
            {'params': (self.alpha_log,), 'lr': learning_rate},
        ], lr=learning_rate)

    def select_actions(self, states):
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = self.act.get__a_noisy(states)
        return actions.detach().cpu().numpy()

    def update_policy(self, buffer, max_step, batch_size, repeat_times):
        buffer.update__now_len__before_sample()
        """Contribution of SAC (Soft Actor-Critic with maximum entropy)
        1. maximum entropy (Soft Q-learning -> Soft Actor-Critic, good idea)
        2. auto alpha (automating entropy adjustment on temperature parameter alpha for maximum entropy)
        3. SAC use TD3's TwinCritics too

        Modify content of ModSAC
        1. Reliable Lambda is calculated based on Critic's loss function value.
        2. Increasing batch_size and update_times
        3. Auto-TTUR updates parameter in non-integer times.
        4. net_dim of critic is slightly larger than actor.
        """
        k = 1.0 + buffer.now_len / buffer.max_len
        batch_size_ = int(batch_size * k)
        train_steps = int(max_step * k * repeat_times)

        alpha = self.alpha_log.exp().detach()
        update_a = 0
        for update_c in range(1, train_steps):
            with torch.no_grad():
                reward, mask, state, action, next_s = buffer.random_sample(batch_size_)

                next_a_noise, next_log_prob = self.act_target.get__a__log_prob(next_s)
                q_label = reward + mask * (torch.min(*self.cri_target(next_s, next_a_noise)) + next_log_prob * alpha)

            q1, q2 = self.cri(state, action)
            cri_obj = self.criterion(q1, q_label) + self.criterion(q2, q_label)
            self.obj_c = 0.995 * self.obj_c + 0.0025 * cri_obj.item()

            a_noise_pg, log_prob = self.act.get__a__log_prob(state)  # policy gradient

            alpha_obj = (self.alpha_log * (log_prob - self.target_entropy).detach()).mean()
            with torch.no_grad():
                self.alpha_log[:] = self.alpha_log.clamp(-16, 2)

            united_obj = cri_obj + alpha_obj

            lamb = np.exp(-self.obj_c ** 2)
            if_update_a = update_a / update_c < 1 / (2 - lamb)

            if if_update_a:  # auto TTUR
                update_a += 1

                alpha = self.alpha_log.exp().detach()
                act_obj = -(torch.min(*self.cri_target(state, a_noise_pg)) + log_prob * alpha).mean()
                self.obj_a = 0.995 * self.obj_a + 0.005 * q_label.mean().item()
                united_obj += act_obj

            self.optimizer.zero_grad()
            united_obj.backward()
            self.optimizer.step()

            soft_target_update(self.cri_target, self.cri)
            if if_update_a:
                soft_target_update(self.act_target, self.act)


class AgentGaePPO(AgentBase):
    def __init__(self, state_dim, action_dim, net_dim, learning_rate=1e-4):
        AgentBase.__init__(self)

        self.act = ActorPPO(state_dim, action_dim, net_dim).to(self.device)
        self.cri = CriticAdv(state_dim, net_dim).to(self.device)

        self.criterion = nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam([
            {'params': self.act.parameters(), 'lr': learning_rate},
            {'params': self.cri.parameters(), 'lr': learning_rate},
        ], lr=learning_rate)

    def select_actions(self, states):
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)  # plan to detach() here
        a_noise, noise = self.act.get__a_noisy__noise(states)
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

                buffer.append_memo((reward * reward_scale, 0.0 if done else gamma, *state, *action, *noise))

                if done:
                    break
                state = next_state
        return step_counter

    def update_buffer__pipe(self, pipes, buffer, max_step):
        buffer.empty_memories__before_explore()
        env_num = len(pipes)
        env_num2 = env_num // 2
        target_step = buffer.max_len - env_num * max_step

        action, noise = self.select_actions(self.state)
        for i in range(env_num):
            pipes[i].send(np.tanh(action[i]))  # pipes 1 send ppo # not elegant

        trajectories = [list() for _ in range(env_num)]
        for _ in range(target_step // env_num):
            for i_beg, i_end in ((0, env_num2), (env_num2, env_num)):
                for i in range(i_beg, i_end):
                    reward, mask, next_state = pipes[i].recv()
                    trajectories[i].append([reward, mask, *self.state[i], *action[i], *noise[i]])
                    self.state[i] = next_state

                action[i_beg:i_end], noise[i_beg:i_end] = self.select_actions(self.state[i_beg:i_end])
                for i in range(i_beg, i_end):
                    pipes[i].send(np.tanh(action[i]))  # pipes action # not elegant

        if_stops = [trajectory[-1][1] == 0 for trajectory in trajectories]  # trajectory[-1][1]==mask
        for _ in range(target_step // env_num):
            if all(if_stops):
                break

            for i_beg, i_end in ((0, env_num2), (env_num2, env_num)):
                for i in range(i_beg, i_end):
                    if if_stops[i]:
                        continue
                    reward, mask, next_state = pipes[i].recv()
                    trajectories[i].append([reward, mask, *self.state[i], *action[i], *noise[i]])
                    self.state[i] = next_state
                    if_stops[i] = mask == 0

                action[i_beg:i_end], noise[i_beg:i_end] = self.select_actions(self.state[i_beg:i_end])
                for i in range(i_beg, i_end):
                    if if_stops[i]:
                        continue
                    pipes[i].send(np.tanh(action[i]))  # pipes action # not elegant

        steps_sum = 0
        for trajectory in trajectories:
            steps_sum += len(trajectory)
            buffer.extend_memo(memo_tuple=trajectory)
        return steps_sum

    def update_policy(self, buffer, _max_step, batch_size, repeat_times=8):
        buffer.update__now_len__before_sample()
        """Contribution of PPO (Proximal Policy Optimization
        1. the surrogate objective of TRPO, PPO simplified calculation of TRPO
        2. use the advantage function of A3C (Asynchronous Advantage Actor-Critic)
        3. add GAE. ICLR 2016. Generalization Advantage Estimate and use trajectory to calculate Q value
        """

        clip = 0.25  # ratio.clamp(1 - clip, 1 + clip)
        lambda_adv = 0.98  # why 0.98? cannot use 0.99
        lambda_entropy = 0.01  # could be 0.02

        '''the batch for training'''
        max_memo = buffer.now_len
        all_reward, all_mask, all_state, all_action, all_noise = buffer.all_sample()

        all__new_v = list()
        all_log_prob = list()
        with torch.no_grad():
            b_size = 2 ** 10
            a_std_log__sqrt_2pi_log = self.act.a_std_log + self.act.sqrt_2pi_log
            for i in range(0, all_state.size()[0], b_size):
                new_v = self.cri(all_state[i:i + b_size])
                all__new_v.append(new_v)

                log_prob = -(all_noise[i:i + b_size].pow(2) / 2 + a_std_log__sqrt_2pi_log).sum(1)
                all_log_prob.append(log_prob)

            all__new_v = torch.cat(all__new_v, dim=0)
            all_log_prob = torch.cat(all_log_prob, dim=0)

        '''compute old_v (old policy value), adv_v (advantage value) 
        refer: GAE. ICLR 2016. Generalization Advantage Estimate. 
        https://arxiv.org/pdf/1506.02438.pdf
        '''
        all__delta = torch.empty(max_memo, dtype=torch.float32, device=self.device)
        all__old_v = torch.empty(max_memo, dtype=torch.float32, device=self.device)  # old policy value
        all__adv_v = torch.empty(max_memo, dtype=torch.float32, device=self.device)  # advantage value

        prev_old_v = 0  # old q value
        prev_new_v = 0  # new q value
        prev_adv_v = 0  # advantage q value
        for i in range(max_memo - 1, -1, -1):  # could be more elegant
            all__delta[i] = all_reward[i] + all_mask[i] * prev_new_v - all__new_v[i]
            all__old_v[i] = all_reward[i] + all_mask[i] * prev_old_v
            all__adv_v[i] = all__delta[i] + all_mask[i] * prev_adv_v * lambda_adv

            prev_old_v = all__old_v[i]
            prev_new_v = all__new_v[i]
            prev_adv_v = all__adv_v[i]

        all__adv_v = (all__adv_v - all__adv_v.mean()) / (all__adv_v.std() + 1e-5)  # advantage_norm:

        '''mini batch sample'''
        actor_obj = critic_obj = None
        for _ in range(int(repeat_times * max_memo / batch_size)):
            '''random sample'''
            indices = torch.randint(max_memo, size=(batch_size,), device=self.device)

            state = all_state[indices]
            action = all_action[indices]
            advantage = all__adv_v[indices]
            old_value = all__old_v[indices].unsqueeze(1)
            old_log_prob = all_log_prob[indices]

            """Adaptive KL Penalty Coefficient
            loss_KLPEN = surrogate_obj + value_obj * lambda_value + entropy_obj * lambda_entropy
            loss_KLPEN = (value_obj * lambda_value) + (surrogate_obj + entropy_obj * lambda_entropy)
            loss_KLPEN = (critic_obj) + (actor_obj)
            """

            new_log_prob = self.act.compute__log_prob(state, action)  # it is actor_obj
            new_value = self.cri(state)

            critic_obj = (self.criterion(new_value, old_value)) / (old_value.std() + 1e-5)

            ratio = torch.exp(new_log_prob - old_log_prob)
            surrogate_obj0 = advantage * ratio  # surrogate objective of TRPO
            surrogate_obj1 = advantage * ratio.clamp(1 - clip, 1 + clip)
            surrogate_obj = -torch.min(surrogate_obj0, surrogate_obj1).mean()
            loss_entropy = (torch.exp(new_log_prob) * new_log_prob).mean()  # policy entropy

            actor_obj = surrogate_obj + loss_entropy * lambda_entropy

            united_obj = actor_obj + critic_obj
            self.optimizer.zero_grad()
            united_obj.backward()
            self.optimizer.step()

        self.obj_a = actor_obj.item()
        self.obj_c = critic_obj.item()


class AgentBaseDQN:
    def __init__(self):
        self.state = self.action = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.explore_rate = 0.1  # epsilon-greedy: explore env by randomly choosing actions
        self.action_dim = None
        self.softmax = nn.Softmax(dim=1)

        self.act = None
        self.act_target = None

        self.obj_a = 0.0  # average Q value estimation
        self.obj_c = 0.5  # loss function value of Q value estimation
        self.criterion = None
        self.optimizer = None

    def select_actions(self, states):  # for discrete action space
        if rd.rand() < self.explore_rate:
            a_ints = rd.randint(self.action_dim, size=len(states))  # a_int if_discrete
        else:
            states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
            actions = self.act(states)
            a_ints = actions.argmax(dim=1).detach().cpu().numpy()  # a_int if_discrete
        return a_ints

    def update_buffer(self, env, buffer, max_step, reward_scale, gamma):
        for _ in range(max_step):
            action = self.select_actions((self.state,))[0]
            next_state, reward, done, _ = env.step(action)
            buffer.append_memo((reward * reward_scale, 0.0 if done else gamma,
                                *self.state, action,  # a_int if_discrete
                                *next_state))
            self.state = env.reset() if done else next_state
        return max_step

    def update_buffer__pipe(self, pipes, buffer, max_step):
        env_num = len(pipes)
        env_num2 = env_num // 2

        trajectories = [list() for _ in range(env_num)]
        for _ in range(max_step // env_num):
            for i_beg, i_end in ((0, env_num2), (env_num2, env_num)):
                for i in range(i_beg, i_end):
                    reward, mask, next_state = pipes[i].recv()
                    trajectories[i].append([reward, mask, *self.state[i], self.action[i],  # a_int if_discrete
                                            *next_state])
                    self.state[i] = next_state

                self.action[i_beg:i_end] = self.select_actions(self.state[i_beg:i_end])
                for i in range(i_beg, i_end):
                    pipes[i].send(self.action[i])  # pipes action

        steps_sum = 0
        for trajectory in trajectories:
            steps_sum += len(trajectory)
            buffer.extend_memo(memo_tuple=trajectory)
        return steps_sum

    def update_policy(self, buffer, max_step, batch_size, repeat_times):
        buffer.update__now_len__before_sample()
        """Contribution of DQN (Deep Q Network)
        1. Q-table (discrete state space) --> Q-network (continuous state space)
        2. Use experiment replay buffer to train a neural network in RL
        3. Use soft target update for stable training in RL
        """

        q_label = critic_obj = None

        update_times = int(max_step * repeat_times)
        for _ in range(update_times):
            with torch.no_grad():
                rewards, masks, states, actions, next_states = buffer.random_sample(batch_size)

                next_q_label = self.act_target(next_states).max(dim=1, keepdim=True)[0]
                q_label = rewards + masks * next_q_label

            q_eval = self.act(states).gather(1, actions.type(torch.long))
            critic_obj = self.criterion(q_eval, q_label)

            self.optimizer.zero_grad()
            critic_obj.backward()
            self.optimizer.step()
            soft_target_update(self.act_target, self.act)

        self.obj_a = q_label.mean().item()
        self.obj_c = critic_obj.item()

    def save_or_load_model(self, cwd, if_save):
        net, name = (self.act, 'act')
        if net is None:
            return None
        save_path = f'{cwd}/{name}.pth'
        if if_save:
            torch.save(net.state_dict(), save_path)
        elif os.path.exists(save_path):
            net = torch.load(save_path, map_location=lambda storage, loc: storage)
            net.load_state_dict(net)
            print(f"Loaded act and cri: {cwd}")
        else:
            print(f"FileNotFound when load_model: {cwd}")


class AgentDQN(AgentBaseDQN):
    def __init__(self, state_dim, action_dim, net_dim, learning_rate=1e-4):
        AgentBaseDQN.__init__(self)
        self.explore_rate = 0.1  # epsilon-greedy: explore env by randomly choosing actions
        self.action_dim = action_dim

        self.act = QNet(state_dim, action_dim, net_dim).to(self.device)
        self.act_target = QNet(state_dim, action_dim, net_dim).to(self.device)
        self.act_target.load_state_dict(self.act.state_dict())

        self.optimizer = torch.optim.Adam(self.act.parameters(), lr=learning_rate)
        """Contribution of Dueling DQN
        1. Sometimes the q = QNet(state) is independent with actions. 
        2. Advantage function helps RL learns the Q value faster without collecting each action in these state.
        3. Advantage function value + Original Q value --> Dueling Q value = val_q + adv_q - adv_q.mean()
        """


class AgentDuelingDQN(AgentBaseDQN):
    def __init__(self, state_dim, action_dim, net_dim, learning_rate=1e-4):
        AgentBaseDQN.__init__(self)
        self.explore_rate = 0.1  # epsilon-Greedy
        self.action_dim = action_dim

        self.act = QNetTwin(state_dim, action_dim, net_dim).to(self.device)
        self.act_target = QNetTwin(state_dim, action_dim, net_dim).to(self.device)
        self.act_target.load_state_dict(self.act.state_dict())

        self.optimizer = torch.optim.Adam(self.act.parameters(), lr=learning_rate)
        """Contribution of Dueling DQN
        1. Sometimes the q = QNet(state) is independent with actions. 
        2. Advantage function helps RL learns the Q value faster without collecting each action in these state.
        3. Advantage function value + Original Q value --> Dueling Q value = val_q + adv_q - adv_q.mean()
        """


class AgentDoubleDQN(AgentBaseDQN):
    def __init__(self, state_dim, action_dim, net_dim, learning_rate=1e-4, if_dueling=False):
        AgentBaseDQN.__init__(self)
        self.explore_rate = 0.1  # epsilon-greedy: explore env by randomly choosing actions
        self.action_dim = action_dim

        net = QNetDuelTwin if if_dueling else QNetTwin
        self.act = net(state_dim, action_dim, net_dim).to(self.device)
        self.act_target = net(state_dim, action_dim, net_dim).to(self.device)
        self.act_target.load_state_dict(self.act.state_dict())

        self.optimizer = torch.optim.Adam(self.act.parameters(), lr=learning_rate)

    def select_actions(self, states):  # for discrete action space
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = self.act(states)

        if rd.rand() < self.explore_rate:
            a_prob_l = self.softmax(actions).detach().cpu().numpy()
            a_ints = [rd.choice(self.action_dim, p=a_prob) for a_prob in a_prob_l]
        else:
            a_ints = actions.argmax(dim=1).detach().cpu().numpy()
        return a_ints

    def update_policy__twin(self, buffer, max_step, batch_size, repeat_times):
        buffer.update__now_len__before_sample()
        """Contribution of DDQN (Double DQN)
        1. The Q value estimation of Value network (critic) is noisy
        2. Over-estimation will propagated through the Bellman equation because of maximum operation.
        3. Under-estimation won't.
        4. Twin Q-Network. Use q=min(q1, q2) to reduce over-estimation.
        """

        q_label = cri_obj = None

        update_times = int(max_step * repeat_times)
        for _ in range(update_times):
            with torch.no_grad():
                rewards, masks, states, actions, next_states = buffer.random_sample(batch_size)
                actions = actions.type(torch.long)

                next_q = torch.min(*self.act_target.get__q1_q2(next_states))
                next_q = next_q.max(dim=1, keepdim=True)[0]
                q_label = rewards + masks * next_q

            q1, q2 = [qs.gather(1, actions) for qs in self.act.get__q1_q2(states)]
            cri_obj = self.criterion(q1, q_label) + self.criterion(q2, q_label)

            self.optimizer.zero_grad()
            cri_obj.backward()
            self.optimizer.step()
            soft_target_update(self.act_target, self.act)

        self.obj_a = q_label.mean().item()
        self.obj_c = cri_obj.item()


class AgentD3QN(AgentDoubleDQN):
    def __init__(self, state_dim, action_dim, net_dim, learning_rate=1e-4):
        AgentDoubleDQN.__init__(self, state_dim, action_dim, net_dim, learning_rate, if_dueling=True)
        self.explore_rate = 0.1  # epsilon-greedy: explore env by randomly choosing actions
        """Contribution of D3QN (Dueling Double DQN)
        1. DoubleDQN is compatible with Dueling DQN
        2. Anyone who just started deep reinforcement learning can discover D3QN algorithm independently
        """


def soft_target_update(target, current, tau=5e-3):
    # Paper: Taming the noise in reinforcement learning via soft updates. 2016
    for target_param, param in zip(target.parameters(), current.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


"""AgentRun"""


class Arguments:
    def __init__(self, rl_agent=None, env=None, gpu_id=None):
        self.rl_agent = rl_agent
        self.gpu_id = gpu_id
        self.cwd = None  # init cwd in def init_before_training()()
        self.env = env

        '''Arguments for training'''
        self.net_dim = 2 ** 8  # the network width
        self.max_memo = 2 ** 17  # memories capacity (memories: replay buffer)
        self.max_step = 2 ** 10  # max steps in one training episode
        self.batch_size = 2 ** 7  # num of transitions sampled from replay buffer.
        self.repeat_times = 2 ** 0  # repeatedly update network to keep critic's loss small
        self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256
        self.gamma = 0.99  # discount factor of future rewards
        self.rollout_num = 2  # the number of rollout workers (larger is not always faster)
        self.num_threads = 4  # cpu_num for evaluate model, torch.set_num_threads(self.num_threads)

        '''Arguments for evaluate'''
        self.break_step = 2 ** 17  # break training after 'total_step > break_step'
        self.if_break_early = True  # break training after 'eval_reward > target reward'
        self.if_remove_history = True  # remove the cwd folder? (True, False, None:ask me)
        self.show_gap = 2 ** 8  # show the Reward and Loss of actor and critic per show_gap seconds
        self.eval_times1 = 2 ** 3  # evaluation times if 'eval_reward > old_max_reward'
        self.eval_times2 = 2 ** 4  # evaluation times if 'eval_reward > target_reward'
        self.random_seed = 1943  # Github: YonV 1943

    def init_before_training(self):
        if not hasattr(self.env, 'env_name'):
            raise RuntimeError('| What is env.env_name? use env = build_env(env) to decorate env')

        self.gpu_id = sys.argv[-1][-4] if self.gpu_id is None else str(self.gpu_id)
        self.cwd = f'./{self.rl_agent.__name__}/{self.env.env_name}_{self.gpu_id}'
        print(f'| GPU id: {self.gpu_id}, cwd: {self.cwd}')
        whether_remove_history(self.cwd, self.if_remove_history)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        torch.set_num_threads(self.num_threads)
        torch.set_default_dtype(torch.float32)
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)


def whether_remove_history(cwd, is_remove=None):
    import shutil
    if is_remove is None:
        is_remove = bool(input("PRESS 'y' to REMOVE: {}? ".format(cwd)) == 'y')
    if is_remove:
        shutil.rmtree(cwd, ignore_errors=True)
        print("| Remove")
    os.makedirs(cwd, exist_ok=True)
    del shutil


def train_agent(args):  # 2021-02-02
    args.init_before_training()

    '''basic arguments'''
    rl_agent = args.rl_agent
    gpu_id = args.gpu_id
    env = args.env
    cwd = args.cwd

    '''training arguments'''
    gamma = args.gamma
    net_dim = args.net_dim
    max_memo = args.max_memo
    max_step = args.max_step
    batch_size = args.batch_size
    repeat_times = args.repeat_times
    reward_scale = args.reward_scale

    '''evaluate arguments'''
    break_step = args.break_step
    if_break_early = args.if_break_early
    show_gap = args.show_gap
    eval_times1 = args.eval_times1
    eval_times2 = args.eval_times2
    del args  # In order to show these hyper-parameters clearly, I put them above.

    '''init: env'''
    state_dim = env.state_dim
    action_dim = env.action_dim
    if_discrete = env.if_discrete
    target_reward = env.target_reward
    from copy import deepcopy  # built-in library of Python
    env = deepcopy(env)
    env_eval = deepcopy(env)  # 2020-12-12

    '''build rl_agent'''
    agent = rl_agent(state_dim, action_dim, net_dim)  # training agent
    agent.state = env.reset()

    '''build ReplayBuffer'''
    total_step = 0
    if_on_policy = rl_agent.__name__ in {'AgentPPO', 'AgentGaePPO'}

    buffer = ReplayBufferCPU(max_memo, state_dim,
                             action_dim=1 if if_discrete else action_dim,
                             if_on_policy=if_on_policy)
    if if_on_policy:
        steps = 0
    else:
        with torch.no_grad():  # update replay buffer
            steps = explore_before_train(env, buffer, max_step, if_discrete, reward_scale, gamma, action_dim)

        '''pre training and hard update before training loop'''
        buffer.update__now_len__before_sample()
        agent.update_policy(buffer, max_step, batch_size, repeat_times)
        agent.act_target.load_state_dict(agent.act.state_dict()) if 'act_target' in dir(agent) else None
    total_step += steps

    '''build Recorder'''
    recorder = Recorder(eval_times1, eval_times2)
    with torch.no_grad():
        recorder.update_recorder(env_eval, agent.act, agent.device, steps, agent.obj_a, agent.obj_c)

    '''loop'''
    if_solve = False
    while not ((if_break_early and if_solve) or total_step > break_step or os.path.exists(f'{cwd}/stop')):
        with torch.no_grad():  # speed up running
            steps = agent.update_buffer(env, buffer, max_step, reward_scale, gamma)
        total_step += steps

        buffer.update__now_len__before_sample()
        agent.update_policy(buffer, max_step, batch_size, repeat_times)

        with torch.no_grad():  # for saving the GPU buffer
            if_save = recorder.update_recorder(env_eval, agent.act, agent.device, steps, agent.obj_a, agent.obj_c)
            recorder.save_act(cwd, agent.act, gpu_id) if if_save else None

            if_solve = recorder.check__if_solved(target_reward, gpu_id, show_gap, cwd)

    recorder.save_npy__draw_plot(cwd)

    new_cwd = cwd[:-2] + f'_{recorder.r_max:.2f}' + cwd[-2:]
    if not os.path.exists(new_cwd):  # 2020-12-12
        os.rename(cwd, new_cwd)
        cwd = new_cwd
    else:
        print(f'| SavedDir: {new_cwd}    WARNING: file exit')
    print(f'| SavedDir: {cwd}\n'
          f'| UsedTime {time.time() - recorder.start_time:.0f}')

    buffer.print_state_norm(env.neg_state_avg if hasattr(env, 'neg_state_avg') else None,
                            env.div_state_std if hasattr(env, 'div_state_std') else None)  # 2020-12-12


def train_agent_mp(args):  # 2021-01-01
    args.init_before_training()
    act_workers = 0 if args.rollout_num < 2 else args.rollout_num

    import multiprocessing as mp
    process = list()

    exp_pipes = list()
    for i in range(act_workers):
        exp_pipe1, exp_pipe2 = mp.Pipe(duplex=True)
        process.append(mp.Process(target=mp_explore_in_env, args=(args, exp_pipe2, i)))

    eva_pipe1, eva_pipe2 = mp.Pipe(duplex=True)
    process = [mp.Process(target=mp__update_params, args=(args, eva_pipe2, exp_pipes)),
               mp.Process(target=mp_evaluate_agent, args=(args, eva_pipe1)),
               ] + process

    [p.start() for p in process]
    [p.join() for p in process[:2]]
    [p.terminate() for p in process[2:]]


def mp__update_params(args, eva_pipe, pipes):
    rl_agent = args.rl_agent
    max_memo = args.max_memo
    net_dim = args.net_dim
    max_step = args.max_step
    break_step = args.break_step
    batch_size = args.batch_size
    repeat_times = args.repeat_times
    cwd = args.cwd
    env = args.env
    reward_scale = args.reward_scale
    if_break_early = args.if_break_early
    gamma = args.gamma
    del args

    state_dim = env.state_dim
    action_dim = env.action_dim
    if_discrete = env.if_discrete
    env_num = len(pipes)

    '''build agent'''
    agent = rl_agent(state_dim, action_dim, net_dim)  # training agent
    agent.state = [pipe.recv() for pipe in pipes] if env_num else env.reset()

    if_on_policy = rl_agent.__name__ in {'AgentPPO', 'AgentGaePPO'}
    if not if_on_policy:
        agent.action = agent.select_actions(agent.state if env_num else (agent.state,))
        for i in range(env_num):
            pipes[i].send(agent.action[i])  # pipes 1 send

    from copy import deepcopy  # built-in library of Python
    act_cpu = deepcopy(agent.act).to(torch.device("cpu"))
    [setattr(param, 'requires_grad', False) for param in act_cpu.parameters()]

    '''build replay buffer'''
    total_step = 0
    buffer = ReplayBufferGPU(max_memo, state_dim,
                             action_dim=1 if if_discrete else action_dim,
                             if_on_policy=if_on_policy)
    if if_on_policy:
        steps = 0
    else:
        with torch.no_grad():  # initial exploration
            steps = explore_before_train(env, buffer, max_step, if_discrete, reward_scale, gamma, action_dim)

        agent.update_policy(buffer, max_step, batch_size, repeat_times)
        agent.act_target.load_state_dict(agent.act.state_dict()) if 'act_target' in dir(agent) else None

    total_step += steps
    eva_pipe.send((act_cpu, steps, agent.obj_a, agent.obj_c))  # eva_pipe act

    '''loop'''
    if_solve = False
    while not ((if_break_early and if_solve) or total_step > break_step or os.path.exists(f'{cwd}/stop')):
        with torch.no_grad():  # speed up running
            if env_num:  # not elegant
                steps = agent.update_buffer__pipe(pipes, buffer, max_step)  # pipes action inside
            else:
                steps = agent.update_buffer(env, buffer, max_step, reward_scale, gamma)
        total_step += steps

        buffer.update__now_len__before_sample()
        agent.update_policy(buffer, max_step, batch_size, repeat_times)

        act_cpu.load_state_dict(agent.act.state_dict())
        eva_pipe.send((act_cpu, steps, agent.obj_a, agent.obj_c))  # eva_pipe act
        if eva_pipe.poll():
            if_solve = eva_pipe.recv()  # eva_pipe if_solve

    buffer.print_state_norm(env.neg_state_avg if hasattr(env, 'neg_state_avg') else None,
                            env.div_state_std if hasattr(env, 'div_state_std') else None)  # 2020-12-12

    eva_pipe.send('stop')  # eva_pipe stop  # send to mp_evaluate_agent
    time.sleep(4)


def mp_explore_in_env(args, pipe, _act_id):  # 2021-02-02
    env = args.env
    gamma = args.gamma
    reward_scale = args.reward_scale
    del args

    pipe.send(env.reset())  # next_state
    while True:
        action = pipe.recv()
        next_state, reward, done, _ = env.step(action)  # pipes 1 recv, pipes n recv
        pipe.send((reward * reward_scale,  # reward
                   0.0 if done else gamma,  # mask
                   env.reset() if done else next_state,))  # next_state


def mp_evaluate_agent(args, eva_pipe):
    env = args.env
    cwd = args.cwd
    gpu_id = args.gpu_id
    show_gap = args.show_gap
    eval_size1 = args.eval_times1
    eval_size2 = args.eval_times2
    del args

    target_reward = env.target_reward

    '''build recorder'''
    act, step_sum, obj_a, obj_c = eva_pipe.recv()  # eva_pipe act, act == act.to(device_cpu), requires_grad=False
    torch.set_num_threads(4)
    device = torch.device('cpu')
    recorder = Recorder(eval_size1, eval_size2)
    recorder.update_recorder(env, act, device, step_sum, obj_a, obj_c)

    if_train = True
    with torch.no_grad():  # for saving the GPU util
        while if_train:
            is_solve = recorder.check__if_solved(target_reward, gpu_id, show_gap, cwd)
            eva_pipe.send(is_solve)  # eva_pipe is_solve

            while not eva_pipe.poll():  # wait until eva_pipe not empty
                time.sleep(1)

            step_sum = 0
            while eva_pipe.poll():  # receive the latest object from pipe
                q_i_eva_get = eva_pipe.recv()  # eva_pipe act
                if q_i_eva_get == 'stop':
                    if_train = False
                    break  # it should break 'while q_i_eva.qsize():' and 'while if_train:'
                act, steps, obj_a, obj_c = q_i_eva_get
                step_sum += steps
            if step_sum > 0:
                is_saved = recorder.update_recorder(env, act, device, step_sum, obj_a, obj_c)
                recorder.save_act(cwd, act, gpu_id) if is_saved else None

    recorder.save_npy__draw_plot(cwd)

    new_cwd = cwd[:-2] + f'_{recorder.r_max:.2f}' + cwd[-2:]
    if not os.path.exists(new_cwd):  # 2020-12-12
        os.rename(cwd, new_cwd)
        cwd = new_cwd
    else:
        print(f'| SavedDir: {new_cwd}    WARNING: file exit')
    print(f'| SavedDir: {cwd}\n'
          f'| UsedTime {time.time() - recorder.start_time:.0f}')

    while eva_pipe.poll():  # empty the pipe
        eva_pipe.recv()


def explore_before_train(env, buffer, max_step, if_discrete, reward_scale, gamma, action_dim):
    state = env.reset()
    steps = 0
    while steps < max_step:
        action = rd.randint(action_dim) if if_discrete else rd.uniform(-1, 1, size=action_dim)
        next_state, reward, done, _ = env.step(action)
        steps += 1

        scaled_reward = reward * reward_scale
        mask = 0.0 if done else gamma
        memo_tuple = (scaled_reward, mask, *state, action, *next_state) if if_discrete else \
            (scaled_reward, mask, *state, *action, *next_state)  # not elegant
        buffer.append_memo(memo_tuple)

        state = env.reset() if done else next_state
    return steps


class Recorder:
    def __init__(self, eval_size1=3, eval_size2=9):
        self.recorder = [(0., -np.inf, 0., 0., 0.), ]  # total_step, r_avg, r_std, obj_a, obj_c
        self.r_max = -np.inf
        self.is_solved = False
        self.total_step = 0
        self.eva_size1 = eval_size1  # constant
        self.eva_size2 = eval_size2  # constant

        self.used_time = None
        self.start_time = time.time()
        self.print_time = time.time()
        print(f"{'ID':>2}  {'Step':>8}  {'MaxR':>8} |{'avgR':>8}  {'stdR':>8}   {'objA':>8}  {'objC':>8}")

    def update_recorder(self, env, act, device, step_sum, obj_a, obj_c):
        is_saved = False
        reward_list = [get_episode_return(env, act, device)
                       for _ in range(self.eva_size1)]

        r_avg = np.average(reward_list)
        if r_avg > self.r_max:  # check 1
            reward_list.extend([get_episode_return(env, act, device)
                                for _ in range(self.eva_size2 - self.eva_size1)])
            r_avg = np.average(reward_list)
            if r_avg > self.r_max:  # check final
                self.r_max = r_avg
                is_saved = True

        r_std = float(np.std(reward_list))
        self.total_step += step_sum
        self.recorder.append((self.total_step, r_avg, r_std, obj_a, obj_c))
        return is_saved

    def save_act(self, cwd, act, agent_id):
        act_save_path = f'{cwd}/actor.pth'
        torch.save(act.state_dict(), act_save_path)
        print(f"{agent_id:<2}  {self.total_step:8.2e}  {self.r_max:8.2f} |")

    def check__if_solved(self, target_reward, agent_id, show_gap, cwd):
        total_step, r_avg, r_std, obj_a, obj_c = self.recorder[-1]
        if self.r_max > target_reward:
            self.is_solved = True
            if self.used_time is None:
                self.used_time = int(time.time() - self.start_time)
                print(f"{'ID':>2}  {'Step':>8}  {'TargetR':>8} |"
                      f"{'avgR':>8}  {'stdR':>8}   {'UsedTime':>8}  ########\n"
                      f"{agent_id:<2}  {total_step:8.2e}  {target_reward:8.2f} |"
                      f"{r_avg:8.2f}  {r_std:8.2f}   {self.used_time:>8}  ########")
        if time.time() - self.print_time > show_gap:
            self.print_time = time.time()
            print(f"{agent_id:<2}  {total_step:8.2e}  {self.r_max:8.2f} |"
                  f"{r_avg:8.2f}  {r_std:8.2f}   {obj_a:8.2f}  {obj_c:8.2f}")
            self.save_npy__draw_plot(cwd)
        return self.is_solved

    def save_npy__draw_plot(self, cwd):  # 2020-12-12
        if len(self.recorder) == 0:
            print(f"| save_npy__draw_plot() WARNING: len(self.recorder) == {len(self.recorder)}")
            return None
        np.save(f'{cwd}/recorder.npy', self.recorder)

        train_time = time.time() - self.start_time
        max_reward = self.r_max

        recorder = np.array(self.recorder[1:], dtype=np.float32)  # 2020-12-12 Compatibility
        if len(recorder) == 0:  # not elegant
            return None

        train_time = int(train_time)
        total_step = int(recorder[-1][0])
        save_title = f"plot_step_time_maxR_{int(total_step)}_{int(train_time)}_{max_reward:.3f}"
        save_path = f"{cwd}/{save_title}.jpg"

        import matplotlib as mpl  # draw figure in Terminal
        mpl.use('Agg')
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2)
        plt.title(save_title, y=2.3)

        steps = recorder[:, 0]
        r_avg, r_std = recorder[:, 1], recorder[:, 2]
        obj_a, obj_c = recorder[:, 3], recorder[:, 4]

        r_color = 'lightcoral'  # episode return (learning curve)
        axs[0].plot(steps, r_avg, label='EvaR', color=r_color)
        axs[0].fill_between(steps, r_avg - r_std, r_avg + r_std, facecolor=r_color, alpha=0.3, )

        axs[1].plot(steps, obj_a, label='objA', color=r_color)
        axs[1].tick_params(axis='y', labelcolor=r_color)
        axs[1].set_ylabel('objA', color=r_color)

        l_color = 'royalblue'  # loss function of critic (objective of critic)
        axs_1_twin = axs[1].twinx()
        axs_1_twin.fill_between(steps, obj_c, facecolor=l_color, alpha=0.2, )
        axs_1_twin.tick_params(axis='y', labelcolor=l_color)
        axs_1_twin.set_ylabel('objC', color=l_color)

        prev_save_names = [name for name in os.listdir(cwd) if name[:9] == save_title[:9]]  # remove previous plot
        os.remove(f'{cwd}/{prev_save_names[0]}') if len(prev_save_names) > 0 else None

        plt.savefig(save_path)
        plt.close()


def get_episode_return(env, act, device) -> float:  # 2021-02-02
    episode_return = 0.0  # sum of rewards in an episode
    max_step = env.max_step if hasattr(env, 'max_step') else 2 ** 10
    if_discrete = env.if_discrete

    state = env.reset()
    for _ in range(max_step):
        s_tensor = torch.as_tensor((state,), device=device)
        a_tensor = act(s_tensor)
        if if_discrete:
            a_tensor = a_tensor.argmax(dim=1)
        action = a_tensor.cpu().numpy()[0]  # not need detach(), because with torch.no_grad() outside

        state, reward, done, _ = env.step(action)
        episode_return += reward
        if done:
            break

    return env.episode_return if hasattr(env, 'episode_return') else episode_return


class ReplayBufferBase:
    def __init__(self, max_len, state_dim, action_dim, if_ppo=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.next_idx = 0
        self.is_full = False
        self.max_len = max_len
        self.now_len = self.max_len if self.is_full else self.next_idx

        self.state_idx = 1 + 1 + state_dim  # reward_dim=1, done_dim=1
        self.action_idx = self.state_idx + action_dim

        state_dim = state_dim if isinstance(state_dim, int) else np.prod(state_dim)  # pixel-level state
        last_dim = action_dim if if_ppo else state_dim
        self.memo_dim = 1 + 1 + state_dim + action_dim + last_dim
        self.memories = None

    @staticmethod
    def convert_tuple(memo_tuple):
        return memo_tuple

    def append_memo(self, memo_tuple):
        self.memories[self.next_idx] = self.convert_tuple(memo_tuple)
        self.next_idx = self.next_idx + 1
        if self.next_idx >= self.max_len:
            self.is_full = True
            self.next_idx = 0

    def extend_memo(self, memo_array):
        size = memo_array.shape[0]
        memo_array = self.convert_tuple(memo_array)

        next_idx = self.next_idx + size
        if next_idx > self.max_len:
            if next_idx > self.max_len:
                self.memories[self.next_idx:self.max_len] = memo_array[:self.max_len - self.next_idx]
            self.is_full = True
            next_idx = next_idx - self.max_len
            self.memories[0:next_idx] = memo_array[-next_idx:]
        else:
            self.memories[self.next_idx:next_idx] = memo_array
        self.next_idx = next_idx

    def random_sample(self, batch_size):
        # indices = rd.choice(self.now_len, batch_size, replace=False)  # why perform worse?
        # indices = rd.choice(self.now_len, batch_size, replace=True)  # why perform better?
        # same as:
        indices = rd.randint(self.now_len, size=batch_size)
        memory = torch.as_tensor(self.memories[indices], device=self.device)
        return (memory[:, 0:1],  # rewards
                memory[:, 1:2],  # masks, mark == (1-float(done)) * gamma
                memory[:, 2:self.state_idx],  # states
                memory[:, self.state_idx:self.action_idx],  # actions
                memory[:, self.action_idx:],)  # next_states

    def all_sample(self):
        tensors = (
            self.memories[:self.now_len, 0:1],  # rewards
            self.memories[:self.now_len, 1:2],  # masks, mark == (1-float(done)) * gamma
            self.memories[:self.now_len, 2:self.state_idx],  # states
            self.memories[:self.now_len, self.state_idx:self.action_idx],  # actions
            self.memories[:self.now_len, self.action_idx:],  # next_states or log_prob_sum
        )
        tensors = [torch.tensor(ary, device=self.device) for ary in tensors]
        return tensors

    def update__now_len__before_sample(self):
        self.now_len = self.max_len if self.is_full else self.next_idx

    def empty_memories__before_explore(self):
        self.next_idx = 0
        self.is_full = False
        self.now_len = 0

    def print_state_norm(self, neg_avg=None, div_std=None):  # non-essential
        batch_state = self.memories[:self.now_len, 2:self.state_idx]
        if isinstance(batch_state, torch.Tensor):
            batch_state = batch_state.detach().cpu().numpy()
        assert isinstance(batch_state, np.ndarray)

        if batch_state.shape[1] > 64:
            print(f"| _print_norm(): state_dim: {batch_state.shape[1]:.0f} is too large to print its norm. ")
            return None

        if np.isnan(batch_state).any():  # 2020-12-12
            batch_state = np.nan_to_num(batch_state)  # nan to 0

        ary_avg = batch_state.mean(axis=0)
        ary_std = batch_state.std(axis=0)
        fix_std = ((np.max(batch_state, axis=0) - np.min(batch_state, axis=0)) / 6
                   + ary_std) / 2

        if neg_avg is not None:  # norm transfer
            ary_avg = ary_avg - neg_avg / div_std
            ary_std = fix_std / div_std

        print(f"| Replay Buffer: avg, fixed std")
        print(f"| avg = np.{repr(ary_avg).replace('=float32', '=np.float32')}")
        print(f"| std = np.{repr(ary_std).replace('=float32', '=np.float32')}")


class ReplayBufferCPU(ReplayBufferBase):
    def __init__(self, max_len, state_dim, action_dim, if_on_policy=False):
        ReplayBufferBase.__init__(self, max_len, state_dim, action_dim, if_on_policy)
        self.memories = np.empty((max_len, self.memo_dim), dtype=np.float32)


class ReplayBufferGPU(ReplayBufferBase):
    def __init__(self, max_len, state_dim, action_dim, if_on_policy=False):
        ReplayBufferBase.__init__(self, max_len, state_dim, action_dim, if_on_policy)
        self.memories = torch.empty((max_len, self.memo_dim), dtype=torch.float32, device=self.device)

    def convert_tuple(self, memo_tuple):
        return torch.as_tensor(memo_tuple, dtype=torch.float32, device=self.device)  # if_gpu

    def random_sample(self, batch_size):
        indices = torch.randint(self.now_len, size=(batch_size,), device=self.device)
        memory = self.memories[indices]
        return (memory[:, 0:1],  # rewards
                memory[:, 1:2],  # masks, mark == (1-float(done)) * gamma
                memory[:, 2:self.state_idx],  # state
                memory[:, self.state_idx:self.action_idx],  # actions
                memory[:, self.action_idx:])  # next_states or actions_noise

    def all_sample(self):  # 2020-11-11 fix bug for ModPPO
        return (self.memories[:self.now_len, 0:1],  # rewards
                self.memories[:self.now_len, 1:2],  # masks, mark == (1-float(done)) * gamma
                self.memories[:self.now_len, 2:self.state_idx],  # state
                self.memories[:self.now_len, self.state_idx:self.action_idx],  # actions
                self.memories[:self.now_len, self.action_idx:])  # next_states or log_prob_sum


"""AgentRun"""


def decorate_env(env, if_print=True):  # important function # 2020-12-12
    if not all([hasattr(env, attr) for attr in (
            'env_name', 'state_dim', 'action_dim', 'target_reward', 'if_discrete')]):
        (env_name, state_dim, action_dim, action_max, if_discrete, target_reward
         ) = get_gym_env_information(env)

        env = get_decorate_env(env, action_max, data_type=np.float32)

        setattr(env, 'env_name', env_name)
        setattr(env, 'state_dim', state_dim)
        setattr(env, 'action_dim', action_dim)
        setattr(env, 'if_discrete', if_discrete)
        setattr(env, 'target_reward', target_reward)
    if if_print:
        print(f"| env_name:  {env.env_name}, action is {'Discrete' if env.if_discrete else 'Continuous'}\n"
              f"| state_dim, action_dim: ({env.state_dim}, {env.action_dim}), target_reward: {env.target_reward}")
    return env


def get_gym_env_information(env) -> (str, int, int, float, bool, float):
    import gym  # gym of OpenAI is not necessary for ElegantRL (even RL)
    gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'

    if not isinstance(env, gym.Env):
        raise RuntimeError('| It is not a standard gym env. Could tell the following values?\n'
                           '| state_dim=int, action_dim=int, target_reward=float, if_discrete=bool')

    '''special rule'''
    env_name = env.unwrapped.spec.id
    if env_name == 'Pendulum-v0':
        env.spec.reward_threshold = -200.0  # target_reward

    state_shape = env.observation_space.shape
    state_dim = state_shape[0] if len(state_shape) == 1 else state_shape  # sometimes state_dim is a list

    target_reward = env.spec.reward_threshold
    if target_reward is None:
        raise RuntimeError('| I do not know how much is target_reward? Maybe set as +np.inf?')

    if_discrete = isinstance(env.action_space, gym.spaces.Discrete)
    if if_discrete:  # make sure it is discrete action space
        action_dim = env.action_space.n
        action_max = int(1)
    elif isinstance(env.action_space, gym.spaces.Box):  # make sure it is continuous action space
        action_dim = env.action_space.shape[0]
        action_max = float(env.action_space.high[0])

        action_edge = np.array([action_max, ] * action_dim)  # need check
        if any(env.action_space.high - action_edge):
            raise RuntimeError(f'| action_space.high should be {action_edge}, but {env.action_space.high}')
        if any(env.action_space.low + action_edge):
            raise RuntimeError(f'| action_space.low should be {-action_edge}, but {env.action_space.low}')
    else:
        raise RuntimeError('| Set these value manually? if_discrete=bool, action_dim=int, action_max=1.0')
    return env_name, state_dim, action_dim, action_max, if_discrete, target_reward


def get_decorate_env(env, action_max=1, state_avg=None, state_std=None, data_type=np.float32):
    if state_avg is None:
        neg_state_avg = 0
        div_state_std = 1
    else:
        state_avg = state_avg.astype(data_type)
        state_std = state_std.astype(data_type)

        neg_state_avg = -state_avg
        div_state_std = 1 / (state_std + 1e-4)
    setattr(env, 'neg_state_avg', neg_state_avg)  # for def print_norm() AgentZoo.py
    setattr(env, 'div_state_std', div_state_std)  # for def print_norm() AgentZoo.py

    '''decorator_step'''
    if state_avg is not None:
        def decorator_step(env_step):
            def new_env_step(action):
                state, reward, done, info = env_step(action * action_max)
                return (state.astype(data_type) + neg_state_avg) * div_state_std, reward, done, info

            return new_env_step
    elif action_max is not None:
        def decorator_step(env_step):
            def new_env_step(action):
                state, reward, done, info = env_step(action * action_max)
                return state.astype(data_type), reward, done, info

            return new_env_step
    else:  # action_max is None:
        def decorator_step(env_step):
            def new_env_step(action):
                state, reward, done, info = env_step(action * action_max)
                return state.astype(data_type), reward, done, info

            return new_env_step
    env.step = decorator_step(env.step)

    '''decorator_reset'''
    if state_avg is not None:
        def decorator_reset(env_reset):
            def new_env_reset():
                state = env_reset()
                return (state.astype(data_type) + neg_state_avg) * div_state_std

            return new_env_reset
    else:
        def decorator_reset(env_reset):
            def new_env_reset():
                state = env_reset()
                return state.astype(data_type)

            return new_env_reset
    env.reset = decorator_reset(env.reset)

    return env


class FinanceMultiStockEnv:  # 2021-02-02
    """FinRL
    Paper: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance
           https://arxiv.org/abs/2011.09607 NeurIPS 2020: Deep RL Workshop.
    Source: Github https://github.com/AI4Finance-LLC/FinRL-Library
    Modify: Github Yonv1943 ElegantRL
    """

    def __init__(self, initial_account=1e6, transaction_fee_percent=1e-3, max_stock=100):
        self.stock_dim = 30
        self.initial_account = initial_account
        self.transaction_fee_percent = transaction_fee_percent
        self.max_stock = max_stock

        self.ary = self.load_training_data_for_multi_stock()
        assert self.ary.shape == (1699, 5 * 30)  # ary: (date, item*stock_dim), item: (adjcp, macd, rsi, cci, adx)

        # reset
        self.day = 0
        self.account = self.initial_account
        self.day_npy = self.ary[self.day]
        self.stocks = np.zeros(self.stock_dim, dtype=np.float32)  # multi-stack
        self.total_asset = self.account + (self.day_npy[:self.stock_dim] * self.stocks).sum()
        self.episode_return = 0.0  # Compatibility for ElegantRL 2020-12-21

        '''env information'''
        self.env_name = 'FinanceStock-v1'
        self.state_dim = 1 + (5 + 1) * self.stock_dim
        self.action_dim = self.stock_dim
        self.if_discrete = False
        self.target_reward = 15
        self.max_step = self.ary.shape[0]

        self.gamma_r = 0.0

    def reset(self):  # 2021-02-03
        self.account = self.initial_account * rd.uniform(0.9, 1.0)  # notice reset()
        self.stocks = np.zeros(self.stock_dim, dtype=np.float32)
        self.total_asset = self.account + (self.day_npy[:self.stock_dim] * self.stocks).sum()
        # total_asset = account + (adjcp * stocks).sum()

        self.day = 0
        self.day_npy = self.ary[self.day]
        self.day += 1

        state = np.hstack((self.account * 2 ** -16,
                           self.day_npy * 2 ** -8,
                           self.stocks * 2 ** -12,), ).astype(np.float32)

        return state

    def step(self, actions):
        actions = actions * self.max_stock

        """bug or sell stock"""
        for index in range(self.stock_dim):
            action = actions[index]
            adj = self.day_npy[index]
            if action > 0:  # buy_stock
                available_amount = self.account // adj
                delta_stock = min(available_amount, action)
                self.account -= adj * delta_stock * (1 + self.transaction_fee_percent)
                self.stocks[index] += delta_stock
            elif self.stocks[index] > 0:  # sell_stock
                delta_stock = min(-action, self.stocks[index])
                self.account += adj * delta_stock * (1 - self.transaction_fee_percent)
                self.stocks[index] -= delta_stock

        """update day"""
        self.day_npy = self.ary[self.day]
        self.day += 1
        done = self.day == self.max_step  # 2020-12-21

        state = np.hstack((
            self.account * 2 ** -16,
            self.day_npy * 2 ** -8,
            self.stocks * 2 ** -12,
        ), ).astype(np.float32)

        next_total_asset = self.account + (self.day_npy[:self.stock_dim] * self.stocks).sum()
        reward = (next_total_asset - self.total_asset) * 2 ** -16  # notice scaling!
        self.total_asset = next_total_asset

        self.gamma_r = self.gamma_r * 0.99 + reward  # notice: gamma_r seems good? Yes
        if done:
            reward += self.gamma_r
            self.gamma_r = 0.0  # env.reset()

            # cumulative_return_rate
            self.episode_return = next_total_asset / self.initial_account

        return state, reward, done, None

    @staticmethod
    def load_training_data_for_multi_stock(if_load=True):  # need more independent
        npy_path = './Result/FinanceMultiStock.npy'
        if if_load and os.path.exists(npy_path):
            data_ary = np.load(npy_path).astype(np.float32)
            assert data_ary.shape[1] == 5 * 30
            return data_ary
        else:
            raise RuntimeError(
                f'| FinanceMultiStockEnv(): Can you download and put it into: {npy_path}\n'
                f'| https://github.com/Yonv1943/ElegantRL/blob/master/Result/FinanceMultiStock.npy'
                f'| Or you can use the following code to generate it from a csv file.'
            )

        # from preprocessing.preprocessors import pd, data_split, preprocess_data, add_turbulence
        #
        # # the following is same as part of run_model()
        # preprocessed_path = "done_data.csv"
        # if if_load and os.path.exists(preprocessed_path):
        #     data = pd.read_csv(preprocessed_path, index_col=0)
        # else:
        #     data = preprocess_data()
        #     data = add_turbulence(data)
        #     data.to_csv(preprocessed_path)
        #
        # df = data
        # rebalance_window = 63
        # validation_window = 63
        # i = rebalance_window + validation_window
        #
        # unique_trade_date = data[(data.datadate > 20151001) & (data.datadate <= 20200707)].datadate.unique()
        # train__df = data_split(df, start=20090000, end=unique_trade_date[i - rebalance_window - validation_window])
        # # print(train__df) # df: DataFrame of Pandas
        #
        # train_ary = train__df.to_numpy().reshape((-1, 30, 12))
        # '''state_dim = 1 + 6 * stock_dim, stock_dim=30
        # n   item    index
        # 1   ACCOUNT -
        # 30  adjcp   2
        # 30  stock   -
        # 30  macd    7
        # 30  rsi     8
        # 30  cci     9
        # 30  adx     10
        # '''
        # data_ary = np.empty((train_ary.shape[0], 5, 30), dtype=np.float32)
        # data_ary[:, 0] = train_ary[:, :, 2]  # adjcp
        # data_ary[:, 1] = train_ary[:, :, 7]  # macd
        # data_ary[:, 2] = train_ary[:, :, 8]  # rsi
        # data_ary[:, 3] = train_ary[:, :, 9]  # cci
        # data_ary[:, 4] = train_ary[:, :, 10]  # adx
        #
        # data_ary = data_ary.reshape((-1, 5 * 30))
        #
        # os.makedirs(npy_path[:npy_path.rfind('/')])
        # np.save(npy_path, data_ary.astype(np.float16))  # save as float16 (0.5 MB), float32 (1.0 MB)
        # print('| FinanceMultiStockEnv(): save in:', npy_path)
        # return data_ary


def train__demo():
    args = Arguments(rl_agent=None, env=None, gpu_id=0)

    '''DEMO 1: Discrete action env: CartPole-v0 of gym'''
    import gym  # gym of OpenAI is not necessary for ElegantRL (even RL)
    gym.logger.set_level(40)  # Block warning: "WARN: Box bound precision lowered by casting to float32"
    args.env = decorate_env(env=gym.make('CartPole-v0'), if_print=True)
    args.rl_agent = AgentD3QN  # Dueling Double DQN (off-policy)
    args.break_step = int(1e5 * 8)  # UsedTime 60s (reach target_reward 195)
    args.net_dim = 2 ** 7  # network dimension (width)
    train_agent(args)  # training using single process. Easy to find errors
    # train_agent_mp(args)  # training using multiple processes. Speed up training
    exit()

    '''DEMO 2: Continuous action env: LunarLanderContinuous-v2 of gym.box2D'''
    import gym
    gym.logger.set_level(40)
    args.env = decorate_env(env=gym.make('LunarLanderContinuous-v2'), if_print=True)
    args.rl_agent = AgentModSAC  # Modified SAC (off-policy)
    args.break_step = int(6e4 * 8)  # UsedTime 900s (reach target_reward 200)
    args.net_dim = 2 ** 7
    train_agent_mp(args)  # train_agent(args)
    exit()

    '''DEMO 3: Custom Continuous action env: FinanceStock-v1'''
    args.env = FinanceMultiStockEnv()  # a standard env for ElegantRL, not need decorate_env()
    args.rl_agent = AgentGaePPO  # PPO+GAE (on-policy)
    args.eval_times1 = 1
    args.eval_times2 = 1
    args.rollout_num = 4
    args.if_break_early = True

    args.reward_scale = 2 ** 0  # (0) 1.1 ~ 15 (19)
    args.break_step = int(1e6)  # 5e6 * 4)  # 5e6 (15e6) UsedTime 3,000s (9,000s)
    args.net_dim = 2 ** 8
    args.max_step = 1699
    args.max_memo = (args.max_step - 1) * 16
    args.batch_size = 2 ** 11
    args.repeat_times = 2 ** 4
    args.init_before_training()
    train_agent_mp(args)  # train_agent(args)
    exit()


def train__discrete_action():
    args = Arguments(rl_agent=AgentD3QN, env=None, gpu_id=None)

    import gym  # gym of OpenAI is not necessary for ElegantRL (even RL)
    gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'
    args.rollout_num = 4

    # args.env = decorate_env(gym.make("CartPole-v0"), if_print=True)
    # args.break_step = int(1e4 * 8)  # (3e5) 1e4, UsedTime 20s
    # args.reward_scale = 2 ** 0  # 0 ~ 200
    # args.net_dim = 2 ** 6
    # args.init_before_training()()
    # train_agent_mp(args)  # train_agent(args)
    # exit()

    args.env = decorate_env(gym.make("LunarLander-v2"), if_print=True)
    args.break_step = int(1e5 * 8)  # (2e4) 1e5 (3e5), UsedTime: (200s) 1000s (2000s)
    args.reward_scale = 2 ** -1  # (-1000) -150 ~ 200 (285)
    args.net_dim = 2 ** 7
    args.init_before_training()
    train_agent_mp(args)  # train_agent(args)
    exit()


def train__continuous_action__off_policy():
    args = Arguments(rl_agent=None, env=None, gpu_id=None)
    # args.rl_agent in {AgentModSAC, }

    args.if_break_early = True  # break training if reach the target reward (total return of an episode)
    args.if_remove_history = True  # delete the historical directory

    import gym  # gym of OpenAI is not necessary for ElegantRL (even RL)
    gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'

    args.env = decorate_env(gym.make("Pendulum-v0"), if_print=True)
    args.rl_agent = AgentModSAC
    args.break_step = int(1e4 * 8)  # 1e4 means the average total training step of InterSAC to reach target_reward
    args.reward_scale = 2 ** -2  # (-1800) -1000 ~ -200 (-50)
    train_agent(args)  # Train agent using single process. Recommend run on PC.
    # train_agent_mp(args)  # Train using multi process. Recommend run on Server. Mix CPU(eval) GPU(train)
    exit()

    args.env = decorate_env(gym.make("LunarLanderContinuous-v2"), if_print=True)
    args.rl_agent = AgentModSAC
    args.break_step = int(5e5 * 8)  # (2e4) 5e5, UsedTime 1500s
    args.reward_scale = 2 ** -3  # (-800) -200 ~ 200 (302)
    args.init_before_training()
    train_agent_mp(args)  # train_agent(args)
    exit()

    args.env = decorate_env(gym.make("BipedalWalker-v3"), if_print=True)
    args.rl_agent = AgentModSAC
    args.break_step = int(2e5 * 8)  # (1e5) 2e5, UsedTime 3500s
    args.reward_scale = 2 ** -1  # (-200) -140 ~ 300 (341)
    args.init_before_training()
    train_agent_mp(args)  # train_agent(args)
    exit()

    # args.env = decorate_env(gym.make("BipedalWalkerHardcore-v3"), if_print=True)
    # args.rl_agent = AgentModSAC
    # args.reward_scale = 2 ** 0  # (-200) -150 ~ 300 (334), UsedTime 24 hours
    # args.break_step = int(4e6 * 8)  # (2e6) 4e6
    # args.net_dim = int(2 ** 8)  # int(2 ** 8.5) #
    # args.max_memo = int(2 ** 21)
    # args.batch_size = int(2 ** 8)
    # args.eval_times2 = 2 ** 5  # for Recorder
    # args.show_gap = 2 ** 8  # for Recorder
    # args.init_before_training()
    # train_agent_mp(args)  # train_offline_policy(args)
    # exit()

    import gym  # import gym before import pybullet_envs
    import pybullet_envs  # for PyBullet
    dir(pybullet_envs)

    args.env = decorate_env(gym.make("ReacherBulletEnv-v0"), if_print=True)  # PyBullet
    args.rl_agent = AgentModSAC
    args.break_step = int(5e4 * 8)  # (4e4) 5e4
    args.reward_scale = 2 ** 0  # (-37) 0 ~ 18 (29)
    args.init_before_training()
    train_agent_mp(args)  # train_agent(args)
    exit()

    args.env = decorate_env(gym.make("AntBulletEnv-v0"), if_print=True)  # PyBullet
    args.rl_agent = AgentModSAC
    args.break_step = int(1e6 * 8)  # (5e5) 1e6, UsedTime: (15,000s) 30,000s
    args.reward_scale = 2 ** -3  # (-50) 0 ~ 2500 (3340)
    args.batch_size = 2 ** 8
    args.max_memo = 2 ** 20
    args.eva_size = 2 ** 3  # for Recorder
    args.show_gap = 2 ** 8  # for Recorder
    args.init_before_training()
    train_agent_mp(args)  # train_agent(args)
    exit()

    args.env = decorate_env(gym.make("MinitaurBulletEnv-v0"), if_print=True)  # PyBullet
    args.rl_agent = AgentModSAC
    args.break_step = int(4e6 * 4)  # (2e6) 4e6
    args.reward_scale = 2 ** 5  # (-2) 0 ~ 16 (20)
    args.batch_size = (2 ** 8)
    args.net_dim = int(2 ** 8)
    args.max_step = 2 ** 11
    args.max_memo = 2 ** 20
    args.eval_times2 = 3  # for Recorder
    args.eval_times2 = 9  # for Recorder
    args.show_gap = 2 ** 9  # for Recorder
    args.init_before_training()
    train_agent_mp(args)  # train_agent(args)
    exit()


def train__continuous_action__on_policy():
    args = Arguments(rl_agent=None, env=None, gpu_id=None)
    # args.rl_agent in {AgentPPO, }

    args.net_dim = 2 ** 8
    args.max_memo = 2 ** 12
    args.batch_size = 2 ** 9
    args.repeat_times = 2 ** 4
    args.reward_scale = 2 ** 0  # unimportant hyper-parameter in PPO which do normalization on Q value

    import gym  # gym of OpenAI is not necessary for ElegantRL (even RL)
    gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'

    args.env = decorate_env(gym.make("Pendulum-v0"), if_print=True)
    args.rl_agent = AgentGaePPO
    args.break_step = int(8e4 * 8)  # 5e5 means the average total training step of ModPPO to reach target_reward
    args.reward_scale = 2 ** 0  # (-1800) -1000 ~ -200 (-50), UsedTime  (100s) 200s
    args.gamma = 0.9  # discount factor
    train_agent_mp(args)  # train_agent(args)
    exit()

    args.env = decorate_env(gym.make("LunarLanderContinuous-v2"), if_print=True)
    args.rl_agent = AgentGaePPO
    args.break_step = int(3e5 * 8)  # (2e5) 3e5 , UsedTime: (400s) 600s
    args.reward_scale = 2 ** 0  # (-800) -200 ~ 200 (301)
    args.gamma = 0.99
    args.init_before_training()
    train_agent_mp(args)
    exit()

    args.env = decorate_env(gym.make("BipedalWalker-v3"), if_print=True)
    args.rl_agent = AgentGaePPO
    args.break_step = int(8e5 * 8)  # (4e5) 8e5 (4e6), UsedTime: (600s) 1500s (8000s)
    args.reward_scale = 2 ** 0  # (-150) -90 ~ 300 (325)
    args.gamma = 0.96
    args.init_before_training()
    train_agent_mp(args)
    exit()

    import gym  # import gym before import pybullet_envs
    import pybullet_envs  # for PyBullet
    dir(pybullet_envs)

    args.env = decorate_env(gym.make("ReacherBulletEnv-v0"), if_print=True)  # PyBullet
    args.rl_agent = AgentGaePPO
    args.break_step = int(2e6 * 8)  # (1e6) 2e6 (4e6), UsedTime: 2000s (6000s)
    args.reward_scale = 2 ** 0  # (-15) 0 ~ 18 (25)
    args.gamma = 0.95
    args.init_before_training()
    train_agent_mp(args)  # train_agent(args)
    exit()

    args.env = decorate_env(gym.make("AntBulletEnv-v0"), if_print=True)  # PyBullet
    args.rl_agent = AgentGaePPO
    args.break_step = int(5e6 * 8)  # (1e6) 5e6 UsedTime 25697s
    args.reward_scale = 2 ** -3  #
    args.gamma = 0.99
    args.net_dim = 2 ** 9
    args.init_before_training()
    train_agent_mp(args)
    exit()

    args.env = decorate_env(gym.make("MinitaurBulletEnv-v0"), if_print=True)  # PyBullet
    args.rl_agent = AgentGaePPO
    args.break_step = int(1e6 * 8)  # (4e5) 1e6 (8e6)
    args.reward_scale = 2 ** 4  # (-2) 0 ~ 16 (PPO 34)
    args.gamma = 0.95
    args.net_dim = 2 ** 8
    args.max_memo = 2 ** 11
    args.batch_size = 2 ** 9
    args.repeat_times = 2 ** 4
    args.init_before_training()
    train_agent_mp(args)
    exit()


if __name__ == '__main__':
    train__demo()
