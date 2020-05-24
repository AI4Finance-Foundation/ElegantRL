import os
from time import time as timer

import gym
import numpy as np
import numpy.random as rd
import torch
import torch.nn as nn

from AgentNetwork import QNetwork  # QLearning
from AgentNetwork import Actor, Critic
from AgentNetwork import ActorCritic  # IntelAC
from AgentNetwork import CriticTwin  # TD3, SAC
from AgentNetwork import ActorCriticPPO  # PPO
from AgentNetwork import ActorSAC  # SAC

"""
2019-07-01 Zen4Jia1Hao2, GitHub: YonV1943 DL_RL_Zoo RL
2019-11-11 Issay-0.0 [Essay Consciousness]
2020-02-02 Issay-0.1 Deep Learning Techniques (spectral norm, DenseNet, etc.) 
2020-04-04 Issay-0.1 [An Essay of Consciousness by YonV1943], IntelAC
2020-04-20 Issay-0.2 SN_AC, IntelAC_UnitedLoss
2020-05-20 Issay-0.3 [Essay, LongDear's Cerebellum (Little Brain)]

I consider that Reinforcement Learning Algorithms before 2020 have not consciousness
They feel more like a Cerebellum (Little Brain) for Machines.

Refer: (TD3) https://github.com/sfujim/TD3
Refer: (TD3) https://github.com/nikhilbarhate99/TD3-PyTorch-BipedalWalker-v2
Refer: (PPO) https://github.com/zhangchuheng123/Reinforcement-Implementation/blob/master/code/ppo.py
Refer: (PPO) https://github.com/Jiankai-Sun/Proximal-Policy-Optimization-in-Pytorch/blob/master/ppo.py
Refer: (PPO) https://github.com/openai/baselines/tree/master/baselines/ppo2
Refer: (SAC) https://github.com/TianhongDai/reinforcement-learning-algorithms/tree/master/rl_algorithms/sac
"""


class AgentSNAC:
    def __init__(self, state_dim, action_dim, net_dim):
        use_densenet = True
        use_spectral_norm = True
        self.lr_c = 4e-4  # learning rate of critic
        self.lr_a = 1e-4  # learning rate of actor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''dim and idx'''
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_idx = 1 + 1 + state_dim  # reward_dim==1, done_dim==1
        self.action_idx = self.state_idx + action_dim

        '''network'''
        actor_dim = net_dim
        act = Actor(state_dim, action_dim, actor_dim, use_densenet).to(self.device)
        act.train()
        self.act = act
        self.act_optimizer = torch.optim.Adam(act.parameters(), lr=self.lr_a, )  # betas=(0.5, 0.99))

        act_target = Actor(state_dim, action_dim, actor_dim, use_densenet).to(self.device)
        act_target.eval()
        self.act_target = act_target
        self.act_target.load_state_dict(act.state_dict())

        '''critic'''
        critic_dim = int(net_dim * 1.25)
        cri = Critic(state_dim, action_dim, critic_dim, use_densenet, use_spectral_norm).to(self.device)
        cri.train()
        self.cri = cri
        self.cri_optimizer = torch.optim.Adam(cri.parameters(), lr=self.lr_c, )  # betas=(0.5, 0.99))

        cri_target = Critic(state_dim, action_dim, critic_dim, use_densenet, use_spectral_norm).to(self.device)
        cri_target.eval()
        self.cri_target = cri_target
        self.cri_target.load_state_dict(cri.state_dict())

        self.criterion = nn.SmoothL1Loss()

        self.update_counter = 0
        self.loss_c_sum = 0.0
        self.rho = 0.5

    def inactive_in_env(self, env, memories, max_step, explore_noise, action_max, r_norm):
        self.act.eval()

        rewards = list()
        steps = list()

        step_counter = 0
        while step_counter < max_step:
            t = 0
            reward_sum = 0
            state = env.reset()

            for t in range(max_step):
                action = self.select_actions(state[np.newaxis],
                                             explore_noise if rd.rand() < 0.5 else 0.0)[0]

                # action_max == None, when action space is 'Discrete'
                next_state, reward, done, _ = env.step((action * action_max) if action_max else action)
                memories.add_memo(np.hstack((r_norm(reward), 1 - float(done), state, action, next_state)))

                state = next_state
                reward_sum += reward

                if done:
                    break

            rewards.append(reward_sum)
            t += 1
            steps.append(t)
            step_counter += t
        return rewards, steps

    def update_parameter(self, memories, iter_num, batch_size, policy_noise, update_gap, gamma):  # 2020-02-02
        loss_a_sum = 0.0
        loss_c_sum = 0.0

        k = 1.0 + memories.now_len / memories.max_len
        batch_size = int(batch_size * k)
        iter_num = int(iter_num * k)

        for _ in range(iter_num):
            with torch.no_grad():
                memory = memories.random_sample(batch_size)
                memory = torch.tensor(memory, device=self.device)

                reward = memory[:, 0:1]
                undone = memory[:, 1:2]
                state = memory[:, 2:self.state_idx]
                action = memory[:, self.state_idx:self.action_idx]
                next_state = memory[:, self.action_idx:]

                # next_action0 = self.act_target(next_state, policy_noise)
                # q_target = self.cri_target(next_state, next_action)
                next_action0 = self.act_target(next_state)
                next_action1 = self.act_target.add_noise(next_action0, policy_noise)
                q_target0 = self.cri_target(next_state, next_action0)
                q_target1 = self.cri_target(next_state, next_action1)
                q_target = (q_target0 + q_target1) * 0.5
                q_target = reward + undone * gamma * q_target

            '''loss C'''
            q_eval = self.cri(state, action)
            critic_loss = self.criterion(q_eval, q_target)
            loss_c_tmp = critic_loss.item()
            loss_c_sum += loss_c_tmp
            self.loss_c_sum += loss_c_tmp

            self.cri_optimizer.zero_grad()
            critic_loss.backward()
            self.cri_optimizer.step()

            '''loss A'''
            action_cur = self.act(state)
            actor_loss = -self.cri(state, action_cur).mean()
            loss_a_sum += actor_loss.item()

            self.act_optimizer.zero_grad()
            actor_loss.backward()
            self.act_optimizer.step()

            self.update_counter += 1
            if self.update_counter == update_gap:
                self.update_counter = 0

                self.act_target.load_state_dict(self.act.state_dict())
                self.cri_target.load_state_dict(self.cri.state_dict())

                rho = np.exp(-(self.loss_c_sum / update_gap) ** 2)
                self.rho = self.rho * 0.75 + rho * 0.25
                self.act_optimizer.param_groups[0]['lr'] = self.lr_a * self.rho
                self.loss_c_sum = 0.0

        return loss_a_sum / iter_num, loss_c_sum / iter_num,

    def select_actions(self, states, explore_noise=0.0):  # CPU array to GPU tensor to CPU array
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = self.act(states, explore_noise).cpu().data.numpy()
        return actions

    @staticmethod
    def soft_update(target, source, tau=0.005):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def save_or_load_model(self, cwd, is_save):  # 2020-04-30
        act_save_path = '{}/actor.pth'.format(cwd)
        cri_save_path = '{}/critic.pth'.format(cwd)

        if is_save:
            torch.save(self.act.state_dict(), act_save_path)
            torch.save(self.cri.state_dict(), cri_save_path)
            # print("Saved act and cri:", cwd)
        elif os.path.exists(act_save_path):
            act_dict = torch.load(act_save_path, map_location=lambda storage, loc: storage)
            self.act.load_state_dict(act_dict)
            self.act_target.load_state_dict(act_dict)
            cri_dict = torch.load(cri_save_path, map_location=lambda storage, loc: storage)
            self.cri.load_state_dict(cri_dict)
            self.cri_target.load_state_dict(cri_dict)
            print("Load act and cri:", cwd)
        else:
            print("FileNotFound when load_model:", cwd)
            # pass


class AgentQLearning(AgentSNAC):
    def __init__(self, state_dim, action_dim, net_dim):  # 2020-04-30
        super(AgentSNAC, self).__init__()
        learning_rate = 4e-4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''dim and idx'''
        self.state_dim = state_dim
        self.action_dim = action_dim
        memo_action_dim = 1  # Discrete action space
        self.state_idx = 1 + 1 + state_dim  # reward_dim==1, done_dim==1
        self.action_idx = self.state_idx + memo_action_dim

        '''network'''
        actor_dim = net_dim
        act = QNetwork(state_dim, action_dim, actor_dim).to(self.device)
        act.train()
        self.act = act
        self.act_optimizer = torch.optim.Adam(act.parameters(), lr=learning_rate)

        act_target = QNetwork(state_dim, action_dim, actor_dim).to(self.device)
        act_target.eval()
        self.act_target = act_target
        self.act_target.load_state_dict(act.state_dict())

        self.criterion = nn.SmoothL1Loss()

        self.update_counter = 0

    def update_parameter(self, memories, iter_num, batch_size, policy_noise, update_gap, gamma):  # 2020-02-02
        loss_a_sum = 0.0
        loss_c_sum = 0.0

        k = 1.0 + memories.now_len / memories.max_len
        batch_size = int(batch_size * k)
        iter_num = int(iter_num * k)

        for _ in range(iter_num):
            with torch.no_grad():
                memory = memories.random_sample(batch_size)
                memory = torch.tensor(memory, device=self.device)

                reward = memory[:, 0:1]
                undone = memory[:, 1:2]
                state = memory[:, 2:self.state_idx]
                action = memory[:, self.state_idx:self.action_idx]
                next_state = memory[:, self.action_idx:]

                q_target = self.act_target(next_state).max(dim=1, keepdim=True)[0]
                q_target = reward + undone * gamma * q_target

            self.act.train()
            action = action.type(torch.long)
            q_eval = self.act(state).gather(1, action)
            critic_loss = self.criterion(q_eval, q_target)
            loss_c_sum += critic_loss.item()

            self.act_optimizer.zero_grad()
            critic_loss.backward()
            self.act_optimizer.step()

            self.update_counter += 1
            if self.update_counter == update_gap:
                self.update_counter = 0
                self.act_target.load_state_dict(self.act.state_dict())

        return loss_a_sum / iter_num, loss_c_sum / iter_num,

    def select_actions(self, states, explore_noise=0.0):  # state -> ndarray shape: (1, state_dim)
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = self.act(states, explore_noise).argmax(dim=1).cpu().data.numpy()
        return actions

    def save_or_load_model(self, mod_dir, is_save):
        act_save_path = '{}/actor.pth'.format(mod_dir)

        if is_save:
            torch.save(self.act.state_dict(), act_save_path)
            print("Saved act and cri:", mod_dir)
        elif os.path.exists(act_save_path):
            act_dict = torch.load(act_save_path, map_location=lambda storage, loc: storage)
            self.act.load_state_dict(act_dict)
            self.act_target.load_state_dict(act_dict)
        else:
            print("FileNotFound when load_model: {}".format(mod_dir))


class AgentIntelAC(AgentSNAC):
    def __init__(self, state_dim, action_dim, net_dim):
        super(AgentSNAC, self).__init__()
        use_densenet = True
        self.learning_rate = 4e-4

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''dim and idx'''
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_idx = 1 + 1 + state_dim  # reward_dim==1, done_dim==1
        self.action_idx = self.state_idx + action_dim

        '''network'''
        self.act = ActorCritic(state_dim, action_dim, net_dim, use_densenet).to(self.device)
        self.act.train()
        self.act_target = ActorCritic(state_dim, action_dim, net_dim, use_densenet).to(self.device)
        self.act_target.eval()
        self.act_target.load_state_dict(self.act.state_dict())

        self.net_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate)
        self.criterion = nn.SmoothL1Loss()

        self.update_counter = 0
        self.loss_c_sum = 0.0
        self.rho = 0.5

    def update_parameter(self, memories, iter_num, batch_size, policy_noise, update_gap, gamma):  # 2020-02-02
        loss_a_sum = 0.0
        loss_c_sum = 0.0

        batch_size = int(batch_size * (1.0 + memories.now_len / memories.max_len))
        iter_num_c = int(iter_num * (1.0 + memories.now_len / 2 ** 18))
        iter_num_a = 0

        for _ in range(iter_num_c):
            with torch.no_grad():
                memory = memories.random_sample(batch_size)
                memory = torch.tensor(memory, device=self.device)

                reward = memory[:, 0:1]
                undone = memory[:, 1:2]
                state = memory[:, 2:self.state_idx]
                action = memory[:, self.state_idx:self.action_idx]
                next_state = memory[:, self.action_idx:]

                q_target, next_action0 = self.act_target.next__q_a_fix_bug(state, next_state, policy_noise)
                q_target = reward + undone * gamma * q_target

            '''loss C'''
            q_eval = self.act.critic(state, action)
            critic_loss = self.criterion(q_eval, q_target)
            loss_c_tmp = critic_loss.item()
            loss_c_sum += loss_c_tmp
            self.loss_c_sum += loss_c_tmp

            '''term A'''
            actor_term = self.criterion(self.act(next_state), next_action0)

            '''loss A'''
            action_cur = self.act(state)
            actor_loss = -self.act_target.critic(state, action_cur).mean()
            loss_a_sum += actor_loss.item()
            iter_num_a += 1

            '''united loss'''
            united_loss = critic_loss + actor_term * (1 - self.rho) + actor_loss * (self.rho * 0.5)

            self.net_optimizer.zero_grad()
            united_loss.backward()
            self.net_optimizer.step()

            self.update_counter += 1
            if self.update_counter == update_gap:
                self.update_counter = 0

                rho = np.exp(-(self.loss_c_sum / update_gap) ** 2)
                self.rho = self.rho * 0.75 + rho * 0.25
                self.loss_c_sum = 0.0

                # self.net_optimizer.param_groups[0]['lr'] = self.learning_rate * max(self.rho, 0.1)
                if self.rho > 0.1:
                    self.act_target.load_state_dict(self.act.state_dict())

        loss_a_avg = (loss_a_sum / iter_num_a) if iter_num_a else 0.0
        loss_c_avg = (loss_c_sum / iter_num_c) if iter_num_c else 0.0
        return loss_a_avg, loss_c_avg

    def save_or_load_model(self, mod_dir, is_save):
        act_save_path = '{}/actor.pth'.format(mod_dir)
        # cri_save_path = '{}/critic.pth'.format(mod_dir)

        if is_save:
            torch.save(self.act.state_dict(), act_save_path)
            # torch.save(self.cri.state_dict(), cri_save_path)
            # print("Saved act and cri:", mod_dir)
        elif os.path.exists(act_save_path):
            act_dict = torch.load(act_save_path, map_location=lambda storage, loc: storage)
            self.act.load_state_dict(act_dict)
            self.act_target.load_state_dict(act_dict)
            # cri_dict = torch.load(cri_save_path, map_location=lambda storage, loc: storage)
            # self.cri.load_state_dict(cri_dict)
            # self.cri_target.load_state_dict(cri_dict)
        else:
            print("FileNotFound when load_model: {}".format(mod_dir))


class AgentTD3(AgentSNAC):
    def __init__(self, state_dim, action_dim, net_dim):
        super(AgentSNAC, self).__init__()
        use_densenet = False
        learning_rate = 4e-4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''dim and idx'''
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_idx = 1 + 1 + state_dim  # reward_dim==1, done_dim==1
        self.action_idx = self.state_idx + action_dim

        '''network'''
        actor_dim = net_dim
        act = Actor(state_dim, action_dim, actor_dim, use_densenet).to(self.device)
        act.train()
        self.act = act
        self.act_optimizer = torch.optim.Adam(act.parameters(), lr=learning_rate / 4)

        act_target = Actor(state_dim, action_dim, actor_dim, use_densenet).to(self.device)
        act_target.eval()
        self.act_target = act_target
        self.act_target.load_state_dict(act.state_dict())

        '''critic'''
        critic_dim = int(net_dim * 1.25)
        cri = CriticTwin(state_dim, action_dim, critic_dim).to(self.device)  # TD3
        cri.train()
        self.cri = cri
        self.cri_optimizer = torch.optim.Adam(cri.parameters(), lr=learning_rate)

        cri_target = CriticTwin(state_dim, action_dim, critic_dim).to(self.device)  # TD3
        cri_target.eval()
        self.cri_target = cri_target
        self.cri_target.load_state_dict(cri.state_dict())

        self.criterion = nn.SmoothL1Loss()

        self.update_counter = 0
        self.loss_c_sum = 0.0
        self.rho = 0.5

    def update_parameter(self, memories, iter_num, batch_size, policy_noise, update_gap, gamma):  # 2020-02-02
        loss_a_sum = 0.0
        loss_c_sum = 0.0

        k = 1.0 + memories.now_len / memories.max_len
        batch_size = int(batch_size * k)
        iter_num = int(iter_num * k)

        for _ in range(iter_num):
            with torch.no_grad():
                memory = memories.random_sample(batch_size)
                memory = torch.tensor(memory, device=self.device)

                reward = memory[:, 0:1]
                undone = memory[:, 1:2]
                state = memory[:, 2:self.state_idx]
                action = memory[:, self.state_idx:self.action_idx]
                next_state = memory[:, self.action_idx:]

                next_action = self.act_target(next_state, policy_noise)
                q_target1, q_target2 = self.cri_target.get_q1_q2(next_state, next_action)
                q_target = torch.min(q_target1, q_target2)  # TD3
                q_target = reward + undone * gamma * q_target

            '''loss C'''
            q_eval1, q_eval2 = self.cri.get_q1_q2(state, action)  # TD3
            critic_loss = self.criterion(q_eval1, q_target) + self.criterion(q_eval2, q_target)
            loss_c_tmp = critic_loss.item() * 0.5  # TD3
            loss_c_sum += loss_c_tmp
            self.loss_c_sum += loss_c_tmp

            self.cri_optimizer.zero_grad()
            critic_loss.backward()
            self.cri_optimizer.step()

            '''loss A'''
            action_cur = self.act(state)
            actor_loss = -self.cri(state, action_cur).mean()
            loss_a_sum += actor_loss.item()

            self.act_optimizer.zero_grad()
            actor_loss.backward()
            self.act_optimizer.step()

            self.update_counter += 1
            if self.update_counter == update_gap:
                self.update_counter = 0

                # self.act_target.load_state_dict(self.act.state_dict())
                # self.cri_target.load_state_dict(self.cri.state_dict())
                self.soft_update(self.act_target, self.act, tau=0.01)
                self.soft_update(self.cri_target, self.cri, tau=0.01)

                rho = np.exp(-(self.loss_c_sum / update_gap) ** 2)
                self.rho = self.rho * 0.75 + rho * 0.25
                self.act_optimizer.param_groups[0]['lr'] = 1e-4 * self.rho
                self.loss_c_sum = 0.0

        return loss_a_sum / iter_num, loss_c_sum / iter_num,


class AgentPPO:
    def __init__(self, state_dim, action_dim, net_dim):
        # super(AgentSNAC, self).__init__()
        """
        """
        self.act_lr = 4e-4  # learning rate of actor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''network'''
        act = ActorCriticPPO(state_dim, action_dim, net_dim).to(self.device)
        act.train()
        self.act = act
        self.act_optimizer = torch.optim.Adam(act.parameters(), lr=self.act_lr)  # , betas=(0.5, 0.99))

        self.criterion = nn.SmoothL1Loss()

        self.clip = 0.5  # constant
        self.update_counter = 0
        self.loss_c_sum = 0.0
        self.loss_coeff_value = 0.5
        self.loss_coeff_entropy = 0.02  # 0.01

    def inactive_in_env_ppo(self, env, max_step, max_memo, max_action, state_norme):
        memory = MemoryList()
        rewards = []
        steps = []

        step_counter = 0
        while step_counter < max_memo:
            state = env.reset()
            reward_sum = 0
            t = 0

            state = state_norme(state)  # if state_norm:
            for t in range(max_step):
                actions, log_probs, q_values = self.select_actions(state[np.newaxis], explore_noise=True)
                action = actions[0]
                log_prob = log_probs[0]
                q_value = q_values[0]

                next_state, reward, done, _ = env.step(action * max_action)
                reward_sum += reward

                next_state = state_norme(next_state)  # if state_norm:
                mask = 0 if done else 1

                # memory.push(state, q_value, action, log_prob, mask, next_state, reward)
                memory.push(state, q_value, action, log_prob, mask, reward)

                if done:
                    break

                state = next_state
            rewards.append(reward_sum)

            t += 1
            steps.append(t)
            step_counter += t
        return rewards, steps, memory

    def update_parameter_ppo(self, memory, batch_size, gamma, ep_ratio):
        clip = 0.2
        lamda = 0.97
        num_epoch = 10

        all_batch = memory.random_sample()
        max_memo = len(memory)

        all_reward = torch.tensor(all_batch.reward, dtype=torch.float32, device=self.device)
        all_value = torch.tensor(all_batch.value, dtype=torch.float32, device=self.device)
        all_mask = torch.tensor(all_batch.mask, dtype=torch.float32, device=self.device)
        all_action = torch.tensor(all_batch.action, dtype=torch.float32, device=self.device)
        all_state = torch.tensor(all_batch.state, dtype=torch.float32, device=self.device)
        all_log_prob = torch.tensor(all_batch.log_prob, dtype=torch.float32, device=self.device)
        # next_state not use?

        '''calculate prev (return, value, advantage)'''
        all_deltas = torch.empty(max_memo, dtype=torch.float32, device=self.device)
        all_returns = torch.empty(max_memo, dtype=torch.float32, device=self.device)
        all_advantages = torch.empty(max_memo, dtype=torch.float32, device=self.device)

        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        for i in range(max_memo - 1, -1, -1):
            all_deltas[i] = all_reward[i] + gamma * prev_value * all_mask[i] - all_value[i]
            all_returns[i] = all_reward[i] + gamma * prev_return * all_mask[i]
            # ref: https://arxiv.org/pdf/1506.02438.pdf (generalization advantage estimate)
            all_advantages[i] = all_deltas[i] + gamma * lamda * prev_advantage * all_mask[i]

            prev_return = all_returns[i]
            prev_value = all_value[i]
            prev_advantage = all_advantages[i]

        all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-6)  # if advantage_norm:

        '''mini all_batch sample'''
        loss_total = loss_value = None

        for i_epoch in range(int(num_epoch * max_memo / batch_size)):
            ind = rd.choice(max_memo, batch_size, replace=False)
            states = all_state[ind]
            actions = all_action[ind]
            log_probs = all_log_prob[ind]
            advantages = all_advantages[ind]
            returns = all_returns[ind]

            new_values = self.act.critic(states).flatten()

            new_actions = self.act(states)
            new_log_probs = self.act.get__log_prob(new_actions, actions)

            ratio = torch.exp(new_log_probs - log_probs)
            surr1 = advantages * ratio
            surr2 = advantages * ratio.clamp(1 - self.clip, 1 + self.clip)
            loss_surr = - torch.mean(torch.min(surr1, surr2))

            loss_value = torch.mean((new_values - returns).pow(2)) / (returns.std() * 6.0)

            loss_entropy = torch.mean(torch.exp(new_log_probs) * new_log_probs)

            loss_total = loss_surr + self.loss_coeff_value * loss_value + self.loss_coeff_entropy * loss_entropy
            self.act_optimizer.zero_grad()
            loss_total.backward()
            self.act_optimizer.step()

        '''schedule (clip, adam)'''
        # ep_ratio = 1 - (now_epoch / max_epoch)
        self.clip = clip * ep_ratio
        self.act_optimizer.param_groups[0]['lr'] = self.act_lr * ep_ratio

        # return loss_total.data, loss_surr.data, loss_value.data, loss_entropy.data
        return loss_total.item(), loss_value.item(),

    def select_actions(self, states, explore_noise=0.0):  # CPU array to GPU tensor to CPU array
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = self.act(states)

        if explore_noise == 0.0:
            actions = actions.cpu().data.numpy()
            return actions
        else:
            a_noise, log_prob = self.act.get__a__log_prob(actions)
            a_noise = a_noise.cpu().data.numpy()

            log_prob = log_prob.cpu().data.numpy()

            q_value = self.act.critic(states)
            q_value = q_value.cpu().data.numpy()
            return a_noise, log_prob, q_value

    def save_or_load_model(self, mod_dir, is_save):
        act_save_path = '{}/actor.pth'.format(mod_dir)
        # cri_save_path = '{}/critic.pth'.format(mod_dir)

        if is_save:
            torch.save(self.act.state_dict(), act_save_path)
            # torch.save(self.cri.state_dict(), cri_save_path)
            # print("Saved act and cri:", mod_dir)
        elif os.path.exists(act_save_path):
            act_dict = torch.load(act_save_path, map_location=lambda storage, loc: storage)
            self.act.load_state_dict(act_dict)
            # self.act_target.load_state_dict(act_dict)
            # cri_dict = torch.load(cri_save_path, map_location=lambda storage, loc: storage)
            # self.cri.load_state_dict(cri_dict)
            # self.cri_target.load_state_dict(cri_dict)
        else:
            print("FileNotFound when load_model: {}".format(mod_dir))


class AgentSAC:
    def __init__(self, env, state_dim, action_dim, net_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = 2e-4

        '''network'''
        self.act = ActorSAC(state_dim, action_dim, net_dim).to(self.device)
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate * 0.5)

        self.act_target = ActorSAC(state_dim, action_dim, net_dim).to(self.device)

        self.cri = CriticTwin(state_dim, action_dim, int(net_dim * 1.25)).to(self.device)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate)

        # from copy import deepcopy
        # self.cri_target = copy.deepcopy(self.cri).to(self.device)
        # same as:
        self.cri_target = CriticTwin(state_dim, action_dim, int(net_dim * 1.25)).to(self.device)

        self.criterion = nn.MSELoss()

        '''extension: alpha and entropy'''
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam((self.log_alpha,), lr=self.learning_rate)
        self.target_entropy = -1

        '''training'''
        self.state = env.reset()
        self.reward_sum = 0.0
        self.step_sum = 0
        self.update_counter = 0

    @staticmethod
    def soft_target_update(target, source, tau=4e-3):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def select_actions(self, states, explore_noise=0.0):  # CPU array to GPU tensor to CPU array
        states = torch.tensor(states, dtype=torch.float32, device=self.device)

        if explore_noise == 0.0:
            actions = self.act(states)
        else:
            a_means, a_stds = self.act.actor(states)
            actions = torch.normal(a_means, a_stds)

        actions = actions.tanh()
        actions = actions.cpu().data.numpy()
        return actions

    def save_or_load_model(self, mod_dir, is_save):
        act_save_path = '{}/actor.pth'.format(mod_dir)
        cri_save_path = '{}/critic.pth'.format(mod_dir)

        if is_save:
            torch.save(self.act.state_dict(), act_save_path)
            torch.save(self.cri.state_dict(), cri_save_path)
            # print("Saved act and cri:", mod_dir)
        elif os.path.exists(act_save_path):
            act_dict = torch.load(act_save_path, map_location=lambda storage, loc: storage)
            self.act.load_state_dict(act_dict)
            # self.act_target.load_state_dict(act_dict)
            cri_dict = torch.load(cri_save_path, map_location=lambda storage, loc: storage)
            self.cri.load_state_dict(cri_dict)
            # self.cri_target.load_state_dict(cri_dict)
        else:
            print("FileNotFound when load_model: {}".format(mod_dir))

    def update_buffer(self, env, memo, max_step, max_action, reward_scale, gamma):
        rewards = list()
        steps = list()
        for t in range(max_step):
            '''inactive with environment'''
            action = self.select_actions((self.state,), explore_noise=True)[0]
            next_state, reward, done, _ = env.step(action * max_action)

            self.reward_sum += reward
            self.step_sum += 1

            '''update memory (replay buffer)'''
            reward_ = reward * reward_scale
            mask = 0.0 if done else gamma
            memo.add_memo((reward_, mask, self.state, action, next_state))

            self.state = next_state
            if done:
                rewards.append(self.reward_sum)
                self.reward_sum = 0.0

                steps.append(self.step_sum)
                self.step_sum = 0

                self.state = env.reset()
        memo.init_after_add_memo()
        return rewards, steps

    def update_parameters(self, memo, max_step, batch_size, update_gap):
        loss_a_sum = 0.0
        loss_c_sum = 0.0
        for _ in range(max_step):
            with torch.no_grad():
                rewards, marks, states, actions, next_states = memo.random_sample(batch_size, self.device)

            """actor loss"""

            '''stochastic policy'''
            actions_noise, log_prob = self.act.get__a__log_prob(states, self.device)
            '''auto alpha for actor'''
            alpha_loss = -(self.log_alpha * (self.target_entropy + log_prob).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            '''actor loss'''
            alpha = self.log_alpha.exp()
            q0_min = torch.min(*self.cri(states, actions_noise))
            actor_loss = (alpha * log_prob - q0_min).mean()
            loss_a_sum += actor_loss.item()
            self.act_optimizer.zero_grad()
            actor_loss.backward()
            self.act_optimizer.step()

            """critic loss1"""

            '''q0_target'''
            with torch.no_grad():
                next_actions_noise, next_log_prob = self.act_target.get__a__log_prob(next_states, self.device)

                next_q0_min = torch.min(*self.cri_target(next_states, next_actions_noise))
                next_q0_target = next_q0_min - next_log_prob * alpha
                q0_target = rewards + marks * next_q0_target
            '''q1 and q2'''
            q1_value, q2_value = self.cri(states, actions)
            '''critic loss'''
            critic_loss = self.criterion(q1_value, q0_target) + self.criterion(q2_value, q0_target)
            loss_c_sum += critic_loss.item() * 0.5
            self.cri_optimizer.zero_grad()
            critic_loss.backward()
            self.cri_optimizer.step()

            """critic loss2"""
            with torch.no_grad():
                rewards, marks, states, actions, next_states = memo.random_sample(batch_size, self.device)
            '''q0_target'''
            with torch.no_grad():
                next_actions_noise, next_log_prob = self.act_target.get__a__log_prob(next_states, self.device)

                next_q0_min = torch.min(*self.cri_target(next_states, next_actions_noise))
                next_q0_target = next_q0_min - next_log_prob * alpha
                q0_target = rewards + marks * next_q0_target
            '''q1 and q2'''
            q1_value, q2_value = self.cri(states, actions)
            '''critic loss'''
            critic_loss = self.criterion(q1_value, q0_target) + self.criterion(q2_value, q0_target)
            loss_c_sum += critic_loss.item() * 0.5
            self.cri_optimizer.zero_grad()
            critic_loss.backward()
            self.cri_optimizer.step()

            """target update"""
            self.soft_target_update(self.act_target, self.act)  # todo soft
            self.soft_target_update(self.cri_target, self.cri)  # todo soft
            # self.update_counter += 1
            # if self.update_counter > update_gap:
            #     self.update_counter = 0
            #     self.cri_target.load_state_dict(self.cri.state_dict())  # hard target update

        loss_a = loss_a_sum / max_step
        loss_c = loss_c_sum / max_step
        return loss_a, loss_c


def initial_exploration(env, memo, max_step, max_action, reward_scale, gamma, action_dim):
    state = env.reset()

    rewards = list()
    reward_sum = 0.0
    steps = list()
    step = 0

    global_step = 0
    while global_step < max_step:
        # action = np.tanh(rd.normal(0, 0.25, size=action_dim))  # zero-mean gauss exploration
        action = rd.uniform(-1.0, +1.0, size=action_dim)  # uniform exploration

        next_state, reward, done, _ = env.step(action * max_action)
        reward_sum += reward
        step += 1

        adjust_reward = reward * reward_scale
        mask = 0.0 if done else gamma
        memo.add_memo((adjust_reward, mask, state, action, next_state))

        state = next_state
        if done:
            rewards.append(reward_sum)
            steps.append(step)
            global_step += step

            state = env.reset()  # reset the environment
            reward_sum = 0.0
            step = 1

    memo.init_after_add_memo()
    return rewards, steps


"""utils"""


class MemoryList:  # todo del, for PPO
    def __init__(self, ):
        self.memory = []
        from collections import namedtuple
        self.transition = namedtuple(
            'Transition',
            # ('state', 'value', 'action', 'log_prob', 'mask', 'next_state', 'reward')
            ('state', 'value', 'action', 'log_prob', 'mask', 'reward')
        )

    def push(self, *args):
        self.memory.append(self.transition(*args))

    def sample(self):
        return self.transition(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)


class BufferList:
    def __init__(self, memo_max_len):
        self.memories = list()

        self.max_len = memo_max_len
        self.now_len = len(self.memories)

    def add_memo(self, memory_tuple):
        self.memories.append(memory_tuple)

    def init_after_add_memo(self):
        del_len = len(self.memories) - self.max_len
        if del_len > 0:
            del self.memories[:del_len]
            # print('Length of Deleted Memories:', del_len)

        self.now_len = len(self.memories)

    def random_sample(self, batch_size, device):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # indices = rd.choice(self.memo_len, batch_size, replace=False)  # why perform worse?
        # indices = rd.choice(self.memo_len, batch_size, replace=True)  # why perform better?
        # same as:
        indices = rd.randint(self.now_len, size=batch_size)

        '''convert list into array'''
        arrays = [list()
                  for _ in range(5)]  # len(self.memories[0]) == 5
        for index in indices:
            items = self.memories[index]
            for item, array in zip(items, arrays):
                array.append(item)

        '''convert array into torch.tensor'''
        tensors = [torch.tensor(np.array(ary), dtype=torch.float32, device=device)
                   for ary in arrays]
        return tensors


class BufferTuple:  # todo plan for PPO
    def __init__(self, memo_max_len):
        self.memories = list()

        self.max_len = memo_max_len
        self.now_len = None  # init in init_after_add_memo()

        from collections import namedtuple
        self.transition = namedtuple(
            'Transition', ('reward', 'mask', 'state', 'action', 'next_state',)
        )

    def add_memo(self, args):
        self.memories.append(self.transition(*args))

    def init_after_add_memo(self):
        del_len = len(self.memories) - self.max_len
        if del_len > 0:
            del self.memories[:del_len]
            # print('Length of Deleted Memories:', del_len)

        self.now_len = len(self.memories)

    def random_sample(self, batch_size, device):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # indices = rd.choice(self.memo_len, batch_size, replace=False)  # why perform worse?
        # indices = rd.choice(self.memo_len, batch_size, replace=True)  # why perform better?
        # same as:
        indices = rd.randint(self.now_len, size=batch_size)

        '''convert tuple into array'''
        arrays = self.transition(*zip(*[self.memories[i] for i in indices]))

        '''convert array into torch.tensor'''
        tensors = [torch.tensor(np.array(ary), dtype=torch.float32, device=device)
                   for ary in arrays]
        return tensors


class BufferArray:  # 2020-05-20
    def __init__(self, memo_max_len, state_dim, action_dim, ):
        memo_dim = 1 + 1 + state_dim + action_dim + state_dim
        self.memories = np.empty((memo_max_len, memo_dim), dtype=np.float32)

        self.next_idx = 0
        self.is_full = False
        self.max_len = memo_max_len
        self.now_len = self.max_len if self.is_full else self.next_idx

        self.state_idx = 1 + 1 + state_dim  # reward_dim==1, done_dim==1
        self.action_idx = self.state_idx + action_dim

    def add_memo(self, memo_tuple):
        self.memories[self.next_idx] = np.hstack(memo_tuple)
        self.next_idx = self.next_idx + 1
        if self.next_idx >= self.max_len:
            self.is_full = True
            self.next_idx = 0

    def init_after_add_memo(self):
        self.now_len = self.max_len if self.is_full else self.next_idx

    def random_sample(self, batch_size, device):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # indices = rd.choice(self.memo_len, batch_size, replace=False)  # why perform worse?
        # indices = rd.choice(self.memo_len, batch_size, replace=True)  # why perform better?
        # same as:
        indices = rd.randint(self.now_len, size=batch_size)

        memory = self.memories[indices]
        memory = torch.tensor(memory, device=device)

        '''convert array into torch.tensor'''
        tensors = (
            memory[:, 0:1],  # rewards
            memory[:, 1:2],  # masks, mark == (1-float(done)) * gamma
            memory[:, 2:self.state_idx],  # states
            memory[:, self.state_idx:self.action_idx],  # actions
            memory[:, self.action_idx:],  # next_states
        )
        return tensors


class Recorder:
    def __init__(self, agent, max_step, max_action, target_reward,
                 env_name, eva_size=100, show_gap=2 ** 7, smooth_kernel=2 ** 4,
                 state_norm=None, **_kwargs):
        self.show_gap = show_gap
        self.smooth_kernel = smooth_kernel

        '''get_eva_reward(agent, env_list, max_step, max_action)'''
        self.agent = agent
        self.env_list = [gym.make(env_name) for _ in range(eva_size)]
        self.max_step = max_step
        self.max_action = max_action
        self.e1 = 3
        self.e2 = int(eva_size // np.e)
        self.running_stat = state_norm

        '''reward'''
        self.rewards = get_eva_reward(agent, self.env_list[:5], max_step, max_action, self.running_stat)
        self.reward_avg = np.average(self.rewards)
        self.reward_std = float(np.std(self.rewards))
        self.reward_target = target_reward
        self.reward_max = self.reward_avg

        self.record_epoch = list()  # record_epoch.append((epoch_reward, actor_loss, critic_loss, iter_num))
        self.record_eval = [(0, self.reward_avg, self.reward_std), ]  # [(epoch, reward_avg, reward_std), ]
        self.total_step = 0

        self.epoch = 0
        self.train_time = 0  # train_time
        self.train_timer = timer()  # train_time
        self.start_time = self.show_time = timer()
        print("epoch|   reward   r_max    r_ave    r_std |  loss_A loss_C |step")

    def show_reward(self, epoch_rewards, iter_numbers, loss_a, loss_c):
        self.train_time += timer() - self.train_timer  # train_time
        self.epoch += len(epoch_rewards)

        if isinstance(epoch_rewards, float):
            epoch_rewards = (epoch_rewards,)
            iter_numbers = (iter_numbers,)
        for reward, iter_num in zip(epoch_rewards, iter_numbers):
            self.record_epoch.append((reward, loss_a, loss_c, iter_num))
            self.total_step += iter_num

        if timer() - self.show_time > self.show_gap:
            self.rewards = get_eva_reward(self.agent, self.env_list[:self.e1], self.max_step, self.max_action,
                                          self.running_stat)
            self.reward_avg = np.average(self.rewards)
            self.reward_std = float(np.std(self.rewards))
            self.record_eval.append((len(self.record_epoch), self.reward_avg, self.reward_std))

            slice_reward = np.array(self.record_epoch[-self.smooth_kernel:])[:, 0]
            smooth_reward = np.average(slice_reward, axis=0)
            print("{:4} |{:8.2f} {:8.2f} {:8.2f} {:8.2f} |{:8.2f} {:6.2f} |{:.2e}".format(
                len(self.record_epoch),
                smooth_reward, self.reward_max, self.reward_avg, self.reward_std,
                loss_a, loss_c, self.total_step))

            self.show_time = timer()  # reset show_time after get_eva_reward_batch !
        else:
            self.rewards = list()

    def check_reward(self, cwd, loss_a, loss_c):  # 2020-05-05
        is_solved = False
        if self.reward_avg >= self.reward_max:  # and len(self.rewards) > 1:  # 2020-04-30
            self.rewards.extend(get_eva_reward(self.agent, self.env_list[:self.e2], self.max_step, self.max_action,
                                               self.running_stat))
            self.reward_avg = np.average(self.rewards)

            if self.reward_avg >= self.reward_max:
                self.reward_max = self.reward_avg

                '''NOTICE! Recorder saves the agent with max reward automatically. '''
                self.agent.save_or_load_model(cwd, is_save=True)

                if self.reward_max >= self.reward_target:
                    res_env_len = len(self.env_list) - len(self.rewards)
                    self.rewards.extend(get_eva_reward(
                        self.agent, self.env_list[:res_env_len], self.max_step, self.max_action,
                        self.running_stat))
                    self.reward_avg = np.average(self.rewards)
                    self.reward_max = self.reward_avg

                    if self.reward_max >= self.reward_target:
                        print("########## Solved! ###########")
                        is_solved = True

            self.reward_std = float(np.std(self.rewards))
            self.record_eval[-1] = (len(self.record_epoch), self.reward_avg, self.reward_std)  # refresh
            print("{:4} |{:8} {:8.2f} {:8.2f} {:8.2f} |{:8.2f} {:6.2f} |{:.2e}".format(
                len(self.record_epoch),
                '', self.reward_max, self.reward_avg, self.reward_std,
                loss_a, loss_c, self.total_step, ))

        self.train_timer = timer()  # train_time
        return is_solved

    def show_and_save(self, env_name, cwd):  # 2020-04-30
        iter_used = self.total_step  # int(sum(np.array(self.record_epoch)[:, -1]))
        time_used = int(timer() - self.start_time)
        print('Used Time:', time_used)
        self.train_time = int(self.train_time)  # train_time
        print('TrainTime:', self.train_time)  # train_time

        print_str = "{}-{:.2f}AVE-{:.2f}STD-{}E-{}S-{}T".format(
            env_name, self.reward_max, self.reward_std, self.epoch, self.train_time, iter_used)  # train_time
        print(print_str)
        nod_path = '{}/{}.txt'.format(cwd, print_str)
        os.mknod(nod_path, ) if not os.path.exists(nod_path) else None

        np.save('%s/record_epoch.npy' % cwd, self.record_epoch)
        np.save('%s/record_eval.npy' % cwd, self.record_eval)
        print("Saved record_*.npy in:", cwd)

        return self.train_time


class RewardNormalization:
    def __init__(self, n_max, n_min, size=2 ** 7):
        self.k = size / (n_max - n_min)
        # print(';;RewardNorm', n_max, n_min)
        # print(';;RewardNorm', self(n_max), self(n_min))

    def __call__(self, n):
        return n * self.k


class RunningStat:  # for class AutoNormalization
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        # assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            pre_memo = self._M.copy()
            self._M[...] = pre_memo + (x - pre_memo) / self._n
            self._S[...] = self._S + (x - pre_memo) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


class AutoNormalization:
    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update:
            self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    @staticmethod
    def output_shape(input_space):
        return input_space.shape


class OrnsteinUhlenbeckProcess(object):
    def __init__(self, size, theta=0.15, sigma=0.3, x0=0.0, dt=1e-2):
        """
        Source: https://github.com/slowbull/DDPG/blob/master/src/explorationnoise.py
        I think that:
        It makes Zero-mean Gaussian Noise more stable.
        It helps agent explore better in a inertial system.
        """
        self.theta = theta
        self.sigma = sigma
        self.x0 = x0
        self.dt = dt
        self.size = size

    def __call__(self):
        noise = self.sigma * np.sqrt(self.dt) * rd.normal(size=self.size)
        x = self.x0 - self.theta * self.x0 * self.dt + noise
        self.x0 = x  # update x0
        return x


def get_eva_reward(agent, env_list, max_step, max_action, running_state=None):  # class Recorder 2020-01-11
    act = agent.act  # agent.net,
    act.eval()

    env_list_copy = env_list.copy()
    eva_size = len(env_list_copy)  # 100

    sum_rewards = [0.0, ] * eva_size
    states = [env.reset() for env in env_list_copy]

    reward_sums = list()
    for iter_num in range(max_step):
        if running_state:
            states = [running_state(state, update=False) for state in states]  # if state_norm:
        actions = agent.select_actions(states)

        next_states = list()
        done_list = list()
        if max_action:  # Continuous action space
            actions *= max_action
        for i in range(len(env_list_copy) - 1, -1, -1):
            next_state, reward, done, _ = env_list_copy[i].step(actions[i])

            next_states.insert(0, next_state)
            sum_rewards[i] += reward
            done_list.insert(0, done)
            if done:
                reward_sums.append(sum_rewards[i])
                del sum_rewards[i]
                del env_list_copy[i]
        states = next_states

        if len(env_list_copy) == 0:
            break
    else:
        reward_sums.extend(sum_rewards)
    act.train()

    return reward_sums
