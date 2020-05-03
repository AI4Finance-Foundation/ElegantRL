import os
from time import time as timer

import gym
import numpy as np
import numpy.random as rd
import torch
import torch.nn as nn

from AgentNetwork import Actor, Critic  # class AgentSNAC
from AgentNetwork import ActorCritic  # class AgentIntelAC
from AgentNetwork import QNetwork  # class AgentQLearning
from AgentNetwork import CriticTwin  # class AgentTD3

"""
2019-07-01 Zen4Jia1Hao2, GitHub: YonV1943 DL_RL_Zoo RL
2019-11-11 Issay-0.0 [Essay Consciousness]
2020-02-02 Issay-0.1 Deep Learning Techniques (spectral norm, DenseNet, etc.) 
2020-04-04 Issay-0.1 [An Essay of Consciousness by YonV1943], IntelAC
2020-04-20 Issay-0.2 SN_AC, IntelAC_UnitedLoss
2020-04-22 Issay-0.2 [Essay, LongDear's Cerebellum (Little Brain)]

I consider that Reinforcement Learning Algorithms before 2020 have not consciousness
They feel more like a Cerebellum (Little Brain) for Machines.
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
                memories.add(np.hstack((r_norm(reward), 1 - float(done), state, action, next_state)))

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

        k = 1.0 + memories.size / memories.max_size
        batch_size = int(batch_size * k)
        iter_num = int(iter_num * k)

        for _ in range(iter_num):
            with torch.no_grad():
                memory = memories.sample(batch_size)
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
    def soft_update(target, source, tau=0.01):
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

        k = 1.0 + memories.size / memories.max_size
        batch_size = int(batch_size * k)
        iter_num = int(iter_num * k)

        for _ in range(iter_num):
            with torch.no_grad():
                memory = memories.sample(batch_size)
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
        learning_rate = 4e-4

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

        self.net_optimizer = torch.optim.Adam(self.act.parameters(), lr=learning_rate)
        self.criterion = nn.SmoothL1Loss()

        self.update_counter = 0
        self.loss_c_sum = 0.0
        self.rho = 0.5

    def update_parameter(self, memories, iter_num, batch_size, policy_noise, update_gap, gamma):  # 2020-02-02
        loss_a_sum = 0.0
        loss_c_sum = 0.0

        k = 1.0 + memories.size / memories.max_size
        batch_size = int(batch_size * k)
        iter_num_c = int(iter_num * k)
        iter_num_a = 0

        for _ in range(iter_num_c):
            with torch.no_grad():
                memory = memories.sample(batch_size)
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
        use_spectral_norm = False
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
        cri = CriticTwin(state_dim, action_dim, critic_dim,
                         use_densenet, use_spectral_norm).to(self.device)  # TD3
        cri.train()
        self.cri = cri
        self.cri_optimizer = torch.optim.Adam(cri.parameters(), lr=learning_rate)

        cri_target = CriticTwin(state_dim, action_dim, critic_dim,
                                use_densenet, use_spectral_norm).to(self.device)  # TD3
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

        k = 1.0 + memories.size / memories.max_size
        batch_size = int(batch_size * k)
        iter_num = int(iter_num * k)

        for _ in range(iter_num):
            with torch.no_grad():
                memory = memories.sample(batch_size)
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


"""utils"""


class Memories:  # Experiment Replay Buffer 2020-02-02
    def __init__(self, max_size, memo_dim):
        self.ptr_u = 0  # pointer_for_update
        self.ptr_s = 0  # pointer_for_sample
        self.is_full = False
        self.indices = np.arange(max_size)

        self.size = 0  # real-time memories size
        self.max_size = max_size

        self.memories = np.empty((max_size, memo_dim), dtype=np.float32)

    def add(self, memory):
        self.memories[self.ptr_u, :] = memory

        self.ptr_u += 1
        if self.ptr_u == self.max_size:
            self.ptr_u = 0
            self.is_full = True
            print('Memories is_full')
        self.size = self.max_size if self.is_full else self.ptr_u

    def sample(self, batch_size):
        self.ptr_s += batch_size
        if self.ptr_s >= self.size:
            self.ptr_s = batch_size
            rd.shuffle(self.indices[:self.size])

        batch_memories = self.memories[self.indices[self.ptr_s - batch_size:self.ptr_s]]
        return batch_memories

    def save_or_load_memo(self, net_dir, is_save):
        save_path = "%s/memories.npy" % net_dir
        if is_save:
            ptr_u = self.max_size if self.is_full else self.ptr_u
            np.save(save_path, self.memories[:ptr_u])
            print('Saved memories.npy in:', net_dir)
        elif not os.path.exists(save_path):
            print("FileNotFound when load_memories:", save_path)
        else:  # exist
            memories = np.load(save_path)

            memo_len = memories.shape[0]
            if memo_len > self.max_size:
                memo_len = self.max_size
                self.ptr_u = self.max_size
                print("Memories_num change:", memo_len)
            else:
                self.ptr_u = memo_len
                self.size = memo_len
                print("Memories_num:", self.ptr_u)

            self.memories[:self.ptr_u] = memories[:memo_len]
            if self.ptr_u == self.max_size:
                self.ptr_u = 0
                self.is_full = True
                print('Memories is_full!')

            print("Load Memories:", save_path)


class Recorder:
    def __init__(self, agent, max_step, max_action, target_reward,
                 env_name, eva_size=100, show_gap=2 ** 7, smooth_kernel=2 ** 4,
                 running_stat=None):
        self.show_gap = show_gap
        self.smooth_kernel = smooth_kernel

        '''get_eva_reward(agent, env_list, max_step, max_action)'''
        self.agent = agent
        self.env_list = [gym.make(env_name) for _ in range(eva_size)]
        self.max_step = max_step
        self.max_action = max_action
        self.e1 = 3
        self.e2 = int(eva_size // np.e)

        '''reward'''
        self.rewards = get_eva_reward(agent, self.env_list[:5], max_step, max_action)
        self.reward_avg = np.average(self.rewards)
        self.reward_std = float(np.std(self.rewards))
        self.reward_target = target_reward
        self.reward_max = self.reward_avg

        self.record_epoch = list()  # record_epoch.append((epoch_reward, actor_loss, critic_loss, iter_num))
        self.record_eval = [(0, self.reward_avg, self.reward_std), ]  # [(epoch, reward_avg, reward_std), ]
        self.total_step = 0
        self.running_stat = running_stat

        self.epoch = 0
        self.train_time = 0  # train_time
        self.train_timer = timer()  # train_time
        self.start_time = self.show_time = timer()
        print("epoch|   reward   r_max    r_ave    r_std | loss_A  loss_C |step")

    def show_reward(self, epoch, epoch_rewards, iter_numbers, actor_loss, critic_loss):
        self.train_time += timer() - self.train_timer  # train_time

        self.epoch = epoch

        if isinstance(epoch_rewards, float):
            epoch_rewards = (epoch_rewards,)
            iter_numbers = (iter_numbers,)
        for reward, iter_num in zip(epoch_rewards, iter_numbers):
            self.record_epoch.append((reward, actor_loss, critic_loss, iter_num))
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
                actor_loss, critic_loss, self.total_step))

            self.show_time = timer()  # reset show_time after get_eva_reward_batch !
        else:
            self.rewards = list()

    def check_reward(self, cwd, actor_loss, critic_loss):  # 2020-04-30
        is_solved = False
        if self.reward_avg >= self.reward_max:  # and len(self.rewards) > 1:  # 2020-04-30
            self.rewards.extend(get_eva_reward(self.agent, self.env_list[:self.e2], self.max_step, self.max_action,
                                               self.running_stat))
            self.reward_avg = np.average(self.rewards)

            if self.reward_avg >= self.reward_max:
                self.reward_max = self.reward_avg
                self.agent.save_or_load_model(cwd, is_save=True)

                if self.reward_avg >= self.reward_target:
                    res_env_len = len(self.env_list) - len(self.rewards)
                    self.rewards.extend(get_eva_reward(
                        self.agent, self.env_list[:res_env_len], self.max_step, self.max_action))
                    self.reward_avg = np.average(self.rewards)

                    if self.reward_avg >= self.reward_target:
                        print("########## Solved! ###########")
                        is_solved = True

            self.reward_std = float(np.std(self.rewards))
            self.record_eval[-1] = (len(self.record_epoch), self.reward_avg, self.reward_std)  # refresh
            print("{:4} |{:8} {:8.2f} {:8.2f} {:8.2f} |{:8.2f} {:6.2f} |{:.2e}".format(
                len(self.record_epoch),
                '', self.reward_max, self.reward_avg, self.reward_std,
                actor_loss, critic_loss, self.total_step, ))

        self.train_timer = timer()  # train_time
        return is_solved

    def show_and_save(self, env_name, cwd):  # 2020-04-30
        iter_used = self.total_step  # int(sum(np.array(self.record_epoch)[:, -1]))
        time_used = int(timer() - self.start_time)
        print('Used Time:', time_used)
        self.train_time = int(self.train_time)  # train_time
        print('TrainTime:', self.train_time)  # train_time

        print_str = "{}-{:.2f}AVE-{:.2f}STD-{}E-{}S-{}T".format(
            env_name, self.reward_avg, self.reward_std, self.epoch, self.train_time, iter_used)  # train_time
        print(print_str)
        nod_path = '{}/{}.txt'.format(cwd, print_str)
        os.mknod(nod_path, ) if not os.path.exists(nod_path) else None

        np.save('%s/record_epoch.npy' % cwd, self.record_epoch)
        np.save('%s/record_eval.npy' % cwd, self.record_eval)
        print("Saved record_*.npy in:", cwd)

        return self.train_time


class RewardNorm:
    def __init__(self, n_max, n_min):
        self.k = 2 * 128 / (n_max - n_min)
        # print(';;RewardNorm', n_max, n_min)
        # print(';;RewardNorm', self(n_max), self(n_min))

    def __call__(self, n):
        return n * self.k


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
