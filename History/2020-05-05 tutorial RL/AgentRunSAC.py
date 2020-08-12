import argparse
import random
import copy
import os
from abc import ABC

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import Distribution

from AgentRun import get_env_info, draw_plot_with_npy
from AgentRun import Recorder

"""
Reference: https://github.com/TianhongDai/reinforcement-learning-algorithms/tree/master/rl_algorithms/sac
Modify: Yonv1943 Zen4 Jia1Hao2.
"""


class ActorSAC(nn.Module):
    def __init__(self, state_dim, action_dim, mid_net):
        super(ActorSAC, self).__init__()
        self.log_std_min = -20
        self.log_std_max = 2

        self.net = nn.Sequential(nn.Linear(state_dim, mid_net), nn.ReLU(),
                                 nn.Linear(mid_net, mid_net), nn.ReLU(), )
        self.net_mean = nn.Linear(mid_net, action_dim)
        self.net_log_std = nn.Linear(mid_net, action_dim)

    def forward(self, state):
        x = self.net(state)
        action_mean = self.net_mean(x)
        return action_mean

    def actor(self, state):
        x = self.net(state)
        action_mean = self.net_mean(x)
        log_std = self.net_log_std(x)
        log_std = log_std.clamp(self.log_std_min, self.log_std_max)
        action_std = log_std.exp()
        return action_mean, action_std

    def get__a__log_prob(self, states):
        a_mean, a_std = self.actor(states)
        noise = torch.randn_like(a_mean, requires_grad=True)  # device=self.device
        pre_tanh_value = a_mean + a_std * noise
        actions_noise = pre_tanh_value.tanh()

        log_prob = Normal(a_mean, a_std).log_prob(pre_tanh_value) - (-actions_noise.pow(2) + (1 + 1e-6)).log()
        return actions_noise, log_prob


class CriticSAC(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim, use_densenet, use_spectral_norm):
        super(CriticSAC, self).__init__()

        self.net = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                 nn.Linear(mid_dim, 1), )

        # layer_norm(self.net[0], std=1.0)
        # layer_norm(self.net[-1], std=1.0)

    def forward(self, s, a):
        x = torch.cat((s, a), dim=1)
        q = self.net(x)
        return q


class ReplayBuffer:
    def __init__(self, memory_size):
        self.storage = []
        self.memory_size = memory_size
        self.next_idx = 0

    # add the samples
    def add(self, obs, action, reward, obs_, done):
        data = (obs, action, reward, obs_, done)
        if self.next_idx >= len(self.storage):
            self.storage.append(data)
        else:
            self.storage[self.next_idx] = data
        # get the next idx
        self.next_idx = (self.next_idx + 1) % self.memory_size

    # encode samples
    def _encode_sample(self, idx):
        obses, actions, rewards, obses_, dones = [], [], [], [], []
        for i in idx:
            data = self.storage[i]
            obs, action, reward, obs_, done = data
            obses.append(np.array(obs, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_.append(np.array(obs_, copy=False))
            dones.append(done)
        return np.array(obses), np.array(actions), np.array(rewards), np.array(obses_), np.array(dones)

    # sample from the memory
    def sample(self, batch_size):
        idxes = [random.randint(0, len(self.storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class AgentSAC:
    def __init__(self, env, state_dim, action_dim, net_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = 3e-4

        use_densenet = False
        use_spectral_norm = False

        '''network'''
        self.act = ActorSAC(state_dim, action_dim, net_dim).to(self.device)
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate * 0.5)

        cri_dim = int(net_dim * 1.25)
        self.cri = CriticSAC(state_dim, action_dim, cri_dim, use_densenet, use_spectral_norm).to(self.device)
        self.cri2 = CriticSAC(state_dim, action_dim, cri_dim, use_densenet, use_spectral_norm).to(self.device)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate)
        self.cri2_optimizer = torch.optim.Adam(self.cri2.parameters(), lr=self.learning_rate)

        self.cri_target = copy.deepcopy(self.cri).to(self.device)
        self.cri2_target = copy.deepcopy(self.cri2).to(self.device)

        '''extension'''
        self.target_entropy = -1
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)

        self.alpha_optim = torch.optim.Adam((self.log_alpha,), lr=self.learning_rate)

        '''training'''
        self.state = env.reset()
        self.reward_sum = 0.0
        self.step_sum = 0

    @staticmethod
    def soft_target_update(target, source, tau=5e-3):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def select_actions(self, states, explore_noise=0.0):  # CPU array to GPU tensor to CPU array
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = self.act(states)

        if explore_noise != 0.0:
            pis = self.act.actor(states)
            actions = Normal(*pis).sample()
        actions = actions.tanh()
        actions = actions.cpu().data.numpy()
        return actions

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

    def inactive_in_env_sac(self, env, memo, max_step, max_action, reward_scale, gamma):
        rewards = list()
        steps = list()
        for t in range(max_step):
            action = self.select_actions((self.state,), explore_noise=True)[0]

            next_state, reward, done, _ = env.step(action * max_action)
            res_reward = reward * reward_scale
            mask = 0.0 if done else gamma

            self.reward_sum += reward
            self.step_sum += 1

            memo.add(self.state, action, res_reward, next_state, mask)
            self.state = next_state
            if done:
                rewards.append(self.reward_sum)
                self.reward_sum = 0.0
                steps.append(self.step_sum)
                self.step_sum = 0

                # reset the environment
                self.state = env.reset()
        return rewards, steps

    def update_parameter_sac(self, memo, max_step, batch_size):
        loss_a_sum = 0.0
        loss_c_sum = 0.0
        iter_num = max_step
        for _ in range(iter_num):
            with torch.no_grad():
                # smaple batch of samples from the replay buffer
                states, actions, rewards, next_states, marks = [
                    torch.tensor(ary, dtype=torch.float32, device=self.device)
                    for ary in memo.sample(batch_size)
                ]
                rewards = rewards.unsqueeze(-1)
                marks = marks.unsqueeze(-1)  # mark == (1-float(done)) * gamma

            actions_noise, log_prob = self.act.get__a__log_prob(states)

            '''auto alpha'''
            alpha_loss = -(self.log_alpha * (self.target_entropy + log_prob).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            '''actor loss'''
            alpha = self.log_alpha.exp()
            q0_min = torch.min(self.cri(states, actions_noise), self.cri2(states, actions_noise))
            actor_loss = (alpha * log_prob - q0_min).mean()
            self.act_optimizer.zero_grad()
            actor_loss.backward()
            self.act_optimizer.step()

            '''critic loss'''
            q1_value = self.cri(states, actions)
            q2_value = self.cri2(states, actions)
            with torch.no_grad():
                next_actions_noise, next_log_prob = self.act.get__a__log_prob(next_states)

                next_q0_min = torch.min(self.cri_target(next_states, next_actions_noise),
                                        self.cri2_target(next_states, next_actions_noise))
                next_target_q_value = next_q0_min - next_log_prob * alpha
                target_q_value = rewards + marks * next_target_q_value
            qf1_loss = (q1_value - target_q_value).pow(2).mean()
            qf2_loss = (q2_value - target_q_value).pow(2).mean()
            # qf1
            self.cri_optimizer.zero_grad()
            qf1_loss.backward()
            self.cri_optimizer.step()
            # qf2
            self.cri2_optimizer.zero_grad()
            qf2_loss.backward()
            self.cri2_optimizer.step()

            loss_a_sum += actor_loss.item()
            loss_c_sum += (qf1_loss.item() + qf2_loss.item()) * 0.5
            self.soft_target_update(self.cri_target, self.cri)
            self.soft_target_update(self.cri2_target, self.cri2)

        loss_a = loss_a_sum / iter_num
        loss_c = loss_c_sum / iter_num
        return loss_a, loss_c


def train_agent_sac(agent_class, env_name, cwd, net_dim, max_step, max_memo, max_epoch,  # env
                    batch_size, gamma,
                    **_kwargs):  # 2020-0430
    reward_scale = 1
    env = gym.make(env_name)
    state_dim, action_dim, max_action, target_reward = get_env_info(env)

    agent = agent_class(env, state_dim, action_dim, net_dim)

    memo = ReplayBuffer(max_memo)
    recorder = Recorder(agent, max_step, max_action, target_reward, env_name)

    agent.inactive_in_env_sac(env, memo, max_step, max_action, reward_scale, gamma)  # init memory before training

    try:
        for epoch in range(max_epoch):
            with torch.no_grad():
                rewards, steps = agent.inactive_in_env_sac(env, memo, max_step, max_action, reward_scale, gamma)

            loss_a, loss_c = agent.update_parameter_sac(memo, max_step, batch_size)

            with torch.no_grad():  # for saving the GPU memory
                recorder.show_reward(epoch, rewards, steps, loss_a, loss_c)
                is_solved = recorder.check_reward(cwd, loss_a, loss_c)
                if is_solved:
                    break
    except KeyboardInterrupt:
        print("raise KeyboardInterrupt while training.")
    except AssertionError:  # for BipedWalker BUG 2020-03-03
        print("AssertionError: OpenAI gym r.LengthSquared() > 0.0f ??? Please run again.")
        return False

    train_time = recorder.show_and_save(env_name, cwd)

    # agent.save_or_load_model(cwd, is_save=True)  # save max reward agent in Recorder
    # memo.save_or_load_memo(cwd, is_save=True)

    draw_plot_with_npy(cwd, train_time)
    return True


def run__sac(gpu_id=0, cwd='AC_SAC'):
    from AgentRun import Arguments
    args = Arguments(AgentSAC)
    args.gpu_id = gpu_id
    args.reward_scale = 1.0  # important

    # args.env_name = "BipedalWalker-v3"
    # args.cwd = './{}/BW_{}'.format(cwd, gpu_id)
    # args.init_for_training()
    # while not train_agent_sac(**vars(args)):
    #     args.random_seed += 42

    args.env_name = "LunarLanderContinuous-v2"
    args.cwd = './{}/LL_{}'.format(cwd, gpu_id)

    args.init_for_training()
    while not train_agent_sac(**vars(args)):
        args.random_seed += 42


if __name__ == '__main__':
    run__sac(gpu_id=2, cwd='AC_SAC')
