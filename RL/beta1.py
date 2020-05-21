import copy
import os

import gym
import numpy as np
import numpy.random as rd
import torch
import torch.nn as nn

from AgentRun import get_env_info, draw_plot_with_npy
from AgentRun import Recorder

"""
Refer: https://github.com/TianhongDai/reinforcement-learning-algorithms/tree/master/rl_algorithms/sac
TianhongDai's code is ok but not elegant.
Modify: ZenJiaHao. Github: YonV1943 DL_RL_Zoo RL AgentXXX.py
"""


class ActorSAC(nn.Module):
    def __init__(self, state_dim, action_dim, mid_net):
        super(ActorSAC, self).__init__()
        self.log_std_min = -20
        self.log_std_max = 2
        self.constant_log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))

        '''network'''
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

    def get__a__log_prob(self, states, device):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        a_mean, a_std = self.actor(states)

        '''add noise to action, stochastic policy'''
        noise = torch.randn_like(a_mean, requires_grad=True, device=device)
        a_noise = a_mean + a_std * noise
        a_noise_tanh = a_noise.tanh()

        '''calculate log_prob according to mean and std of action (stochastic policy)'''
        # from torch.distributions.normal import Normal
        # log_prob_noise = Normal(a_mean, a_std).log_prob(a_noise)
        # same as:
        # log_prob_noise = -(a_noise - a_mean).pow(2) /(2* a_std.pow(2)) - a_std.log() - np.log(np.sqrt(2 * np.pi))
        # same as:
        # self.constant_log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))
        log_prob_noise = -(((a_noise - a_mean) / a_std).pow(2) * 0.5 + a_std.log() + self.constant_log_sqrt_2pi)

        # log_prob = log_prob_noise - (1 - a_noise_tanh.pow(2) + epsilon).log() # epsilon = 1e-6
        # same as:
        log_prob = log_prob_noise - (-a_noise_tanh.pow(2) + 1.000001).log()
        return a_noise_tanh, log_prob.sum(1, keepdim=True)


class CriticTwin(nn.Module):  # TwinSAC <- TD3(TwinDDD) <- DoubleDQN -< Double Q-learning
    def __init__(self, state_dim, action_dim, mid_dim):
        super(CriticTwin, self).__init__()

        def build_cri_net():
            net = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                nn.Linear(mid_dim, 1), )
            # layer_norm(self.net[0], std=1.0)
            # layer_norm(self.net[-1], std=1.0)
            return net
        self.net1 = build_cri_net()
        self.net2 = build_cri_net()

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        q_value1 = self.net1(x)
        q_value2 = self.net2(x)
        return q_value1, q_value2


class MemoryArray:
    def __init__(self, memo_max_len, state_dim, action_dim, ):
        memo_dim = 1 + 1 + state_dim + action_dim + state_dim
        self.memories = np.empty((memo_max_len, memo_dim), dtype=np.float32)

        self.next_idx = 0
        self.is_full = False
        self.memo_max_len = memo_max_len
        self.memo_len = self.memo_max_len if self.is_full else self.next_idx

        self.state_idx = 1 + 1 + state_dim  # reward_dim==1, done_dim==1
        self.action_idx = self.state_idx + action_dim

    def add_memo(self, memo_tuple):
        self.memories[self.next_idx] = np.hstack(memo_tuple)
        self.next_idx = self.next_idx + 1
        if self.next_idx >= self.memo_max_len:
            self.is_full = True
            self.next_idx = 0

    def del_memo(self):
        self.memo_len = self.memo_max_len if self.is_full else self.next_idx

    def random_sample(self, batch_size, device):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # indices = rd.choice(self.memo_len, batch_size, replace=False)  # why perform worse?
        # indices = rd.choice(self.memo_len, batch_size, replace=True)  # why perform better?
        # same as:
        indices = rd.randint(self.memo_len, size=batch_size)

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


class AgentSAC:
    def __init__(self, env, state_dim, action_dim, net_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = 3e-4

        '''network'''
        self.act = ActorSAC(state_dim, action_dim, net_dim).to(self.device)
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate * 0.5)

        self.cri = CriticTwin(state_dim, action_dim, int(net_dim * 1.25)).to(self.device)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate)
        self.cri_target = copy.deepcopy(self.cri).to(self.device)

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
    def soft_target_update(target, source, tau=5e-3):
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

    def inactive_in_env_sac(self, env, memo, max_step, max_action, reward_scale, gamma):
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
        memo.del_memo()
        return rewards, steps

    def update_parameter_sac(self, memo, max_step, batch_size, update_gap):
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

            """critic loss"""

            '''q0_target'''
            with torch.no_grad():
                next_actions_noise, next_log_prob = self.act.get__a__log_prob(next_states, self.device)

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
            # self.soft_target_update(self.cri_target, self.cri)
            # self.soft_target_update(self.cri2_target, self.cri2)
            self.update_counter += 1
            if self.update_counter > update_gap:
                self.update_counter = 0
                self.cri_target.load_state_dict(self.cri.state_dict())  # hard target update

        loss_a = loss_a_sum / max_step
        loss_c = loss_c_sum / max_step
        return loss_a, loss_c


def train_agent_sac(agent_class, env_name, cwd, net_dim, max_step, max_memo, max_epoch,  # env
                    batch_size, gamma, update_gap, reward_scale,
                    **_kwargs):  # 2020-0430
    env = gym.make(env_name)
    state_dim, action_dim, max_action, target_reward = get_env_info(env)

    agent = agent_class(env, state_dim, action_dim, net_dim)

    memo = MemoryArray(max_memo, state_dim, action_dim)
    recorder = Recorder(agent, max_step, max_action, target_reward, env_name, show_gap=2 ** 8)

    with torch.no_grad():
        rewards, steps = agent.inactive_in_env_sac(env, memo, max_step, max_action, reward_scale, gamma)
    recorder.show_reward(rewards, steps, 0, 0)

    try:
        for epoch in range(max_epoch):
            with torch.no_grad():
                rewards, steps = agent.inactive_in_env_sac(env, memo, max_step, max_action, reward_scale, gamma)

            loss_a, loss_c = agent.update_parameter_sac(memo, max_step, batch_size, update_gap)

            with torch.no_grad():  # for saving the GPU memory
                recorder.show_reward(rewards, steps, loss_a, loss_c)
                is_solved = recorder.check_reward(cwd)
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

    args.env_name = "BipedalWalker-v3"
    args.cwd = './{}/BW_{}'.format(cwd, gpu_id)
    args.init_for_training()
    while not train_agent_sac(**vars(args)):
        args.random_seed += 42

    args.env_name = "LunarLanderContinuous-v2"
    args.cwd = './{}/LL_{}'.format(cwd, gpu_id)
    args.init_for_training()
    while not train_agent_sac(**vars(args)):
        args.random_seed += 42


if __name__ == '__main__':
    # run__sac(gpu_id=0, cwd='AC_SAC')
    from AgentRun import run__multi_process

    run__multi_process(run__sac, gpu_tuple=(2, 3), cwd='AC_SAC')
