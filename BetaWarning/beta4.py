from AgentRun import *
from AgentZoo import *
from AgentNet import *

import os
import sys
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as nn_f
from torch.distributions import Categorical

"""Soft Actor-Critic for Discrete Action Settings
https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch
bad+ (why should I install nn_builder and TensorFlow2 in a PyTorch implement?)
https://github.com/ku2482/sac-discrete.pytorch
normal--

beta1 memory
beta2 state1d
"""

'''agent'''


def soft_target_update(target, source, tau=5e-3):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


class QNetTwin(nn.Module):
    def __init__(self, state_dim, action_dim, net_dim):
        super(QNetTwin, self).__init__()

        self.net = nn.Sequential(nn.Linear(state_dim, net_dim), nn.ReLU(),
                                 DenseNet(net_dim), )

        self.net1 = nn.Sequential(nn.Linear(net_dim * 4, net_dim), nn.ReLU(),
                                  nn.Linear(net_dim, action_dim), )
        self.net2 = nn.Sequential(nn.Linear(net_dim * 4, net_dim), nn.ReLU(),
                                  nn.Linear(net_dim, action_dim), )

    def forward(self, states):
        x = self.net(states)
        q1 = self.net1(x)
        q2 = self.net2(x)
        return q1, q2


class QNet(nn.Module):

    def __init__(self, state_dim, action_dim, net_dim):
        super(QNet, self).__init__()  # super().__init__()

        self.net = nn.Sequential(nn.Linear(state_dim, net_dim), nn.ReLU(),
                                 DenseNet(net_dim),
                                 nn.Linear(net_dim * 4, action_dim), )

    def forward(self, states):
        action_logits = self.net(states)
        greedy_actions = torch.argmax(action_logits, dim=1, keepdim=True)
        return greedy_actions

    def sample(self, states):
        action_probs = nn_f.softmax(self.net(states), dim=1)
        action_dist = Categorical(action_probs)
        actions = action_dist.sample().view(-1, 1)

        # Avoid numerical instability.
        z = (action_probs == 0.0).float() * 1e-8
        log_action_probs = torch.log(action_probs + z)

        return actions, action_probs, log_action_probs


class SacdAgent:
    def __init__(self, env, test_env, log_dir,
                 state_dim, action_dim, net_dim,
                 num_steps=100000, batch_size=64,
                 lr=2e-4, gamma=0.99, multi_step=1,
                 target_entropy_ratio=0.98, start_steps=20000,
                 update_interval=4, target_update_interval=8000,
                 use_per=False, num_eval_steps=125000,
                 max_episode_steps=27000, log_interval=10, eval_interval=1000,
                 cuda=True, **_kwargs):
        self.env = env
        self.test_env = test_env

        self.device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")

        self.steps = 0
        self.learning_steps = 0
        self.episodes = 0
        self.best_eval_score = -np.inf
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.gamma_n = gamma ** multi_step
        self.start_steps = start_steps
        self.update_interval = update_interval
        self.target_update_interval = target_update_interval
        self.use_per = use_per
        self.num_eval_steps = num_eval_steps
        self.max_episode_steps = max_episode_steps
        self.log_interval = log_interval
        self.eval_interval = eval_interval

        self.log_dir = log_dir
        self.model_dir = os.path.join(log_dir, 'model')
        # self.summary_dir = os.path.join(log_dir, 'summary')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Define networks.
        self.act = QNet(state_dim, action_dim, net_dim).to(self.device)
        self.act_target = QNet(state_dim, action_dim, net_dim).to(self.device).eval()
        self.act_target.load_state_dict(self.act.state_dict())

        self.cri = QNetTwin(state_dim, action_dim, net_dim).to(device=self.device)
        self.cri_target = QNetTwin(state_dim, action_dim, net_dim).to(device=self.device).eval()
        self.cri_target.load_state_dict(self.cri.state_dict())

        self.act_optim = torch.optim.Adam(self.act.parameters(), lr=lr)

        self.cri_optim = torch.optim.Adam(self.cri.parameters(), lr=lr)
        # self.q1_optim = Adam(self.online_critic.net1.parameters(), lr=lr)
        # self.q2_optim = Adam(self.online_critic.net2.parameters(), lr=lr)

        # Target entropy is -log(1/|A|) * ratio (= maximum entropy * ratio).
        self.target_entropy = -np.log(1.0 / action_dim) * target_entropy_ratio

        # We optimize log(alpha), instead of alpha.
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr)

        self.state = None  # env.reset()
        self.reward_sum = 0.0
        self.step = 0

    def explore(self, state):
        state = torch.tensor((state,), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action, _, _ = self.act.sample(state)
        return action.item()

    def exploit(self, state):
        state = torch.tensor((state,), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action = self.act(state)
        return action.item()

    def update_target(self):
        self.cri_target.load_state_dict(self.cri.state_dict())

    def is_update(self):
        return self.steps % self.update_interval == 0 \
               and self.steps >= self.start_steps

    def evaluate(self):
        num_episodes = 0
        num_steps = 0
        total_return = 0.0

        rewards = list()
        reward_sum = 0.0

        for e in range(16):
            for i in range(1000):
                state = self.test_env.reset()
                state = torch.tensor((state,), dtype=torch.float32, device=self.device)
                action = self.act(state).item()
                next_state, reward, done, _ = self.test_env.step(action)

                reward_sum += reward

                if done:
                    break

                state = next_state

            rewards.append(reward_sum)

        return rewards


'''run'''


def run():
    config = {
        'num_steps': 300000,
        'batch_size': 64,
        'lr': 2e-4,
        'memory_size': 300000,
        'gamma': 0.99,
        'multi_step': 1,
        'target_entropy_ratio': 0.98,
        'start_steps': 20000,
        'update_interval': 4,
        'target_update_interval': 8000,
        'use_per': False,
        'dueling_net': False,
        'num_eval_steps': 2 ** 12,
        'max_episode_steps': 27000,
        'log_interval': 10,
        'eval_interval': 5000,
    }
    gpu_id = sys.argv[-1][-4]
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    net_dim = 2 ** 7
    gamma = 0.99
    batch_size = 2 ** 8
    max_step = 2 ** 10
    reward_scale = 1

    # from sac_d_env import make_pytorch_env
    # env_name = 'MsPacmanNoFrameskip-v4'
    # env = make_pytorch_env(env_name, clip_rewards=False)
    # test_env = make_pytorch_env(env_name, episode_life=False, clip_rewards=False)

    # env_name = "CartPole-v0"
    env_name = "LunarLander-v2"
    env = gym.make(env_name)
    test_env = gym.make(env_name)
    state_dim, action_dim, max_action, target_reward, is_discrete = get_env_info(
        env, is_print=True)

    assert is_discrete
    a_int_dim = 1

    log_dir = os.path.join('logs', env_name, f'sac_discrete')
    agent = SacdAgent(env, test_env, log_dir,
                      state_dim, action_dim, net_dim,
                      cuda=True, **config)

    buffer = BufferArray(2 ** 17, state_dim, a_int_dim)  # experiment replay buffer
    criterion = nn.SmoothL1Loss()

    import numpy.random as rd
    with torch.no_grad():  # update replay buffer
        initial_exploration(env, buffer, max_step, max_action, reward_scale, gamma, action_dim)
    self = agent
    self.state = env.reset()

    for e in range(2 ** 10):
        '''update buffer'''
        done = False
        buffer.init_before_sample()
        rewards = list()
        steps = list()

        for _ in range(max_step):
            if rd.rand() > 0.5:
                states = torch.tensor((self.state, ), dtype=torch.float32, device=self.device)
                action = self.act(states).item()
            else:  # stochacstic
                action = self.explore(self.state)

            next_state, reward, done, _ = self.env.step(action)

            self.reward_sum += reward
            self.step += 1

            mask = 0.0 if done else gamma
            buffer.add_memo((reward, mask, self.state, action, next_state))

            self.state = next_state
            if done:
                rewards.append(self.reward_sum)
                self.reward_sum = 0.0

                steps.append(self.step)
                self.step = 0

                self.state = env.reset()

        print(f'R: {np.average(rewards):<8.2f}     S: {np.average(steps):8.2f}')

        '''update parameters'''
        for _ in range(max_step):
            reward, mask, states, action, next_state = buffer.random_sample(batch_size, self.device)
            with torch.no_grad():
                _, action_probs, log_action_probs = self.act_target.sample(next_state)
                next_q1, next_q2 = self.cri_target(next_state)
                next_q = (action_probs * (torch.min(next_q1, next_q2) - self.alpha * log_action_probs)
                          ).sum(dim=1, keepdim=True)
                target_q = reward + mask * next_q

            curr_q1, curr_q2 = self.cri(states)
            curr_q1 = curr_q1.gather(1, action.long())
            curr_q2 = curr_q2.gather(1, action.long())

            q1_loss = criterion(curr_q1, target_q)
            q2_loss = criterion(curr_q2, target_q)
            critic_loss = q1_loss + q2_loss

            self.cri_optim.zero_grad()
            critic_loss.backward()
            self.cri_optim.step()

            _, action_probs, log_action_probs = self.act.sample(states)

            entropies = -torch.sum(action_probs * log_action_probs, dim=1, keepdim=True)
            with torch.no_grad():
                q1, q2 = self.cri(states)
            entropy_loss = (self.log_alpha * (- entropies.detach() + self.target_entropy)).mean()
            self.alpha_optim.zero_grad()
            entropy_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            q = torch.sum(torch.min(q1, q2) * action_probs, dim=1, keepdim=True)
            policy_loss = (self.alpha * entropies - q).mean()
            self.act_optim.zero_grad()
            policy_loss.backward()
            self.act_optim.step()

            soft_target_update(self.cri_target, self.cri)
            soft_target_update(self.act_target, self.act)

        if e % 2 == 0:
            eva_rewards = self.evaluate()
            print(f'Eva_R: {np.average(eva_rewards):<8.2f}')


if __name__ == '__main__':
    run()
