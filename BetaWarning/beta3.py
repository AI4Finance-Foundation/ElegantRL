from AgentRun import *
from AgentZoo import *
from AgentNet import *

# from sac_d_zoo import SacdAgent
from sac_d_env import make_pytorch_env

import os
import sys
import numpy as np
from collections import deque

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import functional as F
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


def initialize_weights_he(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class BaseNetwork(nn.Module):
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class DQNBase(BaseNetwork):

    def __init__(self, num_channels):
        super(DQNBase, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            Flatten(),
        ).apply(initialize_weights_he)

    def forward(self, states):
        return self.net(states)


class QNetwork(BaseNetwork):

    def __init__(self, num_channels, num_actions, shared=False,
                 dueling_net=False):
        super().__init__()

        if not shared:
            self.conv = DQNBase(num_channels)

        if not dueling_net:
            self.head = nn.Sequential(
                nn.Linear(7 * 7 * 64, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, num_actions))
        else:
            self.a_head = nn.Sequential(
                nn.Linear(7 * 7 * 64, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, num_actions))
            self.v_head = nn.Sequential(
                nn.Linear(7 * 7 * 64, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 1))

        self.shared = shared
        self.dueling_net = dueling_net

    def forward(self, states):
        if not self.shared:
            states = self.conv(states)

        if not self.dueling_net:
            return self.head(states)
        else:
            a = self.a_head(states)
            v = self.v_head(states)
            return v + a - a.mean(1, keepdim=True)


class QNetTwin(nn.Module):
    def __init__(self, state_dim, action_dim, net_dim):
        super(QNetTwin, self).__init__()

        self.net = nn.Sequential(nn.Linear(state_dim, net_dim), nn.ReLU(),
                                 nn.Linear(net_dim, net_dim), )

        self.net1 = nn.Sequential(nn.Linear(net_dim, net_dim), nn.ReLU(),
                                  nn.Linear(net_dim, action_dim), )
        self.net2 = nn.Sequential(nn.Linear(net_dim, net_dim), nn.ReLU(),
                                  nn.Linear(net_dim, action_dim), )

    def forward(self, states):
        x = self.net(states)
        q1 = self.net1(x)
        q2 = self.net2(x)
        return q1, q2


class QNet(BaseNetwork):

    def __init__(self, state_dim, action_dim, net_dim):
        super(QNet, self).__init__()  # super().__init__() todo?

        self.net = nn.Sequential(nn.Linear(state_dim, net_dim), nn.ReLU(),
                                 nn.Linear(net_dim, net_dim), nn.ReLU(),
                                 nn.Linear(net_dim, action_dim), )

    def act(self, states):
        action_logits = self.net(states)
        greedy_actions = torch.argmax(action_logits, dim=1, keepdim=True)
        return greedy_actions

    def sample(self, states):
        action_probs = F.softmax(self.net(states), dim=1)
        action_dist = Categorical(action_probs)
        actions = action_dist.sample().view(-1, 1)

        # Avoid numerical instability.
        z = (action_probs == 0.0).float() * 1e-8
        log_action_probs = torch.log(action_probs + z)

        return actions, action_probs, log_action_probs


class MultiStepBuff:

    def __init__(self, maxlen=3):
        super(MultiStepBuff, self).__init__()
        self.maxlen = int(maxlen)
        self.reset()

    def append(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def get(self, gamma=0.99):
        assert len(self.rewards) > 0
        state = self.states.popleft()
        action = self.actions.popleft()
        reward = self._nstep_return(gamma)
        return state, action, reward

    def _nstep_return(self, gamma):
        r = np.sum([r * (gamma ** i) for i, r in enumerate(self.rewards)])
        self.rewards.popleft()
        return r

    def reset(self):
        # Buffer to store n-step transitions.
        self.states = deque(maxlen=self.maxlen)
        self.actions = deque(maxlen=self.maxlen)
        self.rewards = deque(maxlen=self.maxlen)

    def is_empty(self):
        return len(self.rewards) == 0

    def is_full(self):
        return len(self.rewards) == self.maxlen

    def __len__(self):
        return len(self.rewards)


class LazyMultiStepMemory(dict):

    def __init__(self, capacity, state_shape, device, gamma=0.99,
                 multi_step=3):
        super(LazyMultiStepMemory, self).__init__()
        self.capacity = int(capacity)
        self.state_shape = state_shape
        self.device = device
        self.reset()

        self.gamma = gamma
        self.multi_step = int(multi_step)
        if self.multi_step != 1:
            self.buff = MultiStepBuff(maxlen=self.multi_step)

    def append(self, state, action, reward, next_state, done):
        if self.multi_step != 1:
            self.buff.append(state, action, reward)

            if self.buff.is_full():
                state, action, reward = self.buff.get(self.gamma)
                self._append(state, action, reward, next_state, done)

            if done:
                while not self.buff.is_empty():
                    state, action, reward = self.buff.get(self.gamma)
                    self._append(state, action, reward, next_state, done)
        else:
            self._append(state, action, reward, next_state, done)

    def reset(self):
        self['state'] = []
        self['next_state'] = []

        self['action'] = np.empty((self.capacity, 1), dtype=np.int64)
        self['reward'] = np.empty((self.capacity, 1), dtype=np.float32)
        self['done'] = np.empty((self.capacity, 1), dtype=np.float32)

        self._n = 0
        self._p = 0

    def truncate(self):
        while len(self['state']) > self.capacity:
            del self['state'][0]
            del self['next_state'][0]

    def sample(self, batch_size):
        indices = np.random.randint(low=0, high=len(self), size=batch_size)
        return self._sample(indices, batch_size)

    def __len__(self):
        return self._n

    def _sample(self, indices, batch_size):
        bias = -self._p if self._n == self.capacity else 0

        states = np.empty(
            (batch_size, *self.state_shape), dtype=np.uint8)
        next_states = np.empty(
            (batch_size, *self.state_shape), dtype=np.uint8)

        for i, index in enumerate(indices):
            _index = np.mod(index + bias, self.capacity)
            states[i, ...] = self['state'][_index]
            next_states[i, ...] = self['next_state'][_index]

        states = torch.ByteTensor(states).to(self.device).float() / 255.
        next_states = torch.ByteTensor(
            next_states).to(self.device).float() / 255.
        actions = torch.LongTensor(self['action'][indices]).to(self.device)
        rewards = torch.FloatTensor(self['reward'][indices]).to(self.device)
        dones = torch.FloatTensor(self['done'][indices]).to(self.device)

        return states, actions, rewards, next_states, dones

    def _append(self, state, action, reward, next_state, done):
        self['state'].append(state)
        self['next_state'].append(next_state)
        self['action'][self._p] = action
        self['reward'][self._p] = reward
        self['done'][self._p] = done

        self._n = min(self._n + 1, self.capacity)
        self._p = (self._p + 1) % self.capacity

        self.truncate()


def update_params(optim, loss, retain_graph=False):
    optim.zero_grad()
    loss.backward(retain_graph=retain_graph)
    optim.step()


def disable_gradients(network):
    # Disable calculations of gradients.
    for param in network.parameters():
        param.requires_grad = False


class RunningMeanStats:

    def __init__(self, n=10):
        self.n = n
        self.stats = deque(maxlen=n)

    def append(self, x):
        self.stats.append(x)

    def get(self):
        return np.mean(self.stats)


class SacdAgent:
    def __init__(self, env, test_env, log_dir,
                 state_dim, action_dim, net_dim,
                 num_steps=100000, batch_size=64,
                 lr=0.0003, memory_size=1000000, gamma=0.99, multi_step=1,
                 target_entropy_ratio=0.98, start_steps=20000,
                 update_interval=4, target_update_interval=8000,
                 use_per=False, dueling_net=False, num_eval_steps=125000,
                 max_episode_steps=27000, log_interval=10, eval_interval=1000,
                 cuda=True, seed=0):
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

        self.train_return = RunningMeanStats(log_interval)

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

        # Disable gradient calculations of the target network.
        disable_gradients(self.act_target)
        disable_gradients(self.cri_target)

        self.act_optim = torch.optim.Adam(self.act.parameters(), lr=lr)

        self.cri_optim = torch.optim.Adam(self.cri.parameters(), lr=lr)
        # self.q1_optim = Adam(self.online_critic.net1.parameters(), lr=lr)
        # self.q2_optim = Adam(self.online_critic.net2.parameters(), lr=lr)

        # Target entropy is -log(1/|A|) * ratio (= maximum entropy * ratio).
        self.target_entropy = -np.log(1.0 / action_dim) * target_entropy_ratio

        # We optimize log(alpha), instead of alpha.
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = Adam([self.log_alpha], lr=lr)

    def explore(self, state):
        state = torch.tensor((state,), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action, _, _ = self.act_target.sample(state)
        return action.item()

    def exploit(self, state):
        state = torch.tensor((state,), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action = self.act_target.act(state)
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

        while True:
            state = self.test_env.reset()
            episode_steps = 0
            episode_return = 0.0
            done = False
            while (not done) and episode_steps <= self.max_episode_steps:
                action = self.exploit(state)
                next_state, reward, done, _ = self.test_env.step(action)
                num_steps += 1
                episode_steps += 1
                episode_return += reward
                state = next_state

            num_episodes += 1
            total_return += episode_return

            if num_steps > self.num_eval_steps:
                break

        mean_return = total_return / num_episodes

        if mean_return > self.best_eval_score:
            self.best_eval_score = mean_return
            # self.save_models(os.path.join(self.model_dir, 'best'))

        # self.writer.add_scalar(
        #     'reward/test', mean_return, self.steps)
        # print('-' * 60)
        print(f'Num steps: {self.steps:<5}  '
              f'return: {mean_return:<5.1f}')
        # print('-' * 60)


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
    net_dim = 2 ** 8
    gamma = 0.99
    batch_size = 2 ** 8
    max_step = 2 ** 9
    reward_scale = 1

    # env_name = 'MsPacmanNoFrameskip-v4'
    # env = make_pytorch_env(env_name, clip_rewards=False)
    # test_env = make_pytorch_env(env_name, episode_life=False, clip_rewards=False)

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
                      cuda=True, seed=0, **config)

    buffer = BufferArray(2 ** 17, state_dim, a_int_dim)  # experiment replay buffer

    with torch.no_grad():  # update replay buffer
        rewards, steps = initial_exploration(
            env, buffer, max_step, max_action, reward_scale, gamma, action_dim)
    self = agent
    while self.steps < self.num_steps:
        self.episodes += 1
        episode_return = 0.
        episode_steps = 0

        done = False
        state = self.env.reset()
        buffer.init_before_sample()

        while (not done) and episode_steps <= self.max_episode_steps:

            if self.start_steps > self.steps:
                action = self.env.action_space.sample()
            else:
                action = self.explore(state)

            next_state, reward, done, _ = self.env.step(action)

            # buffer.append(state, action, reward, next_state, done)
            mask = 0.0 if done else gamma
            buffer.add_memo((reward, mask, state, action, next_state))

            self.steps += 1
            episode_steps += 1
            episode_return += reward
            state = next_state

            self.learning_steps += 1

            # batch == (reward, mask, state, action, next_state)
            reward, mask, states, action, next_state = buffer.random_sample(batch_size, self.device)

            curr_q1, curr_q2 = self.cri(states)
            curr_q1 = curr_q1.gather(1, action.long())
            curr_q2 = curr_q2.gather(1, action.long())

            with torch.no_grad():
                _, action_probs, log_action_probs = self.act.sample(next_state)
                next_q1, next_q2 = self.cri_target(next_state)
                next_q = (action_probs * (
                        torch.min(next_q1, next_q2) - self.alpha * log_action_probs
                )).sum(dim=1, keepdim=True)
            target_q = reward + mask * next_q

            q1_loss = torch.mean((curr_q1 - target_q).pow(2))
            q2_loss = torch.mean((curr_q2 - target_q).pow(2))

            _, action_probs, log_action_probs = self.act.sample(states)

            with torch.no_grad():
                q1, q2 = self.cri(states)

            entropies = -torch.sum(action_probs * log_action_probs, dim=1, keepdim=True)
            q = torch.sum(torch.min(q1, q2) * action_probs, dim=1, keepdim=True)
            policy_loss = (- q - self.alpha * entropies).mean()
            entropies = entropies.detach()

            entropy_loss = -(self.log_alpha * (- entropies + self.target_entropy)).mean()

            critic_loss = q1_loss+q2_loss
            self.cri_optim.zero_grad()
            critic_loss.backward()
            self.cri_optim.step()
            # update_params(self.q1_optim, q1_loss)
            # update_params(self.q2_optim, q2_loss)
            update_params(self.act_optim, policy_loss)
            update_params(self.alpha_optim, entropy_loss)

            self.alpha = self.log_alpha.exp()

            soft_target_update(self.cri_target, self.cri)
            soft_target_update(self.act_target, self.act)

            if self.steps % self.eval_interval == 0:
                self.evaluate()
                # self.save_models(os.path.join(self.model_dir, 'final'))

        # We log running mean of training rewards.
        self.train_return.append(episode_return)

def soft_target_update(target, source, tau=5e-3):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


if __name__ == '__main__':
    run()
