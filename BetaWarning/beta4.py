import gym
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random
from itertools import count
import torch.nn.functional as F

from AgentRun import *
from AgentNet import *
from AgentZoo import *

"""
refer: (DUEL) https://github.com/gouxiangchen/dueling-DQN-pytorch good+
"""


class QNetDuel(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()

        self.net__head = nn.Sequential(
            nn.Linear(state_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, mid_dim), nn.ReLU(),
        )
        self.net_val = nn.Sequential(  # value
            nn.Linear(mid_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, 1),
        )
        self.net_adv = nn.Sequential(  # advantage value
            nn.Linear(mid_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, action_dim),
        )

    def forward(self, state, noise_std=0.0):
        x = self.net__head(state)
        val = self.net_val(x)
        adv = self.net_adv(x)
        q = val + adv - adv.mean(dim=1, keepdim=True)
        return q


class Memory(object):
    def __init__(self, memory_size: int) -> None:
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)

    def add(self, experience) -> None:
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size: int, continuous: bool = True):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if continuous:
            rand = random.randint(0, len(self.buffer) - batch_size)
            return [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
            return [self.buffer[i] for i in indexes]

    def clear(self):
        self.buffer.clear()


class AgentDuelingDQN(AgentBasicAC):  # 2020-07-07
    def __init__(self, state_dim, action_dim, net_dim):
        super(AgentBasicAC, self).__init__()
        self.learning_rate = 2e-4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''network'''
        self.act = QNetDuel(state_dim, action_dim, net_dim).to(self.device)
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate)
        self.act.train()

        self.act_target = QNetDuel(state_dim, action_dim, net_dim).to(self.device)
        self.act_target.load_state_dict(self.act.state_dict())
        self.act_target.eval()

        self.criterion = nn.SmoothL1Loss()
        self.softmax = nn.Softmax(dim=1)
        self.action_dim = action_dim

        '''training record'''
        self.state = None  # env.reset()
        self.reward_sum = 0.0
        self.step = 0
        self.update_counter = 0

        '''extension: rho and loss_c'''
        self.explore_rate = 0.25  # explore rate when update_buffer()
        self.explore_noise = True  # standard deviation of explore noise

    def update_parameters(self, buffer, max_step, batch_size, repeat_times):
        self.act.train()

        # loss_a_sum = 0.0
        loss_c_sum = 0.0

        k = 1.0 + buffer.now_len / buffer.max_len
        batch_size_ = int(batch_size * k)
        update_times = int(max_step * k)

        for _ in range(update_times):
            with torch.no_grad():
                rewards, masks, states, actions, next_states = buffer.random_sample(batch_size_, self.device)

                q_target_next = self.act_target(next_states).max(dim=1, keepdim=True)[0]
                q_target = rewards + masks * q_target_next

            self.act.train()
            a_ints = actions.type(torch.long)
            q_eval = self.act(states).gather(1, a_ints)
            critic_loss = self.criterion(q_eval, q_target)
            loss_c_tmp = critic_loss.item()
            loss_c_sum += loss_c_tmp

            self.act_optimizer.zero_grad()
            critic_loss.backward()
            self.act_optimizer.step()

            soft_target_update(self.act_target, self.act)

        loss_a_avg = 0.0
        loss_c_avg = loss_c_sum / update_times
        return loss_a_avg, loss_c_avg

    def select_actions(self, states, explore_noise=0.0):  # 2020-07-07
        states = torch.tensor(states, dtype=torch.float32, device=self.device)  # state.size == (1, state_dim)
        actions = self.act(states, 0)

        # discrete action space
        if explore_noise == 0.0:
            a_ints = actions.argmax(dim=1).cpu().data.numpy()
        else:
            a_prob = self.softmax(actions).cpu().data.numpy()
            a_ints = [rd.choice(self.action_dim, p=prob)
                      for prob in a_prob]
            # a_ints = rd.randint(self.action_dim, size=)
        return a_ints


def run__dqn(gpu_id=0, cwd='RL_DuelDQN'):  # 2020-07-07
    # import AgentZoo as Zoo
    class_agent = AgentDuelingDQN
    args = ArgumentsBeta(class_agent, gpu_id, cwd, env_name="CartPole-v0")
    args.init_for_training()
    train_agent_discrete(**vars(args))

    args = ArgumentsBeta(class_agent, gpu_id, cwd, env_name="LunarLander-v2")
    args.init_for_training()
    train_agent_discrete(**vars(args))


if __name__ == '__main__':
    run__dqn()
