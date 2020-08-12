import gym
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random
from itertools import count
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from AgentRun import get_env_info

"""
refer: (DUEL) https://github.com/gouxiangchen/dueling-DQN-pytorch good
"""


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()

        self.net__head = nn.Sequential(
            nn.Linear(state_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, mid_dim), nn.ReLU(),
        )
        self.net_value = nn.Sequential(
            # nn.Linear(mid_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, 1),
        )
        self.net_advtg = nn.Sequential(
            # nn.Linear(mid_dim, mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, action_dim),
        )

    def forward(self, state):
        x = self.net__head(state)
        value = self.net_value(x)
        advtg = self.net_advtg(x)
        q = value + advtg - advtg.mean(dim=1, keepdim=True)
        return q

    def select_action(self, state):
        Q = self.forward(state)
        a_int = torch.argmax(Q, dim=1)
        return a_int.item()


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


def run():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net_dim = 2 ** 7
    # env = gym.make('CartPole-v0')
    env = gym.make('LunarLander-v2')
    state_dim, action_dim, max_action, target_reward, is_discrete = get_env_info(env, is_print=True)

    onlineQNetwork = QNetwork(state_dim, action_dim, net_dim).to(device)
    targetQNetwork = QNetwork(state_dim, action_dim, net_dim).to(device)
    targetQNetwork.load_state_dict(onlineQNetwork.state_dict())

    optimizer = torch.optim.Adam(onlineQNetwork.parameters(), lr=1e-4)

    GAMMA = 0.99
    EXPLORE = 20000
    INITIAL_EPSILON = 0.1
    FINAL_EPSILON = 0.0001
    REPLAY_MEMORY = 50000
    BATCH = 2 ** 7

    UPDATE_STEPS = 4

    memory_replay = Memory(REPLAY_MEMORY)

    epsilon = INITIAL_EPSILON
    learn_steps = 0
    writer = SummaryWriter('logs/ddqn')
    begin_learn = False

    episode_reward = 0

    # onlineQNetwork.load_state_dict(torch.load('ddqn-policy.para'))
    for epoch in count():

        state = env.reset()
        episode_reward = 0
        for time_steps in range(200):
            p = random.random()
            if p < epsilon:
                action = random.randint(0, 1)
            else:
                tensor_state = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = onlineQNetwork.select_action(tensor_state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            memory_replay.add((state, next_state, action, reward, done))
            if memory_replay.size() > 128:
                if begin_learn is False:
                    print('learn begin!')
                    begin_learn = True
                learn_steps += 1
                if learn_steps % UPDATE_STEPS == 0:
                    targetQNetwork.load_state_dict(onlineQNetwork.state_dict())
                batch = memory_replay.sample(BATCH, False)
                batch_state, batch_next_state, batch_action, batch_reward, batch_done = zip(*batch)

                batch_state = torch.FloatTensor(batch_state).to(device)
                batch_next_state = torch.FloatTensor(batch_next_state).to(device)
                batch_action = torch.FloatTensor(batch_action).unsqueeze(1).to(device)
                batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1).to(device)
                batch_done = torch.FloatTensor(batch_done).unsqueeze(1).to(device)

                with torch.no_grad():
                    onlineQ_next = onlineQNetwork(batch_next_state)
                    targetQ_next = targetQNetwork(batch_next_state)
                    online_max_action = torch.argmax(onlineQ_next, dim=1, keepdim=True)
                    y = batch_reward + (1 - batch_done) * GAMMA * targetQ_next.gather(1, online_max_action.long())

                loss = F.mse_loss(onlineQNetwork(batch_state).gather(1, batch_action.long()), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                writer.add_scalar('loss', loss.item(), global_step=learn_steps)

                if epsilon > FINAL_EPSILON:
                    epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

            if done:
                break
            state = next_state

        writer.add_scalar('episode reward', episode_reward, global_step=epoch)
        if episode_reward > target_reward:
            print('Ep {}\tMoving average score: {:.2f}\t'.format(epoch, episode_reward))
            print('####### solve #######')
            break
        if epoch % 64 == 0:
            torch.save(onlineQNetwork.state_dict(), 'ddqn-policy.para')
            print('Ep {}\tMoving average score: {:.2f}\t'.format(epoch, episode_reward))


if __name__ == '__main__':
    run()
