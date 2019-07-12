import numpy as np
import numpy.random as rd

import torch
import torch.nn as nn
import torch.nn.functional as F


# import torch.utils.data as data


def f_hard_swish(x):
    return F.relu6(x + 3) / 6 * x


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, mod_dim):
        super(Actor, self).__init__()
        inp_dim = state_dim
        out_dim = action_dim
        self.dense0 = nn.Linear(inp_dim, mod_dim * 1)
        self.dense1 = nn.Linear(mod_dim * 1, mod_dim * 1)
        self.dense2 = nn.Linear(mod_dim * 2, mod_dim * 2)
        self.dense3 = nn.Linear(mod_dim * 4, out_dim)

    def forward(self, x0):
        x1 = f_hard_swish(self.dense0(x0))
        x2 = torch.cat((x1, f_hard_swish(self.dense1(x1))), dim=1)
        x3 = torch.cat((x2, f_hard_swish(self.dense2(x2))), dim=1)
        x3 = F.dropout(x3, p=rd.uniform(0.0, 0.5), training=self.training)
        x4 = torch.tanh(self.dense3(x3))
        return x4


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, mod_dim):
        super(Critic, self).__init__()
        inp_dim = state_dim + action_dim
        out_dim = 1
        self.dense0 = nn.Linear(inp_dim, mod_dim * 1)
        self.dense1 = nn.Linear(mod_dim * 1, mod_dim)
        self.dense2 = nn.Linear(mod_dim * 2, mod_dim * 2)
        self.dense3 = nn.Linear(mod_dim * 4, out_dim)

    def forward(self, s, a):
        x0 = torch.cat((s, a), dim=1)
        x1 = f_hard_swish(self.dense0(x0))
        x2 = torch.cat((x1, f_hard_swish(self.dense1(x1))), dim=1)
        x3 = torch.cat((x2, f_hard_swish(self.dense2(x2))), dim=1)
        x3 = F.dropout(x3, p=rd.uniform(0.0, 0.5), training=self.training)
        x4 = self.dense3(x3)
        return x4


class AgentDelayDDPG:
    def __init__(self, state_dim, action_dim, mod_dim,
                 gamma, policy_noise, update_gap):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ''''''
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_idx = 1 + 1 + state_dim  # reward_dim==1, done_dim==1, state_dim
        self.action_idx = self.state_idx + action_dim

        from torch import optim
        self.act = Actor(state_dim, action_dim, mod_dim).to(self.device)
        self.act_optimizer = optim.Adam(self.act.parameters(), lr=4e-4)
        self.act.train()

        self.act_target = Actor(state_dim, action_dim, mod_dim).to(self.device)
        self.act_target.load_state_dict(self.act.state_dict())
        self.act_target.eval()

        self.cri = Critic(state_dim, action_dim, mod_dim).to(self.device)
        self.cri_optimizer = optim.Adam(self.cri.parameters(), lr=1e-3)
        self.cri.train()

        self.cri_target = Critic(state_dim, action_dim, mod_dim).to(self.device)
        self.cri_target.load_state_dict(self.cri.state_dict())
        self.cri_target.eval()

        self.criterion = nn.SmoothL1Loss()

        self.update_counter = 0
        self.update_gap = update_gap
        self.policy_noise = policy_noise
        self.gamma = gamma

    def select_action(self, state):
        state = torch.tensor((state,), dtype=torch.float32).to(self.device)
        action = self.act(state).cpu().data.numpy()
        return action[0]

    def update(self, memories, iter_num, batch_size):
        actor_loss_avg, critic_loss_avg = 0, 0

        k = 1 + memories.size / memories.memories_num
        iter_num = int(k * iter_num)
        batch_size = int(k * batch_size)

        for i in range(iter_num):
            with torch.no_grad():
                memory = memories.sample(batch_size)
                memory = torch.tensor(memory, dtype=torch.float32).to(self.device)
                reward = memory[:, 0:1]
                undone = memory[:, 1:2]
                state = memory[:, 2:self.state_idx]
                action = memory[:, self.state_idx:self.action_idx]
                next_state = memory[:, self.action_idx:]

                noise = torch.randn(action.size(), dtype=torch.float32, device=self.device) * self.policy_noise

            next_action = self.act_target(next_state) + noise
            next_action = next_action.clamp(-1.0, 1.0)

            with torch.no_grad():
                q_target = self.cri_target(next_state, next_action)
                q_target = reward + undone * self.gamma * q_target

            q_eval = self.cri(state, action)
            critic_loss = self.criterion(q_eval, q_target)
            critic_loss_avg += critic_loss.item()
            self.cri_optimizer.zero_grad()
            critic_loss.backward()
            self.cri_optimizer.step()

            actor_loss = -self.cri(state, self.act(state)).mean()
            actor_loss_avg += actor_loss.item()
            self.act_optimizer.zero_grad()
            actor_loss.backward()
            self.act_optimizer.step()

            self.update_counter += 1
            if self.update_counter == self.update_gap:
                self.update_counter = 0
                self.act_target.load_state_dict(self.act.state_dict())
                self.cri_target.load_state_dict(self.cri.state_dict())

        actor_loss_avg /= iter_num
        critic_loss_avg /= iter_num
        return actor_loss_avg, critic_loss_avg

    def save(self, mod_dir):
        torch.save(self.act.state_dict(), '%s/actor.pth' % (mod_dir,))
        torch.save(self.act_target.state_dict(), '%s/actor_target.pth' % (mod_dir,))

        torch.save(self.cri.state_dict(), '%s/critic.pth' % (mod_dir,))
        torch.save(self.cri_target.state_dict(), '%s/critic_target.pth' % (mod_dir,))
        print("Saved:", mod_dir)

    def load(self, mod_dir, load_actor_only=False):
        print("Loading:", mod_dir)
        self.act.load_state_dict(
            torch.load('%s/actor.pth' % (mod_dir,), map_location=lambda storage, loc: storage))
        self.act_target.load_state_dict(
            torch.load('%s/actor_target.pth' % (mod_dir,), map_location=lambda storage, loc: storage))

        if load_actor_only:
            print("load_actor_only!")
        else:
            self.cri.load_state_dict(
                torch.load('%s/critic.pth' % (mod_dir,), map_location=lambda storage, loc: storage))
            self.cri_target.load_state_dict(
                torch.load('%s/critic_target.pth' % (mod_dir,), map_location=lambda storage, loc: storage))


class Memories:
    ptr_u = 0  # pointer_for_update
    ptr_s = 0  # pointer_for_sample
    is_full = False

    def __init__(self, memories_num, state_dim, action_dim, ):
        self.size = 0

        memories_num = int(memories_num)
        self.memories_num = memories_num

        reward_dim = 1
        done_dim = 1
        memories_dim = reward_dim + done_dim + state_dim + action_dim + state_dim
        self.memories = np.empty((memories_num, memories_dim), dtype=np.float32)
        self.indices = np.arange(memories_num)

    def add(self, memory):
        self.memories[self.ptr_u, :] = memory

        self.ptr_u += 1
        if self.ptr_u == self.memories_num:
            self.ptr_u = 0
            if not self.is_full:
                self.is_full = True
                print('Memories is_full!')
        self.size = self.memories_num if self.is_full else self.ptr_u

    def sample(self, batch_size):
        self.ptr_s += batch_size
        if self.ptr_s >= self.size:
            self.ptr_s = batch_size
            rd.shuffle(self.indices[:self.size])

        batch_memory = self.memories[self.indices[self.ptr_s - batch_size:self.ptr_s]]
        return batch_memory
