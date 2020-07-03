import os
import sys
from time import time as timer

import gym
import numpy as np
import numpy.random as rd

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
2019-07-01 Zen4Jia1Hao2, GitHub: Yonv1943
2019-08-01 soft_update
2019-10-10 spectral normalization (should not use soft_update)

LunarLanderContinuous-v2: 659s 47E, 1582s 79e, 1991s 87e, 542s 53e, 1831s 77e
LunarLander-v2 (Discrete): 1012s 99e, 1034s 91e, 1996s 129e
BipedalWalker-v2: 2100s E165
"""


class Arguments:
    env_name = "LunarLanderContinuous-v2"
    max_step = 2000  # max steps in one epoch
    max_epoch = 1000  # max num of train_epoch

    '''device'''
    gpu_id = sys.argv[0][-4]
    mod_dir = 'DDPG_%s' % gpu_id
    is_remove = True  # remove the pre-training data? (True, False, None:ask me)
    random_seed = 1943

    '''training'''
    actor_dim = 2 ** 8  # the network width of actor_net
    critic_dim = int(actor_dim * 1.25)  # the network width of critic_net
    memories_size = int(2 ** 16)  # memories capacity (memories: replay buffer)
    batch_size = 2 ** 8  # num of transitions sampled from replay buffer.

    update_gap = 2 ** 7  # update the target_net, delay update
    soft_update_tau = 1  # could be 0.005

    gamma = 0.99  # discount for future rewards
    explore_noise = 0.4  # action = select_action(state) + noise, 'explore_noise': sigma of noise
    policy_noise = 0.8  # actor_target(next_state) + noise,  'policy_noise': sigma of noise

    '''plot'''
    show_gap = 2 ** 5  # print the Reward, actor_loss, critic_loss
    eval_epoch = 4  # reload and evaluate the target policy network(actor)
    smooth_kernel = 2 ** 4  # smooth the reward/loss curves

    def __init__(self):
        self.env_name = "LunarLander-v2"
        self.explore_noise = 0.2
        self.policy_noise = 0.0

        # self.env_name = "BipedalWalker-v2"

        # self.env_name = "LunarLanderContinuous-v2"
        # self.explore_noise = 0.2
        # self.policy_noise = 0.2


'''train'''


def run_train():
    args = Arguments()

    gpu_id = args.gpu_id
    env_name = args.env_name
    mod_dir = args.mod_dir

    memories_size = args.memories_size
    batch_size = args.batch_size
    update_gap = args.update_gap
    soft_update_tau = args.soft_update_tau
    actor_dim = args.actor_dim
    critic_dim = args.critic_dim

    show_gap = args.show_gap
    max_step = args.max_step
    max_epoch = args.max_epoch

    gamma = args.gamma
    explore_noise = args.explore_noise
    policy_noise = args.policy_noise
    random_seed = args.random_seed
    smooth_kernel = args.smooth_kernel
    is_remove = args.is_remove

    whether_remove_history(remove=is_remove, mod_dir=mod_dir)

    '''env init'''
    env = gym.make(env_name)
    state_dim, action_dim, action_max, target_reward = get_env_info(env)

    '''mod init'''
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    policy = AgentDelayDDPG(state_dim, action_dim, actor_dim, critic_dim, gamma, policy_noise,
                            update_gap, soft_update_tau)

    memories = Memories(memories_size, state_dim, action_dim)
    torch.set_num_threads(8)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    '''train loop'''
    rd_normal = np.random.normal
    recorders = list()
    rewards = list()

    start_time = show_time = timer()
    try:
        for epoch in range(max_epoch):
            state = env.reset()
            epoch_reward = 0
            iter_num = 0

            policy.act.eval()
            for iter_num in range(max_step):
                action = policy.select_action(state)

                action += rd_normal(0, explore_noise, size=action_dim)  # add explore noise
                action = action.clip(-1.0, 1.0)

                next_state, reward, done, _ = env.step(adapt_action(action, action_max, action_dim, is_train=True))
                memories.add(np.hstack(((reward, 1 - float(done)), state, action, next_state)))
                state = next_state

                epoch_reward += reward

                if done:
                    break

            al, cl = policy.update_parameter(memories, iter_num, batch_size)

            recorders.append((epoch, al, cl))
            rewards.append(epoch_reward)
            smooth_reward = np.average(rewards[-smooth_kernel:])

            if timer() - show_time > show_gap:
                show_time = timer()
                print("%3i\tSmoR: %3i\tEpiR %3i\t|A %.3f, C %.3f"
                      % (epoch, smooth_reward, epoch_reward, al, cl))
            if epoch_reward > target_reward:  # eval and break
                print("Eval: %.2f" % epoch_reward)
                policy.act.eval()

                eva_rewards = list()
                eva_epoch = 100
                for eval_epoch in range(eva_epoch):
                    state = env.reset()
                    eva_reward = 0
                    for _ in range(max_step):
                        action = policy.select_action(state)
                        state, reward, done, _ = env.step(adapt_action(action, action_max, action_dim, is_train=False))

                        eva_reward += reward
                        # env.render()
                        if done:
                            break
                    eva_rewards.append(eva_reward)

                    temp_target_reward = target_reward * (len(eva_rewards) / eva_epoch)
                    if np.average(eva_rewards) < temp_target_reward:
                        break  # break the evaluating loop ahead of time.

                if np.average(eva_rewards) > target_reward:
                    print("########## Solved! ###########")
                    print("%3i\tSmoR: %3i\tEpiR %3i\t|A %.3f, C %.3f"
                          % (epoch, smooth_reward, epoch_reward, al, cl))
                    break

                policy.act.train()

    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    finally:
        print('TimeUsed:', int(timer() - start_time))
        policy.save_model(mod_dir)

    recorders = np.concatenate((np.array(rewards)[:, np.newaxis],
                                recorders), axis=1)
    draw_plot(recorders, smooth_kernel, mod_dir)


def run_eval():
    args = Arguments()

    gpu_id = args.gpu_id
    mod_dir = args.mod_dir
    env_name = args.env_name
    eval_epoch = args.eval_epoch
    max_step = args.max_step
    actor_dim = args.actor_dim

    '''env init'''
    env = gym.make(env_name)
    state_dim, action_dim, action_max, target_reward = get_env_info(env)

    '''mod init'''
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    act = Actor(state_dim, action_dim, actor_dim).to(device)
    act.load_state_dict(torch.load('%s/actor.pth' % (mod_dir,), map_location=lambda storage, loc: storage))
    act.eval()

    # import cv2
    for epoch in range(eval_epoch):
        epoch_reward = 0
        state = env.reset()
        for iter_num in range(max_step):
            '''select_action'''
            state = torch.tensor((state,), dtype=torch.float32, device=device)
            action = act(state).cpu().data.numpy()[0]

            state, reward, done, _ = env.step(adapt_action(action, action_max, action_dim, is_train=False))
            epoch_reward += reward

            env.render()
            # cv2.imwrite('%s/img_%4i.png'%(mod_dir, iter_num), env.render(mode='rgb_array'))

            if done:
                break

        print("%3i\tEpiR %3i" % (epoch, epoch_reward))
    env.close()


'''Network'''


class Memories:
    def __init__(self, memories_num, state_dim, action_dim):
        self.ptr_u = 0  # pointer_for_update
        self.ptr_s = 0  # pointer_for_sample
        self.is_full = False
        self.size = 0

        memories_num = int(memories_num)
        self.memories_num = memories_num

        reward_dim = 1
        done_dim = 1
        slice_ids = [0, reward_dim, done_dim, state_dim, action_dim, state_dim]
        slice_ids = [sum(slice_ids[:i + 1]) for i in range(len(slice_ids))]
        self.slice_dim = slice_ids

        memories_dim = slice_ids[-1]
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

    def extend(self, memory):
        len_m = len(memory)

        if self.ptr_u + len_m >= self.memories_num:
            len_m1 = self.memories_num - self.ptr_u
            len_m2 = len_m - len_m1

            # print(213, len_m, len_m1, len_m2, self.ptr_u, self.memories_num)
            self.memories[self.ptr_u:, :] = memory[:len_m1]
            if len_m2:  # != 0
                self.memories[:len_m2, :] = memory[-len_m2:]
            self.ptr_u = len_m2

            self.is_full = True
            print('Memories is_full!')
        else:
            _ptr_u = self.ptr_u
            self.ptr_u += len_m

            self.memories[_ptr_u:self.ptr_u, :] = memory

        self.size = self.memories_num if self.is_full else self.ptr_u

    def rd_extend(self, memory):
        len_m = len(memory)

        if self.ptr_u + len_m >= self.memories_num:
            len_m1 = self.memories_num - self.ptr_u
            len_m2 = len_m - len_m1

            # print(213, len_m, len_m1, len_m2, self.ptr_u, self.memories_num)
            self.memories[self.ptr_u:, :] = memory[:len_m1]
            if len_m2:  # != 0
                self.memories[:len_m2, :] = memory[-len_m2:]
            self.ptr_u = len_m2

            self.is_full = True
            print('Memories is_full!')
        else:
            _ptr_u = self.ptr_u
            self.ptr_u += len_m

            rd_idx = rd.randint(0, self.memories_num, len_m)
            self.memories[_ptr_u:self.ptr_u, :] = memory[:rd_idx.shape[0]]

        self.size = self.memories_num if self.is_full else self.ptr_u

    def sample(self, batch_size):
        self.ptr_s += batch_size
        if self.ptr_s >= self.size:
            self.ptr_s = batch_size
            rd.shuffle(self.indices[:self.size])

        batch_memory = self.memories[self.indices[self.ptr_s - batch_size:self.ptr_s]]
        return batch_memory

    def save(self, mod_dir):
        ptr_u = self.memories_num if self.is_full else self.ptr_u
        save_path = "%s/memories.npy" % mod_dir
        np.save(save_path, self.memories[:ptr_u])

    def load(self, mod_dir):
        save_path = "%s/memories.npy" % mod_dir
        if os.path.exists(save_path):
            memories = np.load(save_path)

            memo_len = memories.shape[0]
            if memo_len > self.memories_num:
                memo_len = self.memories_num
                self.ptr_u = self.memories_num
                print("Memories_num change:", memo_len)
            else:
                self.ptr_u = memo_len
                self.size = memo_len
                print("Memories_num:", self.ptr_u)

            self.memories[:self.ptr_u] = memories[:memo_len]
            if self.ptr_u == self.memories_num:
                self.ptr_u = 0
                self.is_full = True
                print('Memories is_full!')

            print("Load Memories:", save_path)
        else:
            print("FileNotFound:", save_path)


class AgentDelayDDPG:
    def __init__(self, state_dim, action_dim, actor_dim, critic_dim, gamma, policy_noise,
                 update_gap, soft_update_tau):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ''''''
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_idx = 1 + 1 + state_dim  # reward_dim==1, done_dim==1, state_dim
        self.action_idx = self.state_idx + action_dim

        from torch import optim
        self.act = Actor(state_dim, action_dim, actor_dim)
        self.act = self.act.to(self.device)
        self.act_optimizer = optim.Adam(self.act.parameters(), lr=2e-4, betas=(0.5, 0.99))
        self.act.train()

        self.act_target = Actor(state_dim, action_dim, actor_dim)
        self.act_target = self.act_target.to(self.device)
        self.act_target.load_state_dict(self.act.state_dict())
        self.act_target.eval()

        self.cri = Critic(state_dim, action_dim, critic_dim)
        self.cri = self.cri.to(self.device)
        self.cri_optimizer = optim.Adam(self.cri.parameters(), lr=8e-4, betas=(0.5, 0.99))
        self.cri.train()

        self.cri_target = Critic(state_dim, action_dim, critic_dim)
        self.cri_target = self.cri_target.to(self.device)
        self.cri_target.load_state_dict(self.cri.state_dict())
        self.cri_target.eval()

        self.criterion = nn.SmoothL1Loss()

        self.update_counter = 0
        self.update_gap = update_gap
        self.policy_noise = policy_noise
        self.gamma = gamma
        self.tau = soft_update_tau

        self.act_op_k = 0.5
        self.actor_loss_avg = 1.0
        self.critic_loss_avg = 1.0

    def select_action(self, state):
        state = torch.tensor((state,), dtype=torch.float32, device=self.device)
        action = self.act(state).cpu().data.numpy()
        return action[0]

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def update_parameter(self, memories, iter_num, batch_size):
        self.act_optimizer.param_groups[0]['lr'] = np.exp(-(self.critic_loss_avg / 4) ** 2) * 4e-4
        self.actor_loss_avg = self.critic_loss_avg = 0

        k = 1 + memories.size / memories.memories_num
        iter_num = int(k * iter_num)
        batch_size = int(k * batch_size)

        for i in range(iter_num):
            with torch.no_grad():
                memory = memories.sample(batch_size)
                memory = torch.tensor(memory, dtype=torch.float32, device=self.device)
                reward = memory[:, 0:1]
                undone = memory[:, 1:2]
                state = memory[:, 2:self.state_idx]
                action = memory[:, self.state_idx:self.action_idx]
                next_state = memory[:, self.action_idx:]

                noise = torch.randn(action.size(), dtype=torch.float32, device=self.device) * self.policy_noise

                next_action = self.act_target(next_state) + noise
                next_action = next_action.clamp(-1.0, 1.0)

                q_target = self.cri_target(next_state, next_action)
                q_target = reward + undone * self.gamma * q_target

            self.cri.train()
            q_eval = self.cri(state, action)
            critic_loss = self.criterion(q_eval, q_target)
            self.critic_loss_avg += critic_loss.item()
            self.cri_optimizer.zero_grad()
            critic_loss.backward()
            self.cri_optimizer.step()

            if self.update_counter % 2 == 1:
                self.cri.eval()
                actor_loss = -self.cri(state, self.act(state)).mean()
                self.actor_loss_avg += actor_loss.item()
                self.act_optimizer.zero_grad()
                actor_loss.backward()
                self.act_optimizer.step()

            self.update_counter += 1
            if self.update_counter == self.update_gap:
                self.update_counter = 0
                self.act_target.load_state_dict(self.act.state_dict())
                self.cri_target.load_state_dict(self.cri.state_dict())
                # self.soft_update(self.act_target, self.act)
                # self.soft_update(self.cri_target, self.cri)

        self.actor_loss_avg /= iter_num // 2
        self.critic_loss_avg /= iter_num
        return self.actor_loss_avg, self.critic_loss_avg

    def save_model(self, mod_dir):
        torch.save(self.act.state_dict(), '%s/actor.pth' % (mod_dir,))
        torch.save(self.act_target.state_dict(), '%s/actor_target.pth' % (mod_dir,))

        torch.save(self.cri.state_dict(), '%s/critic.pth' % (mod_dir,))
        torch.save(self.cri_target.state_dict(), '%s/critic_target.pth' % (mod_dir,))
        print("Saved:", mod_dir)

    def load_model(self, mod_dir):
        print("Loading:", mod_dir)
        self.act.load_state_dict(
            torch.load('%s/actor.pth' % (mod_dir,), map_location=lambda storage, loc: storage))
        self.act_target.load_state_dict(
            torch.load('%s/actor_target.pth' % (mod_dir,), map_location=lambda storage, loc: storage))

        self.cri.load_state_dict(
            torch.load('%s/critic.pth' % (mod_dir,), map_location=lambda storage, loc: storage))
        self.cri_target.load_state_dict(
            torch.load('%s/critic_target.pth' % (mod_dir,), map_location=lambda storage, loc: storage))


class DenseNet(nn.Module):
    def __init__(self, inp_dim, mid_dim, out_dim):
        super(DenseNet, self).__init__()
        self.dense00 = (nn.Linear(inp_dim, mid_dim * 1))
        self.dense10 = (nn.Linear(mid_dim * 1, mid_dim * 1))
        self.dense11 = (nn.Linear(mid_dim * 1, mid_dim * 1))
        self.dense20 = (nn.Linear(mid_dim * 2, mid_dim * 2))
        self.dense21 = (nn.Linear(mid_dim * 2, mid_dim * 2))
        self.dense_o = (nn.Linear(mid_dim * 4, out_dim))
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x00):
        x10 = f_hard_swish(self.dense00(x00))

        x11 = f_hard_swish(self.dense10(x10))
        x12 = f_hard_swish(self.dense11(x11)) + x11
        x20 = torch.cat((x10, x12), dim=1)

        x21 = f_hard_swish(self.dense20(x20))
        x22 = f_hard_swish(self.dense21(x21)) + x21
        x30 = torch.cat((x20, x22), dim=1)

        self.dropout.p = rd.uniform(0.125, 0.375)
        x_o = self.dropout(x30)
        x_o = self.dense_o(x_o)
        return x_o


class DenseNetSN(nn.Module):
    def __init__(self, inp_dim, mid_dim, out_dim):
        super(DenseNetSN, self).__init__()
        self.dense00 = (nn.Linear(inp_dim, mid_dim * 1))
        self.dense10 = (nn.Linear(mid_dim * 1, mid_dim * 1))
        self.dense11 = (nn.Linear(mid_dim * 1, mid_dim * 1))
        self.dense20 = (nn.Linear(mid_dim * 2, mid_dim * 2))
        self.dense21 = nn.utils.spectral_norm(nn.Linear(mid_dim * 2, mid_dim * 2), n_power_iterations=1)
        self.dense_o = (nn.Linear(mid_dim * 4, out_dim))
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x00):
        x10 = f_hard_swish(self.dense00(x00))

        x11 = f_hard_swish(self.dense10(x10))
        x12 = f_hard_swish(self.dense11(x11)) + x11
        x20 = torch.cat((x10, x12), dim=1)

        x21 = f_hard_swish(self.dense20(x20))
        x22 = f_hard_swish(self.dense21(x21)) + x21
        x30 = torch.cat((x20, x22), dim=1)

        self.dropout.p = rd.uniform(0.125, 0.375)
        x_o = self.dropout(x30)
        x_o = self.dense_o(x_o)
        return x_o


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim):
        super(Actor, self).__init__()
        self.net = DenseNet(inp_dim=state_dim, mid_dim=mid_dim, out_dim=action_dim)

    def forward(self, x00):
        x_o = self.net(x00)
        return torch.tanh(x_o)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim):
        super(Critic, self).__init__()
        self.net = DenseNetSN(inp_dim=state_dim + action_dim, mid_dim=mid_dim, out_dim=1)

    def forward(self, s, a):
        x00 = torch.cat((s, a), dim=1)
        x_o = self.net(x00)
        return x_o


def f_hard_swish(x):
    return F.relu6(x + 3) / 6 * x


"""utils"""


def draw_plot(recorders, smooth_kernel, mod_dir):
    np.save('%s/recorders.npy' % mod_dir, recorders)
    # recorders = np.load('%s/recorders.npy'% mod_dir)
    # report_plot(recorders=np.load('recorders.npy', ), smooth_kernel=32, mod_dir=0, save_name='TD.png')

    save_name = "%s_plot.png" % (mod_dir,)
    if recorders is list():
        return print('Record is empty')
    else:
        print("Matplotlib Plot:", save_name)
    import matplotlib.pyplot as plt

    y_reward = np.array(recorders[:, 0])
    y_reward_smooth = np.convolve(y_reward, np.ones(smooth_kernel) / smooth_kernel, mode='valid')

    x_epoch = np.array(recorders[:, 1])

    fig, axs = plt.subplots(3)
    plt.title(save_name, y=3.5)

    axs[0].plot(x_epoch, y_reward, label='Reward', linestyle=':')
    x_beg = int(smooth_kernel // 2)
    x_end = x_beg + len(y_reward_smooth)
    axs[0].plot(x_epoch[x_beg:x_end], y_reward_smooth, label='Smooth R')
    # axs[0].plot(x_epoch[smooth_kernel-1:], y_reward_smooth, label='Smooth R')
    axs[0].legend()

    axs[1].plot(x_epoch, recorders[:, 2], label='loss_A')
    axs[2].plot(x_epoch, recorders[:, 3], label='loss_C')

    plt.savefig("%s/%s" % (mod_dir, save_name))
    plt.show()


def get_env_info(env):
    state_dim = env.observation_space.shape[0]

    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n  # Discrete
        action_max = None
        print('action_space: Discrete:', action_dim)
    elif isinstance(env.action_space, gym.spaces.Box):
        action_dim = env.action_space.shape[0]  # Continuous
        action_max = float(env.action_space.high[0])
    else:
        action_dim = None
        action_max = None
        print('Error: env.action_space in get_env_info(env)')
        exit()

    target_reward = env.spec.reward_threshold
    return state_dim, action_dim, action_max, target_reward


def adapt_action(action, action_max, action_dim, is_train):
    """
    action belongs to range(-1, 1), makes it suit for env.step(action)
    :return: state, reward, done, _
    """
    if action_max:  # action_space: Continuous
        return action * action_max
    else:  # action_space: Discrete
        if is_train:
            action_prob = action + 1.00001
            action_prob /= sum(action_prob)
            return rd.choice(action_dim, p=action_prob)
        else:
            return np.argmax(action)


def whether_remove_history(mod_dir, remove=None):
    print('  Model: %s' % mod_dir)
    if remove is None:
        remove = bool(input("  'y' to REMOVE: %s? " % mod_dir) == 'y')
    if remove:
        import shutil
        shutil.rmtree(mod_dir, ignore_errors=True)
        print("| Remove")
        del shutil

    if not os.path.exists(mod_dir):
        os.mkdir(mod_dir)


if __name__ == '__main__':
    run_train()
    run_eval()
