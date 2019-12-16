import os
import sys
from time import time as timer

import gym
import numpy as np
import numpy.random as rd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

"""
2019-07-01 Zen4Jia1Hao2, GitHub: Yonv1943
2019-08-01 soft_update
2019-10-10 spectral normalization (should not use soft_update)

LunarLanderContinuous-v2: 659s 47E, 1582s 79e, 1991s 87e, 542s 53e, 1831s 77e, 1526s 94e, 1025s 80e
LunarLander-v2 (Discrete): 1012s 99e, 1034s 91e, 1996s 129e
BipedalWalker-v2: 883s 100e, 1157s 154e, 1850s 202e

beta1 DenseNet4, dropout0~0.2
"""


class Arguments:
    env_name = "LunarLander-v2"
    max_step = 2000  # max steps in one epoch
    max_epoch = 1000  # max num of train_epoch

    '''device'''
    gpu_id = sys.argv[0][-4]
    mod_dir = 'DDPG_%s' % gpu_id
    is_remove = True  # remove the pre-training data? (True, False, None:ask me)
    random_seed = 1943  # + int(gpu_id)

    '''training'''
    actor_dim = 2 ** 8  # the network width of actor_net
    critic_dim = int(actor_dim * 1.25)  # the network width of critic_net
    memories_size = int(2 ** 16)  # memories capacity (memories: replay buffer)
    batch_size = 2 ** 8  # num of transitions sampled from replay buffer.

    update_gap = 2 ** 7  # update the target_net, delay update
    soft_update_tau = 1  # could be 0.005
    iter_num_k = 1.25

    gamma = 0.99  # discount for future rewards
    explore_noise = 0.4  # action = select_action(state) + noise, 'explore_noise': sigma of noise
    policy_noise = 0.4  # actor_target(next_state) + noise,  'policy_noise': sigma of noise

    '''plot'''
    show_gap = 2 ** 5  # print the Reward, actor_loss, critic_loss
    eval_epoch = 4  # reload and evaluate the target policy network(actor)
    smooth_kernel = 2 ** 4  # smooth the reward/loss curves

    def __init__(self):
        # self.env_name = "CartPole-v1"
        # self.actor_dim = 2 ** 6  # the network width of actor_net
        # self.critic_dim = int(self.actor_dim * 1.25)  # the network width of critic_net

        self.env_name = "LunarLander-v2"


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

    iter_num_k = args.iter_num_k
    gamma = args.gamma
    explore_noise = args.explore_noise
    policy_noise = args.policy_noise
    random_seed = args.random_seed
    smooth_kernel = args.smooth_kernel
    is_remove = args.is_remove

    print('  GPUid: %s' % gpu_id)
    print('  Model: %s' % mod_dir)
    whether_remove_history(remove=is_remove, mod_dir=mod_dir)

    '''env init'''
    env = gym.make(env_name)
    state_dim, action_dim, action_max, target_reward = get_env_info(env)
    if action_max is not None:
        print("Error: Action space should be discrete for DQN.")
        exit()

    '''mod init'''
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    policy = AgentDelayDQN(state_dim, action_dim, actor_dim, critic_dim, gamma, policy_noise,
                           update_gap, soft_update_tau, iter_num_k)
    policy.load_model(mod_dir)

    memories = Memories(memories_size, state_dim, action_dim)
    memories.load(mod_dir)

    torch.set_num_threads(8)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    '''train loop'''
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
                action = policy.select_action(state, explore_noise)
                next_state, reward, done, _ = env.step(action)

                epoch_reward += reward
                memories.add(np.hstack((reward, 1 - float(done), action, state, next_state)))
                state = next_state

                if done:
                    break

            al, cl = policy.update_parameter(memories, iter_num, batch_size)
            eva_reward = get_eva_reward(policy, env, max_step, action_max, action_dim, target_reward)

            recorders.append((epoch, al, cl))
            rewards.append((eva_reward, epoch_reward,))

            if timer() - show_time > show_gap and len(rewards) > 1:
                show_time = timer()
                smooth_eva_r, smooth_epoch_r = np.average(np.array(rewards[-smooth_kernel:]), axis=0)
                print("%3i\tEvaR: %3i\tEpoR %3i\t|A %.3f, C %.3f"
                      % (epoch, smooth_eva_r, smooth_epoch_r, al, cl))

            if eva_reward > target_reward:
                print("########## Solved! ###########")
                print("%3i\tSmoR: %3i\tEpiR %3i\t|A %.3f, C %.3f"
                      % (epoch, eva_reward, epoch_reward, al, cl))
                break

    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    print('TimeUsed:', int(timer() - start_time))
    policy.save_model(mod_dir)
    memories.save(mod_dir)

    recorders = np.concatenate((rewards, recorders), axis=1)
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

    act = QNetwork(state_dim, action_dim, actor_dim).to(device)
    act.load_state_dict(torch.load('%s/actor.pth' % (mod_dir,), map_location=lambda storage, loc: storage))
    act.eval()

    # import cv2
    for epoch in range(eval_epoch):
        epoch_reward = 0
        state = env.reset()
        for iter_num in range(max_step):
            state = torch.tensor((state,), dtype=torch.float32, device=device)
            action = act(state).argmax().item()

            state, reward, done, _ = env.step(action)
            epoch_reward += reward

            env.render()
            # cv2.imwrite('%s/img_%4i.png'%(mod_dir, iter_num), env.render(mode='rgb_array'))

            if done:
                break

        print("%3i\tEpiR %3i" % (epoch, epoch_reward))
    env.close()


def run_test():  # 2019-11-06
    args = Arguments()

    # env = gym.make(args.env_name)
    # get_env_info(env)

    draw_plot(None, args.smooth_kernel, args.mod_dir, save_name=None)


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
        action_dim = 1  # discrete, torch.long
        slice_ids = [0, reward_dim, done_dim, action_dim, state_dim, state_dim]
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
        print("Save memo:", save_path)

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


class AgentDelayDQN:
    def __init__(self, state_dim, action_dim, actor_dim, critic_dim, gamma, policy_noise,
                 update_gap, soft_update_tau, iter_num_k):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ''''''
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_idx = 1 + 1 + 1 + state_dim  # reward_dim==1, done_dim==1, state_dim

        from torch import optim
        self.act = QNetwork(state_dim, action_dim, actor_dim)
        self.act = self.act.to(self.device)
        self.act_optimizer = optim.Adam(self.act.parameters(), lr=8e-4, betas=(0.5, 0.99))
        self.act.train()

        self.act_target = QNetwork(state_dim, action_dim, actor_dim)
        self.act_target = self.act_target.to(self.device)
        self.act_target.load_state_dict(self.act.state_dict())
        self.act_target.eval()

        self.criterion = nn.SmoothL1Loss()

        self.iter_num_k = iter_num_k
        self.update_counter = 0
        self.update_counter1 = 0
        self.update_gap = update_gap
        self.policy_noise = policy_noise
        self.gamma = gamma
        self.tau = soft_update_tau

        self.act_op_k = 0.5
        self.actor_loss_avg = 1.0
        self.critic_loss_avg = 1.0

    def select_action(self, state, explore_noise=0.0):  # state -> ndarray shape: (1, state_dim)
        # if rd.rand() > explore_noise:
        #     state = torch.tensor((state,), dtype=torch.float32, device=self.device)
        #     action = self.act(state).argmax().item()
        # else:
        #     action = rd.randint(self.action_dim)
        state = torch.tensor((state,), dtype=torch.float32, device=self.device)
        action = self.act(state, explore_noise).argmax().item()
        return action

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def update_parameter(self, memories, iter_num, batch_size):
        self.critic_loss_avg = 0

        k = 1 + memories.size / memories.memories_num
        batch_size = int(k * batch_size)
        iter_num = int(k * iter_num * self.iter_num_k)

        for i in range(iter_num):
            with torch.no_grad():
                memory = memories.sample(batch_size)
                memory = torch.tensor(memory, dtype=torch.float32, device=self.device)

                reward = memory[:, 0:1]
                undone = memory[:, 1:2]
                action = memory[:, 2:3]
                state = memory[:, 3:self.state_idx]
                next_state = memory[:, self.state_idx:]

                q_target = self.act_target(next_state).max(dim=1, keepdim=True)[0]
                q_target = reward + undone * self.gamma * q_target

            self.act.train()
            action = action.type(torch.long)
            q_eval = self.act(state).gather(1, action)
            critic_loss = self.criterion(q_eval, q_target)
            self.critic_loss_avg += critic_loss.item()
            self.act_optimizer.zero_grad()
            critic_loss.backward()
            self.act_optimizer.step()

            self.update_counter += 1
            if self.update_counter == self.update_gap:
                self.update_counter = 0
                self.act_target.load_state_dict(self.act.state_dict())
                # self.soft_update(self.act_target, self.act)

        self.critic_loss_avg /= iter_num
        return self.actor_loss_avg, self.critic_loss_avg

    def save_model(self, mod_dir):
        torch.save(self.act.state_dict(), '%s/actor.pth' % (mod_dir,))
        torch.save(self.act_target.state_dict(), '%s/actor_target.pth' % (mod_dir,))
        print("Saved:", mod_dir)

    def load_model(self, mod_dir):  # 2019-11-13
        print("Loading:", mod_dir)
        if os.path.exists('%s/actor.pth' % (mod_dir,)):
            self.act.load_state_dict(
                torch.load('%s/actor.pth' % (mod_dir,), map_location=lambda storage, loc: storage))
            self.act_target.load_state_dict(
                torch.load('%s/actor_target.pth' % (mod_dir,), map_location=lambda storage, loc: storage))
        else:
            print("FileNotFound in mod_dir:%s" % mod_dir)


class DenseNet(nn.Module):
    def __init__(self, inp_dim, mid_dim, out_dim, sn=False):
        super(DenseNet, self).__init__()
        self.dense0 = (nn.Linear(inp_dim, mid_dim * 1))
        self.dense0 = (nn.Linear(inp_dim, mid_dim * 1))
        self.dense1 = (nn.Linear(mid_dim * 1, mid_dim * 1))
        self.dense2 = (nn.Linear(mid_dim * 2, mid_dim * 2))

        self.dense_o = nn.Linear(mid_dim * 4, out_dim)
        if sn:
            self.dense_o = nn.utils.spectral_norm(self.dense_o)

        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x0):
        x1 = f_hard_swish(self.dense0(x0))

        x2 = torch.cat((x1, f_hard_swish(self.dense1(x1))), dim=1)
        x3 = torch.cat((x2, f_hard_swish(self.dense2(x2))), dim=1)

        self.dropout.p = rd.uniform(0, 0.25)
        x_o = self.dropout(x3)
        x_o = self.dense_o(x_o)
        return x_o


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim):
        super(QNetwork, self).__init__()
        self.net = DenseNet(inp_dim=state_dim, mid_dim=mid_dim, out_dim=action_dim, sn=True)

    def forward(self, x00, noise=0):
        x_o = self.net(x00)
        if noise:
            x_o += torch.randn_like(x_o, dtype=torch.float32).cuda() * noise
        # return torch.tanh(x_o)
        return x_o


def f_hard_swish(x):
    return F.relu6(x + 3) / 6 * x


"""utils"""


def draw_plot(recorders, smooth_kernel, mod_dir, save_name=None):  # 2019-11-08 16
    load_path = '%s/recorders.npy' % mod_dir
    if recorders is None:
        recorders = np.load(load_path)
        print(recorders.shape)
    else:
        np.save(load_path, recorders)

    if len(recorders) == 0:
        return print('Record is empty')
    else:
        print("Matplotlib Plot:", save_name)

    if save_name is None:
        save_name = "%s_plot.png" % (mod_dir,)

    import matplotlib.pyplot as plt

    # plt.style.use('ggplot')

    x_epoch = np.array(recorders[:, 2])

    fig, axs = plt.subplots(2)
    plt.title(save_name, y=2.3)

    r_avg, r_std = calculate_avg_std(recorders[:, 0], smooth_kernel)
    ax11 = axs[0]
    ax11_color = 'darkcyan'
    ax11_label = 'Eval R'
    ax11.plot(x_epoch, r_avg, label=ax11_label, color=ax11_color)
    ax11.set_ylabel(ylabel=ax11_label, color=ax11_color)
    ax11.fill_between(x_epoch, r_avg - r_std, r_avg + r_std, facecolor=ax11_color, alpha=0.1, )
    ax11.tick_params(axis='y', labelcolor=ax11_color)
    # ax11.legend(loc='best')
    # ax11.set_facecolor('#f0f0f0')
    # ax11.grid(color='white', linewidth=1.5)

    r_avg, r_std = calculate_avg_std(recorders[:, 1], smooth_kernel)
    ax12 = axs[0].twinx()
    ax12_color = 'royalblue'
    ax12_label = 'Epoch R'
    ax12.plot(x_epoch, r_avg, label=ax12_label, color=ax12_color)
    ax12.set_ylabel(ylabel=ax12_label, color=ax12_color)
    ax12.fill_between(x_epoch, r_avg - r_std, r_avg + r_std, facecolor=ax12_color, alpha=0.1, )
    ax12.tick_params(axis='y', labelcolor=ax12_color)

    ax21 = axs[1]
    ax21_color = 'darkcyan'
    ax21_label = '- loss A'
    ax21.set_ylabel(ax21_label, color=ax21_color)
    ax21.plot(x_epoch, -recorders[:, 3], label=ax21_label, color=ax21_color)  # negative loss A
    ax21.tick_params(axis='y', labelcolor=ax21_color)

    ax22 = axs[1].twinx()
    ax22_color = 'royalblue'
    ax22_label = 'loss C'
    ax22.set_ylabel(ax22_label, color=ax22_color)
    ax22.fill_between(x_epoch, recorders[:, 4], facecolor=ax22_color, alpha=0.25, )
    ax22.tick_params(axis='y', labelcolor=ax22_color)

    plt.savefig("%s/%s" % (mod_dir, save_name))
    plt.show()


def calculate_avg_std(y_reward, smooth_kernel):
    r_avg = list()
    r_std = list()
    for i in range(len(y_reward)):
        i_beg = i - smooth_kernel // 2
        i_end = i_beg + smooth_kernel

        i_beg = 0 if i_beg < 0 else i_beg
        rewards = y_reward[i_beg:i_end]
        r_avg.append(np.average(rewards))
        r_std.append(np.std(rewards))
    r_avg = np.array(r_avg)
    r_std = np.array(r_std)
    return r_avg, r_std


def get_env_info(env):  # 2019-11-06
    state_dim = env.observation_space.shape[0]

    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n  # Discrete
        action_max = None
    elif isinstance(env.action_space, gym.spaces.Box):
        action_dim = env.action_space.shape[0]  # Continuous
        action_max = float(env.action_space.high[0])
    else:
        action_dim = None
        action_max = None
        print('! Error with env.action_space:', env.action_space)
        exit()

    target_reward = env.spec.reward_threshold
    if target_reward is None:
        print('! Error with target_reward:', target_reward)
        exit()

    env_dict = {'env_name': repr(env)[10:-1],
                'state_dim': state_dim,
                'action_dim': action_dim,
                'action_max': (action_max, 'Discrete' if action_max is None else 'Continuous'),
                'target_reward': target_reward, }
    for key, value in env_dict.items():
        print("%16s\t%s" % (key, value))
    return state_dim, action_dim, action_max, target_reward


def get_eva_reward(policy, env, max_step, action_max, action_dim, target_reward):  # 2019-11-11
    policy.act.eval()

    eva_rewards = list()
    eva_epoch = 100
    for eval_epoch in range(eva_epoch):
        state = env.reset()
        eva_reward = 0
        for _ in range(max_step):
            action = policy.select_action(state)
            state, reward, done, _ = env.step(action)

            eva_reward += reward
            # env.render()
            if done:
                break
        eva_rewards.append(eva_reward)

        temp_target_reward = target_reward * (len(eva_rewards) / eva_epoch)
        if np.average(eva_rewards) < temp_target_reward:
            break  # break the evaluating loop ahead of time.
        if eval_epoch == 0 and eva_reward < target_reward:
            break

    policy.act.train()

    eva_reward = np.average(eva_rewards)
    eva_r_std = float(np.std(eva_rewards))
    if eva_reward > target_reward:
        print("Eval| avg: %.2f std: %.2f" % (eva_reward, eva_r_std))

    return eva_reward


def whether_remove_history(mod_dir, remove=None):
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
    # run_test()
