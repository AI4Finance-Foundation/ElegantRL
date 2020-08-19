import os
import sys
from time import time as timer

import cv2
import gym
import numpy as np
import numpy.random as rd

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
beta2 PPO ZFlt stable, running state mean std, def run_eval()
beta1 GPU, def get_eva_reward()
fixbug: not save running state std mean
"""


class Arguments:
    env_name = "LunarLanderContinuous-v2"
    max_step = 2000  # max steps in one epoch
    max_epoch = 1000  # max num of train_epoch

    '''device'''
    gpu_id = sys.argv[0][-4]
    mod_dir = 'PPO_%s' % gpu_id
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
        # self.env_name = "BipedalWalker-v2"  # 17837s 124e
        # self.env_name = "LunarLanderContinuous-v2" # 14554s 132e
        self.env_name = 'CarRacing-v0'  # 71640s


def state2d_1d(state):
    img = state[:-8, 6:-6, 1]
    img = cv2.GaussianBlur(img, (5, 5), 1)
    img = cv2.resize(img, (14, 14))
    img = img.flatten()
    img = img / 128.0 - 0.5
    return img


class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update:
            self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def output_shape(self, input_space):
        return input_space.shape


from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'value', 'action', 'logproba', 'mask', 'next_state', 'reward'))


class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self):
        return Transition(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, layer_norm=True):
        super(ActorCritic, self).__init__()

        mid_dim = 128

        self.actor_fc1 = nn.Linear(num_inputs, mid_dim)
        self.actor_fc2 = nn.Linear(mid_dim, mid_dim)
        self.actor_fc3 = nn.Linear(mid_dim, num_outputs)
        self.actor_logstd = nn.Parameter(torch.zeros(1, num_outputs))

        self.critic_fc1 = nn.Linear(num_inputs, mid_dim)
        self.critic_fc2 = nn.Linear(mid_dim, mid_dim)
        self.critic_fc3 = nn.Linear(mid_dim, 1)

        if layer_norm:
            self.layer_norm(self.actor_fc1, std=1.0)
            self.layer_norm(self.actor_fc2, std=1.0)
            self.layer_norm(self.actor_fc3, std=0.01)

            self.layer_norm(self.critic_fc1, std=1.0)
            self.layer_norm(self.critic_fc2, std=1.0)
            self.layer_norm(self.critic_fc3, std=1.0)

    @staticmethod
    def layer_norm(layer, std=1.0, bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)

    def forward(self, states):
        """
        run policy network (actor) as well as value network (critic)
        :param states: a Tensor2 represents states
        :return: 3 Tensor2
        """
        action_mean, action_logstd = self._forward_actor(states)
        critic_value = self._forward_critic(states)
        return action_mean, action_logstd, critic_value

    def _forward_actor(self, states):
        x = f_hard_swish(self.actor_fc1(states))
        x = f_hard_swish(self.actor_fc2(x))
        action_mean = self.actor_fc3(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        return action_mean, action_logstd

    def _forward_critic(self, states):
        x = f_hard_swish(self.critic_fc1(states))
        x = f_hard_swish(self.critic_fc2(x))
        critic_value = self.critic_fc3(x)
        return critic_value

    def select_action(self, action_mean, action_logstd, return_logproba=True):
        """
        given mean and std, sample an action from normal(mean, std)
        also returns probability of the given chosen
        """
        action_std = torch.exp(action_logstd)
        action = torch.normal(action_mean, action_std)
        if return_logproba:
            logproba = self._normal_logproba(action, action_mean, action_logstd, action_std)
        return action, logproba

    @staticmethod
    def _normal_logproba(x, mean, logstd, std=None):
        if std is None:
            std = torch.exp(logstd)

        std_sq = std.pow(2)
        logproba = - 0.5 * np.log(2 * np.pi) - logstd - (x - mean).pow(2) / (2 * std_sq)
        return logproba.sum(1)

    def get_logproba(self, states, actions):
        """
        return probability of chosen the given actions under corresponding states of current network
        :param states: Tensor
        :param actions: Tensor
        """
        action_mean, action_logstd = self._forward_actor(states)
        logproba = self._normal_logproba(actions, action_mean, action_logstd)
        return logproba


def f_hard_swish(x):
    return F.relu6(x + 3) / 6 * x


"""train"""


def run_train():  # 2019-11-24
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

    '''PPO'''
    num_episode = 500
    batch_size = 2048
    max_step_per_round = 2000
    gamma = 0.995
    lamda = 0.97
    log_num_episode = 1
    num_epoch = 10
    minibatch_size = 256
    clip = 0.2
    loss_coeff_value = 0.5
    loss_coeff_entropy = 0.02  # 0.01
    lr = 3e-4
    num_parallel_run = 5
    layer_norm = True
    state_norm = False
    advantage_norm = True
    lossvalue_norm = True
    schedule_adam = 'linear'
    schedule_clip = 'linear'
    clip_now = clip

    whether_remove_history(remove=is_remove, mod_dir=mod_dir)

    '''env init'''
    env = gym.make(env_name)
    state_dim, action_dim, action_max, target_reward = get_env_info(env)

    '''mod init'''
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    network = ActorCritic(state_dim, action_dim, layer_norm=True).to(device)
    running_state = ZFilter((state_dim,), clip=5.0)
    from torch.optim import Adam
    optimizer = Adam(network.parameters(), lr=lr)

    torch.set_num_threads(8)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    '''train loop'''
    rd_normal = np.random.normal
    recorders = list()
    rewards = list()

    start_time = show_time = timer()
    EPS = 1e-10
    reward_record = []
    global_steps = 0
    from torch import Tensor
    try:
        for i_episode in range(num_episode):
            # step1: perform current policy to collect trajectories
            # this is an on-policy method!
            memory = Memory()
            num_steps = 0
            reward_list = []
            len_list = []
            while num_steps < batch_size:
                state = env.reset()
                state = state2d_1d(state)  # For CarRacing-v0
                reward_sum = 0
                t = 0
                if state_norm:
                    state = running_state(state)
                for t in range(max_step_per_round):
                    state_ten = torch.tensor((state,), dtype=torch.float32, device=device)
                    action_mean, action_logstd, value = network(state_ten)
                    action, logproba = network.select_action(action_mean, action_logstd)
                    action = action.cpu().data.numpy()[0]
                    logproba = logproba.cpu().data.numpy()[0]

                    next_state, reward, done, _ = env.step(action)  # For CarRacing-v0
                    next_state = state2d_1d(next_state)
                    if np.sum(next_state > 1.1) > 185:  # For CarRacing-v0, outside
                        reward = -100
                        done = True

                    reward_sum += reward

                    if state_norm:
                        next_state = running_state(next_state)
                    mask = 0 if done else 1

                    memory.push(state, value, action, logproba, mask, next_state, reward)

                    if done:
                        break

                    state = next_state

                num_steps += (t + 1)
                global_steps += (t + 1)
                reward_list.append(reward_sum)
                len_list.append(t + 1)
            reward_record.append({
                'episode': i_episode,
                'steps': global_steps,
                'meanepreward': np.mean(reward_list),
                'meaneplen': np.mean(len_list)})

            batch = memory.sample()
            batch_size = len(memory)

            rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device)
            values = torch.tensor(batch.value, dtype=torch.float32, device=device)
            masks = torch.tensor(batch.mask, dtype=torch.float32, device=device)
            actions = torch.tensor(batch.action, dtype=torch.float32, device=device)
            states = torch.tensor(batch.state, dtype=torch.float32, device=device)
            oldlogproba = torch.tensor(batch.logproba, dtype=torch.float32, device=device)

            prev_return = 0
            prev_value = 0
            prev_advantage = 0

            returns = torch.empty(batch_size, dtype=torch.float32, device=device)
            deltas = torch.empty(batch_size, dtype=torch.float32, device=device)
            advantages = torch.empty(batch_size, dtype=torch.float32, device=device)
            for i in reversed(range(batch_size)):
                returns[i] = rewards[i] + gamma * prev_return * masks[i]
                deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values[i]
                # ref: https://arxiv.org/pdf/1506.02438.pdf (generalization advantage estimate)
                advantages[i] = deltas[i] + gamma * lamda * prev_advantage * masks[i]

                prev_return = returns[i]
                prev_value = values[i]
                prev_advantage = advantages[i]

            if advantage_norm:
                advantages = (advantages - advantages.mean()) / (advantages.std() + EPS)

            for i_epoch in range(int(num_epoch * batch_size / minibatch_size)):
                # sample from current batch
                minibatch_ind = np.random.choice(batch_size, minibatch_size, replace=False)
                minibatch_states = states[minibatch_ind]
                minibatch_actions = actions[minibatch_ind]
                minibatch_oldlogproba = oldlogproba[minibatch_ind]
                minibatch_newlogproba = network.get_logproba(minibatch_states, minibatch_actions)
                minibatch_advantages = advantages[minibatch_ind]
                minibatch_returns = returns[minibatch_ind]
                minibatch_newvalues = network._forward_critic(minibatch_states).flatten()

                ratio = torch.exp(minibatch_newlogproba - minibatch_oldlogproba)
                surr1 = ratio * minibatch_advantages
                surr2 = ratio.clamp(1 - clip_now, 1 + clip_now) * minibatch_advantages
                loss_surr = - torch.mean(torch.min(surr1, surr2))

                # not sure the value loss should be clipped as well
                # clip example: https://github.com/Jiankai-Sun/Proximal-Policy-Optimization-in-Pytorch/blob/master/ppo.py
                # however, it does not make sense to clip score-like value by a dimensionless clipping parameter
                # moreover, original paper does not mention clipped value
                if lossvalue_norm:
                    minibatch_return_6std = 6 * minibatch_returns.std()
                    loss_value = torch.mean((minibatch_newvalues - minibatch_returns).pow(2)) / minibatch_return_6std
                else:
                    loss_value = torch.mean((minibatch_newvalues - minibatch_returns).pow(2))

                loss_entropy = torch.mean(torch.exp(minibatch_newlogproba) * minibatch_newlogproba)

                total_loss = loss_surr + loss_coeff_value * loss_value + loss_coeff_entropy * loss_entropy
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            if schedule_clip == 'linear':
                ep_ratio = 1 - (i_episode / num_episode)
                clip_now = clip * ep_ratio

            if schedule_adam == 'linear':
                ep_ratio = 1 - (i_episode / num_episode)
                lr_now = lr * ep_ratio
                # set learning rate
                # ref: https://stackoverflow.com/questions/48324152/
                for g in optimizer.param_groups:
                    g['lr'] = lr_now

            eva_reward = get_eva_reward(env, network, state_norm, running_state, max_step,
                                        target_reward, device)

            if i_episode % log_num_episode == 0:
                print('E: {:4} |R: {:8.3f} EvaR: {:8.2f} |L: {:6.3f} = {:6.3f} + {} * {:6.3f} + {} * {:6.3f}'.format(
                    i_episode, reward_record[-1]['meanepreward'], eva_reward,
                    total_loss.data, loss_surr.data,
                    loss_coeff_value, loss_value.data,
                    loss_coeff_entropy, loss_entropy.data,
                ))
            if eva_reward > target_reward:
                print("########## Solved! ###########")
                print('E: {:4} |R: {:8.3f}  EvaR: {:8.2f}'.format(
                    i_episode, reward_record[-1]['meanepreward'], eva_reward, ))
                break

    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    except Exception as e:
        print(next_state.shape)
        print(next_state)
        print("Error: {}".format(e))
    print('TimeUsed:', int(timer() - start_time))

    rs = running_state.rs
    np.savez('state_mean_std.npz', (rs.mean, rs.std))
    # print("State.mean", repr(rs.mean))
    # print("State.std ", repr(rs.std))

    torch.save(network.state_dict(), '%s/PPO.pth' % (mod_dir,))
    np.save('{}/reward_record.npy'.format(mod_dir), reward_record)
    print("Save in Mod_dir:", mod_dir)
    reward_record = np.load('{}/reward_record.npy'.format(args.mod_dir), allow_pickle=True)
    recorders = np.array([(i['episode'], i['meanepreward'], i['meaneplen'])
                          for i in reward_record])
    draw_plot_ppo(recorders, args.smooth_kernel, args.mod_dir)


def run_eval():  # 2019-11-24
    args = Arguments()

    env = gym.make(args.env_name)
    state_dim, action_dim, action_max, target_reward = get_env_info(env)

    '''mod init'''
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    network = ActorCritic(state_dim, action_dim, layer_norm=True)
    network.load_state_dict(torch.load('%s/PPO.pth' % (args.mod_dir,), map_location=lambda storage, loc: storage))
    network.eval()

    state_mean, state_std = np.load('state_mean_std.npz', allow_pickle=True)['arr_0']

    def noise_filter(s):
        return (s - state_mean) / state_std

    state_norm = False

    # import cv2
    for epoch in range(args.eval_epoch):
        epoch_reward = 0
        state = env.reset()
        state = state2d_1d(state)  # For CarRacing-v0
        for t in range(args.max_step):
            if state_norm:
                state = noise_filter(state)

            state_tensor = torch.tensor((state,), dtype=torch.float32)
            action_mean, action_logstd, value = network(state_tensor)

            # action, logproba = network.select_action(action_mean, action_logstd)
            # action = action.cpu().data.numpy()[0]
            action = action_mean.cpu().data.numpy()[0]

            next_state, reward, done, _ = env.step(action)  # For CarRacing-v0
            next_state = state2d_1d(next_state)
            # print(np.sum(next_state > 1.1))
            epoch_reward += reward

            env.render()
            if done:
                break

            state = next_state

        print("%3i\tEpiR %3i" % (epoch, epoch_reward))
    env.close()


def run_test():  # todo test
    args = Arguments()
    env = gym.make(args.env_name)
    state_dim, action_dim, action_max, target_reward = get_env_info(env)

    '''mod init'''
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    network = ActorCritic(state_dim, action_dim, layer_norm=True)  # todo gpu
    network = network.to(device)

    rs = network(torch.randn(2, state_dim, dtype=torch.float32, device=device))
    print([r.size() for r in rs])


"""utils"""


def get_eva_reward(env, network, state_norm, running_state, max_step, target_reward,
                   device):  # 2019-11-20
    network.eval()

    eva_rewards = list()
    eva_epoch = 100
    for eval_epoch in range(eva_epoch):
        state = env.reset()
        state = state2d_1d(state)  # For CarRacing-v0
        eva_reward = 0
        for _ in range(max_step):
            if state_norm:
                state = running_state(state)
            state_ten = torch.tensor((state,), dtype=torch.float32, device=device)
            action_mean, action_logstd, value = network(state_ten)

            action = action_mean.cpu().data.numpy()[0]

            state, reward, done, _ = env.step(action)
            state = state2d_1d(state)  # For CarRacing-v0

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

    network.train()

    eva_reward = np.average(eva_rewards)
    eva_r_std = float(np.std(eva_rewards))
    if eva_reward > target_reward:
        print("Eval| avg: %.2f std: %.2f" % (eva_reward, eva_r_std))

    return eva_reward


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

    state_dim = 14 ** 2  # For CarRacing-v0Racing-v0

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


def draw_plot_ppo(recorders, smooth_kernel, mod_dir, save_name=None):  # 2019-11-08 16
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

    x_epoch = np.array(recorders[:, 0])

    fig, axs = plt.subplots(2)
    plt.title(save_name, y=2.3)

    r_avg, r_std = calculate_avg_std(recorders[:, 1], smooth_kernel)
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

    ax21 = axs[1]
    ax21_color = 'darkcyan'
    ax21_label = 'mean e len'
    ax21.set_ylabel(ax21_label, color=ax21_color)
    ax21.plot(x_epoch, -recorders[:, 1], label=ax21_label, color=ax21_color)  # negative loss A
    ax21.tick_params(axis='y', labelcolor=ax21_color)

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


if __name__ == '__main__':
    run_train()
    run_eval()
    # run_test()
