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
beta1 stable
2019-07-01 ZenJiaHao, GitHub: Yonv1943
2019-08-01 soft_update
2019-08-02 multi-action

LunarLanderContinuous-v2: 1182s 78E
LunarLander-v2 (Discrete): 1487s E147
BipedalWalker-v2: 2100s E165
"""


class Arguments:
    """
    All the hyper-parameter is here.

    If you are not familiar with this algorithm,
    then I do not recommend that you modify other parameters that are not listed here.
    The comments below are all my subjective guesses for reference only!

    If you wanna change this code, please keep READABILITY and ELEGANT! (read 'import this')
    Write by GitHub: Yonv1943 Zen4 Jia1Hao2, 2019-07-07
    """

    '''device'''
    gpu_id = sys.argv[0][-4]
    mod_dir = 'DelayDDPG_%s' % gpu_id
    is_remove = True  # remove the pre-training data?
    # is_remove = True,  yes, remove the directory of model
    # is_remove = None,  ask me when the program is running
    # is_remove = False, keep the pre-training data and load it when necessary
    random_seed = 1943  # random_seed for py_torch and gym.env

    '''training'''
    mod_dim = 2 ** 8  # the network width of actor_net and critic_net
    # low mod_dim should correspond to low dropout_rate
    memories_size = int(2 ** 18)  # memories capacity (memories: replay buffer)
    # low memories capacity leads to low reward in the later stage of training.
    batch_size = 2 ** 8  # num of transitions sampled from replay buffer.
    # big batch_size makes training more stable.
    update_gap = 2 ** 7  # update the target_net, delay update
    # big update_gap will lengthen the training time, but get a better policy network
    soft_update_tau = 0.5  # could be 0.9
    # big soft_update_tau require a small update_gap, and it makes the training more unstable.
    eval_epoch = 2 ** 2  # eval this model after training. and render the env

    '''break'''
    target_reward = None  # target_reward = env.spec.reward_threshold if isinstance(target_reward, int) target_reward
    # when 'epoch_reward > target_reward', break the training loop
    smooth_kernel = 16  # smooth the reward curve
    # big smooth_kernel makes the curve more smooth. Recommended range(16, 64)
    print_gap = 2 ** 5  # print the Reward, actor_loss, critic_loss
    # print the information every 'print_gap'sec
    max_epoch = 1000  # max num of train_epoch
    # if 'epoch > max_epoch' or 'epoch_reward > target_reward', break the training loop
    max_step = 2000  # max steps in one epoch
    # if 'iter_num > max_step' or 'done', break. Then reset the env and start a new round of training

    '''algorithm'''
    gamma = 0.99  # discount for future rewards
    # big gamma leads to a long-term strategy
    explore_noise = 0.4  # action = select_action(state) + noise, 'explore_noise': sigma of noise
    # big explore_noise is suitable when the fault tolerant rate of ENV is high.
    # low explore_noise delays the time when the model reaches high reward
    policy_noise = 0.8  # actor_target(next_state) + noise,  'policy_noise': sigma of noise
    # low policy_noise lead to a stable training, but a longer learning period and clumsy movements
    # Epsilon-Greedy, the variance of noise don not decay here.
    # 'explore_noise' and 'explore_noise' act on 'action' (in range(-1, 1)), before 'action*action_max'

    """Continuous_Action"""
    if 'LunarLanderContinuous-v2':
        env_name = "LunarLanderContinuous-v2"
    # if 'Pendulum-v0':
    #     env_name = "Pendulum-v0"
    #     max_step = 200
    # if "BipedalWalker-v2":
    #     env_name = "BipedalWalker-v2"
    #     target_reward = 300
    # if "BipedalWalkerHardcore-v2":
    #     env_name = "BipedalWalkerHardcore-v2"
    #     target_reward = 300  # 300
    #     mod_dim = 2 ** 9  # the network width of actor_net and critic_net
    #     memories_size = int(2 ** 19)  # memories capacity (memories: replay buffer)
    #     max_step = 8000  # max steps in one epoch
    #     max_epoch = 4000  # max num of train_epoch
    #     is_remove = False

    """Discrete_Action"""
    # if 'LunarLander-v2':
    #     env_name = "LunarLander-v2"
    # if 'CartPole-v0':
    #     env_name = "CartPole-v0"
    #     target_reward = 195
    #     print_gap = 2 ** 2
    #     explore_noise = 0.1
    #     policy_noise = 0.8
    #     memories_size = 2 ** 16
    #     batch_size = 2 ** 7
    #     mod_dim = 2 ** 7


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
                 gamma, policy_noise, update_gap, soft_update_tau):
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
        self.tau = soft_update_tau

    def select_action(self, state):
        state = torch.tensor((state,), dtype=torch.float32, device=self.device)
        action = self.act(state).cpu().data.numpy()
        return action[0]

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, memories, iter_num, batch_size):
        actor_loss_avg, critic_loss_avg = 0, 0

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
                # self.act_target.load_state_dict(self.act.state_dict())
                # self.cri_target.load_state_dict(self.cri.state_dict())
                self.soft_update(self.act_target, self.act)
                self.soft_update(self.cri_target, self.cri)

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


def report_plot(recorders, smooth_kernel, mod_dir, save_name):
    # np.save('%s/recorders.npy'% mod_dir, recorders)
    # recorders = np.load('%s/recorders.npy'% mod_dir)
    # report_plot(recorders=np.load('recorders.npy', ), smooth_kernel=32, mod_dir=0, save_name='TD.png')
    if recorders is list():
        return print('Record is empty')
    else:
        print("Matplotlib Plot:", save_name)
    import matplotlib.pyplot as plt

    y_reward = np.array(recorders[:, 0]).clip(-500, 500)
    y_reward_smooth = np.pad(y_reward, (smooth_kernel - 1, 0), mode='reflect')
    y_reward_smooth = np.convolve(y_reward_smooth, np.ones(smooth_kernel) / smooth_kernel, mode='valid')

    x_epoch = np.array(recorders[:, 1])

    fig, axs = plt.subplots(3)
    plt.title(save_name, y=3.5)

    axs[0].plot(x_epoch, y_reward, label='Reward', linestyle=':')
    axs[0].plot(x_epoch, y_reward_smooth, label='Smooth R')
    axs[0].legend()

    axs[1].plot(x_epoch, recorders[:, 2], label='loss_A')
    axs[2].plot(x_epoch, recorders[:, 3], label='loss_C')

    plt.savefig("%s/%s" % (mod_dir, save_name))
    plt.show()


def train():
    args = Arguments()

    gpu_id = args.gpu_id
    env_name = args.env_name
    mod_dir = args.mod_dir

    memories_size = args.memories_size
    batch_size = args.batch_size
    update_gap = args.update_gap
    soft_update_tau = args.soft_update_tau
    mod_dim = args.mod_dim

    target_reward = args.target_reward
    smooth_kernel = args.smooth_kernel
    print_gap = args.print_gap
    max_step = args.max_step
    max_epoch = args.max_epoch

    gamma = args.gamma
    explore_noise = args.explore_noise
    policy_noise = args.policy_noise
    random_seed = args.random_seed

    def whether_remove_history(remove=None):
        print('  GPUid: %s' % gpu_id)
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

    whether_remove_history(remove=args.is_remove)

    '''env init'''
    env = gym.make(env_name)
    env.seed(random_seed)
    target_reward = target_reward if isinstance(target_reward, int) \
        else env.spec.reward_threshold
    state_dim = env.observation_space.shape[0]
    is_continuous = bool(str(env.action_space)[:3] == 'Box')  # Continuous or Discrete
    if is_continuous:
        action_dim = env.action_space.shape[0]
        action_max = float(env.action_space.high[0])
    else:
        action_dim = env.action_space.n  # Discrete
        action_max = None
        print('action_space: Discrete:', action_dim)

    '''mod init'''
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    policy = AgentDelayDDPG(state_dim, action_dim, mod_dim,
                            gamma, policy_noise, update_gap, soft_update_tau)

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

            al, cl = policy.update(memories, iter_num, batch_size)

            recorders.append((epoch, al, cl))
            rewards.append(epoch_reward)
            smooth_reward = np.average(rewards[-smooth_kernel:])

            if timer() - show_time > print_gap:
                show_time = timer()
                print("%3i\tSmoR: %3i\tEpiR %3i\t|A %.3f, C %.3f"
                      % (epoch, smooth_reward, epoch_reward, al, cl))
            if smooth_reward > target_reward and epoch_reward > target_reward:
                print("########## Solved! ###########")
                print("%3i\tSmoR: %3i\tEpiR %3i\t|A %.3f, C %.3f"
                      % (epoch, smooth_reward, epoch_reward, al, cl))
                break

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
        policy.save(mod_dir)

    recorders = np.concatenate((np.array(rewards)[:, np.newaxis],
                                recorders), axis=1)
    report_plot(recorders, smooth_kernel, mod_dir,
                save_name="%s_plot.png" % (mod_dir,))


def evals():
    args = Arguments()

    gpu_id = args.gpu_id
    mod_dir = args.mod_dir
    env_name = args.env_name
    eval_epoch = args.eval_epoch
    max_step = args.max_step
    mod_dim = args.mod_dim

    '''env init'''
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    try:
        action_dim = env.action_space.shape[0]
        action_max = float(env.action_space.high[0])
    except IndexError:
        action_dim = env.action_space.n  # Discrete
        action_max = None
        print('action_space: Discrete:', action_dim)

    '''mod init'''
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    policy = AgentDelayDDPG(state_dim, action_dim, mod_dim,
                            gamma=0, policy_noise=0, update_gap=0, soft_update_tau=0,
                            )  # these parameters are not required for evaluating
    policy.load(mod_dir, load_actor_only=True)
    policy.act.eval()
    policy.cri.eval()

    # import cv2
    for epoch in range(eval_epoch):
        epoch_reward = 0
        state = env.reset()
        for iter_num in range(max_step):
            action = policy.select_action(state)
            state, reward, done, _ = env.step(adapt_action(action, action_max, action_dim, is_train=False))
            epoch_reward += reward

            env.render()
            # cv2.imwrite('%s/img_%4i.png'%(mod_dir, iter_num), env.render(mode='rgb_array'))

            if done:
                break

        print("%3i\tEpiR %3i" % (epoch, epoch_reward))
    env.close()


def run():
    train()
    evals()


if __name__ == '__main__':
    run()
