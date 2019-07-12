import os
import sys
from time import time as timer

import gym
import numpy as np
import numpy.random as rd

import torch
from agent import Memories
from agent import AgentDelayDDPG

"""
beta2
 Epoch (112, 143) TimeUsed: (823, 1200)sec
dropout, break adjust, f_hswish, 2425s(68, 99)!!!!!
beta2: DelayDDPG_stable 
explore_noise; policy_noise; TimeUsed; Epoch; 
926s E51,1048s E73
dropout 0.50, E92, 1531s, stable

LunarLanderContinuous-v2
1531s E92, 1455s E97

LunarLander-v2 (Discrete)
dropout 0.50: 1530s E147, 3036s E277, 

BipedalWalker-v2
1620s E172
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
    gpu_id = 1  # sys.argv[0][-4]  # ! !!!!!!!!! !!!! !!!
    mod_dir = 'DelayDDPG_%s' % gpu_id  # ! !!!!!!!!! !!!! !!!
    env_name = "LunarLanderContinuous-v2"
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
    update_gap = 2 ** 8  # update the target_net, delay update
    # big update_gap will lengthen the training time, but get a better policy network
    eval_epoch = 2 ** 2  # eval this model after training. and render the env

    '''break'''
    target_reward = 200  # when 'epoch_reward > target_reward', break the training loop
    # "LunarLanderContinuous-v2" Recommended range(100, 200)
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

    if 'LunarLanderContinuous-v2':
        env_name = "LunarLanderContinuous-v2"
    # if 'Pendulum-v0':
    #     env_name = "Pendulum-v0"
    #     max_step = 200
    # if "BipedalWalker-v2":
    #     env_name = "BipedalWalker-v2"
    #     target_reward = 100  # 300
    # if "BipedalWalkerHardcore-v2":
    #     env_name = "BipedalWalkerHardcore-v2"
    #     target_reward = 200  # 300
    #     mod_dim = 2 ** 9  # the network width of actor_net and critic_net
    #     memories_size = int(2 ** 19)  # memories capacity (memories: replay buffer)
    #     max_step = 8000  # max steps in one epoch
    #     is_remove = None  # remove the pre-training data?

    # """Discrete_Action"""
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


def train():
    args = Arguments()

    gpu_id = args.gpu_id
    env_name = args.env_name
    mod_dir = args.mod_dir

    memories_size = args.memories_size
    batch_size = args.batch_size
    update_gap = args.update_gap
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
                            gamma, policy_noise, update_gap)

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

                next_state, reward, done, _ = env.step(adapt_action(action, action_max, action_dim))
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
                        state, reward, done, _ = env.step(adapt_action(action, action_max, action_dim=False))

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
                            gamma=0, policy_noise=0, update_gap=0)
    # (gamma=0, policy_noise=0, update_gap=0) are not required for evaluating
    policy.load(mod_dir, load_actor_only=True)
    policy.act.eval()
    policy.cri.eval()

    for epoch in range(eval_epoch):
        epoch_reward = 0
        state = env.reset()
        for iter_num in range(max_step):
            action = policy.select_action(state)
            state, reward, done, _ = env.step(adapt_action(action, action_max, action_dim=False))
            epoch_reward += reward
            env.render()
            # Image.fromarray(env.render(mode='rgb_array')).save('%s/img_%4i.png'%(mod_dir, iter_num))
            if done:
                break

        print("%3i\tEpiR %3i" % (epoch, epoch_reward))
    env.close()


def adapt_action(action, action_max, action_dim):
    """
    :param action: belongs to range(-1, 1), makes it suit for env.step(action)
    :param action_max: if it is False, means DISCRETE action_space
    :param action_dim: if it is False, means DISCRETE action_space and not train.
    :return: a compatible action for env
    """
    if action_max:  # action_space: Continuous
        return action * action_max
    elif action_dim:  # action_space: Discrete and is_train
        action_prob = action + 1.00001
        action_prob /= sum(action_prob)
        return rd.choice(action_dim, p=action_prob)
    else:  # action_space: Discrete and is_eval
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


if __name__ == '__main__':
    train()
    evals()
