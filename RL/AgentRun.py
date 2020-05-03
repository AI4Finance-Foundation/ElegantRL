import os
import sys

import gym
import torch
import numpy as np

from AgentZoo import AgentSNAC
from AgentZoo import Memories, Recorder, RewardNorm

"""
2019-07-01 Zen4Jia1Hao2, GitHub: YonV1943 DL_RL_Zoo RL
2019-11-11 Issay-0.0 [Essay Consciousness]
2020-02-02 Issay-0.1 Deep Learning Techniques (spectral norm, DenseNet, etc.) 
2020-04-04 Issay-0.1 [An Essay of Consciousness by YonV1943], IntelAC
2020-04-20 Issay-0.2 SN_AC, IntelAC_UnitedLoss
2020-04-22 Issay-0.2 [Essay, LongDear's Cerebellum (Little Brain)]

I consider that Reinforcement Learning Algorithms before 2020 have not consciousness
They feel more like a Cerebellum (Little Brain) for Machines.

2020-04-28 Add Discrete Env CartPole, Pendulum
"""


class Arguments:  # default working setting and hyper-parameter
    def __init__(self):
        self.agent_class = AgentSNAC
        self.env_name = "LunarLanderContinuous-v2"
        self.net_dim = 2 ** 8  # the network width
        self.max_step = 2 ** 10  # max steps in one epoch
        self.max_memo = 2 ** 17  # memories capacity (memories: replay buffer)
        self.max_epoch = 2 ** 10  # max num of train_epoch
        self.batch_size = 2 ** 7  # num of transitions sampled from replay buffer.
        self.update_gap = 2 ** 7  # update the target_net, delay update

        self.gamma = 0.99  # discount factor of future rewards
        self.exp_noise = 0.25  # action = select_action(state) + noise, 'explore_noise': sigma of noise
        self.pol_noise = 0.5  # actor_target(next_state) + noise,  'policy_noise': sigma of noise

        self.is_remove = True  # remove the pre-training data? (True, False, None:ask me)
        self.cwd = 'AC_Methods_LL'  # current work directory
        gpu_id = sys.argv[0][-4]
        self.gpu_id = gpu_id if gpu_id.isdigit() else '0'
        self.random_seed = 1943 + int(self.gpu_id)

    def init_for_training(self):  # remove cwd, choose GPU, set random seed, set CPU threads
        print('GPU: {} | CWD: {}'.format(self.gpu_id, self.cwd))
        whether_remove_history(self.cwd, self.is_remove)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(8)


def train_agent(agent_class, env_name, cwd, net_dim, max_step, max_memo, max_epoch,  # env
                batch_size, update_gap, gamma, exp_noise, pol_noise,  # update
                **_kwargs):  # 2020-0430
    env = gym.make(env_name)
    state_dim, action_dim, max_action, target_reward = get_env_info(env)

    agent = agent_class(state_dim, action_dim, net_dim)
    agent.save_or_load_model(cwd, is_save=False)

    memo_action_dim = action_dim if max_action else 1  # Discrete action space
    memo = Memories(max_memo, memo_dim=1 + 1 + state_dim + memo_action_dim + state_dim)
    memo.save_or_load_memo(cwd, is_save=False)

    recorder = Recorder(agent, max_step, max_action, target_reward, env_name)
    r_norm = RewardNorm(n_max=target_reward, n_min=recorder.reward_avg)

    try:
        for epoch in range(max_epoch):
            with torch.no_grad():  # just the GPU memory
                epoch_reward, iter_num = agent.inactive_in_env(
                    env, memo, max_step, exp_noise, max_action, r_norm)

            actor_loss, critic_loss = agent.update_parameter(
                memo, iter_num, batch_size, pol_noise, update_gap, gamma)

            if np.isnan(actor_loss) or np.isnan(critic_loss):
                print("ValueError: loss value should not be 'nan'. Please run again.")
                return False

            with torch.no_grad():  # just the GPU memory
                is_solved = recorder.show_and_check_reward(
                    epoch, epoch_reward, iter_num, actor_loss, critic_loss, cwd)
                if is_solved:
                    break

    except KeyboardInterrupt:
        print("raise KeyboardInterrupt while training.")
    except AssertionError:  # for BipedWalker BUG 2020-03-03
        print("AssertionError: OpenAI gym r.LengthSquared() > 0.0f ??? Please run again.")
        return False

    train_time = recorder.show_and_save(env_name, cwd)

    # agent.save_or_load_model(cwd, is_save=True)  # save max reward agent in Recorder
    memo.save_or_load_memo(cwd, is_save=True)

    draw_plot_with_npy(cwd, train_time)
    return True


"""utils"""


def get_env_info(env):  # 2020-02-02
    state_dim = env.observation_space.shape[0]

    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n  # Discrete
        action_max = None
    elif isinstance(env.action_space, gym.spaces.Box):
        action_dim = env.action_space.shape[0]  # Continuous
        action_max = float(env.action_space.high[0]) * 0.999999
        # np.float32(0.9999999), np.float16(0.999)
    else:
        action_dim = None
        action_max = None
        print('! Error with env.action_space:', env.action_space)
        exit()

    target_reward = env.spec.reward_threshold
    if target_reward is None:
        print('! Error: target_reward is None', target_reward)
        exit()

    print("| env_name: {}".format(repr(env)[10:-1]))
    print("| state_dim: {}, action_dim: {}, target_reward: {}".format(
        state_dim, action_dim, target_reward))
    return state_dim, action_dim, action_max, target_reward


def draw_plot_with_npy(mod_dir, train_time):  # 2020-04-40
    record_epoch = np.load('%s/record_epoch.npy' % mod_dir)
    # record_epoch.append((epoch_reward, actor_loss, critic_loss, iter_num))
    record_eval = np.load('%s/record_eval.npy' % mod_dir)
    # record_eval.append((epoch, eval_reward, eval_std))

    if len(record_eval.shape) == 1:
        record_eval = np.array([[0., 0., 0.]])

    train_time = int(train_time)
    iter_num = int(sum(record_epoch[:, -1]))
    epoch_num = int(record_eval[-1, 0])
    save_title = "plot_{:04}E_{}T_{}s".format(epoch_num, iter_num, train_time)
    save_path = "{}/{}.png".format(mod_dir, save_title)

    """plot"""
    import matplotlib as mpl  # draw figure in Terminal
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    # plt.style.use('ggplot')

    fig, axs = plt.subplots(2)
    plt.title(save_title, y=2.3)

    ax13 = axs[0].twinx()
    ax13.fill_between(np.arange(record_epoch.shape[0]), record_epoch[:, 3],
                      facecolor='grey', alpha=0.1, )

    ax11 = axs[0]
    ax11_color = 'royalblue'
    ax11_label = 'Epo R'
    ax11.set_ylabel(ylabel=ax11_label, color=ax11_color)
    ax11.tick_params(axis='y', labelcolor=ax11_color)
    ax11.plot(record_epoch[:, 0], label=ax11_label, color=ax11_color)

    ax12 = axs[0]
    ax12_color = 'lightcoral'
    ax12_label = 'Epoch R'
    ax12.set_ylabel(ylabel=ax12_label, color=ax12_color)
    ax12.tick_params(axis='y', labelcolor=ax12_color)

    xs = record_eval[:, 0]
    r_avg = record_eval[:, 1]
    r_std = record_eval[:, 2]
    ax12.plot(xs, r_avg, label=ax12_label, color=ax12_color)
    ax12.fill_between(xs, r_avg - r_std, r_avg + r_std, facecolor=ax12_color, alpha=0.3, )

    ax21 = axs[1]
    ax21_color = 'darkcyan'
    ax21_label = '- loss A'
    ax21.set_ylabel(ax21_label, color=ax21_color)
    ax21.plot(-record_epoch[:, 1], label=ax21_label, color=ax21_color)  # negative loss A
    ax21.tick_params(axis='y', labelcolor=ax21_color)

    ax22 = axs[1].twinx()
    ax22_color = 'darkcyan'
    ax22_label = 'loss C'
    ax22.set_ylabel(ax22_label, color=ax22_color)
    ax22.fill_between(np.arange(record_epoch.shape[0]), record_epoch[:, 2], facecolor=ax22_color, alpha=0.2, )
    ax22.tick_params(axis='y', labelcolor=ax22_color)

    plt.savefig(save_path)
    # plt.show()
    # plt.ion()
    # plt.pause(4)


def whether_remove_history(cwd, remove=None):  # 2020-03-03
    import shutil

    if remove is None:
        remove = bool(input("PRESS 'y' to REMOVE: {}? ".format(cwd)) == 'y')
    if remove:
        shutil.rmtree(cwd, ignore_errors=True)
        print("| Remove")

    os.makedirs(cwd, exist_ok=True)

    shutil.copy(sys.argv[-1], cwd)  # copy *.py to cwd
    print('copy {} to {}'.format(sys.argv[-1], cwd))
    del shutil


"""demo"""


def run__demo():
    """
    Default Agent: AgentSNAC (Spectral Normalization Actor-critic methods)
    Default Environment: LunarLanderContinuous-v2
    Default setting see 'class Arguments()' for details
    """
    args = Arguments()
    args.init_for_training()
    while not train_agent(**vars(args)):
        args.random_seed += 42


def run__sn_ac(gpu_id, cwd='AC_SNAC'):
    from AgentZoo import AgentSNAC
    args = Arguments()

    args.agent_class = AgentSNAC
    args.gpu_id = gpu_id
    args.gamma = 0.995

    args.env_name = "LunarLanderContinuous-v2"
    args.cwd = './{}/LL_{}'.format(cwd, gpu_id)
    args.init_for_training()
    while not train_agent(**vars(args)):
        args.random_seed += 42

    args.env_name = "BipedalWalker-v3"
    args.cwd = './{}/BW_{}'.format(cwd, gpu_id)
    args.init_for_training()
    while not train_agent(**vars(args)):
        args.random_seed += 42

    # args.env_name = "BipedalWalkerHardcore-v3"
    # args.cwd = './{}/BWHC_{}'.format(cwd, gpu_id)
    # args.net_dim = int(2 ** 9)
    # args.max_memo = 2 ** 16 * 24
    # args.batch_size = int(2 ** 9 * 1.5)
    # args.max_epoch = 2 ** 14
    # args.init_for_training()
    # while not run_train(**vars(args)):
    #     args.random_seed += 42

    # import pybullet_envs  # for python-bullet-gym
    # args.env_name = "MinitaurBulletEnv-v0"
    # args.cwd = './{}/Minitaur_{}'.format(cwd, args.gpu_id)
    # args.max_epoch = 2 ** 13
    # args.max_memo = 2 ** 18
    # args.is_remove = True
    # args.init_for_training()
    # while not run_train(**vars(args)):
    #     args.random_seed += 42


def run__intel_ac(gpu_id, cwd='AC_IntelAC'):
    from AgentZoo import AgentIntelAC
    args = Arguments()

    args.agent_class = AgentIntelAC
    args.gpu_id = gpu_id

    args.env_name = "LunarLanderContinuous-v2"
    args.cwd = './{}/LL_{}'.format(cwd, gpu_id)
    args.init_for_training()
    while not train_agent(**vars(args)):
        args.random_seed += 42

    args.env_name = "BipedalWalker-v3"
    args.cwd = './{}/BW_{}'.format(cwd, gpu_id)
    args.init_for_training()
    while not train_agent(**vars(args)):
        args.random_seed += 42

    # args.env_name = "BipedalWalkerHardcore-v3"
    # args.cwd = './{}/BWHC_{}'.format(cwd, gpu_id)
    # args.net_dim = int(2 ** 9)
    # args.max_memo = 2 ** 16 * 24
    # args.batch_size = int(2 ** 9 * 1.5)
    # args.max_epoch = 2 ** 14
    # args.init_for_training()
    # while not run_train(**vars(args)):
    #     args.random_seed += 42

    # import pybullet_envs  # for python-bullet-gym
    # args.env_name = "MinitaurBulletEnv-v0"
    # args.cwd = './{}/Minitaur_{}'.format(cwd, args.gpu_id)
    # args.max_epoch = 2 ** 13
    # args.max_memo = 2 ** 18
    # args.is_remove = True
    # args.init_for_training()
    # while not run_train(**vars(args)):
    #     args.random_seed += 42


def run__td3(gpu_id, cwd):
    from AgentZoo import AgentTD3
    args = Arguments()

    '''
    DenseNet, SN, Hard Update
    '''

    args.agent_class = AgentTD3
    args.gpu_id = gpu_id
    args.exp_noise = 0.1
    args.pol_noise = 0.2
    args.max_epoch = 2 ** 12
    args.max_memo = 2 ** 18

    args.init_for_training()

    args.env_name = "LunarLanderContinuous-v2"
    args.cwd = './{}/LL_{}'.format(cwd, gpu_id)
    while not train_agent(**vars(args)):
        args.random_seed += 42

    args.env_name = "BipedalWalker-v3"
    args.cwd = './{}/BW_{}'.format(cwd, gpu_id)
    while not train_agent(**vars(args)):
        args.random_seed += 42


def run__multi_process(target_func, gpu_tuple=(0, 1), cwd='AC_Methods_MP'):
    os.makedirs(cwd, exist_ok=True)  # all the files save in here

    '''run in multiprocessing'''
    import multiprocessing as mp
    os.system('cp {} {}/'.format(sys.argv[-1], cwd))
    processes = [mp.Process(target=target_func, args=(gpu_id, cwd)) for gpu_id in gpu_tuple]
    [process.start() for process in processes]
    [process.join() for process in processes]


if __name__ == '__main__':
    # run__demo()

    # run__td3(gpu_id=0, cwd='AC_TD3_ResNet')
    # run__multi_process(run__td3, gpu_tuple=(2, 3), cwd='AC_TD3_HardUpdate')

    # run__sn_ac(gpu_id=0, cwd='AC_SNAC')
    # run__multi_process(run__sn_ac, gpu_tuple=(2, 3), cwd='AC_SNAC_AdamGamma995')

    # run__intel_ac(gpu_id=0, cwd='AC_SNAC')
    run__multi_process(run__intel_ac, gpu_tuple=(0, 1, 2, 3), cwd='AC_IntelAC')

    pass
