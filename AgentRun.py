import os
import sys

import gym
import torch
import numpy as np

from AgentZoo import Recorder
from AgentZoo import BufferArray, initial_exploration

"""
2019-07-01 Zen4Jia1Hao2, GitHub: YonV1943 DL_RL_Zoo RL
2019-11-11 Issay-0.0 [Essay Consciousness]
2020-02-02 Issay-0.1 Deep Learning Techniques (spectral norm, DenseNet, etc.) 
2020-04-04 Issay-0.1 [An Essay of Consciousness by YonV1943], IntelAC
2020-04-20 Issay-0.2 SN_AC, IntelAC_UnitedLoss
2020-04-22 Issay-0.2 [Essay, LongDear's Cerebellum (Little Brain)]
2020-06-06 Issay-0.3 check PPO, SAC. Plan to add discrete SAC, EBM(soft-q-learning)

I consider that Reinforcement Learning Algorithms before 2020 have not consciousness
They feel more like a Cerebellum (Little Brain) for Machines.
In my opinion, before 2020, the policy gradient algorithm agent didn't learn s policy.
Actually, they "learn game feel" or "get a soft touch". In Chinese "shou3 gan3". 
Learn more about policy gradient algorithms in:
https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html

2020-04-28 Add Discrete Env CartPole, Pendulum
"""


class Arguments:  # default working setting and hyper-parameter
    def __init__(self, class_agent):
        self.class_agent = class_agent
        self.net_dim = 2 ** 7  # the network width
        self.max_step = 2 ** 10  # max steps in one epoch
        self.max_memo = 2 ** 17  # memories capacity (memories: replay buffer)
        self.max_epoch = 2 ** 10  # max times of train_epoch
        self.batch_size = 2 ** 7  # num of transitions sampled from replay buffer.
        self.repeat_times = 1  # Two-time Update Rule (TTUR)
        self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256
        self.gamma = 0.99  # discount factor of future rewards

        self.gpu_id = 0
        self.random_seed = 19430
        self.is_remove = True  # remove the pre-training data? (True, False, None:ask me)
        self.env_name = "LunarLanderContinuous-v2"
        self.cwd = 'AC_Methods_LL'  # current work directory

        self.show_gap = 2 ** 7  # show the Reward and Loss of actor and critic per show_gap seconds

    def init_for_training(self):  # remove cwd, choose GPU, set random seed, set CPU threads
        print('GPU: {} | CWD: {}'.format(self.gpu_id, self.cwd))
        whether_remove_history(self.cwd, self.is_remove)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)

        # env.seed()  # env has random seed too.
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(8)


def train_agent__off_policy(
        class_agent, net_dim, batch_size, repeat_times, gamma, reward_scale, cwd,
        env_name, max_step, max_memo, max_epoch, **_kwargs):  # 2020-06-01
    env = gym.make(env_name)
    state_dim, action_dim, max_action, target_reward, is_discrete = get_env_info(env, is_print=False)
    assert not is_discrete

    '''init'''
    agent = class_agent(state_dim, action_dim, net_dim)  # training agent
    agent.state = env.reset()
    buffer = BufferArray(max_memo, state_dim, action_dim)  # experiment replay buffer
    recorder = Recorder(agent, max_step, max_action, target_reward, env_name, **_kwargs)  # unnecessary

    '''loop'''
    with torch.no_grad():  # update replay buffer
        # rewards, steps = agent.update_buffer(env, buffer, max_step, max_action, reward_scale, gamma)
        rewards, steps = initial_exploration(env, buffer, max_step, max_action, reward_scale, gamma, action_dim)
    recorder.show_reward(rewards, steps, loss_a=0, loss_c=0)
    try:
        for epoch in range(max_epoch):
            # update replay buffer by interact with environment
            with torch.no_grad():  # for saving the GPU buffer
                rewards, steps = agent.update_buffer(
                    env, buffer, max_step, max_action, reward_scale, gamma)

            # update network parameters by random sampling buffer for gradient descent
            buffer.init_before_sample()
            loss_a, loss_c = agent.update_parameters(
                buffer, max_step, batch_size, repeat_times)

            # show/check the reward, save the max reward actor
            with torch.no_grad():  # for saving the GPU buffer
                # NOTICE! Recorder saves the agent with max reward automatically.
                recorder.show_reward(rewards, steps, loss_a, loss_c)

                is_solved = recorder.check_reward(cwd, loss_a, loss_c)
            if is_solved:
                break
    except KeyboardInterrupt:
        print("| raise KeyboardInterrupt and break training loop")
    # except AssertionError:  # for BipedWalker BUG 2020-03-03
    #     print("AssertionError: OpenAI gym r.LengthSquared() > 0.0f ??? Please run again.")

    train_time = recorder.print_and_save_npy(env_name, cwd)

    if is_solved:
        agent.save_or_load_model(cwd, is_save=True)
    draw_plot_with_npy(cwd, train_time)


def train_agent__on_policy(
        class_agent, net_dim, batch_size, repeat_times, gamma, reward_scale, cwd,
        env_name, max_step, max_memo, max_epoch, **_kwargs):  # 2020-0430
    env = gym.make(env_name)
    state_dim, action_dim, max_action, target_reward, is_discrete = get_env_info(env, is_print=True)

    agent = class_agent(state_dim, action_dim, net_dim)
    agent.save_or_load_model(cwd, is_save=False)

    recorder = Recorder(agent, max_step, max_action, target_reward, env_name, **_kwargs)

    try:
        for epoch in range(max_epoch):
            with torch.no_grad():  # just the GPU memory
                rewards, steps, buffer = agent.update_buffer_online(
                    env, max_step, max_memo, max_action, reward_scale, gamma)

            loss_a, loss_c = agent.update_parameters_online(
                buffer, batch_size, repeat_times)

            with torch.no_grad():  # just the GPU memory
                recorder.show_reward(rewards, steps, loss_a, loss_c)
                is_solved = recorder.check_reward(cwd, loss_a, loss_c)
                if is_solved:
                    break

    except KeyboardInterrupt:
        print("raise KeyboardInterrupt while training.")
    except AssertionError:  # for BipedWalker BUG 2020-03-03
        print("AssertionError: OpenAI gym r.LengthSquared() > 0.0f ??? Please run again.")
        return False

    train_time = recorder.print_and_save_npy(env_name, cwd)

    draw_plot_with_npy(cwd, train_time)
    return True


def train_agent_discrete(
        class_agent, net_dim, batch_size, repeat_times, gamma, reward_scale, cwd,
        env_name, max_step, max_memo, max_epoch, **_kwargs):  # 2020-05-20
    env = gym.make(env_name)
    state_dim, action_dim, max_action, target_reward, is_discrete = get_env_info(env, is_print=True)
    assert is_discrete

    '''init'''
    agent = class_agent(state_dim, action_dim, net_dim)  # training agent
    agent.state = env.reset()
    buffer = BufferArray(max_memo, state_dim, action_dim=1)  # experiment replay buffer
    recorder = Recorder(agent, max_step, max_action, target_reward, env_name, **_kwargs)

    '''loop'''
    with torch.no_grad():  # update replay buffer
        rewards, steps = initial_exploration(
            env, buffer, max_step, max_action, reward_scale, gamma, action_dim)
    recorder.show_reward(rewards, steps, loss_a=0, loss_c=0)
    try:
        for epoch in range(max_epoch):
            # update replay buffer by interact with environment
            with torch.no_grad():  # for saving the GPU buffer
                rewards, steps = agent.update_buffer(
                    env, buffer, max_step, max_action, reward_scale, gamma)

            # update network parameters by random sampling buffer for gradient descent
            buffer.init_before_sample()
            loss_a, loss_c = agent.update_parameters(buffer, max_step, batch_size, repeat_times)

            # show/check the reward, save the max reward actor
            with torch.no_grad():  # for saving the GPU buffer
                # NOTICE! Recorder saves the agent with max reward automatically.
                recorder.show_reward(rewards, steps, loss_a, loss_c)

                is_solved = recorder.check_reward(cwd, loss_a, loss_c)
            if is_solved:
                break
    except KeyboardInterrupt:
        print("| raise KeyboardInterrupt and break training loop")
    # except AssertionError:  # for BipedWalker BUG 2020-03-03
    #     print("AssertionError: OpenAI gym r.LengthSquared() > 0.0f ??? Please run again.")

    train_time = recorder.print_and_save_npy(env_name, cwd)

    if is_solved:
        agent.save_or_load_model(cwd, is_save=True)
    draw_plot_with_npy(cwd, train_time)


"""utils"""


def get_env_info(env, is_print):  # 2020-06-06
    state_dim = env.observation_space.shape[0]

    try:
        is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        if is_discrete:  # discrete
            action_dim = env.action_space.n
            action_max = int(1)
        elif isinstance(env.action_space, gym.spaces.Box):  # make sure it is continuous action space
            action_dim = env.action_space.shape[0]
            action_max = float(env.action_space.high[0])
        else:
            raise AttributeError
    except AttributeError:
        print("| Could you assign these value manually? \n"
              "| I need: state_dim, action_dim, action_max, target_reward, is_discrete")
        raise AttributeError

    target_reward = env.spec.reward_threshold
    if target_reward is None:
        print("| Could you assign these value manually? \n"
              "| I need: target_reward")
        raise ValueError

    if is_print:
        print("| env_name: {}, action space: {}".format(repr(env)[10:-1], 'Discrete' if is_discrete else 'Continuous'))
        print("| state_dim: {}, action_dim: {}, action_max: {}, target_reward: {}".format(
            state_dim, action_dim, action_max, target_reward))
    return state_dim, action_dim, action_max, target_reward, is_discrete


def draw_plot_with_npy(mod_dir, train_time):  # 2020-04-40
    record_epoch = np.load('%s/record_epoch.npy' % mod_dir)  # , allow_pickle=True)
    # record_epoch.append((epoch_reward, actor_loss, critic_loss, iter_num))
    record_eval = np.load('%s/record_eval.npy' % mod_dir)  # , allow_pickle=True)
    # record_eval.append((epoch, eval_reward, eval_std))

    # print(';record_epoch:', record_epoch.shape)
    # print(';record_eval:', record_eval.shape)
    # print(record_epoch)
    # # print(record_eval)
    # exit()

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


def whether_remove_history(cwd, is_remove=None):  # 2020-03-04
    import shutil

    if is_remove is None:
        is_remove = bool(input("PRESS 'y' to REMOVE: {}? ".format(cwd)) == 'y')
    if is_remove:
        shutil.rmtree(cwd, ignore_errors=True)
        print("| Remove")

    os.makedirs(cwd, exist_ok=True)

    # shutil.copy(sys.argv[-1], "{}/AgentRun-py-backup".format(cwd))  # copy *.py to cwd
    # shutil.copy('AgentZoo.py', "{}/AgentZoo-py-backup".format(cwd))  # copy *.py to cwd
    # shutil.copy('AgentNet.py', "{}/AgentNetwork-py-backup".format(cwd))  # copy *.py to cwd
    del shutil


"""demo"""


def run__demo(gpu_id, cwd='RL_BasicAC'):
    from AgentZoo import AgentBasicAC

    args = Arguments(class_agent=AgentBasicAC)
    args.gpu_id = gpu_id

    args.env_name = "LunarLanderContinuous-v2"
    args.cwd = './{}/LL_{}'.format(cwd, gpu_id)
    args.init_for_training()
    train_agent__off_policy(**vars(args))

    args.env_name = "BipedalWalker-v3"
    args.cwd = './{}/BW_{}'.format(cwd, gpu_id)
    args.init_for_training()
    train_agent__off_policy(**vars(args))


def run__zoo(gpu_id, cwd='RL_Zoo'):
    import AgentZoo as Zoo
    class_agent = Zoo.AgentDeepSAC

    assert class_agent in {
        Zoo.AgentDDPG, Zoo.AgentTD3, Zoo.ActorSAC, Zoo.AgentDeepSAC,
        Zoo.AgentBasicAC, Zoo.AgentSNAC, Zoo.AgentInterAC, Zoo.AgentInterSAC,
    }  # you can't run PPO here. goto run__ppo(). PPO need its hyper-parameters

    args = Arguments(class_agent)
    args.gpu_id = gpu_id

    args.env_name = "LunarLanderContinuous-v2"
    args.cwd = './{}/LL_{}'.format(cwd, gpu_id)
    args.init_for_training()
    train_agent__off_policy(**vars(args))

    args.env_name = "BipedalWalker-v3"
    args.cwd = './{}/BW_{}'.format(cwd, gpu_id)
    args.init_for_training()
    train_agent__off_policy(**vars(args))

    # args.env_name = "BipedalWalkerHardcore-v3"
    # args.cwd = './{}/BWHC_{}'.format(cwd, gpu_id)
    # args.net_dim = int(2 ** 8.5)
    # args.max_memo = int(2 ** 20)
    # args.batch_size = int(2 ** 9)
    # args.max_epoch = 2 ** 14
    # args.reward_scale = int(2 ** 6.5)
    # args.is_remove = None
    # args.init_for_training()
    # while not train_agent(**vars(args)):
    #     args.random_seed += 42

    # import pybullet_envs  # for python-bullet-gym
    # dir(pybullet_envs)
    # args.env_name = "MinitaurBulletEnv-v0"
    # args.cwd = './{}/Minitaur_{}'.format(cwd, args.gpu_id)
    # args.max_epoch = 2 ** 13
    # args.max_memo = 2 ** 20
    # args.net_dim = 2 ** 9
    # args.max_step = 2 ** 12
    # args.batch_size = 2 ** 8
    # args.reward_scale = 2 ** 3
    # args.is_remove = True
    # args.eva_size = 2 ** 5  # for Recorder
    # args.show_gap = 2 ** 8  # for Recorder
    # args.init_for_training()
    # while not train_agent(**vars(args)):
    #     args.random_seed += 42

    # import pybullet_envs  # for python-bullet-gym
    # dir(pybullet_envs)
    # args.env_name = "AntBulletEnv-v0"
    # args.cwd = './{}/Ant_{}'.format(cwd, args.gpu_id)
    # args.max_epoch = 2 ** 13
    # args.max_memo = 2 ** 20
    # args.max_step = 2 ** 10
    # args.net_dim = 2 ** 8
    # args.batch_size = 2 ** 8
    # args.reward_scale = 2 ** -3
    # args.is_remove = True
    # args.eva_size = 2 ** 5  # for Recorder
    # args.show_gap = 2 ** 8  # for Recorder
    # args.init_for_training()
    # while not train_agent(**vars(args)):
    #     args.random_seed += 42


def run__ppo(gpu_id, cwd='RL_PPO'):
    import AgentZoo as Zoo
    class_agent = Zoo.AgentGAE

    assert class_agent in {Zoo.AgentPPO, Zoo.AgentGAE}
    args = Arguments(class_agent)

    args.gpu_id = gpu_id
    args.max_memo = 2 ** 12
    args.batch_size = 2 ** 9
    args.repeat_times = 2 ** 4
    args.net_dim = 2 ** 8
    args.gamma = 0.99

    args.env_name = "LunarLanderContinuous-v2"
    args.cwd = './{}/LL_{}'.format(cwd, gpu_id)
    args.init_for_training()
    while not train_agent__on_policy(**vars(args)):
        args.random_seed += 42

    args.env_name = "BipedalWalker-v3"
    args.cwd = './{}/BW_{}'.format(cwd, gpu_id)
    args.init_for_training()
    while not train_agent__on_policy(**vars(args)):
        args.random_seed += 42


def run__ppo_discrete(gpu_id, cwd='RL_DiscreteGAE'):
    import AgentZoo as Zoo
    class_agent = Zoo.AgentDiscreteGAE

    assert class_agent in {Zoo.AgentDiscreteGAE, }
    args = Arguments(class_agent=class_agent)

    args.gpu_id = gpu_id
    args.max_memo = 2 ** 10
    args.batch_size = 2 ** 8
    args.repeat_times = 2 ** 4
    args.net_dim = 2 ** 7

    args.env_name = "CartPole-v0"
    args.cwd = './{}/CP_{}'.format(cwd, gpu_id)
    args.init_for_training()
    while not train_agent__on_policy(**vars(args)):
        args.random_seed += 42

    args.gpu_id = gpu_id
    args.max_memo = 2 ** 12
    args.batch_size = 2 ** 9
    args.repeat_times = 2 ** 4
    args.net_dim = 2 ** 8

    args.env_name = "LunarLander-v2"
    args.cwd = './{}/LL_{}'.format(cwd, gpu_id)
    args.init_for_training()
    while not train_agent__on_policy(**vars(args)):
        args.random_seed += 42


def run__dqn(gpu_id, cwd='RL_DQN'):
    import AgentZoo as Zoo
    class_agent = Zoo.AgentDoubleDQN
    assert class_agent in {Zoo.AgentDQN, Zoo.AgentDoubleDQN}

    args = Arguments(class_agent)
    args.gpu_id = gpu_id
    args.show_gap = 2 ** 5

    args.env_name = "CartPole-v0"
    args.cwd = '{}/{}'.format(cwd, args.env_name)
    args.init_for_training()
    train_agent_discrete(**vars(args))

    args.env_name = "LunarLander-v2"
    args.cwd = '{}/{}'.format(cwd, args.env_name)
    args.init_for_training()
    train_agent_discrete(**vars(args))


"""demo plan to do multi-agent"""


def run__multi_process(target_func, gpu_tuple=(0, 1), cwd='RL_MP'):
    os.makedirs(cwd, exist_ok=True)  # all the files save in here

    '''run in multiprocessing'''
    import multiprocessing as mp
    processes = [mp.Process(target=target_func, args=(gpu_id, cwd)) for gpu_id in gpu_tuple]
    [process.start() for process in processes]
    [process.join() for process in processes]


def process__buffer(q_aggr, qs_dist, args, **_kwargs):
    max_memo = args.max_memo
    env_name = args.env_name
    max_step = args.max_step
    batch_size = args.batch_size
    repeat_times = 2

    # reward_scale = args.reward_scale
    # gamma = args.gamma

    '''init'''
    env = gym.make(env_name)
    state_dim, action_dim, max_action, target_reward, is_discrete = get_env_info(env, is_print=False)
    buffer = BufferArray(max_memo, state_dim, action_dim)  # experiment replay buffer

    workers_num = len(qs_dist)

    '''loop'''
    is_training = True
    while is_training:
        for i in range(workers_num):
            memo_array, is_solved = q_aggr.get()
            buffer.extend_memo(memo_array)
            if is_solved:
                is_training = False

        buffer.init_before_sample()
        for i in range(max_step * repeat_times):
            # batch_arrays = buffer.random_sample(batch_size, device=None) # faster but worse
            for q_dist in qs_dist:
                batch_arrays = buffer.random_sample(batch_size, device=None)  # slower but better
                q_dist.put(batch_arrays)

    print('|| Exit: process__buffer')


def process__workers(gpu_id, root_cwd, q_aggr, q_dist, args, **_kwargs):
    class_agent = args.class_agent
    env_name = args.env_name
    cwd = args.cwd
    net_dim = args.net_dim
    max_step = args.max_step
    # max_memo = args.max_memo
    max_epoch = args.max_epoch
    batch_size = args.batch_size * 1.5
    gamma = args.gamma
    update_gap = args.update_gap
    reward_scale = args.reward_scale

    cwd = '{}/{}_{}'.format(root_cwd, cwd, gpu_id)
    os.makedirs(cwd, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    random_seed = 42 + gpu_id
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(8)

    env = gym.make(env_name)
    is_solved = False

    class BufferArrayMP(BufferArray):
        def init_before_sample(self):
            q_aggr.put((self.memories, is_solved))
            # self.now_len = self.max_len if self.is_full else self.next_idx

        def random_sample(self, _batch_size, device=None):
            batch_arrays = q_dist.get()

            '''convert array into torch.tensor'''
            tensors = [torch.tensor(ary, device=device) for ary in batch_arrays]
            return tensors

    '''init'''
    state_dim, action_dim, max_action, target_reward, is_discrete = get_env_info(env, is_print=True)
    agent = class_agent(env, state_dim, action_dim, net_dim)  # training agent
    buffer = BufferArrayMP(max_step, state_dim, action_dim)  # experiment replay buffer
    recorder = Recorder(agent, max_step, max_action, target_reward, env_name, **_kwargs)

    '''loop'''
    # with torch.no_grad():  # update replay buffer
    #     # rewards, steps = agent.update_buffer(
    #     #     env, buffer, max_step, max_action, reward_scale, gamma)
    #     rewards, steps = initial_exploration(
    #         env, buffer, max_step, max_action, reward_scale, gamma, action_dim)
    # recorder.show_reward(rewards, steps, 0, 0)
    try:
        for epoch in range(max_epoch):
            '''update replay buffer by interact with environment'''
            with torch.no_grad():  # for saving the GPU buffer
                rewards, steps = agent.update_buffer(env, buffer, max_step, max_action, reward_scale, gamma)

            '''update network parameters by random sampling buffer for stochastic gradient descent'''
            loss_a, loss_c = agent.update_parameters(buffer, max_step, batch_size, update_gap)

            '''show/check the reward, save the max reward actor'''
            with torch.no_grad():  # for saving the GPU buffer
                '''NOTICE! Recorder saves the agent with max reward automatically. '''
                recorder.show_reward(rewards, steps, loss_a, loss_c)

                is_solved = recorder.check_reward(cwd, loss_a, loss_c)
            if is_solved:
                break
    except KeyboardInterrupt:
        print("raise KeyboardInterrupt while training.")
    # except AssertionError:  # for BipedWalker BUG 2020-03-03
    #     print("AssertionError: OpenAI gym r.LengthSquared() > 0.0f ??? Please run again.")
    #     return False

    train_time = recorder.print_and_save_npy(env_name, cwd)

    # agent.save_or_load_model(cwd, is_save=True)  # save max reward agent in Recorder
    # buffer.save_or_load_memo(cwd, is_save=True)

    draw_plot_with_npy(cwd, train_time)
    return True


def run__multi_workers(gpu_tuple=(0, 1), root_cwd='RL_MP'):
    print('GPU: {} | CWD: {}'.format(gpu_tuple, root_cwd))
    whether_remove_history(root_cwd, is_remove=True)

    from AgentZoo import AgentSAC
    args = Arguments(AgentSAC)
    args.env_name = "BipedalWalker-v3"
    # args.env_name = "LunarLanderContinuous-v2"

    args.show_gap = 2 ** 8  # for Recorder

    '''run in multiprocessing'''
    import multiprocessing as mp
    workers_num = len(gpu_tuple)
    queue_aggr = mp.Queue(maxsize=workers_num)  # queue of aggregation
    queues_dist = [mp.Queue(maxsize=args.max_step) for _ in range(workers_num)]  # queue of distribution

    processes = [mp.Process(target=process__buffer, args=(queue_aggr, queues_dist, args))]
    processes.extend([mp.Process(target=process__workers, args=(gpu_id, root_cwd, queue_aggr, queue_dist, args))
                      for gpu_id, queue_dist in zip(gpu_tuple, queues_dist)])

    [process.start() for process in processes]
    # [process.join() for process in processes]
    [process.close() for process in processes]


if __name__ == '__main__':
    run__demo(gpu_id=0, cwd='AC_BasicAC')
    # run__zoo(gpu_id=0, cwd='AC_SAC')
    # run__ppo(gpu_id=1, cwd='AC_PPO')

    # run__multi_process(run__zoo, gpu_tuple=(0, 1, 2, 3), cwd='AC_ZooMP')
    # run__multi_process(run__ppo, gpu_tuple=(2, 3), cwd='AC_PPO')
    # run__multi_workers(gpu_tuple=(2, 3), root_cwd='AC_SAC_MP')

    # '''Discrete action space'''
    # run__dqn(gpu_id=sys.argv[-1][-4], cwd='RL_DQN')

    # '''multi worker'''
    # run__multi_workers(gpu_tuple=(2, 3), root_cwd='AC_SAC_MP')

    print('Finish:', sys.argv[-1])
