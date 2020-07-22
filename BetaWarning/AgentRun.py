import os
import sys
import time  # for reward recorder

import gym
import torch
import numpy as np
import numpy.random as rd

from AgentZoo import initial_exploration
from AgentZoo import BufferArray, BufferArrayGPU

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
    def __init__(self):
        self.class_agent = None
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
        self.eva_size = 2 ** 6  # for evaluated reward average

    def init_for_training(self):  # remove cwd, choose GPU, set random seed, set CPU threads
        assert self.class_agent is not None
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

    '''init: agent, buffer, recorder'''
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
                rewards, steps = agent.update_replay_buffer(
                    env, buffer, max_step, max_action, reward_scale, gamma)

            # update network parameters by random sampling buffer for gradient descent
            buffer.init_before_sample()
            loss_a, loss_c = agent.update_network_param(buffer, max_step, batch_size, repeat_times)

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


def get_env_info(env, is_print=True):  # 2020-06-06
    env_name = env.unwrapped.spec.id

    state_shape = env.observation_space.shape
    assert len(state_shape) == 1
    state_dim = state_shape[0]

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
    '''some extra rulues'''
    if env_name == 'Pendulum-v0':
        target_reward = -200.0

    if target_reward is None:
        print("| Could you assign these value manually? \n"
              "| I need: target_reward. ")
        assert target_reward is not None

    if is_print:
        print("| env_name: {}, action space: {}".format(env_name, 'is_discrete' if is_discrete else 'Continuous'))
        print("| state_dim: {}, action_dim: {}, action_max: {}, target_reward: {}".format(
            state_dim, action_dim, action_max, target_reward))
    return state_dim, action_dim, action_max, target_reward, is_discrete


def draw_plot_with_2npy(cwd, train_time):  # 2020-07-07
    record_explore = np.load('%s/record_explore.npy' % cwd)  # , allow_pickle=True)
    # record_explore.append((total_step, exp_r_avg, loss_a_avg, loss_c_avg))
    record_evaluate = np.load('%s/record_evaluate.npy' % cwd)  # , allow_pickle=True)
    # record_evaluate.append((total_step, eva_r_avg, eva_r_std))

    if len(record_evaluate.shape) == 1:
        record_evaluate = np.array([[0., 0., 0.]])

    train_time = int(train_time)
    total_step = int(record_evaluate[-1][0])
    save_title = f"plot_Step_Time_{total_step:06}_{train_time}"
    save_path = "{}/{}.png".format(cwd, save_title)

    """plot"""
    import matplotlib as mpl  # draw figure in Terminal
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    # plt.style.use('ggplot')

    fig, axs = plt.subplots(2)
    plt.title(save_title, y=2.3)

    ax11 = axs[0]
    ax11_color = 'royalblue'
    ax11_label = 'explore R'
    exp_step = record_explore[:, 0]
    exp_reward = record_explore[:, 1]
    ax11.plot(exp_step, exp_reward, label=ax11_label, color=ax11_color)

    ax12 = axs[0]
    ax12_color = 'lightcoral'
    ax12_label = 'Epoch R'
    eva_step = record_evaluate[:, 0]
    r_avg = record_evaluate[:, 1]
    r_std = record_evaluate[:, 2]
    ax12.plot(eva_step, r_avg, label=ax12_label, color=ax12_color)
    ax12.fill_between(eva_step, r_avg - r_std, r_avg + r_std, facecolor=ax12_color, alpha=0.3, )

    ax21 = axs[1]
    ax21_color = 'darkcyan'
    ax21_label = '-lossA'
    exp_loss_a = -record_explore[:, 2]
    ax21.set_ylabel(ax21_label, color=ax21_color)
    ax21.plot(exp_step, -exp_loss_a, label=ax21_label, color=ax21_color)  # negative loss A
    ax21.tick_params(axis='y', labelcolor=ax21_color)

    ax22 = axs[1].twinx()
    ax22_color = 'darkcyan'
    ax22_label = 'lossC'
    exp_loss_c = record_explore[:, 3]
    ax22.set_ylabel(ax22_label, color=ax22_color)
    ax22.fill_between(exp_step, exp_loss_c, facecolor=ax22_color, alpha=0.2, )
    ax22.tick_params(axis='y', labelcolor=ax22_color)

    plt.savefig(save_path)
    # plt.pause(4)
    # plt.show()


def draw_plot_with_npy(cwd, train_time):  # 2020-04-40
    record_epoch = np.load('%s/record_epoch.npy' % cwd)  # , allow_pickle=True)
    # record_epoch.append((epoch_reward, actor_loss, critic_loss, iter_num))
    record_eval = np.load('%s/record_eval.npy' % cwd)  # , allow_pickle=True)
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
    save_path = "{}/{}.png".format(cwd, save_title)

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


class Recorder:
    def __init__(self, agent, max_step, max_action, target_reward,
                 env_name, eva_size=100, show_gap=2 ** 7, smooth_kernel=2 ** 4,
                 state_norm=None, **_kwargs):
        self.show_gap = show_gap
        self.smooth_kernel = smooth_kernel

        '''get_eva_reward(agent, env_list, max_step, max_action)'''
        self.agent = agent
        self.env_list = [gym.make(env_name) for _ in range(eva_size)]
        self.max_step = max_step
        self.max_action = max_action
        self.e1 = 3
        self.e2 = int(eva_size // np.e)
        self.running_stat = state_norm

        '''reward'''
        self.rewards = get_eva_reward(agent, self.env_list[:5], max_step, max_action, self.running_stat)
        self.reward_avg = np.average(self.rewards)
        self.reward_std = float(np.std(self.rewards))
        self.reward_target = target_reward
        self.reward_max = self.reward_avg

        self.record_epoch = list()  # record_epoch.append((epoch_reward, actor_loss, critic_loss, iter_num))
        self.record_eval = [(0, self.reward_avg, self.reward_std), ]  # [(epoch, reward_avg, reward_std), ]
        self.total_step = 0

        self.epoch = 0
        self.train_time = 0  # train_time
        self.train_time = time.time()  # train_time
        self.start_time = self.show_time = time.time()
        print("epoch|   reward   r_max    r_ave    r_std |  loss_A loss_C |step")

    def show_reward(self, epoch_rewards, iter_numbers, loss_a, loss_c):
        self.train_time += time.time() - self.train_time  # train_time
        self.epoch += len(epoch_rewards)

        if isinstance(epoch_rewards, float):
            epoch_rewards = (epoch_rewards,)
            iter_numbers = (iter_numbers,)
        for reward, iter_num in zip(epoch_rewards, iter_numbers):
            self.record_epoch.append((reward, loss_a, loss_c, iter_num))
            self.total_step += iter_num

        if time.time() - self.show_time > self.show_gap:
            self.rewards = get_eva_reward(self.agent, self.env_list[:self.e1], self.max_step, self.max_action,
                                          self.running_stat)
            self.reward_avg = np.average(self.rewards)
            self.reward_std = float(np.std(self.rewards))
            self.record_eval.append((len(self.record_epoch), self.reward_avg, self.reward_std))

            slice_reward = np.array(self.record_epoch[-self.smooth_kernel:])[:, 0]
            smooth_reward = np.average(slice_reward, axis=0)
            print("{:4} |{:8.2f} {:8.2f} {:8.2f} {:8.2f} |{:8.2f} {:6.2f} |{:.2e}".format(
                len(self.record_epoch),
                smooth_reward, self.reward_max, self.reward_avg, self.reward_std,
                loss_a, loss_c, self.total_step))

            self.show_time = time.time()  # reset show_time after get_eva_reward_batch !
        else:
            self.rewards = list()

    def check_reward(self, cwd, loss_a, loss_c):  # 2020-05-05
        is_solved = False
        if self.reward_avg >= self.reward_max:  # and len(self.rewards) > 1:  # 2020-04-30
            self.rewards.extend(get_eva_reward(self.agent, self.env_list[:self.e2], self.max_step, self.max_action,
                                               self.running_stat))
            self.reward_avg = np.average(self.rewards)

            if self.reward_avg >= self.reward_max:
                self.reward_max = self.reward_avg

                '''NOTICE! Recorder saves the agent with max reward automatically. '''
                self.agent.save_or_load_model(cwd, is_save=True)

                if self.reward_max >= self.reward_target:
                    res_env_len = len(self.env_list) - len(self.rewards)
                    self.rewards.extend(get_eva_reward(
                        self.agent, self.env_list[:res_env_len], self.max_step, self.max_action,
                        self.running_stat))
                    self.reward_avg = np.average(self.rewards)
                    self.reward_max = self.reward_avg

                    if self.reward_max >= self.reward_target:
                        print("########## Solved! ###########")
                        is_solved = True

            self.reward_std = float(np.std(self.rewards))
            self.record_eval[-1] = (len(self.record_epoch), self.reward_avg, self.reward_std)  # refresh
            print("{:4} |{:8} {:8.2f} {:8.2f} {:8.2f} |{:8.2f} {:6.2f} |{:.2e}".format(
                len(self.record_epoch),
                '', self.reward_max, self.reward_avg, self.reward_std,
                loss_a, loss_c, self.total_step, ))

        self.train_time = time.time()  # train_time
        return is_solved

    def print_and_save_npy(self, env_name, cwd):  # 2020-04-30
        iter_used = self.total_step  # int(sum(np.array(self.record_epoch)[:, -1]))
        time_used = int(time.time() - self.start_time)
        print('Used Time:', time_used)
        self.train_time = int(self.train_time)  # train_time
        print('TrainTime:', self.train_time)  # train_time

        print_str = "{}-{:.2f}AVE-{:.2f}STD-{}E-{}S-{}T".format(
            env_name, self.reward_max, self.reward_std, self.epoch, self.train_time, iter_used)  # train_time
        print(print_str)
        nod_path = '{}/{}.txt'.format(cwd, print_str)
        os.mknod(nod_path, ) if not os.path.exists(nod_path) else None

        np.save('%s/record_epoch.npy' % cwd, self.record_epoch)
        np.save('%s/record_eval.npy' % cwd, self.record_eval)
        print("Saved record_*.npy in:", cwd)

        return self.train_time


def get_eva_reward(agent, env_list, max_step, max_action, running_state=None):  # class Recorder 2020-01-11
    # this function is a bit complicated. I don't recommend you to change it.
    # max_action is None, when env is discrete action space

    act = agent.act
    act.eval()

    env_list_copy = env_list.copy()
    eva_size = len(env_list_copy)

    sum_rewards = [0.0, ] * eva_size
    states = [env.reset() for env in env_list_copy]

    reward_sums = list()
    for iter_num in range(max_step):
        if running_state:
            states = [running_state(state, update=False) for state in states]  # if state_norm:
        actions = agent.select_actions(states)

        next_states = list()
        done_list = list()
        if max_action:  # Continuous action space
            actions *= max_action
        for i in range(len(env_list_copy) - 1, -1, -1):
            next_state, reward, done, _ = env_list_copy[i].step(actions[i])

            next_states.insert(0, next_state)
            sum_rewards[i] += reward
            done_list.insert(0, done)
            if done:
                reward_sums.append(sum_rewards[i])
                del sum_rewards[i]
                del env_list_copy[i]
        states = next_states

        if len(env_list_copy) == 0:
            break
    else:
        reward_sums.extend(sum_rewards)
    act.train()

    return reward_sums


def get_episode_reward(env, act, max_step, max_action, ) -> float:
    reward_item = 0.0

    state = env.reset()
    for _ in range(max_step):
        s_tensor = torch.tensor((state,), dtype=torch.float32, requires_grad=False)
        a_tensor = act(s_tensor)
        action = a_tensor.detach_().numpy()[0]

        next_state, reward, done, _ = env.step(action * max_action)
        reward_item += reward

        if done:
            break
        state = next_state
    return reward_item


def get__buffer_reward_step(env, max_step, max_action, reward_scale, gamma, action_dim, is_discrete,
                            **_kwargs) -> (np.ndarray, list, list):
    buffer_list = list()

    reward_list = list()
    reward_item = 0.0

    step_list = list()
    step_item = 0

    state = env.reset()

    global_step = 0
    while global_step < max_step:
        action = rd.randint(action_dim) if is_discrete else rd.uniform(-1, 1, size=action_dim)
        next_state, reward, done, _ = env.step(action * max_action)
        reward_item += reward
        step_item += 1

        adjust_reward = reward * reward_scale
        mask = 0.0 if done else gamma
        buffer_list.append((adjust_reward, mask, state, action, next_state))

        if done:
            global_step += step_item

            reward_list.append(reward_item)
            reward_item = 0.0
            step_list.append(step_item)
            step_item = 0

            state = env.reset()
        else:
            state = next_state

    buffer_array = np.stack([np.hstack(buf_tuple) for buf_tuple in buffer_list])
    return buffer_array, reward_list, step_list


"""demo"""


def run__zoo(gpu_id, cwd='RL_Zoo'):
    import AgentZoo as Zoo
    args = Arguments()
    args.class_agent = Zoo.AgentDeepSAC
    assert args.class_agent in {
        Zoo.AgentDDPG, Zoo.AgentTD3, Zoo.ActorSAC, Zoo.AgentDeepSAC,
        Zoo.AgentBasicAC, Zoo.AgentSNAC, Zoo.AgentInterAC, Zoo.AgentInterSAC,
    }  # you can't run PPO here. goto run__ppo(). PPO need special hyper-parameters
    args.gpu_id = gpu_id
    ''' Compare with other algorithm, DDPG, A2C TRPO is unstable and low effective.
    
    I need DDPG as a tutorial so there are a DDPG implementation.
    DDPG will train for a long time in the following env.
    
    A2C introduce the advantage function and you can find this structure in PPO and SAC.
    TRPO's author use a surrogate object to simplify the KL penalty and get PPO.
    So I didn't provide A2C or TRPO implementation in my code.
    If many people want me to provide A2C or TRPO. I will.
    '''

    """args.env_name = "Pendulum-v0"
    It is a easy task. But it has not default target reward. 
    I had manually set as -200.0 in get_env_info(). Easy to reach -200 (-100 is harder)
    Its reward is in (-inf to 0.0].
    Its continuous action spaces is (-2, +2). Its action_dim == 1.
    """
    args.env_name = "Pendulum-v0"
    args.cwd = './{}/Pendulum_{}'.format(cwd, gpu_id)
    args.init_for_training()
    train_agent__off_policy(**vars(args))
    exit()

    args.env_name = "LunarLanderContinuous-v2"
    args.cwd = './{}/LunarLander_{}'.format(cwd, gpu_id)
    args.init_for_training()
    train_agent__off_policy(**vars(args))

    args.env_name = "BipedalWalker-v3"
    args.cwd = './{}/BipedalWalker_{}'.format(cwd, gpu_id)
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
    args = Arguments()
    args.class_agent = Zoo.AgentGAE
    assert args.class_agent in {Zoo.AgentPPO, Zoo.AgentGAE}
    '''PPO and GAE is online policy. 
    The memory in replay buffer will only be saved for one episode.
    
    TRPO's author use a surrogate object to simplify the KL penalty and get PPO.
    So I provide PPO instead of TRPO here.
    
    GAE is Generalization Advantage Estimate. 
    RL algorithm that use advantage function (such as A2C, PPO, SAC) can use this technique.
    AgentGAE is a PPO using GAE and output log_std of action by an actor network.
    '''

    args.gpu_id = gpu_id
    args.max_memo = 2 ** 12
    args.batch_size = 2 ** 9
    args.repeat_times = 2 ** 4
    args.net_dim = 2 ** 8
    args.gamma = 0.99

    args.env_name = "LunarLanderContinuous-v2"
    args.cwd = './{}/LL_{}'.format(cwd, gpu_id)
    args.init_for_training()
    train_agent__on_policy(**vars(args))

    args.env_name = "BipedalWalker-v3"
    args.cwd = './{}/BW_{}'.format(cwd, gpu_id)
    args.init_for_training()
    train_agent__on_policy(**vars(args))


def run__gae_discrete(gpu_id, cwd='RL_DiscreteGAE'):
    import AgentZoo as Zoo
    args = Arguments()
    args.class_agent = Zoo.AgentDiscreteGAE
    assert args.class_agent in {Zoo.AgentDiscreteGAE, }
    args.gpu_id = gpu_id
    '''DiscreteGAE is a modify PPO+GAE. It is an online policy too.
    The action vector can be a probability of discrete action.
    Although it is design by myself, it is so simple and 
    maybe other people had figured it out many years ago.
    '''

    args.max_memo = 2 ** 10
    args.batch_size = 2 ** 8
    args.repeat_times = 2 ** 4
    args.net_dim = 2 ** 7

    args.env_name = "CartPole-v0"
    args.cwd = './{}/CP_{}'.format(cwd, gpu_id)
    args.init_for_training()
    train_agent__on_policy(**vars(args))

    args.gpu_id = gpu_id
    args.max_memo = 2 ** 12
    args.batch_size = 2 ** 9
    args.repeat_times = 2 ** 4
    args.net_dim = 2 ** 8

    args.env_name = "LunarLander-v2"
    args.cwd = './{}/LL_{}'.format(cwd, gpu_id)
    args.init_for_training()
    train_agent__on_policy(**vars(args))


def run__dqn(gpu_id, cwd='RL_DQN'):
    import AgentZoo as Zoo
    args = Arguments()
    args.class_agent = Zoo.AgentDoubleDQN
    assert args.class_agent in {Zoo.AgentDQN, Zoo.AgentDoubleDQN}
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


"""demo plan to do multi-agent 2020-05-05"""


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
                rewards, steps = agent.update_replay_buffer(env, buffer, max_step, max_action, reward_scale, gamma)

            '''update network parameters by random sampling buffer for stochastic gradient descent'''
            loss_a, loss_c = agent.update_network_param(buffer, max_step, batch_size, update_gap)

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
    args = Arguments()
    args.class_agent = AgentSAC
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
    [process.join() for process in processes]
    [process.close() for process in processes]


'''demo plan to do multi-process mix CPU and GPU 2020-07-07'''


def mp__update_params(args, q_i_buf, q_o_buf, q_i_eva, q_o_eva):  # update network parameters using replay buffer
    class_agent = args.class_agent
    max_memo = args.max_memo
    net_dim = args.net_dim
    max_epoch = args.max_epoch
    max_step = args.max_step
    batch_size = args.batch_size
    repeat_times = args.repeat_times
    args.init_for_training()
    del args

    state_dim, action_dim = q_o_buf.get()  # q_o_buf 1.
    agent = class_agent(state_dim, action_dim, net_dim)

    from copy import deepcopy
    act_cpu = deepcopy(agent.act).to(torch.device("cpu"))
    act_cpu.eval()
    [setattr(param, 'requires_grad', False) for param in act_cpu.parameters()]
    q_i_buf.put(act_cpu)  # q_i_buf 1.
    # q_i_buf.put(act_cpu)  # q_i_buf 2. # warning
    q_i_eva.put(act_cpu)  # q_i_eva 1.

    buffer = BufferArrayGPU(max_memo, state_dim, action_dim)  # experiment replay buffer

    '''initial_exploration'''
    buffer_array, reward_list, step_list = q_o_buf.get()  # q_o_buf 2.
    reward_avg = np.average(reward_list)
    step_sum = sum(step_list)
    buffer.extend_memo(buffer_array)

    q_i_eva.put((act_cpu, reward_avg, step_sum, 0, 0))  # q_i_eva 1.

    for epoch in range(max_epoch):  # epoch is episode
        buffer_array, reward_list, step_list = q_o_buf.get()  # q_o_buf n.
        reward_avg = np.average(reward_list)
        step_sum = sum(step_list)
        buffer.extend_memo(buffer_array)

        buffer.init_before_sample()
        loss_a_avg, loss_c_avg = agent.update_parameters(buffer, max_step, batch_size, repeat_times)

        act_cpu.load_state_dict(agent.act.state_dict())
        q_i_buf.put(act_cpu)  # q_i_buf n.
        q_i_eva.put((act_cpu, reward_avg, step_sum, loss_a_avg, loss_c_avg))  # q_i_eva n.

        if q_o_eva.qsize() > 0:
            is_solved = q_o_eva.get()  # q_o_eva n.
            if is_solved:
                break
    # print('; quit: params')


def mp__update_buffer(args, q_i_buf, q_o_buf):  # update replay buffer by interacting with env
    env_name = args.env_name
    max_step = args.max_step
    reward_scale = args.reward_scale
    gamma = args.gamma
    del args

    torch.set_num_threads(4)

    env = gym.make(env_name)
    state_dim, action_dim, max_action, _, is_discrete = get_env_info(env, is_print=False)
    q_o_buf.put((state_dim, action_dim))  # q_o_buf 1.

    '''build evaluated only actor'''
    q_i_buf_get = q_i_buf.get()  # q_i_buf 1.
    act = q_i_buf_get  # act == act.to(device_cpu), requires_grad=False

    buffer_array, reward_list, step_list = get__buffer_reward_step(
        env, max_step, max_action, reward_scale, gamma, action_dim, is_discrete)

    q_o_buf.put((buffer_array, reward_list, step_list))  # q_o_buf 2.

    explore_noise = True
    state = env.reset()
    is_training = True
    while is_training:
        buffer_list = list()
        reward_list = list()
        reward_item = 0.0
        step_list = list()
        step_item = 0

        global_step = 0
        while global_step < max_step:
            '''select action'''
            s_tensor = torch.tensor((state,), dtype=torch.float32, requires_grad=False)
            a_tensor = act(s_tensor, explore_noise)
            action = a_tensor.detach_().numpy()[0]

            next_state, reward, done, _ = env.step(action * max_action)
            reward_item += reward
            step_item += 1

            adjust_reward = reward * reward_scale
            mask = 0.0 if done else gamma
            buffer_list.append((adjust_reward, mask, state, action, next_state))

            if done:
                global_step += step_item

                reward_list.append(reward_item)
                reward_item = 0.0
                step_list.append(step_item)
                step_item = 0

                state = env.reset()
            else:
                state = next_state

        buffer_array = np.stack([np.hstack(buf_tuple) for buf_tuple in buffer_list])
        q_o_buf.put((buffer_array, reward_list, step_list))  # q_o_buf n.

        try:
            q_i_buf_get = q_i_buf.get()  # q_i_buf n.
        except FileNotFoundError:
            is_training = False
        act = q_i_buf_get  # act == act.to(device_cpu), requires_grad=False
    # print('; quit: buffer')


def mp_evaluate_agent(args, q_i_eva, q_o_eva):  # evaluate agent and get its total reward of an episode
    max_step = args.max_step
    cwd = args.cwd
    eva_size = args.eva_size
    show_gap = args.show_gap
    env_name = args.env_name
    gpu_id = args.gpu_id
    del args

    torch.set_num_threads(4)

    '''recorder'''
    eva_r_max = -np.inf
    exp_r_avg = -np.inf
    total_step = 0
    loss_a_avg = 0
    loss_c_avg = 0
    recorder_exp = list()  # total_step, exp_r_avg, loss_a_avg, loss_c_avg
    recorder_eva = list()  # total_step, eva_r_avg, eva_r_std

    env = gym.make(env_name)
    state_dim, action_dim, max_action, target_reward, is_discrete = get_env_info(env, is_print=True)

    '''build evaluated only actor'''
    q_i_eva_get = q_i_eva.get()  # q_i_eva 1.
    act = q_i_eva_get  # q_i_eva_get == act.to(device_cpu), requires_grad=False

    print(f"{'GPU':3}  {'Step':>8}  {'MaxR':>8} |"
          f"{'avgR':>8}  {'stdR':>8} |"
          f"{'ExpR':>8}  {'LossA':>8}  {'LossC':>8}")

    is_solved = False
    start_time = time.time()
    print_time = time.time()

    used_time = None
    is_training = True
    while is_training:
        '''evaluate actor'''
        reward_list = [get_episode_reward(env, act, max_step, max_action, )
                       for _ in range(8)]
        eva_r_avg = np.average(reward_list)
        if eva_r_avg > eva_r_max:  # check 1
            reward_list.extend([get_episode_reward(env, act, max_step, max_action, )
                                for _ in range(eva_size - len(reward_list))])
            eva_r_avg = np.average(reward_list)
            if eva_r_avg > eva_r_max:  # check final
                eva_r_max = eva_r_avg

                act_save_path = f'{cwd}/actor.pth'
                torch.save(act.state_dict(), act_save_path)
                print(f"{gpu_id:<3}  {total_step:8.2e}  {eva_r_max:8.2f} |")

        eva_r_std = np.std(reward_list)
        recorder_eva.append((total_step, eva_r_avg, eva_r_std))

        if eva_r_max > target_reward:
            is_solved = True
            if used_time is None:
                used_time = int(time.time() - start_time)
                print(f'#### GPU:{gpu_id} solve  '
                      f'Time {used_time}  Step {total_step:8.2e}  '
                      f'avgR {eva_r_avg:8.2f}  stdR {eva_r_std:8.2f} ')

        q_o_eva.put(is_solved)  # q_o_eva n.

        if time.time() - print_time > show_gap:
            print_time = time.time()
            print(f"{gpu_id:<3}  {total_step:8.2e}  {eva_r_max:8.2f} |"
                  f"{eva_r_avg:8.2f}  {eva_r_std:8.2f} |"
                  f"{exp_r_avg:8.2f}  {loss_a_avg:8.2f}  {loss_c_avg:8.2f}")

        '''update actor'''
        while q_i_eva.qsize() == 0:  # wait until q_i_eva has item
            time.sleep(1)
        while q_i_eva.qsize():  # get the latest actor
            try:
                q_i_eva_get = q_i_eva.get()  # q_i_eva n.
            except FileNotFoundError:
                is_training = False
                break
            act, exp_r_avg, exp_s_sum, loss_a_avg, loss_c_avg = q_i_eva_get
            total_step += exp_s_sum
            recorder_exp.append((total_step, exp_r_avg, loss_a_avg, loss_c_avg))
    np.save('%s/record_explore.npy' % cwd, recorder_exp)
    np.save('%s/record_evaluate.npy' % cwd, recorder_eva)
    draw_plot_with_2npy(cwd, train_time=time.time() - start_time)
    # print('; quit: evaluate')


def run__mp(gpu_id=None, cwd='MP__beta'):
    import AgentZoo as Zoo
    args = Arguments()
    args.class_agent = Zoo.AgentDeepSAC
    args.gpu_id = gpu_id if gpu_id is not None else sys.argv[-1][-4]
    assert args.class_agent in {
        Zoo.AgentDDPG, Zoo.AgentTD3, Zoo.ActorSAC, Zoo.AgentDeepSAC,
        Zoo.AgentBasicAC, Zoo.AgentSNAC, Zoo.AgentInterAC, Zoo.AgentInterSAC,
    }

    import multiprocessing as mp
    q_i_buf = mp.Queue(maxsize=8)  # buffer I
    q_o_buf = mp.Queue(maxsize=8)  # buffer O
    q_i_eva = mp.Queue(maxsize=8)  # evaluate I
    q_o_eva = mp.Queue(maxsize=8)  # evaluate O

    def build_for_mp():
        process = [mp.Process(target=mp__update_params, args=(args, q_i_buf, q_o_buf, q_i_eva, q_o_eva)),
                   mp.Process(target=mp__update_buffer, args=(args, q_i_buf, q_o_buf,)),
                   mp.Process(target=mp_evaluate_agent, args=(args, q_i_eva, q_o_eva)), ]
        [p.start() for p in process]
        [p.join() for p in process]
        [p.close() for p in process]

    args.env_name = "LunarLanderContinuous-v2"
    args.cwd = f'./{cwd}/{args.env_name}_{gpu_id}'
    build_for_mp()

    args.env_name = "BipedalWalker-v3"
    args.cwd = f'./{cwd}/{args.env_name}_{gpu_id}'
    build_for_mp()

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)
    args.env_name = "AntBulletEnv-v0"
    args.cwd = f'./{cwd}/{args.env_name}_{gpu_id}'
    args.max_epoch = 2 ** 13
    args.max_memo = 2 ** 20
    args.max_step = 2 ** 10
    args.net_dim = 2 ** 8
    args.batch_size = 2 ** 9
    args.reward_scale = 2 ** -2
    args.is_remove = True
    args.eva_size = 2 ** 5  # for Recorder
    args.show_gap = 2 ** 8  # for Recorder
    build_for_mp()


if __name__ == '__main__':
    run__zoo(gpu_id=0, cwd='AC_SAC')
    # run__ppo(gpu_id=1, cwd='AC_PPO')

    # run__multi_process(run__zoo, gpu_tuple=(0, 1, 2, 3), cwd='AC_ZooMP')
    # run__multi_process(run__ppo, gpu_tuple=(2, 3), cwd='AC_PPO')
    # run__multi_workers(gpu_tuple=(2, 3), root_cwd='AC_SAC_MP')

    # '''Discrete action space'''
    # run__dqn(gpu_id=sys.argv[-1][-4], cwd='RL_DQN')

    # '''multi worker'''
    # run__multi_workers(gpu_tuple=(2, 3), root_cwd='AC_SAC_MP')

    print('Finish:', sys.argv[-1])
