import os
import sys
import time

import gym
import torch
import numpy as np

from AgentZoo import initial_exploration
from AgentZoo import BufferArray, BufferArrayGPU, BufferTupleOnline

"""ZenYiYan ZenJiaHao, GitHub: YonV1943 ElegantRL (Pytorch model-free DRL)
I consider that Reinforcement Learning Algorithms before 2020 have not consciousness
They feel more like a Cerebellum (Little Brain) for Machines.
In my opinion, before 2020, the policy gradient algorithm agent didn't learn s policy.
Actually, they "learn game feel" or "get a soft touch". In Chinese "shou3 gan3 手感". 
Learn more about policy gradient algorithms in:
https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html
"""


class Arguments:  # default working setting and hyper-parameters
    def __init__(self, rl_agent=None, env_name=None, gpu_id=None):
        self.rl_agent = rl_agent
        self.env_name = env_name
        self.gpu_id = gpu_id
        self.cwd = None  # init cwd in def init_for_training()

        '''Arguments for training'''
        self.net_dim = 2 ** 8  # the network width
        self.max_memo = 2 ** 17  # memories capacity (memories: replay buffer)
        self.max_step = 2 ** 10  # max steps in one training episode
        self.batch_size = 2 ** 7  # num of transitions sampled from replay buffer.
        self.repeat_times = 2 ** 0  # repeatedly update network to keep critic's loss small
        self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256
        self.gamma = 0.99  # discount factor of future rewards

        '''Arguments for evaluate'''
        self.break_step = 2 ** 17  # break training after 'total_step > break_step'
        self.if_break_early = True  # break training after 'eval_reward > target reward'
        self.if_remove_history = True  # remove the cwd folder? (True, False, None:ask me)
        self.show_gap = 2 ** 8  # show the Reward and Loss of actor and critic per show_gap seconds
        self.eval_times1 = 2 ** 3  # evaluation times if 'eval_reward > old_max_reward'
        self.eval_times2 = 2 ** 4  # evaluation times if 'eval_reward > target_reward'
        self.random_seed = 1943  # Github: YonV 1943

    def init_for_training(self, cpu_threads=6):
        assert self.rl_agent is not None
        assert self.env_name is not None
        self.gpu_id = sys.argv[-1][-4] if self.gpu_id is None else str(self.gpu_id)
        self.cwd = f'./{self.rl_agent.__name__}/{self.env_name}_{self.gpu_id}'

        print('| GPU: {} | CWD: {}'.format(self.gpu_id, self.cwd))
        whether_remove_history(self.cwd, self.if_remove_history)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        torch.set_num_threads(cpu_threads)
        torch.set_default_dtype(torch.float32)
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        # env.seed(random_seed)  # sometimes env has random seed.

    # def update_args(self, new_dict):  # useless
    #     for key, value in new_dict.items():
    #         setattr(self, key, value)


"""train in single processing"""


def train_agent(  # 2020-11-11
        rl_agent, env_name, gpu_id, cwd,
        net_dim, max_memo, max_step, batch_size, repeat_times, reward_scale, gamma,
        break_step, if_break_early, show_gap, eval_times1, eval_times2, **_kwargs):  # 2020-09-18
    env, state_dim, action_dim, target_reward, if_discrete = build_env(env_name, if_print=False)

    '''init: agent, buffer, recorder'''
    recorder = Recorder(eval_size1=eval_times1, eval_size2=eval_times2)
    agent = rl_agent(state_dim, action_dim, net_dim)  # training agent
    agent.state = env.reset()

    if bool(rl_agent.__name__ in {'AgentPPO', }):
        buffer = BufferTupleOnline(max_memo)
    elif bool(rl_agent.__name__ in {'AgentModPPO', 'AgentInterPPO'}):
        buffer = BufferArray(max_memo + max_step, state_dim, action_dim, if_ppo=True)
    else:
        buffer = BufferArray(max_memo, state_dim, action_dim=1 if if_discrete else action_dim, if_ppo=False)
        with torch.no_grad():  # update replay buffer
            rewards, steps = initial_exploration(env, buffer, max_step, if_discrete, reward_scale, gamma, action_dim)
        recorder.update__record_explore(steps, rewards, loss_a=0, loss_c=0)

        '''pre training and hard update before training loop'''
        buffer.update_pointer_before_sample()
        agent.update_policy(buffer, max_step, batch_size, repeat_times)
        if 'act_target' in dir(agent):
            agent.act_target.load_state_dict(agent.act.state_dict())

    '''loop'''
    if_train = True
    while if_train:
        '''update replay buffer by interact with environment'''
        with torch.no_grad():  # speed up running
            rewards, steps = agent.update_buffer(env, buffer, max_step, reward_scale, gamma)

        '''update network parameters by random sampling buffer for gradient descent'''
        loss_a, loss_c = agent.update_policy(buffer, max_step, batch_size, repeat_times)

        '''saves the agent with max reward'''
        recorder.update__record_explore(steps, rewards, loss_a, loss_c)

        if_save = recorder.update__record_evaluate(env, agent.act, max_step, agent.device, if_discrete)
        recorder.save_act(cwd, agent.act, gpu_id) if if_save else None

        with torch.no_grad():  # for saving the GPU buffer
            if_solve = recorder.check__if_solved(target_reward, gpu_id, show_gap, cwd)

        '''break loop rules'''
        if_train = not ((if_break_early and if_solve)
                        or recorder.total_step > break_step
                        or os.path.exists(f'{cwd}/stop'))

    recorder.save_npy__draw_plot(cwd)
    print(f'SavedDir: {cwd}\n'
          f'UsedTime: {time.time() - recorder.start_time:.0f}')

    buffer.print_state_norm(env.neg_state_avg, env.div_state_std)


"""train in multi processing"""


def mp__update_params(args, q_i_eva, q_o_eva):  # 2020-11-11 update network parameters using replay buffer
    rl_agent = args.rl_agent
    max_memo = args.max_memo
    net_dim = args.net_dim
    max_step = args.max_step
    max_total_step = args.break_step
    batch_size = args.batch_size
    repeat_times = args.repeat_times
    cwd = args.cwd
    env_name = args.env_name
    reward_scale = args.reward_scale
    if_stop = args.if_break_early
    gamma = args.gamma
    del args

    env, state_dim, action_dim, target_reward, if_discrete = build_env(env_name, if_print=False)

    '''build agent'''
    agent = rl_agent(state_dim, action_dim, net_dim)  # training agent
    agent.state = env.reset()

    '''send agent to q_i_eva'''
    from copy import deepcopy
    act_cpu = deepcopy(agent.act).to(torch.device("cpu"))
    act_cpu.eval()
    [setattr(param, 'requires_grad', False) for param in act_cpu.parameters()]
    q_i_eva.put(act_cpu)  # q_i_eva 1.

    '''build replay buffer, init: total_step, reward_avg'''
    total_step = 0
    if bool(rl_agent.__name__ in {'AgentPPO', }):
        buffer = BufferTupleOnline(max_memo)
        with torch.no_grad():
            reward_avg = get_total_return(env, act_cpu, max_step, torch.device("cpu"), if_discrete)
    elif bool(rl_agent.__name__ in {'AgentModPPO', 'AgentInterPPO'}):
        buffer = BufferArrayGPU(max_memo + max_step, state_dim, action_dim, if_ppo=True)  # experiment replay buffer
        with torch.no_grad():
            reward_avg = get_total_return(env, act_cpu, max_step, torch.device("cpu"), if_discrete)
    else:
        buffer = BufferArrayGPU(max_memo, state_dim, action_dim=1 if if_discrete else action_dim, if_ppo=False)
        '''initial exploration'''
        with torch.no_grad():  # update replay buffer
            rewards, steps = initial_exploration(env, buffer, max_step, if_discrete, reward_scale, gamma, action_dim)
        reward_avg = np.average(rewards)
        step_sum = sum(steps)

        '''pre training and hard update before training loop'''
        buffer.update_pointer_before_sample()
        agent.update_policy(buffer, max_step, batch_size, repeat_times)
        if 'act_target' in dir(agent):
            agent.act_target.load_state_dict(agent.act.state_dict())

        q_i_eva.put((act_cpu, reward_avg, step_sum, 0, 0))  # q_i_eva n.
        total_step += step_sum

    '''training loop'''
    if_train = True
    if_solve = False
    while if_train:
        '''update replay buffer by interact with environment'''
        with torch.no_grad():  # speed up running
            rewards, steps = agent.update_buffer(env, buffer, max_step, reward_scale, gamma)
        reward_avg = np.average(rewards) if len(rewards) else reward_avg
        step_sum = sum(steps)
        total_step += step_sum

        '''update network parameters by random sampling buffer for gradient descent'''
        buffer.update_pointer_before_sample()
        loss_a_avg, loss_c_avg = agent.update_policy(buffer, max_step, batch_size, repeat_times)

        '''saves the agent with max reward'''
        act_cpu.load_state_dict(agent.act.state_dict())
        q_i_eva.put((act_cpu, reward_avg, step_sum, loss_a_avg, loss_c_avg))  # q_i_eva n.

        if q_o_eva.qsize() > 0:
            if_solve = q_o_eva.get()  # q_o_eva n.

        '''break loop rules'''
        if_train = not ((if_stop and if_solve)
                        or total_step > max_total_step
                        or os.path.exists(f'{cwd}/stop'))

    env, state_dim, action_dim, target_reward, if_discrete = build_env(env_name, if_print=False)
    buffer.print_state_norm(env.neg_state_avg, env.div_state_std)

    q_i_eva.put('stop')
    while q_i_eva.qsize() > 0:
        time.sleep(1)
    time.sleep(4)
    # print('; quit: params')


def mp_evaluate_agent(args, q_i_eva, q_o_eva):  # evaluate agent and get its total reward of an episode
    env_name = args.env_name
    cwd = args.cwd
    gpu_id = args.gpu_id
    max_step = args.max_memo
    show_gap = args.show_gap
    eval_size1 = args.eval_times1
    eval_size2 = args.eval_times2
    del args

    env, state_dim, action_dim, target_reward, if_discrete = build_env(env_name, if_print=True)

    '''build evaluated only actor'''
    act = q_i_eva.get()  # q_i_eva 1, act == act.to(device_cpu), requires_grad=False

    torch.set_num_threads(4)
    device = torch.device('cpu')
    recorder = Recorder(eval_size1, eval_size2)
    recorder.update__record_evaluate(env, act, max_step, device, if_discrete)

    is_training = True
    with torch.no_grad():  # for saving the GPU buffer
        while is_training:
            is_saved = recorder.update__record_evaluate(env, act, max_step, device, if_discrete)
            recorder.save_act(cwd, act, gpu_id) if is_saved else None

            is_solved = recorder.check__if_solved(target_reward, gpu_id, show_gap, cwd)
            q_o_eva.put(is_solved)  # q_o_eva n.

            '''update actor'''
            while q_i_eva.qsize() == 0:  # wait until q_i_eva has item
                time.sleep(1)
            while q_i_eva.qsize():  # get the latest actor
                q_i_eva_get = q_i_eva.get()  # q_i_eva n.
                if q_i_eva_get == 'stop':
                    is_training = False
                    break
                act, exp_r_avg, exp_s_sum, loss_a_avg, loss_c_avg = q_i_eva_get
                recorder.update__record_explore(exp_s_sum, exp_r_avg, loss_a_avg, loss_c_avg)

    recorder.save_npy__draw_plot(cwd)
    print(f'SavedDir: {cwd}\n'
          f'UsedTime: {time.time() - recorder.start_time:.0f}')

    while q_o_eva.qsize() > 0:
        q_o_eva.get()
    while q_i_eva.qsize() > 0:
        q_i_eva.get()
    # print('; quit: evaluate')


def train_agent_mp(args):  # 2020-1111
    import multiprocessing as mp
    q_i_eva = mp.Queue(maxsize=16)  # evaluate I
    q_o_eva = mp.Queue(maxsize=16)  # evaluate O
    process = [mp.Process(target=mp__update_params, args=(args, q_i_eva, q_o_eva)),
               mp.Process(target=mp_evaluate_agent, args=(args, q_i_eva, q_o_eva)), ]
    [p.start() for p in process]
    [p.join() for p in process]
    print('\n')


"""utils"""


class Recorder:  # 2020-10-12
    def __init__(self, eval_size1=3, eval_size2=9):
        self.eva_r_max = -np.inf
        self.total_step = 0
        self.record_exp = [(0., 0., 0., 0.), ]  # total_step, exp_r_avg, loss_a_avg, loss_c_avg
        self.record_eva = [(0., 0., 0.), ]  # total_step, eva_r_avg, eva_r_std
        self.is_solved = False

        '''constant'''
        self.eva_size1 = eval_size1
        self.eva_size2 = eval_size2

        '''print_reward'''
        self.used_time = None
        self.start_time = time.time()
        self.print_time = time.time()

        print(f"{'ID':>2}  {'Step':>8}  {'MaxR':>8} |"
              f"{'avgR':>8}  {'stdR':>8} |"
              f"{'ExpR':>8}  {'LossA':>8}  {'LossC':>8}")

    def update__record_evaluate(self, env, act, max_step, device, if_discrete):
        is_saved = False
        reward_list = [get_total_return(env, act, max_step, device, if_discrete)
                       for _ in range(self.eva_size1)]

        eva_r_avg = np.average(reward_list)
        if eva_r_avg > self.eva_r_max:  # check 1
            reward_list.extend([get_total_return(env, act, max_step, device, if_discrete)
                                for _ in range(self.eva_size2 - self.eva_size1)])
            eva_r_avg = np.average(reward_list)
            if eva_r_avg > self.eva_r_max:  # check final
                self.eva_r_max = eva_r_avg
                is_saved = True

        eva_r_std = float(np.std(reward_list))
        self.record_eva.append((self.total_step, eva_r_avg, eva_r_std))

        return is_saved

    def update__record_explore(self, exp_s_sum, exp_r_avg, loss_a, loss_c):  # 2020-10-10
        if isinstance(exp_s_sum, int):
            exp_s_sum = (exp_s_sum,)
            exp_r_avg = (exp_r_avg,)
        if loss_c > 64:
            print(f"| ToT: Critic Loss explosion {loss_c:.1f}. Select a smaller reward_scale.")
        for s, r in zip(exp_s_sum, exp_r_avg):
            self.total_step += s
            self.record_exp.append((self.total_step, r, loss_a, loss_c))

    def save_act(self, cwd, act, agent_id):
        act_save_path = f'{cwd}/actor.pth'
        torch.save(act.state_dict(), act_save_path)
        print(f"{agent_id:<2}  {self.total_step:8.2e}  {self.eva_r_max:8.2f} |")

    def check__if_solved(self, target_reward, agent_id, show_gap, cwd):
        if self.eva_r_max > target_reward:
            self.is_solved = True
            if self.used_time is None:
                self.used_time = int(time.time() - self.start_time)
                print(f"{'ID':>2}  {'Step':>8}  {'TargetR':>8} |"
                      f"{'avgR':>8}  {'stdR':>8} |"
                      f"{'ExpR':>8}  {'UsedTime':>8}  ########")

                total_step, eva_r_avg, eva_r_std = self.record_eva[-1]
                total_step, exp_r_avg, loss_a_avg, loss_c_avg = self.record_exp[-1]
                print(f"{agent_id:<2}  {total_step:8.2e}  {target_reward:8.2f} |"
                      f"{eva_r_avg:8.2f}  {eva_r_std:8.2f} |"
                      f"{exp_r_avg:8.2f}  {self.used_time:>8}  ########")

        if time.time() - self.print_time > show_gap:
            self.print_time = time.time()

            total_step, eva_r_avg, eva_r_std = self.record_eva[-1]
            total_step, exp_r_avg, loss_a_avg, loss_c_avg = self.record_exp[-1]
            print(f"{agent_id:<2}  {total_step:8.2e}  {self.eva_r_max:8.2f} |"
                  f"{eva_r_avg:8.2f}  {eva_r_std:8.2f} |"
                  f"{exp_r_avg:8.2f}  {loss_a_avg:8.2f}  {loss_c_avg:8.2f}")
            self.save_npy__draw_plot(cwd)

        return self.is_solved

    def save_npy__draw_plot(self, cwd):  # 2020-10-10
        np.save('%s/record_explore.npy' % cwd, self.record_exp)
        np.save('%s/record_evaluate.npy' % cwd, self.record_eva)
        draw_plot_with_2npy(cwd, train_time=time.time() - self.start_time, max_reward=self.eva_r_max)

    def demo(self):
        pass


def get_total_return(env, act, max_step, device, if_discrete) -> float:
    # better to 'with torch.no_grad()'
    reward_item = 0.0

    state = env.reset()
    for _ in range(max_step):
        s_tensor = torch.tensor((state,), dtype=torch.float32, device=device)

        a_tensor = act(s_tensor).argmax(dim=1) if if_discrete else act(s_tensor)
        action = a_tensor.cpu().data.numpy()[0]

        next_state, reward, done, _ = env.step(action)
        reward_item += reward

        if done:
            break
        state = next_state
    return reward_item


def get_total_returns(agent, env_list, max_step) -> list:  # class Recorder 2020-01-11
    """ Notice:
    this function is a bit complicated. I don't recommend you or me to change it.
    """
    agent.act.eval()

    env_list_copy = env_list.copy()
    eva_size = len(env_list_copy)

    sum_rewards = [0.0, ] * eva_size
    states = [env.reset() for env in env_list_copy]

    reward_sums = list()
    for iter_num in range(max_step):
        actions = agent.select_actions(states)

        next_states = list()
        done_list = list()
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
    agent.act.train()

    return reward_sums


def draw_plot_with_2npy(cwd, train_time, max_reward):  # 2020-09-18
    record_explore = np.load('%s/record_explore.npy' % cwd)  # item: (total_step, exp_r_avg, loss_a_avg, loss_c_avg)
    record_evaluate = np.load('%s/record_evaluate.npy' % cwd)  # item:(total_step, eva_r_avg, eva_r_std)

    if len(record_evaluate.shape) == 1:
        record_evaluate = np.array([[0., 0., 0.]])
    if len(record_explore.shape) == 1:
        record_explore = np.array([[0., 0., 0., 0.]])

    train_time = int(train_time)
    total_step = int(record_evaluate[-1][0])
    save_title = f"plot_step_time_maxR_{int(total_step)}_{int(train_time)}_{max_reward:.3f}"
    save_path = f"{cwd}/{save_title}.jpg"

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
    ax21_color = 'lightcoral'  # same color as ax11 (expR)
    ax21_label = 'lossA'
    exp_loss_a = record_explore[:, 2]
    ax21.set_ylabel(ax21_label, color=ax21_color)
    ax21.plot(exp_step, exp_loss_a, label=ax21_label, color=ax21_color)  # negative loss A
    ax21.tick_params(axis='y', labelcolor=ax21_color)

    ax22 = axs[1].twinx()
    ax22_color = 'darkcyan'
    ax22_label = 'lossC'
    exp_loss_c = record_explore[:, 3]
    ax22.set_ylabel(ax22_label, color=ax22_color)
    ax22.fill_between(exp_step, exp_loss_c, facecolor=ax22_color, alpha=0.2, )
    ax22.tick_params(axis='y', labelcolor=ax22_color)

    # remove prev figure
    prev_save_names = [name for name in os.listdir(cwd) if name[:9] == save_title[:9]]
    os.remove(f'{cwd}/{prev_save_names[0]}') if len(prev_save_names) > 0 else None

    plt.savefig(save_path)
    # plt.pause(4)
    # plt.show()
    plt.close()


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


"""utils: Env"""


def get_env_info(env, is_print=True) -> tuple:  # 2020-10-10
    env_name = env.unwrapped.spec.id

    state_shape = env.observation_space.shape
    if len(state_shape) == 1:
        state_dim = state_shape[0]
    else:
        state_dim = state_shape

    try:
        if_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        if if_discrete:  # discrete
            action_dim = env.action_space.n
            action_max = int(1)
        elif isinstance(env.action_space, gym.spaces.Box):  # make sure it is continuous action space
            action_dim = env.action_space.shape[0]
            action_max = float(env.action_space.high[0])

            action_high = np.array(env.action_space.high)
            action_high[:] = action_max
            action_low = np.array(env.action_space.low)
            action_low[:] = -action_max
            if any(action_high != env.action_space.high) and any(action_low != env.action_space.low):
                print(f'| Warning: '
                      f'act_high {env.action_space.high}  '
                      f'act_low  {env.action_space.low}')
        else:
            raise AttributeError
    except AttributeError:
        print("| Could you assign these value manually? \n"
              "| I need: state_dim, action_dim, action_max, target_reward, if_discrete")
        raise AttributeError

    target_reward = env.spec.reward_threshold
    if target_reward is None:
        assert target_reward is not None

    if is_print:
        print("| env_name: {}, action space: {}".format(env_name, 'if_discrete' if if_discrete else 'Continuous'))
        print("| state_dim: {}, action_dim: {}, action_max: {}, target_reward: {}".format(
            state_dim, action_dim, action_max, target_reward))
    return state_dim, action_dim, action_max, target_reward, if_discrete


def decorate_env(env, action_max=1, state_avg=None, state_std=None, data_type=np.float32):
    if state_avg is None:
        neg_state_avg = 0
        div_state_std = 1
    else:
        state_avg = state_avg.astype(data_type)
        state_std = state_std.astype(data_type)

        neg_state_avg = -state_avg
        div_state_std = 1 / (state_std + 1e-4)

    env.neg_state_avg = neg_state_avg  # for def print_norm() AgentZoo.py
    env.div_state_std = div_state_std  # for def print_norm() AgentZoo.py

    '''decorator_step'''
    if state_avg is not None:
        def decorator_step(env_step):
            def new_env_step(action):
                state, reward, done, info = env_step(action * action_max)
                return (state.astype(data_type) + neg_state_avg) * div_state_std, reward, done, info

            return new_env_step
    elif action_max is not None:
        def decorator_step(env_step):
            def new_env_step(action):
                state, reward, done, info = env_step(action * action_max)
                return state.astype(data_type), reward, done, info

            return new_env_step
    else:  # action_max is None:
        def decorator_step(env_step):
            def new_env_step(action):
                state, reward, done, info = env_step(action * action_max)
                return state.astype(data_type), reward, done, info

            return new_env_step
    env.step = decorator_step(env.step)

    '''decorator_reset'''
    if state_avg is not None:
        def decorator_reset(env_reset):
            def new_env_reset():
                state = env_reset()
                return (state.astype(data_type) + neg_state_avg) * div_state_std

            return new_env_reset
    else:
        def decorator_reset(env_reset):
            def new_env_reset():
                state = env_reset()
                return state.astype(data_type)

            return new_env_reset
    env.reset = decorator_reset(env.reset)

    return env


def build_env(env_name, if_print=True, if_norm=True):
    assert env_name is not None

    '''UserWarning: WARN: Box bound precision lowered by casting to float32
    https://stackoverflow.com/questions/60149105/
    userwarning-warn-box-bound-precision-lowered-by-casting-to-float32
    '''
    gym.logger.set_level(40)  # non-essential

    # env = gym.make(env_name)
    # state_dim, action_dim, action_max, target_reward, if_discrete = get_env_info(env, if_print)
    '''env compatibility'''  # some env need to adjust.
    if env_name == 'Pendulum-v0':
        env = gym.make(env_name)
        env.spec.reward_threshold = -200.0  # target_reward
        state_dim, action_dim, action_max, target_reward, if_discrete = get_env_info(env, if_print)
    elif env_name == 'CarRacing-v0':
        # | state_dim: (96, 96, 3), action_dim: 3, action_max: 1.0, target_reward: 900
        frame_num = 3
        env = gym.make(env_name)
        env = fix_car_racing_env(env, frame_num=frame_num, action_num=frame_num)
        state_dim, action_dim, action_max, target_reward, if_discrete = get_env_info(env, if_print)
        assert len(state_dim)
        state_dim = (frame_num, state_dim[0], state_dim[1])  # two consecutive frame (96, 96)
        # from AgentPixel import CarRacingEnv
        # env = CarRacingEnv(img_stack=4, action_repeat=4)
        # state_dim, action_dim, action_max = (4, 96, 96), 3, 1.0
        # target_reward, if_discrete = 900, False
    elif env_name == 'MultiWalker':
        from multiwalker_base import MultiWalkerEnv, multi_to_single_walker_decorator
        env = MultiWalkerEnv()
        env = multi_to_single_walker_decorator(env)

        state_dim = sum([box.shape[0] for box in env.observation_space])
        action_dim = sum([box.shape[0] for box in env.action_space])
        action_max = 1.0
        target_reward = 50
        if_discrete = False
    else:
        env = gym.make(env_name)
        state_dim, action_dim, action_max, target_reward, if_discrete = get_env_info(env, if_print)

    '''Not necessary: check data type of state (but necessary for WinOS)'''
    '''Not necessary: env normalization'''  # adjust action into [-1, +1] using action_max is necessary.
    avg = None
    std = None
    if if_norm:
        '''norm no need'''
        # if env_name == 'Pendulum-v0':
        #     state_mean = np.array([-0.00968592 -0.00118888 -0.00304381])
        #     std = np.array([0.53825575 0.54198545 0.8671749 ])

        '''norm could be'''
        if env_name == 'LunarLanderContinuous-v2':
            avg = np.array([1.65470898e-02, -1.29684399e-01, 4.26883133e-03, -3.42124557e-02,
                            -7.39076972e-03, -7.67103031e-04, 1.12640885e+00, 1.12409466e+00])
            std = np.array([0.15094465, 0.29366297, 0.23490797, 0.25931464, 0.21603736,
                            0.25886878, 0.277233, 0.27771219])
        elif env_name == "BipedalWalker-v3":
            avg = np.array([1.42211734e-01, -2.74547996e-03, 1.65104509e-01, -1.33418152e-02,
                            -2.43243194e-01, -1.73886203e-02, 4.24114229e-02, -6.57800099e-02,
                            4.53460692e-01, 6.08022244e-01, -8.64884810e-04, -2.08789053e-01,
                            -2.92092949e-02, 5.04791247e-01, 3.33571745e-01, 3.37325723e-01,
                            3.49106580e-01, 3.70363115e-01, 4.04074671e-01, 4.55838055e-01,
                            5.36685407e-01, 6.70771701e-01, 8.80356865e-01, 9.97987386e-01])
            std = np.array([0.84419678, 0.06317835, 0.16532085, 0.09356959, 0.486594,
                            0.55477525, 0.44076614, 0.85030824, 0.29159821, 0.48093035,
                            0.50323634, 0.48110776, 0.69684234, 0.29161077, 0.06962932,
                            0.0705558, 0.07322677, 0.07793258, 0.08624322, 0.09846895,
                            0.11752805, 0.14116005, 0.13839757, 0.07760469])
        elif env_name == 'AntBulletEnv-v0':
            avg = np.array([
                0.4838, -0.047, 0.3500, 1.3028, -0.249, 0.0000, -0.281, 0.0573,
                -0.261, 0.0000, 0.0424, 0.0000, 0.2278, 0.0000, -0.072, 0.0000,
                0.0000, 0.0000, -0.175, 0.0000, -0.319, 0.0000, 0.1387, 0.0000,
                0.1949, 0.0000, -0.136, -0.060])
            std = np.array([
                0.0601, 0.2267, 0.0838, 0.2680, 0.1161, 0.0757, 0.1495, 0.1235,
                0.6733, 0.4326, 0.6723, 0.3422, 0.7444, 0.5129, 0.6561, 0.2732,
                0.6805, 0.4793, 0.5637, 0.2586, 0.5928, 0.3876, 0.6005, 0.2369,
                0.4858, 0.4227, 0.4428, 0.4831])
        # elif env_name == 'MinitaurBulletEnv-v0':
        #     # avg = np.array([0.90172989, 1.54730119, 1.24560906, 1.97365306, 1.9413892,
        #     #                 1.03866835, 1.69646277, 1.18655352, -0.45842347, 0.17845232,
        #     #                 0.38784456, 0.58572877, 0.91414561, -0.45410697, 0.7591031,
        #     #                 -0.07008998, 3.43842258, 0.61032482, 0.86689961, -0.33910894,
        #     #                 0.47030415, 4.5623528, -2.39108079, 3.03559422, -0.36328256,
        #     #                 -0.20753499, -0.47758384, 0.86756409])
        #     # std = np.array([0.34192648, 0.51169916, 0.39370621, 0.55568461, 0.46910769,
        #     #                 0.28387504, 0.51807949, 0.37723445, 13.16686185, 17.51240024,
        #     #                 14.80264211, 16.60461412, 15.72930229, 11.38926597, 15.40598346,
        #     #                 13.03124941, 2.47718145, 2.55088804, 2.35964651, 2.51025567,
        #     #                 2.66379017, 2.37224904, 2.55892521, 2.41716885, 0.07529733,
        #     #                 0.05903034, 0.1314812, 0.0221248])

        # elif env_name == "BipedalWalkerHardcore-v3":
        #     avg = np.array([-3.6378160e-02, -2.5788052e-03, 3.4413573e-01, -8.4189959e-03,
        #                     -9.1864385e-02, 3.2804706e-04, -6.4693891e-02, -9.8939031e-02,
        #                     3.5180664e-01, 6.8103075e-01, 2.2930240e-03, -4.5893672e-01,
        #                     -7.6047562e-02, 4.6414185e-01, 3.9363885e-01, 3.9603019e-01,
        #                     4.0758255e-01, 4.3053803e-01, 4.6186063e-01, 5.0293463e-01,
        #                     5.7822973e-01, 6.9820738e-01, 8.9829963e-01, 9.8080903e-01])
        #     std = np.array([0.5771428, 0.05302362, 0.18906464, 0.10137994, 0.41284004,
        #                     0.68852615, 0.43710527, 0.87153363, 0.3210142, 0.36864948,
        #                     0.6926624, 0.38297284, 0.76805115, 0.33138904, 0.09618598,
        #                     0.09843876, 0.10035378, 0.11045089, 0.11910835, 0.13400233,
        #                     0.15718603, 0.17106676, 0.14363566, 0.10100251])

    env = decorate_env(env, action_max, avg, std, data_type=np.float32)
    return env, state_dim, action_dim, target_reward, if_discrete


"""utils: Fix Env CarRacing-v0 - Box2D"""


def fix_car_racing_env(env, frame_num=3, action_num=3):  # 2020-11-11
    env.old_step = env.step
    """
    comment 'car_racing.py' line 233-234: print('Track generation ...
    comment 'car_racing.py' line 308-309: print("retry to generate track ...
    """

    def rgb2gray(rgb):
        # # rgb image -> gray [0, 1]
        # gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114]).astype(np.float32)
        # if norm:
        #     # normalize
        #     gray = gray / 128. - 1.
        # return gray
        state = rgb[:, :, 1]  # show green
        state[86:, 24:36] = rgb[86:, 24:36, 2]  # show red
        state[86:, 72:] = rgb[86:, 72:, 0]  # show blue
        state = (state - 128) / 128.
        return state

    env.state_stack = None
    env.avg_reward = 0
    env.state = None

    def decorator_step(env_step):
        def new_env_step(action):
            action = action.copy()
            action[1:] = (action[1:] + 1) / 2  # fix action_space.low

            reward_sum = 0
            done = info = None
            try:
                for _ in range(action_num):
                    env.state, reward, done, info = env_step(action)

                    if done:  # don't penalize "die state"
                        reward += 100
                    if env.state.mean() > 192:  # 185.0:  # penalize when outside of road
                        reward -= 0.05

                    env.avg_reward = env.avg_reward * 0.95 + reward * 0.05
                    if env.avg_reward <= -0.1:
                        done = True

                    reward_sum += reward

                    if done:
                        break
            except Exception as error:
                print(f"| CarRacing-v0 Error 'stack underflow'? {error}")
                reward_sum = 0
                done = True
            env.state_stack.pop(0)
            env.state_stack.append(rgb2gray(env.state))

            return np.array(env.state_stack).flatten(), reward_sum, done, info

        return new_env_step

    env.step = decorator_step(env.step)

    def decorator_reset(env_reset):
        def new_env_reset():
            state = rgb2gray(env_reset())
            env.state_stack = [state, ] * frame_num
            return np.array(env.state_stack).flatten()

        return new_env_reset

    env.reset = decorator_reset(env.reset)
    return env


def test_car_racing():
    env_name = 'CarRacing-v0'
    env, state_dim, action_dim, target_reward, is_discrete = build_env(env_name, if_print=True)

    _state = env.reset()
    import cv2
    action = np.array((0, 1.0, -1.0))
    for i in range(321):
        # action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        # env.render
        show = state.reshape(state_dim)
        show = ((show[0] + 1.0) * 128).astype(np.uint8)
        cv2.imshow('', show)
        cv2.waitKey(1)
        if done:
            break
        # env.render()


def run__car_racing(gpu_id=None, random_seed=0):
    print('pixel-level state')
    import AgentZoo as Zoo
    rl_agent = (
        Zoo.AgentModPPO,
        Zoo.AgentInterPPO
    )[1]  # choose DRl algorithm.

    args = Arguments(rl_agent=rl_agent, gpu_id=gpu_id)
    args.if_break_early = True
    args.eval_times2 = 2
    args.eval_times2 = 3  # CarRacing Env is so slow. The GPU-util is low while training CarRacing.

    args.env_name = "CarRacing-v0"
    """
    ID      Step   TargetR |    avgR      stdR |    ExpR  UsedTime  ########
    3   1.93e+05    900.00 |  939.68    113.69 |  227.59      9112  ########
    3   2.37e+05    900.00 |  957.16     36.60 |  365.65     12832  ########
    0   4.49e+05    900.00 |  910.85     97.42 |  634.27     22515  ########
    2   5.14e+05    900.00 |  953.76    104.13 |  752.05     24101  ########
    
    ID      Step      MaxR |    avgR      stdR |    ExpR     LossA     LossC
    2   1.82e+06   1001.72 | 1001.72     31.75 |  852.32     -1.17      1.03
    """

    args.random_seed = 1943 + random_seed
    args.break_step = int(5e5 * 4)  # (2e5) 5e5, used time 25000s
    args.reward_scale = 2 ** -2  # (-1) 80 ~ 900 (1001)
    args.max_memo = 2 ** 11
    args.batch_size = 2 ** 7
    args.repeat_times = 2 ** 4
    args.net_dim = 2 ** 7
    args.max_step = 2 ** 10
    args.show_gap = 2 ** 8  # for Recorder
    args.init_for_training()
    train_agent_mp(args)  # train_agent(**vars(args))


"""demo"""


def run__demo():
    import AgentZoo as Zoo

    # args = Arguments(rl_agent=Zoo.AgentDoubleDQN, env_name="LunarLander-v2", gpu_id=0)
    args = Arguments(rl_agent=Zoo.AgentD3QN, env_name="LunarLander-v2", gpu_id=0)
    args.break_step = int(1e5 * 8)  # used time 600s
    args.net_dim = 2 ** 7
    args.init_for_training()
    train_agent_mp(args)  # train_agent(**vars(args))
    # exit()

    # args = Arguments(rl_agent=Zoo.AgentModSAC, env_name="LunarLanderContinuous-v2", gpu_id=0)
    args = Arguments(rl_agent=Zoo.AgentModSAC, env_name="LunarLanderContinuous-v2", gpu_id=0)
    args.break_step = int(5e5 * 8)  # used time 1500s
    args.net_dim = 2 ** 7
    args.init_for_training()
    train_agent_mp(args)  # train_agent(**vars(args))
    exit()


def run__off_policy():
    import AgentZoo as Zoo
    args = Arguments(rl_agent=None, env_name=None, gpu_id=None)
    args.rl_agent = [
        Zoo.AgentDDPG,  # 2014. simple, simple, slow, unstable
        Zoo.AgentBaseAC,  # 2014+ modify DDPG, faster, more stable
        Zoo.AgentTD3,  # 2018. twin critics, delay target update
        Zoo.AgentSAC,  # 2018. twin critics, maximum entropy, auto alpha, fix log_prob
        Zoo.AgentModSAC,  # 2018+ modify SAC, faster, more stable
        Zoo.AgentInterAC,  # 2019. Integrated AC(DPG)
        Zoo.AgentInterSAC,  # 2020. Integrated SAC(SPG)
    ][4]  # I suggest to use ModSAC (Modify SAC)
    # On-policy PPO is not here 'run__off_policy()'. See PPO in 'run__on_policy()'.

    args.if_break_early = True  # break training if reach the target reward (total return of an episode)
    args.if_remove_history = True  # delete the historical directory

    args.env_name = "Pendulum-v0"  # It is easy to reach target score -200.0 (-100 is harder)
    args.break_step = int(1e4 * 8)  # 1e4 means the average total training step of InterSAC to reach target_reward
    args.reward_scale = 2 ** -2  # (-1800) -1000 ~ -200 (-50)
    args.init_for_training()
    train_agent(**vars(args))  # Train agent using single process. Recommend run on PC.
    # train_agent_mp(args)  # Train using multi process. Recommend run on Server. Mix CPU(eval) GPU(train)
    exit()

    args.env_name = "LunarLanderContinuous-v2"
    args.break_step = int(5e5 * 8)  # (2e4) 5e5, used time 1500s
    args.reward_scale = 2 ** -3  # (-800) -200 ~ 200 (302)
    args.init_for_training()
    train_agent_mp(args)  # train_agent(**vars(args))
    exit()

    args.env_name = "BipedalWalker-v3"
    args.break_step = int(2e5 * 8)  # (1e5) 2e5, used time 3500s
    args.reward_scale = 2 ** -1  # (-200) -140 ~ 300 (341)
    args.init_for_training()
    train_agent_mp(args)  # train_agent(**vars(args))
    exit()

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)
    args.env_name = "ReacherBulletEnv-v0"
    args.break_step = int(5e4 * 8)  # (4e4) 5e4
    args.reward_scale = 2 ** 0  # (-37) 0 ~ 18 (29)
    args.init_for_training()
    train_agent_mp(args)  # train_agent(**vars(args))
    exit()

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)
    args.env_name = "AntBulletEnv-v0"
    args.break_step = int(1e6 * 8)  # (5e5) 1e6, UsedTime: (15,000s) 30,000s
    args.reward_scale = 2 ** -3  # (-50) 0 ~ 2500 (3340)
    args.batch_size = 2 ** 8
    args.max_memo = 2 ** 20
    args.eva_size = 2 ** 3  # for Recorder
    args.show_gap = 2 ** 8  # for Recorder
    args.init_for_training()
    train_agent_mp(args)  # train_agent(**vars(args))
    exit()

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)
    args.env_name = "MinitaurBulletEnv-v0"
    args.break_step = int(4e6 * 4)  # (2e6) 4e6
    args.reward_scale = 2 ** 5  # (-2) 0 ~ 16 (20)
    args.batch_size = (2 ** 8)
    args.net_dim = int(2 ** 8)
    args.max_step = 2 ** 11
    args.max_memo = 2 ** 20
    args.eval_times2 = 3  # for Recorder
    args.eval_times2 = 9  # for Recorder
    args.show_gap = 2 ** 9  # for Recorder
    args.init_for_training()
    train_agent_mp(args)  # train_agent(**vars(args))
    exit()

    args.env_name = "BipedalWalkerHardcore-v3"  # 2020-08-24 plan
    args.reward_scale = 2 ** 0  # (-200) -150 ~ 300 (334)
    args.break_step = int(4e6 * 8)  # (2e6) 4e6
    args.net_dim = int(2 ** 8)  # int(2 ** 8.5) #
    args.max_memo = int(2 ** 21)
    args.batch_size = int(2 ** 8)
    args.eval_times2 = 2 ** 5  # for Recorder
    args.show_gap = 2 ** 8  # for Recorder
    args.init_for_training()
    train_agent_mp(args)  # train_offline_policy(**vars(args))
    exit()


def run__on_policy():
    import AgentZoo as Zoo
    args = Arguments(rl_agent=None, env_name=None, gpu_id=None)
    args.rl_agent = [
        Zoo.AgentPPO,  # 2018. PPO2 + GAE, slow but quite stable, especially in high-dim
        Zoo.AgentModPPO,  # 2019. Reliable Lambda
        Zoo.AgentInterPPO,  # 2020. Integrated Network, useful in pixel-level task (state2D)
    ][1]

    args.net_dim = 2 ** 8
    args.max_memo = 2 ** 12
    args.batch_size = 2 ** 9
    args.repeat_times = 2 ** 4
    args.reward_scale = 2 ** 0  # unimportant hyper-parameter in PPO which do normalization on Q value
    args.gamma = 0.99  # important hyper-parameter, related to episode steps

    args.env_name = "Pendulum-v0"  # It is easy to reach target score -200.0 (-100 is harder)
    args.break_step = int(8e4 * 8)  # 5e5 means the average total training step of ModPPO to reach target_reward
    args.reward_scale = 2 ** 0  # (-1800) -1000 ~ -200 (-50), UsedTime:  (100s) 200s
    args.gamma = 0.9  # important hyper-parameter, related to episode steps
    args.init_for_training()
    train_agent_mp(args)  # train_agent(**vars(args))
    exit()

    args.env_name = "LunarLanderContinuous-v2"
    args.break_step = int(3e5 * 8)  # (2e5) 3e5 , used time: (400s) 600s
    args.reward_scale = 2 ** 0  # (-800) -200 ~ 200 (301)
    args.gamma = 0.99  # important hyper-parameter, related to episode steps
    args.init_for_training()
    train_agent_mp(args)  # train_agent(**vars(args))
    # exit()

    args.env_name = "BipedalWalker-v3"
    args.break_step = int(8e5 * 8)  # (6e5) 8e5 (6e6), UsedTimes: (800s) 1500s (8000s)
    args.reward_scale = 2 ** 0  # (-150) -90 ~ 300 (324)
    args.gamma = 0.95  # important hyper-parameter, related to episode steps
    args.init_for_training()
    train_agent_mp(args)  # train_agent(**vars(args))
    exit()

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)
    args.env_name = "ReacherBulletEnv-v0"
    args.break_step = int(2e6 * 8)  # (1e6) 2e6 (4e6), UsedTimes: 2000s (6000s)
    args.reward_scale = 2 ** 0  # (-15) 0 ~ 18 (25)
    args.gamma = 0.95  # important hyper-parameter, related to episode steps
    args.init_for_training()
    train_agent_mp(args)  # train_agent(**vars(args))
    exit()

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)
    args.env_name = "AntBulletEnv-v0"
    args.break_step = int(5e6 * 8)  # (1e6) 5e6 UsedTime: 25697s
    args.reward_scale = 2 ** -3  #
    args.gamma = 0.99  # important hyper-parameter, related to episode steps
    args.net_dim = 2 ** 9
    args.init_for_training()
    train_agent(**vars(args))
    exit()

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)
    args.env_name = "MinitaurBulletEnv-v0"  # PPO is the best, I don't know why.
    args.break_step = int(5e5 * 8)  # (PPO 3e5) 5e5
    args.reward_scale = 2 ** 4  # (-2) 0 ~ 16 (PPO 34)
    args.gamma = 0.99  # important hyper-parameter, related to episode steps
    args.net_dim = 2 ** 8
    args.max_memo = 2 ** 11
    args.batch_size = 2 ** 9
    args.repeat_times = 2 ** 4
    args.init_for_training()
    train_agent(**vars(args))
    exit()


def run__discrete_action():
    import AgentZoo as Zoo
    args = Arguments(rl_agent=None, env_name=None, gpu_id=None)
    args.rl_agent = [
        Zoo.AgentDQN,  # 2014.
        Zoo.AgentDoubleDQN,  # 2016. stable
        Zoo.AgentDuelingDQN,  # 2016. stable and fast
        Zoo.AgentD3QN,  # 2016+ Dueling + Double DQN (Not a creative work)
    ][3]  # I suggest to use D3QN

    args.env_name = "CartPole-v0"
    args.break_step = int(1e4 * 8)  # (3e5) 1e4, used time 20s
    args.reward_scale = 2 ** 0  # 0 ~ 200
    args.net_dim = 2 ** 6
    args.init_for_training()
    train_agent_mp(args)  # train_agent(**vars(args))
    exit()

    args.env_name = "LunarLander-v2"
    args.break_step = int(1e5 * 8)  # (5e4) 1e5 (3e5), used time (355s) 600s (2000s)
    args.reward_scale = 2 ** 0  # (-1000) -150 ~ 200 (285)
    args.init_for_training()
    train_agent_mp(args)  # train_agent(**vars(args))
    exit()


if __name__ == '__main__':
    run__demo()
    print('Finish:', sys.argv[-1])
