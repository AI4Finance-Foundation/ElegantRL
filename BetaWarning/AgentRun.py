import os
import sys
import time

import torch
import numpy as np
import numpy.random as rd

"""ZenYiYan, GitHub: YonV1943 ElegantRL (Pytorch 3 files model-free DRL Library)
I consider that Reinforcement Learning Algorithms before 2020 have not consciousness
They feel more like a Cerebellum (Little Brain) for Machines.
In my opinion, before 2020, the policy gradient algorithm agent didn't learn s policy.
Actually, they "learn game feel" or "get a soft touch". In Chinese "shǒu gǎn 手感". 
Learn more about policy gradient algorithms in:
https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html
"""


class Arguments:  # default working setting and hyper-parameters
    def __init__(self, rl_agent=None, env=None, gpu_id=None):
        self.rl_agent = rl_agent
        self.gpu_id = gpu_id
        self.cwd = None  # init cwd in def init_for_training()
        self.env = env

        '''Arguments for training'''
        self.net_dim = 2 ** 8  # the network width
        self.max_memo = 2 ** 17  # memories capacity (memories: replay buffer)
        self.max_step = 2 ** 10  # max steps in one training episode
        self.batch_size = 2 ** 7  # num of transitions sampled from replay buffer.
        self.repeat_times = 2 ** 0  # repeatedly update network to keep critic's loss small
        self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256
        self.gamma = 0.99  # discount factor of future rewards
        self.rollout_num = 2  # the number of rollout workers (larger is not always faster)

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
        assert self.env is not None
        if not hasattr(self.env, 'env_name'):
            raise RuntimeError(
                '\n| init_for_training() WARNING: AttributeError. '
                '\n| What is env.env_name? use env = build_env(env) to decorate env'
            )

        self.gpu_id = sys.argv[-1][-4] if self.gpu_id is None else str(self.gpu_id)
        self.cwd = f'./{self.rl_agent.__name__}/{self.env.env_name}_{self.gpu_id}'

        print(f'| GPU id: {self.gpu_id}, cwd: {self.cwd}')  # 2021-01-01
        _whether_remove_history(self.cwd, self.if_remove_history)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        torch.set_num_threads(cpu_threads)
        torch.set_default_dtype(torch.float32)
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        # env.seed(random_seed)  # sometimes env has random seed.

    # def update_args(self, new_dict):  # useless
    #     for key, value in new_dict.items():
    #         setattr(self, key, value)


def _whether_remove_history(cwd, is_remove=None):  # 2020-03-04
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


"""train agent in single or multi processing"""


def train_agent(args):  # 2020-12-12
    rl_agent = args.rl_agent
    env = args.env
    gpu_id = args.gpu_id
    cwd = args.cwd

    '''Arguments for training'''
    gamma = args.gamma
    net_dim = args.net_dim
    max_memo = args.max_memo
    max_step = args.max_step
    batch_size = args.batch_size
    repeat_times = args.repeat_times
    reward_scale = args.reward_scale

    '''Arguments for evaluate'''
    break_step = args.break_step
    if_break_early = args.if_break_early
    show_gap = args.show_gap
    eval_times1 = args.eval_times1
    eval_times2 = args.eval_times2
    del args  # In order to show these hyper-parameters clearly, I put them above.

    '''init: env'''
    state_dim = env.state_dim
    action_dim = env.action_dim
    if_discrete = env.if_discrete
    target_reward = env.target_reward
    from copy import deepcopy  # built-in library of Python
    env = deepcopy(env)
    env_eval = deepcopy(env)  # 2020-12-12

    '''init: agent, buffer, recorder'''
    recorder = Recorder(eval_times1, eval_times2)
    agent = rl_agent(state_dim, action_dim, net_dim)  # training agent
    agent.state = env.reset()

    if bool(rl_agent.__name__ in {'AgentPPO', 'AgentInterPPO'}):
        buffer = BufferArray(max_memo + max_step, state_dim, action_dim, if_ppo=True)
    else:
        buffer = BufferArray(max_memo, state_dim, action_dim=1 if if_discrete else action_dim, if_ppo=False)

        with torch.no_grad():  # update replay buffer
            rewards, steps = explore_before_train(env, buffer, max_step, if_discrete, reward_scale, gamma, action_dim)
        recorder.update__record_explore(steps, rewards, loss_a=0, loss_c=0)

        '''pre training and hard update before training loop'''
        buffer.update__now_len__before_sample()
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

        if_save = recorder.update__record_evaluate(env_eval, agent.act, max_step, agent.device, if_discrete)
        recorder.save_act(cwd, agent.act, gpu_id) if if_save else None

        with torch.no_grad():  # for saving the GPU buffer
            if_solve = recorder.check__if_solved(target_reward, gpu_id, show_gap, cwd)

        '''break loop rules'''
        if_train = not ((if_break_early and if_solve)
                        or recorder.total_step > break_step
                        or os.path.exists(f'{cwd}/stop'))

    recorder.save_npy__draw_plot(cwd)

    new_cwd = cwd[:-2] + f'_{recorder.eva_r_max:.2f}' + cwd[-2:]
    if not os.path.exists(new_cwd):  # 2020-12-12
        os.rename(cwd, new_cwd)
        cwd = new_cwd
    print(f'SavedDir: {cwd}\n'
          f'UsedTime: {time.time() - recorder.start_time:.0f}')

    buffer.print_state_norm(env.neg_state_avg if hasattr(env, 'neg_state_avg') else None,
                            env.div_state_std if hasattr(env, 'div_state_std') else None)  # 2020-12-12


def train_agent_mp(args):  # 2020-12-12
    import multiprocessing as mp
    q_i_eva = mp.Queue(maxsize=16)  # evaluate I
    q_o_eva = mp.Queue(maxsize=16)  # evaluate O
    process = list()

    act_workers = args.rollout_num
    qs_i_exp = [mp.Queue(maxsize=16) for _ in range(act_workers)]
    qs_o_exp = [mp.Queue(maxsize=16) for _ in range(act_workers)]
    for i in range(act_workers):
        process.append(mp.Process(target=mp__explore_a_env, args=(args, qs_i_exp[i], qs_o_exp[i], i)))

    process.extend([
        mp.Process(target=mp_evaluate_agent, args=(args, q_i_eva, q_o_eva)),
        mp.Process(target=mp__update_params, args=(args, q_i_eva, q_o_eva, qs_i_exp, qs_o_exp)),
    ])

    [p.start() for p in process]
    [p.join() for p in process]
    print('\n')


def mp__update_params(args, q_i_eva, q_o_eva, qs_i_exp, qs_o_exp):  # 2020-12-22
    rl_agent = args.rl_agent
    max_memo = args.max_memo
    net_dim = args.net_dim
    max_step = args.max_step
    max_total_step = args.break_step
    batch_size = args.batch_size
    repeat_times = args.repeat_times
    cwd = args.cwd
    env = args.env
    reward_scale = args.reward_scale
    if_stop = args.if_break_early
    gamma = args.gamma
    del args

    workers_num = len(qs_i_exp)

    '''init: env'''
    state_dim = env.state_dim
    action_dim = env.action_dim
    if_discrete = env.if_discrete
    # target_reward = env.target_reward

    '''build agent and act_cpu'''
    agent = rl_agent(state_dim, action_dim, net_dim)  # training agent
    agent.state = env.reset()
    from copy import deepcopy  # built-in library of Python
    act_cpu = deepcopy(agent.act).to(torch.device("cpu"))
    act_cpu.eval()
    [setattr(param, 'requires_grad', False) for param in act_cpu.parameters()]

    '''build replay buffer, init: total_step, reward_avg'''
    reward_avg = None
    total_step = 0
    if bool(rl_agent.__name__ in {'AgentPPO', 'AgentModPPO', 'AgentInterPPO'}):
        buffer = BufferArrayGPU(max_memo + max_step * workers_num, state_dim, action_dim, if_ppo=True)
        exp_step = max_memo // workers_num
    else:
        buffer = BufferArrayGPU(max_memo, state_dim, action_dim=1 if if_discrete else action_dim, if_ppo=False)
        exp_step = max_step // workers_num

        '''initial exploration'''
        with torch.no_grad():  # update replay buffer
            rewards, steps = explore_before_train(env, buffer, max_step, if_discrete, reward_scale, gamma, action_dim)
        reward_avg = np.average(rewards)
        step_sum = sum(steps)

        '''pre training and hard update before training loop'''
        buffer.update__now_len__before_sample()
        agent.update_policy(buffer, max_step, batch_size, repeat_times)
        if 'act_target' in dir(agent):
            agent.act_target.load_state_dict(agent.act.state_dict())

        # q_i_eva.put((act_cpu, reward_avg, step_sum, 0, 0))  # q_i_eva n.
        total_step += step_sum
    if reward_avg is None:
        # reward_avg = _get_episode_return(env, act_cpu, max_step, torch.device("cpu"), if_discrete)
        reward_avg = get_episode_return(env, agent.act, max_step, agent.device, if_discrete)

    q_i_eva.put(act_cpu)  # q_i_eva 1.
    for q_i_exp in qs_i_exp:
        q_i_exp.put((agent.act, exp_step))  # q_i_exp 1.

    '''training loop'''
    if_train = True
    if_solve = False
    while if_train:
        '''update replay buffer by interact with environment'''
        rewards = list()
        steps = list()
        for i, q_i_exp in enumerate(qs_i_exp):
            q_i_exp.put(agent.act)  # q_i_exp n.
            # q_i_exp.put(act_cpu)  # env_cpu--act_cpu
        for i, q_o_exp in enumerate(qs_o_exp):
            _memo_array, _rewards, _steps = q_o_exp.get()  # q_o_exp n.
            buffer.extend_memo(_memo_array)
            rewards.extend(_rewards)
            steps.extend(_steps)
        # with torch.no_grad():  # speed up running
        #     rewards, steps = agent.update_buffer(env, buffer, max_step, reward_scale, gamma)

        reward_avg = np.average(rewards) if len(rewards) else reward_avg
        step_sum = sum(steps)
        total_step += step_sum

        '''update network parameters by random sampling buffer for gradient descent'''
        buffer.update__now_len__before_sample()
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

    buffer.print_state_norm(env.neg_state_avg if hasattr(env, 'neg_state_avg') else None,
                            env.div_state_std if hasattr(env, 'div_state_std') else None)  # 2020-12-12

    for queue in [q_i_eva, ] + list(qs_i_exp):  # quit orderly and safely
        queue.put('stop')
        while queue.qsize() > 0:
            time.sleep(1)
    time.sleep(4)
    # print('; quit: params')


def mp__explore_a_env(args, q_i_exp, q_o_exp, act_id):
    env = args.env
    rl_agent = args.rl_agent
    net_dim = args.net_dim
    reward_scale = args.reward_scale
    gamma = args.gamma

    torch.manual_seed(args.random_seed + act_id)
    np.random.seed(args.random_seed + act_id)
    del args

    '''init: env'''
    state_dim = env.state_dim
    action_dim = env.action_dim
    if_discrete = env.if_discrete
    # target_reward = env.target_reward

    '''build agent'''
    agent = rl_agent(state_dim, action_dim, net_dim)  # training agent
    agent.state = env.reset()
    # agent.device = torch.device('cpu')  # env_cpu--act_cpu a little faster than env_cpu--act_gpu, but high cpu-util

    '''build replay buffer, init: total_step, reward_avg'''
    act, exp_step = q_i_exp.get()  # q_i_exp 1.  # plan to make it elegant: max_memo, max_step, exp_step

    if bool(rl_agent.__name__ in {'AgentPPO', 'AgentModPPO', 'AgentInterPPO'}):
        buffer = BufferArray(exp_step * 2, state_dim, action_dim, if_ppo=True)
    else:
        buffer = BufferArray(exp_step, state_dim, action_dim=1 if if_discrete else action_dim, if_ppo=False)

    with torch.no_grad():  # speed up running
        while True:
            buffer.empty_memories__before_explore()
            q_i_exp_get = q_i_exp.get()  # q_i_exp n.
            if q_i_exp_get == 'stop':
                break
            agent.act = q_i_exp_get  # q_i_exp n.
            rewards, steps = agent.update_buffer(env, buffer, exp_step, reward_scale, gamma)

            buffer.update__now_len__before_sample()
            q_o_exp.put((buffer.memories[:buffer.now_len], rewards, steps))  # q_o_exp n.

    for queue in [q_o_exp, q_i_exp]:  # quit orderly and safely
        while queue.qsize() > 0:
            queue.get()
    # print('; quit: explore')


def mp__explore_pipe_env(args, pipe, _act_id):
    env = args.env
    reward_scale = args.reward_scale
    gamma = args.gamma

    next_state = env.reset()
    pipe.send(next_state)
    while True:
        action = pipe.recv()
        next_state, reward, done, _ = env.step(action)

        reward_mask = np.array((reward * reward_scale, 0.0 if done else gamma), dtype=np.float32)
        if done:
            next_state = env.reset()

        pipe.send((reward_mask, next_state))


def mp_evaluate_agent(args, q_i_eva, q_o_eva):  # 2020-12-12
    env = args.env
    cwd = args.cwd
    gpu_id = args.gpu_id
    max_step = args.max_memo
    show_gap = args.show_gap
    eval_size1 = args.eval_times1
    eval_size2 = args.eval_times2
    del args

    '''init: env'''
    # state_dim = env.state_dim
    # action_dim = env.action_dim
    if_discrete = env.if_discrete
    target_reward = env.target_reward

    '''build evaluated only actor'''
    act = q_i_eva.get()  # q_i_eva 1, act == act.to(device_cpu), requires_grad=False

    torch.set_num_threads(4)
    device = torch.device('cpu')
    recorder = Recorder(eval_size1, eval_size2)
    recorder.update__record_evaluate(env, act, max_step, device, if_discrete)

    if_train = True
    with torch.no_grad():  # for saving the GPU buffer
        while if_train:
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
                    if_train = False
                    break  # it should break 'while q_i_eva.qsize():' and 'while if_train:'
                act, exp_r_avg, exp_s_sum, loss_a_avg, loss_c_avg = q_i_eva_get
                recorder.update__record_explore(exp_s_sum, exp_r_avg, loss_a_avg, loss_c_avg)

    recorder.save_npy__draw_plot(cwd)

    new_cwd = cwd[:-2] + f'_{recorder.eva_r_max:.2f}' + cwd[-2:]
    if not os.path.exists(new_cwd):  # 2020-12-12
        os.rename(cwd, new_cwd)
        cwd = new_cwd
    print(f'SavedDir: {cwd}\n'
          f'UsedTime: {time.time() - recorder.start_time:.0f}')

    for queue in [q_o_eva, q_i_eva]:  # quit orderly and safely
        while queue.qsize() > 0:
            queue.get()
    # print('; quit: evaluate')


def explore_before_train(env, buffer, max_step, if_discrete, reward_scale, gamma, action_dim):
    state = env.reset()

    rewards = list()
    episode_return = 0.0  # 2020-12-12 episode_return
    steps = list()
    step = 0

    if if_discrete:
        def random_action__discrete():
            return rd.randint(action_dim)

        get_random_action = random_action__discrete
    else:
        def random_action__continuous():
            return rd.uniform(-1, 1, size=action_dim)

        get_random_action = random_action__continuous

    global_step = 0
    while global_step < max_step or len(rewards) == 0:  # warning 2020-10-10?
        # action = np.tanh(rd.normal(0, 0.25, size=action_dim))  # zero-mean gauss exploration
        action = get_random_action()
        next_state, reward, done, _ = env.step(action * if_discrete)
        episode_return += reward
        step += 1

        adjust_reward = reward * reward_scale
        mask = 0.0 if done else gamma
        buffer.append_memo((adjust_reward, mask, state, action, next_state))

        state = next_state
        if done:
            episode_return = env.episode_return if hasattr(env, 'episode_return') else episode_return

            rewards.append(episode_return)
            steps.append(step)
            global_step += step

            state = env.reset()  # reset the environment
            episode_return = 0.0
            step = 1

    buffer.update__now_len__before_sample()
    return rewards, steps


"""utils"""


class Recorder:  # 2020-10-12
    def __init__(self, eval_size1=3, eval_size2=9):
        self.eva_r_max = -np.inf
        self.total_step = 0
        self.record_exp = [(0., -np.inf, 0., 0.), ]  # total_step, exp_r_avg, loss_a_avg, loss_c_avg
        self.record_eva = [(0., -np.inf, 0.), ]  # total_step, eva_r_avg, eva_r_std
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
        if self.total_step == self.record_eva[-1][0]:
            return None  # plan to be more elegant

        is_saved = False
        reward_list = [get_episode_return(env, act, max_step, device, if_discrete)
                       for _ in range(self.eva_size1)]

        eva_r_avg = np.average(reward_list)
        if eva_r_avg > self.eva_r_max:  # check 1
            reward_list.extend([get_episode_return(env, act, max_step, device, if_discrete)
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

    def save_npy__draw_plot(self, cwd):  # 2020-12-12
        if len(self.record_exp) == 0 or len(self.record_eva) == 0:
            print(f"| save_npy__draw_plot() WARNNING: len(self.record_exp) == {len(self.record_exp)}")
            print(f"| save_npy__draw_plot() WARNNING: len(self.record_eva) == {len(self.record_eva)}")
            return None

        np.save('%s/record_exp.npy' % cwd, self.record_exp)
        np.save('%s/record_eva.npy' % cwd, self.record_eva)

        # draw_plot_with_2npy(cwd, train_time=time.time() - self.start_time, max_reward=self.eva_r_max)
        train_time = time.time() - self.start_time
        max_reward = self.eva_r_max

        # record_exp = np.load('%s/record_exp.npy' % cwd, allow_pickle=True)  # 2020-12-11 allow_pickle
        # # record_exp.append((total_step, exp_r_avg, loss_a_avg, loss_c_avg))
        # record_eva = np.load('%s/record_eva.npy' % cwd, allow_pickle=True)
        # # record_eva.append((total_step, eva_r_avg, eva_r_std))
        record_exp = np.array(self.record_exp[1:], dtype=np.float32)
        record_eva = np.array(self.record_eva[1:], dtype=np.float32)  # 2020-12-12 Compatibility

        train_time = int(train_time)
        total_step = int(record_eva[-1][0])
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
        exp_step = record_exp[:, 0]
        exp_reward = record_exp[:, 1]
        ax11.plot(exp_step, exp_reward, label=ax11_label, color=ax11_color)

        ax12 = axs[0]
        ax12_color = 'lightcoral'
        ax12_label = 'Epoch R'
        eva_step = record_eva[:, 0]
        r_avg = record_eva[:, 1]
        r_std = record_eva[:, 2]
        ax12.plot(eva_step, r_avg, label=ax12_label, color=ax12_color)
        ax12.fill_between(eva_step, r_avg - r_std, r_avg + r_std, facecolor=ax12_color, alpha=0.3, )

        ax21 = axs[1]
        ax21_color = 'lightcoral'  # same color as ax11 (expR)
        ax21_label = 'lossA'
        exp_loss_a = record_exp[:, 2]
        ax21.set_ylabel(ax21_label, color=ax21_color)
        ax21.plot(exp_step, exp_loss_a, label=ax21_label, color=ax21_color)  # negative loss A
        ax21.tick_params(axis='y', labelcolor=ax21_color)

        ax22 = axs[1].twinx()
        ax22_color = 'darkcyan'
        ax22_label = 'lossC'
        exp_loss_c = record_exp[:, 3]
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

    def demo(self):
        pass


def get_episode_return(env, act, max_step, device, if_discrete) -> float:  # 2020-12-21
    # faster to run 'with torch.no_grad()'
    episode_return = 0.0  # sum of rewards in an episode

    # Compatibility for ElegantRL 2020-12-21
    max_step = env.max_step if hasattr(env, 'max_step') else max_step

    state = env.reset()
    for _ in range(max_step):
        s_tensor = torch.tensor((state,), dtype=torch.float32, device=device)

        a_tensor = act(s_tensor).argmax(dim=1) if if_discrete else act(s_tensor)
        action = a_tensor.cpu().data.numpy()[0]

        next_state, reward, done, _ = env.step(action)
        episode_return += reward

        if done:
            break
        state = next_state

    # Compatibility for ElegantRL 2020-12-21
    episode_return = env.episode_return if hasattr(env, 'episode_return') else episode_return
    return episode_return


''' backup of get_total_returns
def get_total_returns(agent, env_list, max_step) -> list:  # class Recorder 2020-01-11
    # this function is a bit complicated. I don't recommend you or me to change it.

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
'''

"""Environment for training agent"""


def decorate_env(env, if_print=True, if_norm=True):  # important function # 2020-12-12
    assert env is not None

    if all([hasattr(env, attr) for attr in (
            'env_name', 'state_dim', 'action_dim', 'target_reward', 'if_discrete')]):
        pass  # not elegant enough
    else:
        (env_name, state_dim, action_dim, action_max, if_discrete, target_reward
         ) = _get_gym_env_information(env)

        # do normalization on state (if_norm=True) through decorate_env()
        avg, std = _get_gym_env_state_norm(env_name, if_norm)

        # convert state to float32 (for WinOS)
        env = _get_decorate_env(env, action_max, avg, std, data_type=np.float32)

        setattr(env, 'env_name', env_name)
        setattr(env, 'state_dim', state_dim)
        setattr(env, 'action_dim', action_dim)
        setattr(env, 'if_discrete', if_discrete)
        setattr(env, 'target_reward', target_reward)

    if if_print:
        print(f"| env_name:  {env.env_name}, action is {'Discrete' if env.if_discrete else 'Continuous'}\n"
              f"| state_dim: {env.state_dim}, action_dim: {env.action_dim}, target_reward: {env.target_reward}")
    return env


def _get_gym_env_information(env) -> (str, int, int, float, bool, float):
    import gym  # gym of OpenAI is not necessary for ElegantRL (even RL)
    gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'

    if not isinstance(env, gym.Env):
        raise RuntimeError(
            '| It is not a standard gym env. Could tell me the values of the following?\n'
            '| state_dim, action_dim, target_reward, if_discrete = (int, int, float, bool)'
        )

    '''env_name and special rule'''
    env_name = env.unwrapped.spec.id
    if env_name == 'Pendulum-v0':
        env.spec.reward_threshold = -200.0  # target_reward

    '''state_dim'''
    state_shape = env.observation_space.shape
    state_dim = state_shape[0] if len(state_shape) == 1 else state_shape  # sometimes state_dim is a list

    '''target_reward'''
    target_reward = env.spec.reward_threshold
    if target_reward is None:
        raise RuntimeError(
            '| I do not know how much is target_reward.\n'
            '| If you do not either. You can set target_reward=+np.inf. \n'
        )

    '''if_discrete action_dim, action_max'''
    if_discrete = isinstance(env.action_space, gym.spaces.Discrete)
    if if_discrete:  # make sure it is discrete action space
        action_dim = env.action_space.n
        action_max = int(1)
    elif isinstance(env.action_space, gym.spaces.Box):  # make sure it is continuous action space
        action_dim = env.action_space.shape[0]
        action_max = float(env.action_space.high[0])

        action_edge = np.array([action_max, ] * action_dim)  # need check
        if any(env.action_space.high - action_edge):
            raise RuntimeError(
                f'| action_space.high should be {action_edge}, but {env.action_space.high}')
        if any(env.action_space.low + action_edge):
            raise RuntimeError(
                f'| action_space.low should be {-action_edge}, but {env.action_space.low}')
    else:
        raise RuntimeError(
            '| I do not know env.action_space is discrete or continuous.\n'
            '| You can set these value manually: if_discrete, action_dim, action_max\n'
        )
    return env_name, state_dim, action_dim, action_max, if_discrete, target_reward


def _get_gym_env_state_norm(env_name, if_norm):
    avg = None
    std = None
    if if_norm:  # I use def print_norm() to get the following (avg, std)
        # if env_name == 'Pendulum-v0':
        #     state_mean = np.array([-0.00968592 -0.00118888 -0.00304381])
        #     std = np.array([0.53825575 0.54198545 0.8671749 ])
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
        # elif env_name == 'MinitaurBulletEnv-v0': # need check
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
        # elif env_name == "BipedalWalkerHardcore-v3": # need check
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
    return avg, std


def _get_decorate_env(env, action_max=1, state_avg=None, state_std=None, data_type=np.float32):
    if state_avg is None:
        neg_state_avg = 0
        div_state_std = 1
    else:
        state_avg = state_avg.astype(data_type)
        state_std = state_std.astype(data_type)

        neg_state_avg = -state_avg
        div_state_std = 1 / (state_std + 1e-4)

    setattr(env, 'neg_state_avg', neg_state_avg)  # for def print_norm() AgentZoo.py
    setattr(env, 'div_state_std', div_state_std)  # for def print_norm() AgentZoo.py

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


"""Extension: Fix Env CarRacing-v0 - Box2D"""


def fix_car_racing_env(env, frame_num=3, action_num=3):  # 2020-12-12
    setattr(env, 'old_step', env.step)  # env.old_step = env.step
    setattr(env, 'env_name', 'CarRacing-Fix')
    setattr(env, 'state_dim', (frame_num, 96, 96))
    setattr(env, 'action_dim', 3)
    setattr(env, 'if_discrete', False)
    setattr(env, 'target_reward', 700)  # 900 in default

    setattr(env, 'state_stack', None)  # env.state_stack = None
    setattr(env, 'avg_reward', 0)  # env.avg_reward = 0
    """ cancel the print() in environment
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
        state = (state - 128).astype(np.float32) / 128.
        return state

    def decorator_step(env_step):
        def new_env_step(action):
            action = action.copy()
            action[1:] = (action[1:] + 1) / 2  # fix action_space.low

            reward_sum = 0
            done = state = None
            try:
                for _ in range(action_num):
                    state, reward, done, info = env_step(action)
                    state = rgb2gray(state)

                    if done:
                        reward += 100  # don't penalize "die state"
                    if state.mean() > 192:  # 185.0:  # penalize when outside of road
                        reward -= 0.05

                    env.avg_reward = env.avg_reward * 0.95 + reward * 0.05
                    if env.avg_reward <= -0.1:  # done if car don't move
                        done = True

                    reward_sum += reward

                    if done:
                        break
            except Exception as error:
                print(f"| CarRacing-v0 Error 'stack underflow'? {error}")
                reward_sum = 0
                done = True
            env.state_stack.pop(0)
            env.state_stack.append(state)

            return np.array(env.state_stack).flatten(), reward_sum, done, {}

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


def render__car_racing():
    import gym  # gym of OpenAI is not necessary for ElegantRL (even RL)
    gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'
    env = gym.make('CarRacing-v0')
    env = fix_car_racing_env(env)

    state_dim = env.state_dim

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


"""Extension: Finance RL: Github AI4Finance-LLC"""


class FinanceMultiStockEnv:  # 2020-12-24
    """FinRL
    Paper: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance
           https://arxiv.org/abs/2011.09607 NeurIPS 2020: Deep RL Workshop.
    Source: Github https://github.com/AI4Finance-LLC/FinRL-Library
    Modify: Github Yonv1943 ElegantRL
    """

    def __init__(self, initial_account=1e6, transaction_fee_percent=1e-3, max_stock=100):
        self.stock_dim = 30
        self.initial_account = initial_account
        self.transaction_fee_percent = transaction_fee_percent
        self.max_stock = max_stock

        self.ary = self.load_training_data_for_multi_stock()
        assert self.ary.shape == (1699, 5 * 30)  # ary: (date, item*stock_dim), item: (adjcp, macd, rsi, cci, adx)

        # reset
        self.day = 0
        self.account = self.initial_account
        self.day_npy = self.ary[self.day]
        self.stocks = np.zeros(self.stock_dim, dtype=np.float32)  # multi-stack
        self.total_asset = self.account + (self.day_npy[:self.stock_dim] * self.stocks).sum()
        self.episode_return = 0.0  # Compatibility for ElegantRL 2020-12-21

        '''env information'''
        self.env_name = 'FinanceStock-v1'
        self.state_dim = 1 + (5 + 1) * self.stock_dim
        self.action_dim = self.stock_dim
        self.if_discrete = False
        self.target_reward = 15
        self.max_step = self.ary.shape[0]

        self.gamma_r = 0.0

    def reset(self):
        self.account = self.initial_account * rd.uniform(0.99, 1.00)  # notice reset()
        self.stocks = np.zeros(self.stock_dim, dtype=np.float32)
        self.total_asset = self.account + (self.day_npy[:self.stock_dim] * self.stocks).sum()
        # total_asset = account + (adjcp * stocks).sum()

        self.day = 0
        self.day_npy = self.ary[self.day]
        self.day += 1

        state = np.hstack((
            self.account * 2 ** -16,
            self.day_npy * 2 ** -8,
            self.stocks * 2 ** -12,
        ), ).astype(np.float32)

        return state

    def step(self, actions):
        actions = actions * self.max_stock

        """bug or sell stock"""
        for index in range(self.stock_dim):
            action = actions[index]
            adj = self.day_npy[index]
            if action > 0:  # buy_stock
                available_amount = self.account // adj
                delta_stock = min(available_amount, action)
                self.account -= adj * delta_stock * (1 + self.transaction_fee_percent)
                self.stocks[index] += delta_stock
            elif self.stocks[index] > 0:  # sell_stock
                delta_stock = min(-action, self.stocks[index])
                self.account += adj * delta_stock * (1 - self.transaction_fee_percent)
                self.stocks[index] -= delta_stock

        """update day"""
        self.day_npy = self.ary[self.day]
        self.day += 1
        done = self.day == self.max_step  # 2020-12-21

        state = np.hstack((
            self.account * 2 ** -16,
            self.day_npy * 2 ** -8,
            self.stocks * 2 ** -12,
        ), ).astype(np.float32)

        next_total_asset = self.account + (self.day_npy[:self.stock_dim] * self.stocks).sum()
        reward = (next_total_asset - self.total_asset) * 2 ** -16  # notice scaling!
        self.total_asset = next_total_asset

        self.gamma_r = self.gamma_r * 0.99 + reward  # notice: gamma_r seems good? Yes
        if done:
            reward += self.gamma_r
            self.gamma_r = 0.0  # env.reset()

            # cumulative_return_rate
            self.episode_return = next_total_asset / self.initial_account

        return state, reward, done, None

    @staticmethod
    def load_training_data_for_multi_stock(if_load=True):  # need more independent
        npy_path = './Result/FinanceMultiStock.npy'
        if if_load and os.path.exists(npy_path):
            data_ary = np.load(npy_path).astype(np.float32)
            assert data_ary.shape[1] == 5 * 30
            return data_ary
        else:
            raise RuntimeError(
                f'| FinanceMultiStockEnv(): Can you download and put it into: {npy_path}\n'
                f'| https://github.com/Yonv1943/ElegantRL/blob/master/Result/FinanceMultiStock.npy'
                f'| Or you can use the following code to generate it from a csv file.'
            )

        # from preprocessing.preprocessors import pd, data_split, preprocess_data, add_turbulence
        #
        # # the following is same as part of run_model()
        # preprocessed_path = "done_data.csv"
        # if if_load and os.path.exists(preprocessed_path):
        #     data = pd.read_csv(preprocessed_path, index_col=0)
        # else:
        #     data = preprocess_data()
        #     data = add_turbulence(data)
        #     data.to_csv(preprocessed_path)
        #
        # df = data
        # rebalance_window = 63
        # validation_window = 63
        # i = rebalance_window + validation_window
        #
        # unique_trade_date = data[(data.datadate > 20151001) & (data.datadate <= 20200707)].datadate.unique()
        # train__df = data_split(df, start=20090000, end=unique_trade_date[i - rebalance_window - validation_window])
        # # print(train__df) # df: DataFrame of Pandas
        #
        # train_ary = train__df.to_numpy().reshape((-1, 30, 12))
        # '''state_dim = 1 + 6 * stock_dim, stock_dim=30
        # n   item    index
        # 1   ACCOUNT -
        # 30  adjcp   2
        # 30  stock   -
        # 30  macd    7
        # 30  rsi     8
        # 30  cci     9
        # 30  adx     10
        # '''
        # data_ary = np.empty((train_ary.shape[0], 5, 30), dtype=np.float32)
        # data_ary[:, 0] = train_ary[:, :, 2]  # adjcp
        # data_ary[:, 1] = train_ary[:, :, 7]  # macd
        # data_ary[:, 2] = train_ary[:, :, 8]  # rsi
        # data_ary[:, 3] = train_ary[:, :, 9]  # cci
        # data_ary[:, 4] = train_ary[:, :, 10]  # adx
        #
        # data_ary = data_ary.reshape((-1, 5 * 30))
        #
        # os.makedirs(npy_path[:npy_path.rfind('/')])
        # np.save(npy_path, data_ary.astype(np.float16))  # save as float16 (0.5 MB), float32 (1.0 MB)
        # print('| FinanceMultiStockEnv(): save in:', npy_path)
        # return data_ary


# import gym
# class DemoGymEnv(gym.Env):
#     def __init__(self):
#         self.observation_space = gym.spaces.Box(low=-np.inf, high=+np.inf, shape=(4,))  # state_dim = 4
#         self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))  # action_dim = 2
#
#     def reset(self):
#         state = rd.randn(4)  # for example
#         return state
#
#     def step(self, action):
#         state = rd.randn(4)  # for example
#         reward = 1
#         done = False
#         return state, reward, done, dict()  # gym.Env requires it to be a dict, even an empty dict
#
#     def render(self, mode='human'):
#         pass


# def original_download_pandas_data():
#     import yfinance as yf
#     from stockstats import StockDataFrame as Sdf
#
#     # Download and save the data in a pandas DataFrame:
#     data_df = yf.download("AAPL", start="2009-01-01", end="2020-10-23")
#
#     # data_df.shape
#
#     # reset the index, we want to use numbers instead of dates
#     data_df = data_df.reset_index()
#     data_df.head()
#     # data_df.columns
#
#     # convert the column names to standardized names
#     data_df.columns = ['datadate', 'open', 'high', 'low', 'close', 'adjcp', 'volume']
#
#     # save the data to a csv file in your current folder
#     # data_df.to_csv('AAPL_2009_2020.csv')
#
#     """# Part 2: Preprocess Data
#     Data preprocessing is a crucial step for training a high quality machine learning model.
#     We need to check for missing data and do feature engineering
#     in order to convert the data into a model-ready state.
#     """
#
#     # check missing data
#     data_df.isnull().values.any()
#
#     # calculate technical indicators like MACD
#     stock = Sdf.retype(data_df.copy())
#     # we need to use adjusted close price instead of close price
#     stock['close'] = stock['adjcp']
#     data_df['macd'] = stock['macd']
#
#     # check missing data again
#     data_df.isnull().values.any()
#
#     data_df.head()
#
#     # data_df=data_df.fillna(method='bfill')
#
#     # Note that I always use a copy of the original data to try it track step by step.
#     data_clean = data_df.copy()
#
#     data_clean.head()
#
#     data_clean.tail()
#     return data_clean
#
#
# class SingleStockFinEnvForStableBaseLines(gym.Env):  # adjust state, inner df_pandas, beta3 pass
#     """FinRL
#     Paper: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance
#            https://arxiv.org/abs/2011.09607 NeurIPS 2020: Deep RL Workshop.
#     Source: Github https://github.com/AI4Finance-LLC/FinRL-Library
#     Modify: Github Yonv1943 ElegantRL
#     """
#
#     """ Update Log 2020-12-12 by Github Yonv1943
#     change download_preprocess_data: If the data had been downloaded, then don't download again
#
#     # env
#     move reward_memory out of Env
#     move plt.savefig('account_value.png') out of Env
#     cancel SingleStockEnv(gym.Env): There is not need to use OpenAI's gym
#     change pandas to numpy
#     fix bug in comment: ('open', 'high', 'low', 'close', 'adjcp', 'volume', 'macd'), lack 'macd' before
#     change slow 'state'
#     change repeat compute 'begin_total_asset', 'end_total_asset'
#     cancel self.asset_memory
#     cancel self.cost
#     cancel self.trade
#     merge '_sell_stock' and '_bug_stock' to _sell_or_but_stock
#     adjust order of state
#     reserved expansion interface on self.stock self.stocks
#
#     # compatibility
#     move global variable into Env.__init__()
#     cancel matplotlib.use('Agg'): It will cause compatibility issues for ssh connection
#     """
#     """A stock trading environment for OpenAI gym"""
#     metadata = {'render.modes': ['human']}
#
#     def __init__(self, initial_account=100000, transaction_fee_rate=0.001, max_stock=200):
#         self.stock_dim = 1
#
#         # not necessary
#         self.observation_space = gym.spaces.Box(low=0, high=2 ** 24, shape=(4,))
#         self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.stock_dim,))
#
#         self.initial_account = initial_account
#         self.transaction_fee_rate = transaction_fee_rate
#         self.max_stock = max_stock
#         self.state_div_std = np.array((2 ** -14, 2 ** -4, 2 ** 0, 2 ** -11))
#
#         self.ary = self.download_data_as_csv__load_as_array()
#         assert self.ary.shape == (2517, 9)  # ary: (date, item)
#         self.ary = self.ary[1:, 2:].astype(np.float32)
#         assert self.ary.shape == (2516, 7)  # ary: (date, item), item: (open, high, low, close, adjcp, volume, macd)
#         self.ary = np.concatenate((
#             self.ary[:, 4:5],  # adjcp? What is this? unit price?
#             self.ary[:, 6:7],  # macd? What is this?
#         ), axis=1)
#         self.max_day = self.ary.shape[0] - 1
#
#         # reset
#         self.day = 0
#         self.account = self.initial_account
#         self.day_npy = self.ary[self.day]
#         # self.stocks = np.zeros(self.stock_dim, dtype=np.float32) # multi-stack
#         self.stock = 0
#         # self.begin_total_asset = self.account + (self.day_npy[:self.stock_dim] * self.stocks).sum()
#         self.begin_total_asset = self.account + self.day_npy[0] * self.stock
#
#         self.step_sum = 0
#         self.reward_sum = 0.0
#
#     def reset(self):
#         self.reward_sum = 0.0
#
#         self.day = 0
#         self.account = self.initial_account
#         self.day_npy = self.ary[self.day]
#         # self.stocks = np.zeros(self.stock_dim, dtype=np.float32)
#         self.stock = 0
#         # self.begin_total_asset = self.account + (self.day_npy[:self.stock_dim] * self.stocks).sum()
#         self.begin_total_asset = self.account + self.day_npy[0] * self.stock
#         # state = np.hstack((self.account, self.day_npy, self.stocks)
#         #                   ).astype(np.float32) * self.state_div_std
#         state = np.hstack((self.account, self.day_npy, self.stock)
#                           ).astype(np.float32) * self.state_div_std
#         return state
#
#     def step(self, actions):
#         actions = actions * self.max_stock
#
#         """bug or sell stock"""
#         index = 0
#         action = actions[index]
#         adj = self.day_npy[index]
#         if action > 0:  # buy_stock
#             available_amount = self.account // adj
#             delta_stock = min(available_amount, action)
#             self.account -= adj * delta_stock * (1 + self.transaction_fee_rate)
#             # self.stocks[index] += delta_stock
#             self.stock += delta_stock
#         # elif self.stocks[index] > 0:  # sell_stock
#         #     delta_stock = min(-action, self.stocks[index])
#         #     self.account += adj * delta_stock * (1 - self.transaction_fee_percent)
#         #     self.stocks[index] -= delta_stock
#         elif self.stock > 0:  # sell_stock
#             delta_stock = min(-action, self.stock)
#             self.account += adj * delta_stock * (1 - self.transaction_fee_rate)
#             self.stock -= delta_stock
#
#         """update day"""
#         self.day += 1
#         # self.data = self.df.loc[self.day, :]
#         self.day_npy = self.ary[self.day]
#
#         # state = np.hstack((self.account, self.day_npy, self.stocks)
#         #                   ).astype(np.float32) * self.state_div_std
#         state = np.hstack((self.account, self.day_npy, self.stock)
#                           ).astype(np.float32) * self.state_div_std
#
#         # end_total_asset = self.account + (self.day_npy[:self.stock_dim] * self.stocks).sum()
#         end_total_asset = self.account + self.day_npy[0] * self.stock
#         reward = end_total_asset - self.begin_total_asset
#         self.begin_total_asset = end_total_asset
#
#         done = self.day == self.max_day  # 2516 is over
#         reward_ = reward * 2 ** -10
#
#         self.reward_sum += reward * 2 ** -10
#         self.step_sum += 1
#         if done:
#             print(f'{self.step_sum:8}   {self.reward_sum:8.1f}')
#         return state, reward_, done, {}
#
#     def render(self, mode='human'):
#         pass
#
#     @staticmethod
#     def download_data_as_csv__load_as_array(if_load=True):
#         save_path = './AAPL_2009_2020.csv'
#
#         import os
#         if if_load and os.path.isfile(save_path):
#             ary = np.genfromtxt(save_path, delimiter=',')
#             assert isinstance(ary, np.ndarray)
#             return ary
#         import yfinance as yf
#         from stockstats import StockDataFrame as Sdf
#         """ pip install
#         !pip install yfinance
#         !pip install pandas
#         !pip install matplotlib
#         !pip install stockstats
#         """
#
#         """# Part 1: Download Data
#         Yahoo Finance is a website that provides stock data, financial news, financial reports, etc.
#         All the data provided by Yahoo Finance is free.
#         """
#         print('| download_preprocess_data_as_csv: Download Data')
#
#         data_pd = yf.download("AAPL", start="2009-01-01", end="2020-10-23")
#         assert data_pd.shape == (2974, 6)
#
#         data_pd = data_pd.reset_index()
#
#         data_pd.columns = ['datadate', 'open', 'high', 'low', 'close', 'adjcp', 'volume']
#
#         """# Part 2: Preprocess Data
#         Data preprocessing is a crucial step for training a high quality machine learning model.
#         We need to check for missing data and do feature engineering
#         in order to convert the data into a model-ready state.
#         """
#         print('| download_preprocess_data_as_csv: Preprocess Data')
#
#         # check missing data
#         data_pd.isnull().values.any()
#
#         # calculate technical indicators like MACD
#         stock = Sdf.retype(data_pd.copy())
#         # we need to use adjusted close price instead of close price
#         stock['close'] = stock['adjcp']
#         data_pd['macd'] = stock['macd']
#
#         # check missing data again
#         data_pd.isnull().values.any()
#         data_pd.head()
#         # data_pd=data_pd.fillna(method='bfill')
#
#         # Note that I always use a copy of the original data to try it track step by step.
#         data_clean = data_pd.copy()
#         data_clean.head()
#         data_clean.tail()
#
#         data = data_clean[(data_clean.datadate >= '2009-01-01') & (data_clean.datadate < '2019-01-01')]
#         data = data.reset_index(drop=True)  # the index needs to start from 0
#
#         data.to_csv(save_path)  # save *.csv
#         # assert isinstance(data_pd, pd.DataFrame)
#
#         df_pandas = data[(data.datadate >= '2009-01-01') & (data.datadate < '2019-01-01')]
#         df_pandas = df_pandas.reset_index(drop=True)  # the index needs to start from 0
#         ary = df_pandas.to_numpy()
#         return ary


# def test():
#     env = SingleStockFinEnv()
#     ary = env.download_data_as_csv__load_as_array(if_load=True)  # data_frame_pandas
#     print(ary.shape)
#     ary = env.download_data_as_csv__load_as_array(if_load=True)  # data_frame_pandas
#     print(ary.shape)
#
#     env = SingleStockFinEnv(ary)
#     state_dim, action_dim = 4, 1
#
#     # state = env.reset()
#     # done = False
#     reward_sum = 0
#     for i in range(2514):
#         state, reward, done, info = env.step(rd.uniform(-1, 1, size=action_dim))
#         reward_sum += reward
#         # print(f'{i:5} {done:5} {reward:8.1f}', state)
#     print(';', reward_sum)
#
#     # state = env.reset()
#     # done = False
#     for _ in range(4):
#         state, reward, done, info = env.step(rd.uniform(-1, 1, size=action_dim))
#         print(f'{done:5} {reward:8.1f}', state)
#
#
# def train():
#     from AgentRun import Arguments, train_agent_mp
#     from AgentZoo import AgentPPO, AgentModSAC
#     args = Arguments(rl_agent=None, env_name=None, gpu_id=None)
#     args.rl_agent = AgentModSAC
#     """
#     | GPU: 0 | CWD: ./AgentModSAC/FinRL_0
#     ID      Step      MaxR |    avgR      stdR |    ExpR     LossA     LossC
#     0.0
#     0   1.01e+04      1.30 |
#     0   1.76e+04    485.86 |  158.33      0.00 |  430.58      0.67      0.07
#     0   2.01e+04    916.53 |
#     ID      Step   TargetR |    avgR      stdR |    ExpR  UsedTime  ########
#     0   2.01e+04    800.00 |  916.53      0.00 |  141.46       322  ########
#     0   3.27e+04    976.58 |  589.64      0.00 |  379.84      0.26      0.19
#     0   5.78e+04    976.58 |  628.69      0.00 |  517.65     -0.02      0.41
#
#     | GPU: 1 | CWD: ./AgentModSAC/FinRL_1
#     ID      Step      MaxR |    avgR      stdR |    ExpR     LossA     LossC
#     0.0
#     1   5.03e+03      0.40 |
#     1   1.76e+04      6.63 |    6.63      0.00 |   24.55      0.69      0.01
#     1   2.26e+04    652.01 |
#     1   2.77e+04    836.11 |
#     ID      Step   TargetR |    avgR      stdR |    ExpR  UsedTime  ########
#     1   2.77e+04    800.00 |  836.11      0.00 |  650.18       430  ########
#     1   3.02e+04    873.22 |
#     1   3.52e+04    913.21 |
#     1   3.52e+04    913.21 |  913.21      0.00 |  842.45     -0.12      0.35
#     """
#     args.if_break_early = False
#     args.break_step = 2 ** 20
#
#     args.max_memo = 2 ** 16
#     args.gamma = 0.99  # important hyper-parameter, related to episode steps
#     args.reward_scale = 2 ** -2
#     args.max_step = 2515
#     args.eval_times1 = 1
#     args.eval_times2 = 1
#
#     args.env_name = 'FinRL'
#     args.init_for_training()
#     train_agent_mp(args)
#
#     args = Arguments(rl_agent=None, env_name=None, gpu_id=None)
#     args.rl_agent = AgentPPO
#     """
#     | GPU: 3 | CWD: ./AgentPPO/FinRL_3
#     ID      Step      MaxR |    avgR      stdR |    ExpR     LossA     LossC
#     3   2.51e+04     48.87 |
#     3   5.53e+04    441.54 |
#     3   1.06e+05    707.73 |
#     ID      Step   TargetR |    avgR      stdR |    ExpR  UsedTime  ########
#     3   2.01e+05    800.00 |  810.60      0.00 |  748.02       197  ########
#     3   2.66e+05    862.69 |  855.54      0.00 |  797.89     -0.03      0.53
#     3   5.33e+05    915.21 |  888.58      0.00 |  874.78      0.06      0.48
#     3   1.07e+06    942.40 |  938.52      0.00 |  932.97     -0.10      0.44
#     3   1.34e+06    948.64 |  944.51      0.00 |  945.21     -0.05      0.41
#     """
#     args.if_break_early = False
#     args.break_step = 2 ** 21  # 2**20==1e6, 2**21
#
#     args.net_dim = 2 ** 8
#     args.max_memo = 2 ** 12
#     args.batch_size = 2 ** 9
#     args.repeat_times = 2 ** 4
#     # args.reward_scale = 2 ** -10  # unimportant hyper-parameter in PPO which do normalization on Q value
#     args.gamma = 0.95  # important hyper-parameter, related to episode steps
#     args.max_step = 2515 * 2
#     args.eval_times1 = 1
#     args.eval_times2 = 1
#
#     args.env_name = 'FinRL'
#     args.init_for_training()
#     train_agent_mp(args)
#     exit()


'''Utils: Experiment Replay Buffer'''


class BufferArray:  # 2020-11-11
    def __init__(self, max_len, state_dim, action_dim, if_ppo=False):
        state_dim = state_dim if isinstance(state_dim, int) else np.prod(state_dim)  # pixel-level state

        if if_ppo:  # for Offline PPO
            memo_dim = 1 + 1 + state_dim + action_dim + action_dim
        else:
            memo_dim = 1 + 1 + state_dim + action_dim + state_dim

        self.memories = np.empty((max_len, memo_dim), dtype=np.float32)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.max_len = max_len
        self.now_len = 0
        self.next_idx = 0
        self.is_full = False

        self.state_idx = 1 + 1 + state_dim  # reward_dim==1, done_dim==1
        self.action_idx = self.state_idx + action_dim

    def append_memo(self, memo_tuple):
        # memo_array == (reward, mask, state, action, next_state)
        self.memories[self.next_idx] = np.hstack(memo_tuple)
        self.next_idx = self.next_idx + 1
        if self.next_idx >= self.max_len:
            self.is_full = True
            self.next_idx = 0

    def extend_memo(self, memo_array):
        # assert isinstance(memo_array, np.ndarray)
        size = memo_array.shape[0]
        next_idx = self.next_idx + size
        if next_idx > self.max_len:
            if next_idx > self.max_len:
                self.memories[self.next_idx:self.max_len] = memo_array[:self.max_len - self.next_idx]
            self.is_full = True
            next_idx = next_idx - self.max_len
            self.memories[0:next_idx] = memo_array[-next_idx:]
        else:
            self.memories[self.next_idx:next_idx] = memo_array
        self.next_idx = next_idx

    def random_sample(self, batch_size):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # indices = rd.choice(self.memo_len, batch_size, replace=False)  # why perform worse?
        # indices = rd.choice(self.memo_len, batch_size, replace=True)  # why perform better?
        # same as:
        indices = rd.randint(self.now_len, size=batch_size)
        memory = torch.tensor(self.memories[indices], device=self.device)

        '''convert array into torch.tensor'''
        tensors = (
            memory[:, 0:1],  # rewards
            memory[:, 1:2],  # masks, mark == (1-float(done)) * gamma
            memory[:, 2:self.state_idx],  # states
            memory[:, self.state_idx:self.action_idx],  # actions
            memory[:, self.action_idx:],  # next_states
        )
        return tensors

    def all_sample(self):  # 2020-11-11 fix bug for ModPPO
        tensors = (
            self.memories[:self.now_len, 0:1],  # rewards
            self.memories[:self.now_len, 1:2],  # masks, mark == (1-float(done)) * gamma
            self.memories[:self.now_len, 2:self.state_idx],  # states
            self.memories[:self.now_len, self.state_idx:self.action_idx],  # actions
            self.memories[:self.now_len, self.action_idx:],  # next_states or log_prob_sum
        )
        tensors = [torch.tensor(ary, device=self.device) for ary in tensors]
        return tensors

    def update__now_len__before_sample(self):
        self.now_len = self.max_len if self.is_full else self.next_idx

    def empty_memories__before_explore(self):
        self.next_idx = 0
        self.now_len = 0
        self.is_full = False

    def print_state_norm(self, neg_avg=None, div_std=None):  # non-essential
        memory_state = self.memories[:, 2:self.state_idx]
        _print_norm(memory_state, neg_avg, div_std)


class BufferArrayGPU:  # 2020-07-07
    def __init__(self, memo_max_len, state_dim, action_dim, if_ppo=False):
        state_dim = state_dim if isinstance(state_dim, int) else np.prod(state_dim)  # pixel-level state

        if if_ppo:  # for Offline PPO
            memo_dim = 1 + 1 + state_dim + action_dim + action_dim
        else:
            memo_dim = 1 + 1 + state_dim + action_dim + state_dim

        assert torch.cuda.is_available()
        self.device = torch.device("cuda")
        self.memories = torch.empty((memo_max_len, memo_dim), dtype=torch.float32, device=self.device)

        self.next_idx = 0
        self.is_full = False
        self.max_len = memo_max_len
        self.now_len = self.max_len if self.is_full else self.next_idx

        self.state_idx = 1 + 1 + state_dim  # reward_dim==1, done_dim==1
        self.action_idx = self.state_idx + action_dim

    def append_memo(self, memo_tuple):
        """memo_tuple == (reward, mask, state, action, next_state)
        """
        memo_array = np.hstack(memo_tuple)
        self.memories[self.next_idx] = torch.tensor(memo_array, device=self.device)
        self.next_idx = self.next_idx + 1
        if self.next_idx >= self.max_len:
            self.is_full = True
            self.next_idx = 0

    def append_memo_ary(self, memo_array):
        """
        memo_tuple == (reward, mask, state, action, next_state)
        memo_array = np.hstack(memo_tuple)
        """
        self.memories[self.next_idx] = torch.tensor(memo_array, device=self.device)
        self.next_idx = self.next_idx + 1
        if self.next_idx >= self.max_len:
            self.is_full = True
            self.next_idx = 0

    def extend_memo(self, memo_array):  # 2020-07-07
        # assert isinstance(memo_array, np.ndarray)
        size = memo_array.shape[0]
        memo_tensor = torch.tensor(memo_array, device=self.device)

        next_idx = self.next_idx + size
        if next_idx > self.max_len:
            if next_idx > self.max_len:
                self.memories[self.next_idx:self.max_len] = memo_tensor[:self.max_len - self.next_idx]
            self.is_full = True
            next_idx = next_idx - self.max_len
            self.memories[0:next_idx] = memo_tensor[-next_idx:]
        else:
            self.memories[self.next_idx:next_idx] = memo_tensor
        self.next_idx = next_idx

    def update__now_len__before_sample(self):
        self.now_len = self.max_len if self.is_full else self.next_idx

    def empty_memories__before_explore(self):
        self.next_idx = 0
        self.is_full = False
        self.now_len = 0

    def random_sample(self, batch_size):  # _device should remove
        indices = rd.randint(self.now_len, size=batch_size)
        memory = self.memories[indices]

        '''convert array into torch.tensor'''
        tensors = (
            memory[:, 0:1],  # rewards
            memory[:, 1:2],  # masks, mark == (1-float(done)) * gamma
            memory[:, 2:self.state_idx],  # state
            memory[:, self.state_idx:self.action_idx],  # actions
            memory[:, self.action_idx:],  # next_states or actions_noise
        )
        return tensors

    def all_sample(self):  # 2020-11-11 fix bug for ModPPO
        tensors = (
            self.memories[:self.now_len, 0:1],  # rewards
            self.memories[:self.now_len, 1:2],  # masks, mark == (1-float(done)) * gamma
            self.memories[:self.now_len, 2:self.state_idx],  # state
            self.memories[:self.now_len, self.state_idx:self.action_idx],  # actions
            self.memories[:self.now_len, self.action_idx:],  # next_states or log_prob_sum
        )
        return tensors

    def print_state_norm(self, neg_avg=None, div_std=None):  # non-essential
        max_sample_size = 2 ** 14
        if self.now_len > max_sample_size:
            indices = rd.randint(self.now_len, size=min(self.now_len, max_sample_size))
            memory_state = self.memories[indices, 2:self.state_idx]
        else:
            memory_state = self.memories[:, 2:self.state_idx]

        _print_norm(memory_state, neg_avg, div_std)


def _print_norm(batch_state, neg_avg=None, div_std=None):  # 2020-12-12
    if isinstance(batch_state, torch.Tensor):
        batch_state = batch_state.cpu().data.numpy()
    assert isinstance(batch_state, np.ndarray)

    if batch_state.shape[1] > 64:
        print(f"| _print_norm(): state_dim: {batch_state.shape[1]:.0f} is too large to print its norm. ")
        return None

    if np.isnan(batch_state).any():  # 2020-12-12
        batch_state = np.nan_to_num(batch_state)  # nan to 0

    ary_avg = batch_state.mean(axis=0)
    ary_std = batch_state.std(axis=0)
    fix_std = ((np.max(batch_state, axis=0) - np.min(batch_state, axis=0)) / 6
               + ary_std) / 2

    if neg_avg is not None:  # norm transfer
        ary_avg = ary_avg - neg_avg / div_std
        ary_std = fix_std / div_std

    print(f"| print_norm: state_avg, state_fix_std")
    print(f"| avg = np.{repr(ary_avg).replace('=float32', '=np.float32')}")
    print(f"| std = np.{repr(ary_std).replace('=float32', '=np.float32')}")


""" Backup
class BufferList:
    def __init__(self, memo_max_len):
        self.memories = list()

        self.max_len = memo_max_len
        self.now_len = len(self.memories)

    def append_memo(self, memory_tuple):
        self.memories.append(memory_tuple)

    def init_before_sample(self):
        del_len = len(self.memories) - self.max_len
        if del_len > 0:
            del self.memories[:del_len]
            # print('Length of Deleted Memories:', del_len)

        self.now_len = len(self.memories)

    def random_sample(self, batch_size, device):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # indices = rd.choice(self.memo_len, batch_size, replace=False)  # why perform worse?
        # indices = rd.choice(self.memo_len, batch_size, replace=True)  # why perform better?
        # same as:
        indices = rd.randint(self.now_len, size=batch_size)

        '''convert list into array'''
        arrays = [list()
                  for _ in range(5)]  # len(self.memories[0]) == 5
        for index in indices:
            items = self.memories[index]
            for item, array in zip(items, arrays):
                array.append(item)

        '''convert array into torch.tensor'''
        tensors = [torch.tensor(np.array(ary), dtype=torch.float32, device=device)
                   for ary in arrays]
        return tensors


class BufferTuple:
    def __init__(self, memo_max_len):
        self.memories = list()

        self.max_len = memo_max_len
        self.now_len = None  # init in init_after_append_memo()

        from collections import namedtuple
        self.transition = namedtuple(
            'Transition', ('reward', 'mask', 'state', 'action', 'next_state',)
        )

    def append_memo(self, args):
        self.memories.append(self.transition(*args))

    def init_before_sample(self):
        del_len = len(self.memories) - self.max_len
        if del_len > 0:
            del self.memories[:del_len]
            # print('Length of Deleted Memories:', del_len)

        self.now_len = len(self.memories)

    def random_sample(self, batch_size, device):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # indices = rd.choice(self.memo_len, batch_size, replace=False)  # why perform worse?
        # indices = rd.choice(self.memo_len, batch_size, replace=True)  # why perform better?
        # same as:
        indices = rd.randint(self.now_len, size=batch_size)

        '''convert tuple into array'''
        arrays = self.transition(*zip(*[self.memories[i] for i in indices]))

        '''convert array into torch.tensor'''
        tensors = [torch.tensor(np.array(ary), dtype=torch.float32, device=device)
                   for ary in arrays]
        return tensors
        
        
class BufferTupleOnline:
    def __init__(self, max_memo):
        self.max_memo = max_memo
        self.storage_list = list()
        from collections import namedtuple
        self.transition = namedtuple(
            'Transition',
            # ('state', 'value', 'action', 'log_prob', 'mask', 'next_state', 'reward')
            ('reward', 'mask', 'state', 'action', 'log_prob')
        )

    def push(self, *args):
        self.storage_list.append(self.transition(*args))

    def extend_memo(self, storage_list):
        self.storage_list.extend(storage_list)

    def sample_all(self):
        return self.transition(*zip(*self.storage_list))

    def __len__(self):
        return len(self.storage_list)

    def update_pointer_before_sample(self):
        pass  # compatibility

    def print_state_norm(self, neg_avg=None, div_std=None):  # non-essential
        memory_state = np.array([item.state for item in self.storage_list])
        print_norm(memory_state, neg_avg, div_std)

"""

"""DEMO"""


def train__demo():
    pass

    '''DEMO 1: Standard gym env CartPole-v0 (discrete action) using D3QN (DuelingDoubleDQN, off-policy)'''
    import gym  # gym of OpenAI is not necessary for ElegantRL (even RL)
    gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'
    env = gym.make('CartPole-v0')
    env = decorate_env(env, if_print=True)

    from AgentZoo import AgentD3QN
    args = Arguments(rl_agent=AgentD3QN, env=env, gpu_id=0)
    args.break_step = int(1e5 * 8)  # UsedTime: 60s (reach target_reward 195)
    args.net_dim = 2 ** 7
    args.init_for_training()
    train_agent(args)
    exit()

    '''DEMO 2: Standard gym env LunarLanderContinuous-v2 (continuous action) using ModSAC (Modify SAC, off-policy)'''
    import gym  # gym of OpenAI is not necessary for ElegantRL (even RL)
    gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'
    env = gym.make('LunarLanderContinuous-v2')
    env = decorate_env(env, if_print=True)

    from AgentZoo import AgentModSAC
    args = Arguments(rl_agent=AgentModSAC, env=env, gpu_id=0)
    args.break_step = int(6e4 * 8)  # UsedTime: 900s (reach target_reward 200)
    args.net_dim = 2 ** 7
    args.init_for_training()
    # train_agent(args)  # Train agent using single process. Recommend run on PC.
    train_agent_mp(args)  # Train using multi process. Recommend run on Server.
    exit()

    '''DEMO 3: Custom env FinanceStock (continuous action) using PPO (PPO2+GAE, on-policy)'''
    env = FinanceMultiStockEnv()  # 2020-12-24

    from AgentZoo import AgentPPO
    args = Arguments(rl_agent=AgentPPO, env=env)
    args.eval_times1 = 1
    args.eval_times2 = 1
    args.rollout_num = 4
    args.if_break_early = True

    args.reward_scale = 2 ** 0  # (0) 1.1 ~ 15 (19)
    args.break_step = int(5e6 * 4)  # 5e6 (15e6) UsedTime: 4,000s (12,000s)
    args.net_dim = 2 ** 8
    args.max_step = 1699
    args.max_memo = 1699 * 16
    args.batch_size = 2 ** 10
    args.repeat_times = 2 ** 4
    args.init_for_training()
    train_agent_mp(args)  # train_agent(args)
    exit()

    # from AgentZoo import AgentModSAC
    # args = Arguments(rl_agent=AgentModSAC, env=env)  # much slower than on-policy trajectory
    # args.eval_times1 = 1
    # args.eval_times2 = 2
    #
    # args.break_step = 2 ** 22  # UsedTime:
    # args.net_dim = 2 ** 7
    # args.max_memo = 2 ** 18
    # args.batch_size = 2 ** 8
    # args.init_for_training()
    # train_agent_mp(args)  # train_agent(args)


def train__continuous_action__off_policy():
    import AgentZoo as Zoo
    args = Arguments(rl_agent=None, env=None, gpu_id=None)
    args.rl_agent = [
        Zoo.AgentDDPG,  # 2016. simple, simple, slow, unstable
        Zoo.AgentBaseAC,  # 2016+ modify DDPG, faster, more stable
        Zoo.AgentTD3,  # 2018. twin critics, delay target update
        Zoo.AgentSAC,  # 2018. twin critics, maximum entropy, auto alpha, fix log_prob
        Zoo.AgentModSAC,  # 2018+ modify SAC, faster, more stable
        Zoo.AgentInterAC,  # 2019. Integrated AC(DPG)
        Zoo.AgentInterSAC,  # 2020. Integrated SAC(SPG)
    ][4]  # I suggest to use ModSAC (Modify SAC)
    # On-policy PPO is not here 'run__off_policy()'. See PPO in 'run__on_policy()'.

    import gym  # gym of OpenAI is not necessary for ElegantRL (even RL)
    gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'
    args.if_break_early = True  # break training if reach the target reward (total return of an episode)
    args.if_remove_history = True  # delete the historical directory

    env = gym.make('Pendulum-v0')  # It is easy to reach target score -200.0 (-100 is harder)
    args.env = decorate_env(env, if_print=True)
    args.break_step = int(1e4 * 8)  # 1e4 means the average total training step of InterSAC to reach target_reward
    args.reward_scale = 2 ** -2  # (-1800) -1000 ~ -200 (-50)
    args.init_for_training()
    train_agent(args)  # Train agent using single process. Recommend run on PC.
    # train_agent_mp(args)  # Train using multi process. Recommend run on Server. Mix CPU(eval) GPU(train)
    exit()

    args.env = decorate_env(gym.make('LunarLanderContinuous-v2'), if_print=True)
    args.break_step = int(5e5 * 8)  # (2e4) 5e5, used time 1500s
    args.reward_scale = 2 ** -3  # (-800) -200 ~ 200 (302)
    args.init_for_training()
    train_agent_mp(args)  # train_agent(args)
    exit()

    args.env = decorate_env(gym.make('BipedalWalker-v3'), if_print=True)
    args.break_step = int(2e5 * 8)  # (1e5) 2e5, used time 3500s
    args.reward_scale = 2 ** -1  # (-200) -140 ~ 300 (341)
    args.init_for_training()
    train_agent_mp(args)  # train_agent(args)
    exit()

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)
    args.env = decorate_env(gym.make('ReacherBulletEnv-v0'), if_print=True)
    args.break_step = int(5e4 * 8)  # (4e4) 5e4
    args.reward_scale = 2 ** 0  # (-37) 0 ~ 18 (29)
    args.init_for_training()
    train_agent_mp(args)  # train_agent(args)
    exit()

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)
    args.env = decorate_env(gym.make('AntBulletEnv-v0'), if_print=True)
    args.break_step = int(1e6 * 8)  # (5e5) 1e6, UsedTime: (15,000s) 30,000s
    args.reward_scale = 2 ** -3  # (-50) 0 ~ 2500 (3340)
    args.batch_size = 2 ** 8
    args.max_memo = 2 ** 20
    args.eva_size = 2 ** 3  # for Recorder
    args.show_gap = 2 ** 8  # for Recorder
    args.init_for_training()
    train_agent_mp(args)  # train_agent(args)
    exit()

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)
    args.env = decorate_env(gym.make('MinitaurBulletEnv-v0'), if_print=True)
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
    train_agent_mp(args)  # train_agent(args)
    exit()

    args.env = decorate_env(gym.make('BipedalWalkerHardcore-v3'), if_print=True)  # 2020-08-24 plan
    args.reward_scale = 2 ** 0  # (-200) -150 ~ 300 (334)
    args.break_step = int(4e6 * 8)  # (2e6) 4e6
    args.net_dim = int(2 ** 8)  # int(2 ** 8.5) #
    args.max_memo = int(2 ** 21)
    args.batch_size = int(2 ** 8)
    args.eval_times2 = 2 ** 5  # for Recorder
    args.show_gap = 2 ** 8  # for Recorder
    args.init_for_training()
    train_agent_mp(args)  # train_offline_policy(args)
    exit()


def train__continuous_action__on_policy():
    import AgentZoo as Zoo
    args = Arguments(rl_agent=None, env=None, gpu_id=None)
    args.rl_agent = [
        Zoo.AgentPPO,  # 2018. PPO2 + GAE, slow but quite stable, especially in high-dim
        Zoo.AgentInterPPO,  # 2019. Integrated Network, useful in pixel-level task (state2D)
    ][0]

    import gym  # gym of OpenAI is not necessary for ElegantRL (even RL)
    gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'
    args.if_break_early = True  # break training if reach the target reward (total return of an episode)
    args.if_remove_history = True  # delete the historical directory

    args.net_dim = 2 ** 8
    args.max_memo = 2 ** 12
    args.batch_size = 2 ** 9
    args.repeat_times = 2 ** 4
    args.reward_scale = 2 ** 0  # unimportant hyper-parameter in PPO which do normalization on Q value
    args.gamma = 0.99  # important hyper-parameter, related to episode steps

    env = gym.make('Pendulum-v0')  # It is easy to reach target score -200.0 (-100 is harder)
    args.env = decorate_env(env, if_print=True)
    args.break_step = int(8e4 * 8)  # 5e5 means the average total training step of ModPPO to reach target_reward
    args.reward_scale = 2 ** 0  # (-1800) -1000 ~ -200 (-50), UsedTime:  (100s) 200s
    args.gamma = 0.9  # important hyper-parameter, related to episode steps
    args.init_for_training()
    train_agent_mp(args)  # train_agent(args)
    exit()

    args.env = decorate_env(gym.make('LunarLanderContinuous-v2'), if_print=True)
    args.break_step = int(3e5 * 8)  # (2e5) 3e5 , used time: (400s) 600s
    args.reward_scale = 2 ** 0  # (-800) -200 ~ 200 (301)
    args.gamma = 0.99  # important hyper-parameter, related to episode steps
    args.init_for_training()
    train_agent_mp(args)  # train_agent(args)
    # exit()

    args.env = decorate_env(gym.make('BipedalWalker-v3'), if_print=True)
    args.break_step = int(8e5 * 8)  # (4e5) 8e5 (4e6), UsedTimes: (600s) 1500s (8000s)
    args.reward_scale = 2 ** 0  # (-150) -90 ~ 300 (325)
    args.gamma = 0.95  # important hyper-parameter, related to episode steps
    args.init_for_training()
    train_agent_mp(args)  # train_agent(args)
    exit()

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)
    args.env = decorate_env(gym.make('ReacherBulletEnv-v0'), if_print=True)
    args.break_step = int(2e6 * 8)  # (1e6) 2e6 (4e6), UsedTimes: 2000s (6000s)
    args.reward_scale = 2 ** 0  # (-15) 0 ~ 18 (25)
    args.gamma = 0.95  # important hyper-parameter, related to episode steps
    args.init_for_training()
    train_agent_mp(args)  # train_agent(args)
    exit()

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)
    args.env = decorate_env(gym.make('AntBulletEnv-v0'), if_print=True)
    args.break_step = int(5e6 * 8)  # (1e6) 5e6 UsedTime: 25697s
    args.reward_scale = 2 ** -3  #
    args.gamma = 0.99  # important hyper-parameter, related to episode steps
    args.net_dim = 2 ** 9
    args.init_for_training()
    train_agent_mp(args)
    exit()

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)
    args.env = decorate_env(gym.make('MinitaurBulletEnv-v0'), if_print=True)
    args.break_step = int(1e6 * 8)  # (4e5) 1e6 (8e6)
    args.reward_scale = 2 ** 4  # (-2) 0 ~ 16 (PPO 34)
    args.gamma = 0.95  # important hyper-parameter, related to episode steps
    args.net_dim = 2 ** 8
    args.max_memo = 2 ** 11
    args.batch_size = 2 ** 9
    args.repeat_times = 2 ** 4
    args.init_for_training()
    train_agent_mp(args)
    exit()

    # args.env = decorate_env(gym.make('BipedalWalkerHardcore-v3'), if_print=True)  # 2020-08-24 plan
    # on-policy (like PPO) is BAD at learning on a environment with so many random factors (like BipedalWalkerHardcore).
    # exit()

    args.env = fix_car_racing_env(gym.make('CarRacing-v0'))
    # on-policy (like PPO) is GOOD at learning on a environment with less random factors (like 'CarRacing-v0').
    # see 'train__car_racing__pixel_level_state2d()'


def train__discrete_action():
    import AgentZoo as Zoo
    args = Arguments(rl_agent=None, env=None, gpu_id=None)
    args.rl_agent = [
        Zoo.AgentDQN,  # 2014.
        Zoo.AgentDoubleDQN,  # 2016. stable
        Zoo.AgentDuelingDQN,  # 2016. stable and fast
        Zoo.AgentD3QN,  # 2016+ Dueling + Double DQN (Not a creative work)
    ][3]  # I suggest to use D3QN

    import gym  # gym of OpenAI is not necessary for ElegantRL (even RL)
    gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'

    args.env = decorate_env(gym.make('CartPole-v0'), if_print=True)
    args.break_step = int(1e4 * 8)  # (3e5) 1e4, used time 20s
    args.reward_scale = 2 ** 0  # 0 ~ 200
    args.net_dim = 2 ** 6
    args.init_for_training()
    train_agent_mp(args)  # train_agent(args)
    exit()

    args.env = decorate_env(gym.make('LunarLander-v2'), if_print=True)
    args.break_step = int(1e5 * 8)  # (2e4) 1e5 (3e5), used time (200s) 1000s (2000s)
    args.reward_scale = 2 ** -1  # (-1000) -150 ~ 200 (285)
    args.net_dim = 2 ** 7
    args.init_for_training()
    train_agent_mp(args)  # train_agent(args)
    exit()


def train__car_racing__pixel_level_state2d():
    from AgentZoo import AgentPPO

    '''DEMO 4: Fix gym Box2D env CarRacing-v0 (pixel-level 2D-state, continuous action) using PPO'''
    import gym  # gym of OpenAI is not necessary for ElegantRL (even RL)
    gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'
    env = gym.make('CarRacing-v0')
    env = fix_car_racing_env(env)

    args = Arguments(rl_agent=AgentPPO, env=env, gpu_id=None)
    args.if_break_early = True
    args.eval_times2 = 1
    args.eval_times2 = 3  # CarRacing Env is so slow. The GPU-util is low while training CarRacing.
    args.rollout_num = 4  # (num, step, time) (8, 1e5, 1360) (4, 1e4, 1860)
    args.random_seed += 1943

    args.break_step = int(5e5 * 4)  # (1e5) 2e5 4e5 (8e5) used time (7,000s) 10ks 30ks (60ks)
    # Sometimes bad luck (5%), it reach 300 score in 5e5 steps and don't increase.
    # You just need to change the random seed and retrain.
    args.reward_scale = 2 ** -2  # (-1) 50 ~ 700 ~ 900 (1001)
    args.max_memo = 2 ** 11
    args.batch_size = 2 ** 7
    args.repeat_times = 2 ** 4
    args.net_dim = 2 ** 7
    args.max_step = 2 ** 10
    args.show_gap = 2 ** 8  # for Recorder
    args.init_for_training()
    train_agent_mp(args)  # train_agent(args)
    exit()


def run__fin_rl():
    env = FinanceMultiStockEnv()  # 2020-12-24

    from AgentZoo import AgentPPO

    args = Arguments(rl_agent=AgentPPO, env=env)
    args.eval_times1 = 1
    args.eval_times2 = 1
    args.rollout_num = 4
    args.if_break_early = False

    args.reward_scale = 2 ** 0  # (0) 1.1 ~ 15 (19)
    args.break_step = int(5e6 * 4)  # 5e6 (15e6) UsedTime: 4,000s (12,000s)
    args.net_dim = 2 ** 8
    args.max_step = 1699
    args.max_memo = 1699 * 16
    args.batch_size = 2 ** 10
    args.repeat_times = 2 ** 4
    args.init_for_training()
    train_agent_mp(args)  # train_agent(args)
    exit()

    # from AgentZoo import AgentModSAC
    #
    # args = Arguments(rl_agent=AgentModSAC, env=env)  # much slower than on-policy trajectory
    # args.eval_times1 = 1
    # args.eval_times2 = 2
    #
    # args.break_step = 2 ** 22  # UsedTime:
    # args.net_dim = 2 ** 7
    # args.max_memo = 2 ** 18
    # args.batch_size = 2 ** 8
    # args.init_for_training()
    # train_agent_mp(args)  # train_agent(args)


if __name__ == '__main__':
    train__demo()
    # train__off_policy()
    # train__on_policy()
    # train__discrete_action()
    # train__car_racing()
    print('Finish:', sys.argv[-1])
