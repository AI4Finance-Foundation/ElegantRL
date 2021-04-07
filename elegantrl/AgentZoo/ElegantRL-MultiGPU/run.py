import os
import gym
import time
import torch
import numpy as np
import numpy.random as rd
from copy import deepcopy
from elegantrl.replay import ReplayBuffer, ReplayBufferMP
from elegantrl.env import PreprocessEnv

"""[ElegantRL](https://github.com/AI4Finance-LLC/ElegantRL)"""


class Arguments:
    def __init__(self, agent=None, env=None, gpu_id=None, if_on_policy=False):
        self.agent = agent  # Deep Reinforcement Learning algorithm
        self.cwd = None  # current work directory. cwd is None means set it automatically
        self.env = env  # the environment for training
        self.env_eval = None  # the environment for evaluating
        self.gpu_id = gpu_id  # choose the GPU for running. gpu_id is None means set it automatically

        '''Arguments for training (off-policy)'''
        self.net_dim = 2 ** 8  # the network width
        self.batch_size = 2 ** 8  # num of transitions sampled from replay buffer.
        self.repeat_times = 2 ** 0  # repeatedly update network to keep critic's loss small
        self.target_step = 2 ** 10  # collect target_step, then update network
        self.max_memo = 2 ** 17  # capacity of replay buffer
        if if_on_policy:  # (on-policy)
            self.net_dim = 2 ** 9
            self.batch_size = 2 ** 9
            self.repeat_times = 2 ** 4
            self.target_step = 2 ** 12
            self.max_memo = self.target_step
        self.gamma = 0.99  # discount factor of future rewards
        self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256
        self.if_per = False  # Prioritized Experience Replay for sparse reward

        self.rollout_num = 2  # the number of rollout workers (larger is not always faster)
        self.num_threads = 8  # cpu_num for evaluate model, torch.set_num_threads(self.num_threads)

        '''Arguments for evaluate'''
        self.break_step = 2 ** 20  # break training after 'total_step > break_step'
        self.if_remove = True  # remove the cwd folder? (True, False, None:ask me)
        self.if_allow_break = True  # allow break training when reach goal (early termination)
        self.eval_gap = 2 ** 5  # evaluate the agent per eval_gap seconds
        self.eval_times1 = 2 ** 2  # evaluation times
        self.eval_times2 = 2 ** 4  # evaluation times if 'eval_reward > max_reward'
        self.random_seed = 0  # initialize random seed in self.init_before_training()

    def init_before_training(self, if_main=True):
        if self.agent is None:
            raise RuntimeError('\n| Why agent=None? Assignment args.agent = AgentXXX please.')
        if not hasattr(self.agent, 'init'):
            raise RuntimeError('\n| There should be agent=AgentXXX() instead of agent=AgentXXX')
        if self.env is None:
            raise RuntimeError('\n| Why env=None? Assignment args.env = XxxEnv() please.')
        if isinstance(self.env, str) or not hasattr(self.env, 'env_name'):
            raise RuntimeError('\n| What is env.env_name? use env=PreprocessEnv(env). It is a Wrapper.')

        '''set gpu_id automatically'''
        if self.gpu_id is None:  # set gpu_id automatically
            import sys
            self.gpu_id = sys.argv[-1][-4]
        else:
            self.gpu_id = str(self.gpu_id)
        if not self.gpu_id.isdigit():  # set gpu_id as '0' in default
            self.gpu_id = '0'

        '''set cwd automatically'''
        if self.cwd is None:
            agent_name = self.agent.__class__.__name__
            self.cwd = f'./{agent_name}/{self.env.env_name}_{self.gpu_id}'

        if if_main:
            print(f'| GPU id: {self.gpu_id}, cwd: {self.cwd}')

            import shutil  # remove history according to bool(if_remove)
            if self.if_remove is None:
                self.if_remove = bool(input("PRESS 'y' to REMOVE: {}? ".format(self.cwd)) == 'y')
            if self.if_remove:
                shutil.rmtree(self.cwd, ignore_errors=True)
                print("| Remove history")
            os.makedirs(self.cwd, exist_ok=True)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        torch.set_num_threads(self.num_threads)
        torch.set_default_dtype(torch.float32)
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)


'''single process training'''


def train_and_evaluate(args):
    args.init_before_training()

    '''basic arguments'''
    cwd = args.cwd
    env = args.env
    agent = args.agent
    gpu_id = args.gpu_id  # necessary for Evaluator?

    '''training arguments'''
    net_dim = args.net_dim
    max_memo = args.max_memo
    break_step = args.break_step
    batch_size = args.batch_size
    target_step = args.target_step
    repeat_times = args.repeat_times
    if_break_early = args.if_allow_break
    if_per = args.if_per
    gamma = args.gamma
    reward_scale = args.reward_scale

    '''evaluating arguments'''
    eval_gap = args.eval_gap
    eval_times1 = args.eval_times1
    eval_times2 = args.eval_times2
    if args.env_eval is not None:
        env_eval = args.env_eval
    elif args.env_eval in set(gym.envs.registry.env_specs.keys()):
        env_eval = PreprocessEnv(gym.make(env.env_name))
    else:
        env_eval = deepcopy(env)

    del args  # In order to show these hyper-parameters clearly, I put them above.

    '''init: environment'''
    max_step = env.max_step
    state_dim = env.state_dim
    action_dim = env.action_dim
    if_discrete = env.if_discrete

    '''init: Agent, ReplayBuffer, Evaluator'''
    agent.init(net_dim, state_dim, action_dim, if_per)
    if_on_policy = getattr(agent, 'if_on_policy', False)

    buffer = ReplayBuffer(max_len=max_memo + max_step, state_dim=state_dim, action_dim=1 if if_discrete else action_dim,
                          if_on_policy=if_on_policy, if_per=if_per, if_gpu=True)

    evaluator = Evaluator(cwd=cwd, agent_id=gpu_id, device=agent.device, env=env_eval,
                          eval_gap=eval_gap, eval_times1=eval_times1, eval_times2=eval_times2, )

    '''prepare for training'''
    agent.state = env.reset()
    if if_on_policy:
        steps = 0
    else:  # explore_before_training for off-policy
        with torch.no_grad():  # update replay buffer
            steps = explore_before_training(env, buffer, target_step, reward_scale, gamma)

        agent.update_net(buffer, target_step, batch_size, repeat_times)  # pre-training and hard update
        agent.act_target.load_state_dict(agent.act.state_dict()) if getattr(agent, 'act_target', None) else None
        agent.cri_target.load_state_dict(agent.cri.state_dict()) if getattr(agent, 'cri_target', None) else None
    total_step = steps

    '''start training'''
    if_reach_goal = False
    while not ((if_break_early and if_reach_goal)
               or total_step > break_step
               or os.path.exists(f'{cwd}/stop')):
        steps = agent.explore_env(env, buffer, target_step, reward_scale, gamma)
        total_step += steps

        obj_a, obj_c = agent.update_net(buffer, target_step, batch_size, repeat_times)

        if_reach_goal = evaluator.evaluate_save(agent.act, steps, obj_a, obj_c)
        evaluator.draw_plot()

    print(f'| SavedDir: {cwd}\n| UsedTime: {time.time() - evaluator.start_time:.0f}')


'''multiprocessing training'''


def train_and_evaluate_mp(args):
    act_workers = args.rollout_num
    import multiprocessing as mp  # Python built-in multiprocessing library

    pipe1_eva, pipe2_eva = mp.Pipe()  # Pipe() for Process mp_evaluate_agent()
    pipe2_exp_list = list()  # Pipe() for Process mp_explore_in_env()

    process_train = mp.Process(target=mp_train, args=(args, pipe2_eva, pipe2_exp_list))
    process_evaluate = mp.Process(target=mp_evaluate, args=(args, pipe1_eva))
    process = [process_train, process_evaluate]

    for worker_id in range(act_workers):
        exp_pipe1, exp_pipe2 = mp.Pipe(duplex=True)
        pipe2_exp_list.append(exp_pipe1)
        process.append(mp.Process(target=mp_explore, args=(args, exp_pipe2, worker_id)))

    [p.start() for p in process]
    process_evaluate.join()
    process_train.join()
    [p.terminate() for p in process]


def mp_train(args, pipe1_eva, pipe1_exp_list):
    args.init_before_training(if_main=False)

    '''basic arguments'''
    env = args.env
    cwd = args.cwd
    agent = args.agent
    rollout_num = args.rollout_num

    '''training arguments'''
    net_dim = args.net_dim
    max_memo = args.max_memo
    break_step = args.break_step
    batch_size = args.batch_size
    target_step = args.target_step
    repeat_times = args.repeat_times
    if_break_early = args.if_allow_break
    if_per = args.if_per
    del args  # In order to show these hyper-parameters clearly, I put them above.

    '''init: environment'''
    max_step = env.max_step
    state_dim = env.state_dim
    action_dim = env.action_dim
    if_discrete = env.if_discrete

    '''init: Agent, ReplayBuffer'''
    agent.init(net_dim, state_dim, action_dim, if_per)
    if_on_policy = getattr(agent, 'if_on_policy', False)

    '''send'''
    pipe1_eva.send(agent.act)  # send
    # act = pipe2_eva.recv()  # recv

    buffer_mp = ReplayBufferMP(max_len=max_memo + max_step * rollout_num, if_on_policy=if_on_policy,
                               state_dim=state_dim, action_dim=1 if if_discrete else action_dim,
                               rollout_num=rollout_num, if_gpu=True, if_per=if_per)

    '''prepare for training'''
    if if_on_policy:
        steps = 0
    else:  # explore_before_training for off-policy
        with torch.no_grad():  # update replay buffer
            steps = 0
            for i in range(rollout_num):
                pipe1_exp = pipe1_exp_list[i]

                # pipe2_exp.send((buffer.buf_state[:buffer.now_len], buffer.buf_other[:buffer.now_len]))
                buf_state, buf_other = pipe1_exp.recv()

                steps += len(buf_state)
                buffer_mp.extend_buffer(buf_state, buf_other, i)

        agent.update_net(buffer_mp, target_step, batch_size, repeat_times)  # pre-training and hard update
        agent.act_target.load_state_dict(agent.act.state_dict()) if getattr(env, 'act_target', None) else None
        agent.cri_target.load_state_dict(agent.cri.state_dict()) if getattr(env, 'cri_target', None) in dir(
            agent) else None
    total_step = steps
    '''send'''
    pipe1_eva.send((agent.act, steps, 0, 0.5))  # send
    # act, steps, obj_a, obj_c = pipe2_eva.recv()  # recv

    '''start training'''
    if_solve = False
    while not ((if_break_early and if_solve)
               or total_step > break_step
               or os.path.exists(f'{cwd}/stop')):
        '''update ReplayBuffer'''
        steps = 0  # send by pipe1_eva
        for i in range(rollout_num):
            pipe1_exp = pipe1_exp_list[i]
            '''send'''
            pipe1_exp.send(agent.act)
            # agent.act = pipe2_exp.recv()
            '''recv'''
            # pipe2_exp.send((buffer.buf_state[:buffer.now_len], buffer.buf_other[:buffer.now_len]))
            buf_state, buf_other = pipe1_exp.recv()

            steps += len(buf_state)
            buffer_mp.extend_buffer(buf_state, buf_other, i)
        total_step += steps

        '''update network parameters'''
        obj_a, obj_c = agent.update_net(buffer_mp, target_step, batch_size, repeat_times)

        '''saves the agent with max reward'''
        '''send'''
        pipe1_eva.send((agent.act, steps, obj_a, obj_c))
        # q_i_eva_get = pipe2_eva.recv()

        if_solve = pipe1_eva.recv()

        if pipe1_eva.poll():
            '''recv'''
            # pipe2_eva.send(if_solve)
            if_solve = pipe1_eva.recv()

    buffer_mp.print_state_norm(env.neg_state_avg if hasattr(env, 'neg_state_avg') else None,
                               env.div_state_std if hasattr(env, 'div_state_std') else None)  # 2020-12-12

    '''send'''
    pipe1_eva.send('stop')
    # q_i_eva_get = pipe2_eva.recv()
    time.sleep(4)


def mp_explore(args, pipe2_exp, worker_id):
    args.init_before_training(if_main=False)

    '''basic arguments'''
    env = args.env
    agent = args.agent
    rollout_num = args.rollout_num

    '''training arguments'''
    net_dim = args.net_dim
    max_memo = args.max_memo
    target_step = args.target_step
    gamma = args.gamma
    if_per = args.if_per
    reward_scale = args.reward_scale

    random_seed = args.random_seed
    torch.manual_seed(random_seed + worker_id)
    np.random.seed(random_seed + worker_id)
    del args  # In order to show these hyper-parameters clearly, I put them above.

    '''init: environment'''
    max_step = env.max_step
    state_dim = env.state_dim
    action_dim = env.action_dim
    if_discrete = env.if_discrete

    '''init: Agent, ReplayBuffer'''
    agent.init(net_dim, state_dim, action_dim, if_per)
    agent.state = env.reset()

    if_on_policy = getattr(agent, 'if_on_policy', False)
    buffer = ReplayBuffer(max_len=max_memo // rollout_num + max_step, if_on_policy=if_on_policy,
                          state_dim=state_dim, action_dim=1 if if_discrete else action_dim,
                          if_per=if_per, if_gpu=False)

    '''start exploring'''
    exp_step = target_step // rollout_num
    with torch.no_grad():
        if not if_on_policy:
            explore_before_training(env, buffer, exp_step, reward_scale, gamma)

            buffer.update_now_len_before_sample()

            pipe2_exp.send((buffer.buf_state[:buffer.now_len], buffer.buf_other[:buffer.now_len]))
            # buf_state, buf_other = pipe1_exp.recv()

            buffer.empty_buffer_before_explore()

        while True:
            agent.explore_env(env, buffer, exp_step, reward_scale, gamma)

            buffer.update_now_len_before_sample()

            pipe2_exp.send((buffer.buf_state[:buffer.now_len], buffer.buf_other[:buffer.now_len]))
            # buf_state, buf_other = pipe1_exp.recv()

            buffer.empty_buffer_before_explore()

            # pipe1_exp.send(agent.act)
            agent.act = pipe2_exp.recv()


def mp_evaluate(args, pipe2_eva):
    args.init_before_training(if_main=True)

    '''basic arguments'''
    cwd = args.cwd
    env = args.env
    env_eval = env if args.env_eval is None else args.env_eval
    agent_id = args.gpu_id

    '''evaluating arguments'''
    eval_gap = args.eval_gap
    eval_times1 = args.eval_times1
    eval_times2 = args.eval_times2
    del args  # In order to show these hyper-parameters clearly, I put them above.

    '''init: Evaluator'''
    evaluator = Evaluator(cwd=cwd, agent_id=agent_id, device=torch.device("cpu"), env=env_eval,
                          eval_gap=eval_gap, eval_times1=eval_times1, eval_times2=eval_times2, )  # build Evaluator

    '''act_cpu without gradient for pipe1_eva'''
    # pipe1_eva.send(agent.act)
    act = pipe2_eva.recv()

    act_cpu = deepcopy(act).to(torch.device("cpu"))  # for pipe1_eva
    [setattr(param, 'requires_grad', False) for param in act_cpu.parameters()]

    '''start evaluating'''
    with torch.no_grad():  # speed up running
        act, steps, obj_a, obj_c = pipe2_eva.recv()  # pipe2_eva (act, steps, obj_a, obj_c)

        if_loop = True
        while if_loop:
            '''update actor'''
            while not pipe2_eva.poll():  # wait until pipe2_eva not empty
                time.sleep(1)
            steps_sum = 0
            while pipe2_eva.poll():  # receive the latest object from pipe
                '''recv'''
                # pipe1_eva.send((agent.act, steps, obj_a, obj_c))
                # pipe1_eva.send('stop')
                q_i_eva_get = pipe2_eva.recv()

                if q_i_eva_get == 'stop':
                    if_loop = False
                    break
                act, steps, obj_a, obj_c = q_i_eva_get
                steps_sum += steps
            act_cpu.load_state_dict(act.state_dict())
            if_solve = evaluator.evaluate_save(act_cpu, steps_sum, obj_a, obj_c)
            '''send'''
            pipe2_eva.send(if_solve)
            # if_solve = pipe1_eva.recv()

            evaluator.draw_plot()

    print(f'| SavedDir: {cwd}\n| UsedTime: {time.time() - evaluator.start_time:.0f}')

    while pipe2_eva.poll():  # empty the pipe
        pipe2_eva.recv()


'''utils'''


class Evaluator:
    def __init__(self, cwd, agent_id, eval_times1, eval_times2, eval_gap, env, device):
        self.recorder = [(0., -np.inf, 0., 0., 0.), ]  # total_step, r_avg, r_std, obj_a, obj_c
        self.r_max = -np.inf
        self.total_step = 0

        self.cwd = cwd  # constant
        self.device = device
        self.agent_id = agent_id
        self.eval_gap = eval_gap
        self.eval_times1 = eval_times1
        self.eval_times2 = eval_times2
        self.env = env
        self.target_return = env.target_return

        self.used_time = None
        self.start_time = time.time()
        self.eval_time = -1  # a early time
        print(f"{'ID':>2}  {'Step':>8}  {'MaxR':>8} |"
              f"{'avgR':>8}  {'stdR':>8}   {'objA':>8}  {'objC':>8} |"
              f"{'avgS':>6}  {'stdS':>4}")

    def evaluate_save(self, act, steps, obj_a, obj_c) -> bool:
        self.total_step += steps  # update total training steps

        if time.time() - self.eval_time > self.eval_gap:
            self.eval_time = time.time()

            rewards_steps_list = [get_episode_return(self.env, act, self.device) for _ in range(self.eval_times1)]
            r_avg, r_std, s_avg, s_std = self.get_r_avg_std_s_avg_std(rewards_steps_list)

            if r_avg > self.r_max:  # evaluate actor twice to save CPU Usage and keep precision
                rewards_steps_list += [get_episode_return(self.env, act, self.device)
                                       for _ in range(self.eval_times2 - self.eval_times1)]
                r_avg, r_std, s_avg, s_std = self.get_r_avg_std_s_avg_std(rewards_steps_list)
            if r_avg > self.r_max:  # save checkpoint with highest episode return
                self.r_max = r_avg  # update max reward (episode return)

                '''save actor.pth'''
                act_save_path = f'{self.cwd}/actor.pth'
                torch.save(act.state_dict(), act_save_path)
                print(f"{self.agent_id:<2}  {self.total_step:8.2e}  {self.r_max:8.2f} |")  # save policy and print

            self.recorder.append((self.total_step, r_avg, r_std, obj_a, obj_c))  # update recorder

            if_reach_goal = bool(self.r_max > self.target_return)  # check if_reach_goal
            if if_reach_goal and self.used_time is None:
                self.used_time = int(time.time() - self.start_time)
                print(f"{'ID':>2}  {'Step':>8}  {'TargetR':>8} |"
                      f"{'avgR':>8}  {'stdR':>8}   {'UsedTime':>8}  ########\n"
                      f"{self.agent_id:<2}  {self.total_step:8.2e}  {self.target_return:8.2f} |"
                      f"{r_avg:8.2f}  {r_std:8.2f}   {self.used_time:>8}  ########")

            print(f"{self.agent_id:<2}  {self.total_step:8.2e}  {self.r_max:8.2f} |"
                  f"{r_avg:8.2f}  {r_std:8.2f}   {obj_a:8.2f}  {obj_c:8.2f} |"
                  f"{s_avg:6.0f}  {s_std:4.0f}")
        else:
            if_reach_goal = False
        return if_reach_goal

    def draw_plot(self):
        if len(self.recorder) == 0:
            print("| save_npy_draw_plot() WARNNING: len(self.recorder)==0")
            return None

        '''convert to array and save as npy'''
        np.save('%s/recorder.npy' % self.cwd, self.recorder)

        '''draw plot and save as png'''
        train_time = int(time.time() - self.start_time)
        total_step = int(self.recorder[-1][0])
        save_title = f"plot_step_time_maxR_{int(total_step)}_{int(train_time)}_{self.r_max:.3f}"

        save_learning_curve(self.recorder, self.cwd, save_title)

    @staticmethod
    def get_r_avg_std_s_avg_std(rewards_steps_list):
        rewards_steps_ary = np.array(rewards_steps_list)
        r_avg, s_avg = rewards_steps_ary.mean(axis=0)  # average of episode return and episode step
        r_std, s_std = rewards_steps_ary.std(axis=0)  # standard dev. of episode return and episode step
        return r_avg, r_std, s_avg, s_std


def get_episode_return(env, act, device) -> (float, int):
    episode_return = 0.0  # sum of rewards in an episode
    episode_step = 1
    max_step = env.max_step
    if_discrete = env.if_discrete

    state = env.reset()
    for episode_step in range(max_step):
        s_tensor = torch.as_tensor((state,), device=device)
        a_tensor = act(s_tensor)
        if if_discrete:
            a_tensor = a_tensor.argmax(dim=1)
        action = a_tensor.cpu().numpy()[0]  # not need detach(), because with torch.no_grad() outside
        state, reward, done, _ = env.step(action)
        episode_return += reward
        if done:
            break
    episode_return = getattr(env, 'episode_return', episode_return)
    return episode_return, episode_step + 1


def save_learning_curve(recorder, cwd='.', save_title='learning curve'):
    recorder = np.array(recorder)  # recorder_ary.append((self.total_step, r_avg, r_std, obj_a, obj_c))
    steps = recorder[:, 0]  # x-axis is training steps
    r_avg = recorder[:, 1]
    r_std = recorder[:, 2]
    obj_a = recorder[:, 3]
    obj_c = recorder[:, 4]

    '''plot subplots'''
    import matplotlib as mpl
    mpl.use('Agg')
    """Generating matplotlib graphs without a running X server [duplicate]
    write `mpl.use('Agg')` before `import matplotlib.pyplot as plt`
    https://stackoverflow.com/a/4935945/9293137
    """
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2)

    axs0 = axs[0]
    axs0.cla()
    color0 = 'lightcoral'
    axs0.set_xlabel('Total Steps')
    axs0.set_ylabel('Episode Return')
    axs0.plot(steps, r_avg, label='Episode Return', color=color0)
    axs0.fill_between(steps, r_avg - r_std, r_avg + r_std, facecolor=color0, alpha=0.3)

    ax11 = axs[1]
    ax11.cla()
    color11 = 'royalblue'
    axs0.set_xlabel('Total Steps')
    ax11.set_ylabel('objA', color=color11)
    ax11.plot(steps, obj_a, label='objA', color=color11)
    ax11.tick_params(axis='y', labelcolor=color11)

    ax12 = axs[1].twinx()
    color12 = 'darkcyan'
    ax12.set_ylabel('objC', color=color12)
    ax12.fill_between(steps, obj_c, facecolor=color12, alpha=0.2, )
    ax12.tick_params(axis='y', labelcolor=color12)

    '''plot save'''
    plt.title(save_title, y=2.3)
    plt.savefig(f"{cwd}/plot_learning_curve.jpg")
    plt.close('all')  # avoiding warning about too many open figures, rcParam `figure.max_open_warning`
    # plt.show()  # if use `mpl.use('Agg')` to draw figures without GUI, then plt can't plt.show()


def explore_before_training(env, buffer, target_step, reward_scale, gamma) -> int:
    # just for off-policy. Because on-policy don't explore before training.
    if_discrete = env.if_discrete
    action_dim = env.action_dim

    state = env.reset()
    steps = 0

    while steps < target_step:
        action = rd.randint(action_dim) if if_discrete else rd.uniform(-1, 1, size=action_dim)
        next_state, reward, done, _ = env.step(action)
        steps += 1

        scaled_reward = reward * reward_scale
        mask = 0.0 if done else gamma
        other = (scaled_reward, mask, action) if if_discrete else (scaled_reward, mask, *action)
        buffer.append_buffer(state, other)

        state = env.reset() if done else next_state
    return steps
