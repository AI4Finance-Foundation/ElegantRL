import os
import time
import torch
import numpy as np
import numpy.random as rd

'''DEMO'''


class Arguments:
    def __init__(self, agent_rl=None, env=None, gpu_id=None, if_on_policy=False):
        self.agent_rl = agent_rl  # Deep Reinforcement Learning algorithm
        self.gpu_id = gpu_id  # choose the GPU for running. gpu_id is None means set it automatically
        self.cwd = None  # current work directory. cwd is None means set it automatically
        self.env = env  # the environment for training

        '''Arguments for training (off-policy)'''
        self.net_dim = 2 ** 8  # the network width
        self.batch_size = 2 ** 7  # num of transitions sampled from replay buffer.
        self.repeat_times = 2 ** 0  # repeatedly update network to keep critic's loss small
        self.max_memo = 2 ** 17  # capacity of replay buffer
        if if_on_policy:  # (on-policy)
            self.net_dim = 2 ** 9
            self.batch_size = 2 ** 8
            self.repeat_times = 2 ** 4
            self.max_memo = 2 ** 12
        self.max_step = 2 ** 10  # max steps in one training episode
        self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256
        self.gamma = 0.99  # discount factor of future rewards
        self.rollout_num = 2  # the number of rollout workers (larger is not always faster)
        self.num_threads = 4  # cpu_num for evaluate model, torch.set_num_threads(self.num_threads)

        '''Arguments for evaluate'''
        self.if_remove = True  # remove the cwd folder? (True, False, None:ask me)
        self.if_break_early = True  # break training after 'eval_reward > target reward'
        self.break_step = 2 ** 20  # break training after 'total_step > break_step'
        self.eval_times = 2 ** 3  # evaluation times if 'eval_reward > target_reward'
        self.show_gap = 2 ** 8  # show the Reward and Loss value per show_gap seconds
        self.random_seed = 0  # initialize random seed in self.init_before_training(

    def init_before_training(self):
        import sys
        self.gpu_id = sys.argv[-1][-4] if self.gpu_id is None else str(self.gpu_id)
        self.gpu_id = self.gpu_id if self.gpu_id.isdigit() else '0'
        self.cwd = f'./{self.agent_rl.__name__}/{self.env.env_name}_{self.gpu_id}' if self.cwd is None else self.cwd
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


def run__demo():
    import gym  # don't worry about 'WARN: Box bound precision lowered by casting to float32'
    import AgentZoo
    from AgentEnv import decorate_env
    args = Arguments(agent_rl=None, env=None, gpu_id=None)  # see Arguments() to see hyper-parameters

    '''DEMO 1: Discrete action env: CartPole-v0 of gym'''
    args.agent_rl = AgentZoo.AgentDoubleDQN  # choose an DRL algorithm
    # args.env = decorate_env(env=gym.make('CartPole-v0'))
    # args.net_dim = 2 ** 7  # change a default hyper-parameters
    args.env = decorate_env(env=gym.make('LunarLander-v2'))
    args.net_dim = 2 ** 8  # change a default hyper-parameters
    train_and_evaluate(args)
    exit()

    '''DEMO 2: Continuous action env: gym.box2D'''
    args.env = decorate_env(env=gym.make('Pendulum-v0'))
    args.env.target_reward = -200  # set target_reward manually for env 'Pendulum-v0'
    # args.env = decorate_env(env=gym.make('LunarLanderContinuous-v2'))
    # args.env = decorate_env(env=gym.make('BipedalWalker-v3'))  # recommend args.gamma = 0.95
    args.agent_rl = AgentZoo.AgentSAC  # off-policy
    train_and_evaluate(args)
    exit()

    args = Arguments(if_on_policy=True)  # on-policy has different hyper-parameters from off-policy
    # args.env = decorate_env(env=gym.make('Pendulum-v0'))
    args.env = decorate_env(env=gym.make('LunarLanderContinuous-v2'))
    # args.env = decorate_env(env=gym.make('BipedalWalker-v3'))  # recommend args.gamma = 0.95
    args.agent_rl = AgentZoo.AgentPPO  # on-policy
    train_and_evaluate(args)
    exit()

    '''DEMO 3: Custom Continuous action env: FinanceStock-v1'''
    from AgentEnv import FinanceMultiStockEnv
    args = Arguments(if_on_policy=True)
    args.env = FinanceMultiStockEnv()  # a standard env for ElegantRL, not need decorate_env()
    args.agent_rl = AgentZoo.AgentPPO  # PPO+GAE (on-policy)

    args.break_step = int(5e6 * 4)  # 5e6 (15e6) UsedTime 3,000s (9,000s)
    args.net_dim = 2 ** 8
    args.max_step = 1699
    args.max_memo = (args.max_step - 1) * 16
    args.batch_size = 2 ** 11
    args.repeat_times = 2 ** 4
    train_and_evaluate(args)
    exit()


'''single process training'''


def train_and_evaluate(args):
    args.init_before_training()

    agent_rl = args.agent_rl  # basic arguments
    agent_id = args.gpu_id
    env = args.env
    cwd = args.cwd

    gamma = args.gamma  # training arguments
    net_dim = args.net_dim
    max_memo = args.max_memo
    max_step = args.max_step
    batch_size = args.batch_size
    repeat_times = args.repeat_times
    reward_scale = args.reward_scale

    show_gap = args.show_gap  # evaluate arguments
    eval_times = args.eval_times
    break_step = args.break_step
    if_break_early = args.if_break_early
    del args  # In order to show these hyper-parameters clearly, I put them above.

    '''init: env'''
    state_dim = env.state_dim
    action_dim = env.action_dim
    if_discrete = env.if_discrete
    from copy import deepcopy  # built-in library of Python
    env_eval = deepcopy(env)
    del deepcopy

    agent = agent_rl(net_dim, state_dim, action_dim)  # build AgentRL
    agent.state = env.reset()
    evaluator = Evaluator(cwd=cwd, agent_id=agent_id, device=agent.device, env=env_eval,
                          eval_times=eval_times, show_gap=show_gap)  # build Evaluator

    if_on_policy = agent_rl.__name__ in {'AgentPPO', 'AgentGaePPO'}
    buffer = ReplayBuffer(max_memo, state_dim, if_on_policy=if_on_policy,
                          action_dim=1 if if_discrete else action_dim)  # build experience replay buffer
    if if_on_policy:
        steps = 0
    else:
        with torch.no_grad():  # update replay buffer
            steps = explore_before_train(env, buffer, max_step, reward_scale, gamma)
        agent.update_policy(buffer, max_step, batch_size, repeat_times)  # pre-training and hard update
        agent.act_target.load_state_dict(agent.act.state_dict()) if 'act_target' in dir(agent) else None
    total_step = steps

    if_solve = False
    while not ((if_break_early and if_solve)
               or total_step > break_step
               or os.path.exists(f'{cwd}/stop')):
        with torch.no_grad():  # speed up running
            steps = agent.update_buffer(env, buffer, max_step, reward_scale, gamma)
        total_step += steps

        obj_a, obj_c = agent.update_policy(buffer, max_step, batch_size, repeat_times)

        with torch.no_grad():  # speed up running
            if_solve = evaluator.evaluate_act_and_save_checkpoint(agent.act, steps, obj_a, obj_c)


'''multiprocessing training'''


def train_and_evaluate__multiprocessing(args):
    args.init_before_training()

    import multiprocessing as mp  # Python built-in multiprocessing library

    pipe1_eva, pipe2_eva = mp.Pipe()
    process = [
        mp.Process(target=mp__update_params, args=(args, pipe1_eva)),
        mp.Process(target=mp_evaluate_agent, args=(args, pipe2_eva)),
    ]

    [p.start() for p in process]
    [p.join() for p in process]
    print('\n')


def mp__update_params(args, pipe1_eva):
    agent_rl = args.agent_rl  # basic arguments
    env = args.env
    cwd = args.cwd

    gamma = args.gamma  # training arguments
    net_dim = args.net_dim
    max_memo = args.max_memo
    max_step = args.max_step
    batch_size = args.batch_size
    repeat_times = args.repeat_times
    reward_scale = args.reward_scale

    break_step = args.break_step
    if_break_early = args.if_break_early
    del args  # In order to show these hyper-parameters clearly, I put them above.

    '''init: env'''
    state_dim = env.state_dim
    action_dim = env.action_dim
    if_discrete = env.if_discrete

    agent = agent_rl(net_dim, state_dim, action_dim)  # build AgentRL
    agent.state = env.reset()

    '''act_cpu without gradient for pipe1_eva'''
    from copy import deepcopy  # built-in library of Python
    act_cpu = deepcopy(agent.act).to(torch.device("cpu"))  # for pipe1_eva
    act_cpu.eval()
    [setattr(param, 'requires_grad', False) for param in act_cpu.parameters()]

    if_on_policy = agent_rl.__name__ in {'AgentPPO', 'AgentGaePPO'}
    buffer = ReplayBuffer(max_memo, state_dim, if_on_policy=if_on_policy,
                          action_dim=1 if if_discrete else action_dim)  # build experience replay buffer
    if if_on_policy:
        steps = 0
    else:
        with torch.no_grad():  # update replay buffer
            steps = explore_before_train(env, buffer, max_step, reward_scale, gamma)
        agent.update_policy(buffer, max_step, batch_size, repeat_times)  # pre-training and hard update
        agent.act_target.load_state_dict(agent.act.state_dict()) if 'act_target' in dir(agent) else None
    total_step = steps
    pipe1_eva.send((act_cpu, steps, 0, 0.5))  # pipe1_eva (act, steps, obj_a, obj_c)

    if_solve = False
    while not ((if_break_early and if_solve)
               or total_step > break_step
               or os.path.exists(f'{cwd}/stop')):
        with torch.no_grad():  # speed up running
            steps = agent.update_buffer(env, buffer, max_step, reward_scale, gamma)
        total_step += steps

        obj_a, obj_c = agent.update_policy(buffer, max_step, batch_size, repeat_times)

        '''saves the agent with max reward'''
        act_cpu.load_state_dict(agent.act.state_dict())
        pipe1_eva.send((act_cpu, steps, obj_a, obj_c))  # pipe1_eva act_cpu
        if_solve = pipe1_eva.recv()

        if pipe1_eva.poll():
            if_solve = pipe1_eva.recv()  # pipe1_eva if_solve

    buffer.print_state_norm(env.neg_state_avg if hasattr(env, 'neg_state_avg') else None,
                            env.div_state_std if hasattr(env, 'div_state_std') else None)  # 2020-12-12
    pipe1_eva.send('stop')  # eva_pipe stop  # send to mp_evaluate_agent
    time.sleep(4)
    # print('; quit: params')


def mp_evaluate_agent(args, pipe2_eva):
    env = args.env
    cwd = args.cwd
    agent_id = args.gpu_id
    show_gap = args.show_gap  # evaluate arguments
    eval_times = args.eval_times

    from copy import deepcopy  # built-in library of Python
    env_eval = deepcopy(env)
    del deepcopy

    device = torch.device("cpu")
    evaluator = Evaluator(cwd=cwd, agent_id=agent_id, device=device, env=env_eval,
                          eval_times=eval_times, show_gap=show_gap)  # build Evaluator

    with torch.no_grad():  # speed up running
        act, steps, obj_a, obj_c = pipe2_eva.recv()  # pipe2_eva (act, steps, obj_a, obj_c)

        if_loop = True
        while if_loop:
            '''update actor'''
            while not pipe2_eva.poll():  # wait until pipe2_eva not empty
                time.sleep(1)
            steps_sum = 0
            while pipe2_eva.poll():  # receive the latest object from pipe
                q_i_eva_get = pipe2_eva.recv()  # pipe2_eva act
                if q_i_eva_get == 'stop':
                    if_loop = False
                    break
                act, steps, obj_a, obj_c = q_i_eva_get
                steps_sum += steps
            if_solve = evaluator.evaluate_act_and_save_checkpoint(act, steps_sum, obj_a, obj_c)
            pipe2_eva.send(if_solve)

            evaluator.save_npy__draw_plot()

    '''save the model, rename the directory'''
    new_cwd = cwd[:-2] + f'_{evaluator.r_max:.2f}' + cwd[-2:]
    if not os.path.exists(new_cwd):  # 2020-12-12
        os.rename(cwd, new_cwd)
        cwd = new_cwd
    else:
        print(f'| SavedDir: {new_cwd}    WARNING: file exit')
    print(f'| SavedDir: {cwd}\n'
          f'| UsedTime: {time.time() - evaluator.start_time:.0f}')

    while pipe2_eva.poll():  # empty the pipe
        pipe2_eva.recv()
    # print('; quit: evaluate')


'''utils'''


def explore_before_train(env, buffer, target_step, reward_scale, gamma):
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
        buffer.append_memo(state, other)

        state = env.reset() if done else next_state
    return steps


class ReplayBuffer:
    def __init__(self, max_len, state_dim, action_dim, if_on_policy):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.now_len = 0
        self.next_idx = 0
        self.if_full = False
        self.action_dim = action_dim  # for self.sample_for_ppo(

        if if_on_policy:
            other_dim = 1 + 1 + action_dim * 2
            self.buf_other = np.empty((max_len, other_dim), dtype=np.float32)
            self.buf_state = np.empty((max_len, state_dim), dtype=np.float32)
            self.append_memo = self.append_memo__on_policy
        else:
            other_dim = 1 + 1 + action_dim
            self.buf_other = torch.empty((max_len, other_dim), dtype=torch.float32, device=self.device)
            self.buf_state = torch.empty((max_len, state_dim), dtype=torch.float32, device=self.device)
            self.append_memo = self.append_memo__off_policy

    def append_memo__off_policy(self, state, other):
        self.buf_state[self.next_idx, :] = torch.as_tensor(state, device=self.device)
        self.buf_other[self.next_idx] = torch.as_tensor(other, device=self.device)

        self.next_idx += 1
        if self.next_idx >= self.max_len:
            self.if_full = True
            self.next_idx = 0

    def append_memo__on_policy(self, state, other):  # for AgentPPO.update_buffer(
        self.buf_state[self.next_idx] = state
        self.buf_other[self.next_idx] = other

        self.next_idx += 1
        if self.next_idx >= self.max_len:
            self.if_full = True
            self.next_idx = 0

    def random_sample(self, batch_size):
        indices = torch.randint(self.now_len - 1, size=(batch_size,), device=self.device)
        r_m_a = self.buf_other[indices]
        return (r_m_a[:, 0:1],  # reward
                r_m_a[:, 1:2],  # mask = 0.0 if done else gamma
                r_m_a[:, 2:],  # action
                self.buf_state[indices],  # state
                self.buf_state[indices + 1])  # next_state

    def sample_for_ppo(self):
        all_other = torch.as_tensor(self.buf_other[:self.now_len], device=self.device)
        return (all_other[:, 0:1],  # reward
                all_other[:, 1:2],  # mask = 0.0 if done else gamma
                all_other[:, 2:2 + self.action_dim],  # action
                all_other[:, 2 + self.action_dim:],  # noise
                torch.as_tensor(self.buf_state[:self.now_len], device=self.device))  # state

    def update__now_len__before_sample(self):
        self.now_len = self.max_len if self.if_full else self.next_idx

    def empty_memories__before_explore(self):
        self.next_idx = 0
        self.now_len = 0
        self.if_full = False

    def print_state_norm(self, neg_avg=None, div_std=None):  # non-essential
        max_sample_size = 2 ** 14

        '''check if pass'''
        state_shape = self.buf_state.shape
        if len(state_shape) > 2 or state_shape[1] > 64:
            print(f"| print_state_norm(): state_dim: {state_shape:.0f} is too large to print its norm. ")
            return None

        '''sample state'''
        indices = np.arange(self.now_len)
        rd.shuffle(indices)
        indices = indices[:max_sample_size]  # len(indices) = min(self.now_len, max_sample_size)

        batch_state = self.buf_state[indices]

        '''compute state norm'''
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
        fix_std = ((np.max(batch_state, axis=0) - np.min(batch_state, axis=0)) / 6 + ary_std) / 2

        if neg_avg is not None:  # norm transfer
            ary_avg = ary_avg - neg_avg / div_std
            ary_std = fix_std / div_std

        print(f"| print_norm: state_avg, state_fix_std")
        print(f"| avg = np.{repr(ary_avg).replace('=float32', '=np.float32')}")
        print(f"| std = np.{repr(ary_std).replace('=float32', '=np.float32')}")


class Evaluator:
    def __init__(self, cwd, agent_id, eval_times, show_gap, env, device):
        self.recorder = [(0., -np.inf, 0., 0., 0.), ]  # total_step, r_avg, r_std, obj_a, obj_c
        self.r_max = -np.inf
        self.total_step = 0

        self.cwd = cwd  # constant
        self.device = device
        self.agent_id = agent_id
        self.show_gap = show_gap
        self.eva_times = eval_times
        self.env = env
        self.target_reward = env.target_reward

        self.used_time = None
        self.start_time = time.time()
        self.print_time = time.time()
        print(f"{'ID':>2}  {'Step':>8}  {'MaxR':>8} |{'avgR':>8}  {'stdR':>8}   {'objA':>8}  {'objC':>8}")

        import matplotlib as mpl  # draw figure in Terminal
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        # plt.style.use('ggplot')
        self.plt = plt

    def evaluate_act_and_save_checkpoint(self, act, steps, obj_a, obj_c):
        reward_list = [get_episode_return(self.env, act, self.device) for _ in range(self.eva_times)]
        r_avg = np.average(reward_list)  # episode return average
        r_std = float(np.std(reward_list))  # episode return std

        if r_avg > self.r_max:  # save checkpoint with highest episode return
            self.r_max = r_avg

            act_save_path = f'{self.cwd}/actor.pth'
            torch.save(act.state_dict(), act_save_path)
            print(f"{self.agent_id:<2}  {self.total_step:8.2e}  {self.r_max:8.2f} |")
        self.total_step += steps  # update total training steps
        self.recorder.append((self.total_step, r_avg, r_std, obj_a, obj_c))  # update recorder

        if_solve = bool(self.r_max > self.target_reward)  # check if_solve
        if if_solve and self.used_time is None:
            self.used_time = int(time.time() - self.start_time)
            print(f"{'ID':>2}  {'Step':>8}  {'TargetR':>8} |"
                  f"{'avgR':>8}  {'stdR':>8}   {'UsedTime':>8}  ########\n"
                  f"{self.agent_id:<2}  {self.total_step:8.2e}  {self.target_reward:8.2f} |"
                  f"{r_avg:8.2f}  {r_std:8.2f}   {self.used_time:>8}  ########")

        if time.time() - self.print_time > self.show_gap:
            self.print_time = time.time()
            print(f"{self.agent_id:<2}  {self.total_step:8.2e}  {self.r_max:8.2f} |"
                  f"{r_avg:8.2f}  {r_std:8.2f}   {obj_a:8.2f}  {obj_c:8.2f}")
        return if_solve

    def save_npy__draw_plot(self):
        if len(self.recorder) == 0:
            print("| save_npy__draw_plot() WARNNING: len(self.recorder)==0")
            return None

        '''save recorder as npy'''
        np.save('%s/recorder.npy' % self.cwd, self.recorder)

        '''plot subplots'''
        plt = self.plt
        fig, axs = plt.subplots(2)

        recorder = np.array(self.recorder)  # recorder.append((self.total_step, r_avg, r_std, obj_a, obj_c))
        steps = recorder[:, 0]  # x-axis is training steps
        r_avg = recorder[:, 1]
        r_std = recorder[:, 2]
        obj_a = recorder[:, 3]
        obj_c = recorder[:, 4]

        axs0 = axs[0]
        color0 = 'lightcoral'
        axs0.plot(steps, r_avg, label='Episode Return', color=color0)
        axs0.fill_between(steps, r_avg - r_std, r_avg + r_std, facecolor=color0, alpha=0.3)

        axs11 = axs[1]
        color11 = 'royalblue'
        label = 'objA'
        axs11.set_ylabel(label, color=color11)
        axs11.plot(steps, obj_a, label=label, color=color11)
        axs11.tick_params(axis='y', labelcolor=color11)

        ax12 = axs[1].twinx()
        color12 = 'darkcyan'
        ax12.set_ylabel('objC', color=color12)
        ax12.fill_between(steps, obj_c, facecolor=color12, alpha=0.2, )
        ax12.tick_params(axis='y', labelcolor=color12)

        '''plot title'''
        train_time = int(time.time() - self.start_time)
        total_step = int(self.recorder[-1][0])
        save_title = f"plot_step_time_maxR_{int(total_step)}_{int(train_time)}_{self.r_max:.3f}"
        plt.title(save_title, y=2.3)

        '''plot save'''
        save_path = f"{self.cwd}/{save_title}.jpg"
        plt.savefig(save_path)
        # plt.show()
        plt.close()


def get_episode_return(env, act, device) -> float:
    episode_return = 0.0  # sum of rewards in an episode
    max_step = env.max_step if hasattr(env, 'max_step') else 2 ** 10
    if_discrete = env.if_discrete

    state = env.reset()
    for _ in range(max_step):
        s_tensor = torch.as_tensor((state,), device=device)
        a_tensor = act(s_tensor)
        if if_discrete:
            a_tensor = a_tensor.argmax(dim=1)
        action = a_tensor.cpu().numpy()[0]  # not need detach(), because with torch.no_grad() outside

        state, reward, done, _ = env.step(action)
        episode_return += reward
        if done:
            break
    return env.episode_return if hasattr(env, 'episode_return') else episode_return


if __name__ == '__main__':
    run__demo()
