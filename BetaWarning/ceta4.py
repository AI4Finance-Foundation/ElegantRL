from AgentRun import *

"""Compare 
torch.tensor()
torch.as_tensor
torch.from_numpy
"""


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
        # todo tensor one line
        self.memories[self.next_idx] = torch.tensor(np.hstack(memo_tuple), device=self.device)
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


def mp__update_params(args, eva_pipe, pipes):  # 2020-12-22
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

    '''init: env'''
    state_dim = env.state_dim
    action_dim = env.action_dim
    if_discrete = env.if_discrete

    '''build agent and act_cpu'''
    agent = rl_agent(state_dim, action_dim, net_dim)  # training agent
    agent.state = [pipe.recv() for pipe in pipes]
    agent.action = agent.select_actions(agent.state)
    for i in range(len(pipes)):
        pipes[i].send(agent.action[i])

    from copy import deepcopy  # built-in library of Python
    act_cpu = deepcopy(agent.act).to(torch.device("cpu"))
    act_cpu.eval()
    [setattr(param, 'requires_grad', False) for param in act_cpu.parameters()]

    '''build replay buffer, init: total_step, reward_avg'''
    reward_avg = None
    total_step = 0
    if bool(rl_agent.__name__ in {'AgentPPO', 'AgentInterPPO'}):
        buffer = BufferArrayGPU(max_memo + max_step, state_dim, action_dim, if_ppo=True)
    else:
        buffer = BufferArrayGPU(max_memo, state_dim, action_dim=1 if if_discrete else action_dim, if_ppo=False)

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

        total_step += step_sum
    if reward_avg is None:
        reward_avg = get_episode_return(env, agent.act, max_step, agent.device, if_discrete)

    '''training loop'''
    if_train = True
    if_solve = False
    while if_train:
        '''update replay buffer by interact with environment'''
        with torch.no_grad():  # speed up running
            rewards, steps = agent.update_buffer__pipe(pipes, buffer, max_step)

        reward_avg = np.average(rewards) if len(rewards) else reward_avg
        step_sum = sum(steps)
        total_step += step_sum

        '''update network parameters by random sampling buffer for gradient descent'''
        buffer.update__now_len__before_sample()
        loss_a_avg, loss_c_avg = agent.update_policy(buffer, max_step, batch_size, repeat_times)

        '''saves the agent with max reward'''
        act_cpu.load_state_dict(agent.act.state_dict())
        eva_pipe.send((act_cpu, reward_avg, step_sum, loss_a_avg, loss_c_avg))  # eva_pipe act

        if eva_pipe.poll():
            if_solve = eva_pipe.recv()  # eva_pipe if_solve

        '''break loop rules'''
        if_train = not ((if_stop and if_solve)
                        or total_step > max_total_step
                        or os.path.exists(f'{cwd}/stop'))

    buffer.print_state_norm(env.neg_state_avg if hasattr(env, 'neg_state_avg') else None,
                            env.div_state_std if hasattr(env, 'div_state_std') else None)  # 2020-12-12
    eva_pipe.send('stop')  # eva_pipe stop  # send to mp_evaluate_agent
    time.sleep(4)
    # print('; quit: params')

def train_agent_mp(args):  # 2021-01-01
    act_workers = args.rollout_num

    import multiprocessing as mp
    eva_pipe1, eva_pipe2 = mp.Pipe(duplex=True)
    process = list()

    exp_pipe2s = list()
    for i in range(act_workers):
        exp_pipe1, exp_pipe2 = mp.Pipe(duplex=True)
        exp_pipe2s.append(exp_pipe1)
        process.append(mp.Process(target=mp_explore_in_env, args=(args, exp_pipe2, i)))
    process.extend([
        mp.Process(target=mp_evaluate_agent, args=(args, eva_pipe1)),
        mp.Process(target=mp__update_params, args=(args, eva_pipe2, exp_pipe2s)),
    ])

    [p.start() for p in process]
    process[-1].join()
    process[-2].join()
    [p.terminate() for p in process]
    print('\n')


def train__demo():
    pass

    '''DEMO 2: Standard gym env LunarLanderContinuous-v2 (continuous action) using ModSAC (Modify SAC, off-policy)'''
    import gym  # gym of OpenAI is not necessary for ElegantRL (even RL)
    gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'
    env = gym.make("BipedalWalker-v3")
    env = decorate_env(env, if_print=True)

    from AgentZoo import AgentModSAC
    args = Arguments(rl_agent=AgentModSAC, env=env)
    args.rollout_num = 4
    args.break_step = int(2e5 * 8)  # (1e5) 2e5, used time 3500s
    args.reward_scale = 2 ** -1  # (-200) -140 ~ 300 (341)

    args.random_seed = 0
    args.if_break_early = False
    args.break_step = int(1e5)  # todo just for test,  2563 second

    args.init_for_training()
    # train_agent(args)  # Train agent using single process. Recommend run on PC.
    train_agent_mp(args)  # Train using multi process. Recommend run on Server. Mix CPU(eval) GPU(train)
    exit()


if __name__ == '__main__':
    train__demo()
