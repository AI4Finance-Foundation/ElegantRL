import os
import time
import torch
import numpy as np
import numpy.random as rd
import multiprocessing as mp

from copy import deepcopy
from elegantrl.env import build_env
from elegantrl.replay import ReplayBuffer, ReplayBufferMP
from elegantrl.evaluator import Evaluator


class Arguments:
    def __init__(self, env=None, agent=None, if_on_policy=False):
        self.env = env  # the environment for training
        self.agent = agent  # Deep Reinforcement Learning algorithm

        '''Arguments for device'''
        self.env_num = 1  # The Environment number for each worker. env_num == 1 means don't use VecEnv.
        self.worker_num = 2  # rollout workers number pre GPU (adjust it to get high GPU usage)
        self.thread_num = 8  # cpu_num for evaluate model, torch.set_num_threads(self.num_threads)
        self.visible_gpu = '0'  # for example: os.environ['CUDA_VISIBLE_DEVICES'] = '0, 2,'

        '''Arguments for training'''
        self.gamma = 0.99  # discount factor of future rewards
        self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256
        self.learning_rate = 2 ** -14  # 2 ** -14 ~= 6e-5
        self.soft_update_tau = 2 ** -8  # 2 ** -8 ~= 5e-3

        if if_on_policy:  # (on-policy)
            self.net_dim = 2 ** 9  # the network width
            self.batch_size = self.net_dim * 2  # num of transitions sampled from replay buffer.
            self.repeat_times = 2 ** 3  # collect target_step, then update network
            self.target_step = 2 ** 12  # repeatedly update network to keep critic's loss small
            self.max_memo = self.target_step  # capacity of replay buffer
            self.if_per_or_gae = False  # GAE for on-policy sparse reward: Generalized Advantage Estimation.
        else:
            self.net_dim = 2 ** 8  # the network width
            self.batch_size = self.net_dim  # num of transitions sampled from replay buffer.
            self.repeat_times = 2 ** 0  # repeatedly update network to keep critic's loss small
            self.target_step = 2 ** 10  # collect target_step, then update network
            self.max_memo = 2 ** 17  # capacity of replay buffer
            self.if_per_or_gae = False  # PER for off-policy sparse reward: Prioritized Experience Replay.

        '''Arguments for evaluate and save'''
        self.cwd = None  # current work directory. None means set automatically
        self.if_remove = True  # remove the cwd folder? (True, False, None:ask me)
        self.break_step = 2 ** 20  # break training after 'total_step > break_step'
        self.if_allow_break = True  # allow break training when reach goal (early termination)

        self.env_eval = None  # the environment for evaluating. None means set automatically.
        self.eval_gap = 2 ** 6  # evaluate the agent per eval_gap seconds
        self.eval_times1 = 2 ** 4  # number of times that get episode return in first
        self.eval_times2 = 2 ** 6  # number of times that get episode return in second
        self.random_seed = 0  # initialize random seed in self.init_before_training()

    def init_before_training(self, if_main):
        if self.agent is None:
            raise RuntimeError('\n| Why agent=None? Assignment args.agent = AgentXXX please.')
        if not hasattr(self.agent, 'init'):
            raise RuntimeError('\n| Should be agent=AgentXXX() instead of agent=AgentXXX')
        if self.env is None:
            raise RuntimeError('\n| Why env=None? Assignment args.env = XxxEnv() please.')
        if not (isinstance(self.env, str) or hasattr(self.env, 'env_name')):
            raise RuntimeError('\n| What is env.env_name? use env=PreprocessEnv(env).')

        if self.cwd is None:
            agent_name = self.agent.__class__.__name__
            env_name = getattr(self.env, 'env_name', self.env)
            self.cwd = f'./{agent_name}_{env_name}_{self.visible_gpu}'

        if if_main:
            import shutil  # remove history according to bool(if_remove)
            if self.if_remove is None:
                self.if_remove = bool(input(f"| PRESS 'y' to REMOVE: {self.cwd}? ") == 'y')
            elif self.if_remove:
                shutil.rmtree(self.cwd, ignore_errors=True)
                print(f"| Remove cwd: {self.cwd}")
            os.makedirs(self.cwd, exist_ok=True)

        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.set_num_threads(self.thread_num)
        torch.set_default_dtype(torch.float32)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.visible_gpu)
        del self.visible_gpu


'''single processing training'''


def train_and_evaluate(args, agent_id=0):
    args.init_before_training(if_main=True)

    '''init: Agent'''
    env = build_env(args.env, if_print=False)
    agent = args.agent
    agent.init(args.net_dim, env.state_dim, env.action_dim,
               args.learning_rate, args.if_per_or_gae, args.env_num)
    agent.save_or_load_agent(args.cwd, if_save=False)
    if args.env_num > 1:
        agent.explore_env = agent.explore_vec_env

    '''init Evaluator'''
    env_eval = deepcopy(env) if args.env_eval is None else args.env_eval
    evaluator = Evaluator(args.cwd, agent_id, agent.device, env_eval,
                          args.eval_times1, args.eval_times2, args.eval_gap)
    evaluator.save_or_load_recoder(if_save=False)

    '''init ReplayBuffer'''
    if agent.if_on_policy:
        buffer = list()
    else:
        buffer = ReplayBuffer(max_len=args.max_memo, state_dim=env.state_dim,
                              action_dim=1 if env.if_discrete else env.action_dim,
                              if_use_per=args.if_per_or_gae)
        buffer.save_or_load_history(args.cwd, if_save=False)

    '''start training'''
    cwd = args.cwd
    gamma = args.gamma
    break_step = args.break_step
    batch_size = args.batch_size
    target_step = args.target_step
    repeat_times = args.repeat_times
    reward_scale = args.reward_scale
    if_allow_break = args.if_allow_break
    soft_update_tau = args.soft_update_tau
    del args

    if agent.if_on_policy:
        assert isinstance(buffer, list)

        def update_buffer(_trajectory):
            _trajectory = list(map(list, zip(*_trajectory)))  # 2D-list transpose
            ary_state = np.array(_trajectory[0])
            ary_reward = np.array(_trajectory[1], dtype=np.float32) * reward_scale
            ary_mask = (1 - np.array(_trajectory[2], dtype=np.float32)) * gamma
            ary_action = np.array(_trajectory[3])
            ary_noise = np.array(_trajectory[4], dtype=np.float32)

            _steps = ary_reward.shape[0]
            _r_exp = ary_reward.mean()

            buffer[:] = (ary_state, ary_action, ary_noise, ary_reward, ary_mask)
            return _steps, _r_exp
    else:
        assert isinstance(buffer, ReplayBuffer)

        def update_buffer(_trajectory):
            ary_state = np.stack([item[0] for item in _trajectory])
            ary_other = np.stack([item[1] for item in _trajectory])
            ary_other[:, 0] = ary_other[:, 0] * reward_scale  # ary_reward
            ary_other[:, 1] = (1.0 - ary_other[:, 1]) * gamma  # ary_mask = (1.0 - ary_done) * gamma

            _steps = ary_state.shape[0]
            _r_exp = ary_other[:, 0].mean()  # other = (reward, mask, action)

            buffer.extend_buffer(torch.as_tensor(ary_state),
                                 torch.as_tensor(ary_other, dtype=torch.float32))
            return _steps, _r_exp

        if_load = buffer.save_or_load_history(cwd, if_save=False)

        if not if_load:
            trajectory = explore_before_training(env, target_step)
            steps, r_exp = update_buffer(trajectory)
            evaluator.total_step += steps

    agent.states = env.reset()

    if_train = True
    while if_train:
        with torch.no_grad():
            array_tuple = agent.explore_env(env, target_step)
            steps, r_exp = update_buffer(array_tuple)

        logging_tuple = agent.update_net(buffer, batch_size, repeat_times, soft_update_tau)
        with torch.no_grad():
            if_reach_goal = evaluator.evaluate_and_save(agent.act, steps, r_exp, logging_tuple)
            if_train = not ((if_allow_break and if_reach_goal)
                            or evaluator.total_step > break_step
                            or os.path.exists(f'{cwd}/stop'))

    print(f'| UsedTime: {time.time() - evaluator.start_time:.0f} | SavedDir: {cwd}')

    env.close()
    agent.save_or_load_agent(cwd, if_save=True)
    buffer.save_or_load_history(cwd, if_save=True) if not agent.if_on_policy else None
    evaluator.save_or_load_recoder(if_save=True)


def explore_before_training(env, target_step):  # for off-policy only
    trajectory = list()

    if_discrete = env.if_discrete
    action_dim = env.action_dim

    state = env.reset()
    step = 0
    while True:
        if if_discrete:
            action = rd.randint(action_dim)  # assert isinstance(action_int)
            next_s, reward, done, _ = env.step(action)
            other = (reward, done, action)
        else:
            action = rd.uniform(-1, 1, size=action_dim)
            next_s, reward, done, _ = env.step(action)
            other = (reward, done, *action)

        trajectory.append((state, other))
        state = env.reset() if done else next_s

        step += 1
        if done and step > target_step:
            break
    return trajectory


def explore_before_training_vec_env(env, target_step) -> list:  # for off-policy only
    # plan to be elegant: merge this function to explore_before_training()
    assert hasattr(env, 'env_num')
    env_num = env.env_num

    trajectory_list = [list() for _ in range(env_num)]

    if_discrete = env.if_discrete
    action_dim = env.action_dim

    states = env.reset()
    step = 0
    while True:
        if if_discrete:
            actions = rd.randint(action_dim, size=env_num)
            s_r_d_list = env.step(actions)

            next_states = list()
            for env_i in range(env_num):
                next_s, reward, done = s_r_d_list[env_i]
                trajectory_list[env_i].append((states[env_i], (reward, done, actions[env_i])))
                next_states.append(next_s)
        else:
            actions = rd.uniform(-1, 1, size=(env_num, action_dim))
            s_r_d_list = env.step(actions)

            next_states = list()
            for env_i in range(env_num):
                next_s, reward, done = s_r_d_list[env_i]
                trajectory_list[env_i].append((states[env_i], (reward, done, *actions[env_i])))
                next_states.append(next_s)
        states = next_states

        step += 1
        if step > target_step:
            break
    return trajectory_list


'''multiple processing training'''


class CommEvaluate:
    def __init__(self):
        self.pipe = mp.Pipe()

    def evaluate_and_save0(self, act_cpu, evaluator, if_break_early, break_step, cwd):
        act_cpu_dict, steps, r_exp, logging_tuple = self.pipe[0].recv()

        if act_cpu_dict is None:
            if_reach_goal = False
            evaluator.total_step += steps
        else:
            act_cpu.load_state_dict(act_cpu_dict)
            if_reach_goal = evaluator.evaluate_and_save(act_cpu, steps, r_exp, logging_tuple)

        if_train = not ((if_break_early and if_reach_goal)
                        or evaluator.total_step > break_step
                        or os.path.exists(f'{cwd}/stop'))
        self.pipe[0].send(if_train)
        return if_train

    def evaluate_and_save1(self, agent_act, steps, r_exp, logging_tuple, if_train):
        if self.pipe[1].poll():  # if_evaluator_idle
            if_train = self.pipe[1].recv()

            act_cpu_dict = {k: v.cpu() for k, v in agent_act.state_dict().items()}
        else:
            act_cpu_dict = None

        self.pipe[1].send((act_cpu_dict, steps, r_exp, logging_tuple))
        return if_train


def mp_evaluator(args, comm_eva, agent_id=0):
    args.init_before_training(if_main=False)

    '''init: Agent'''
    env = build_env(args.env, if_print=False)
    agent = args.agent
    agent.init(args.net_dim, env.state_dim, env.action_dim,
               args.learning_rate, args.if_per_or_gae, args.env_num)
    agent.save_or_load_agent(args.cwd, if_save=False)

    device = torch.device("cpu")
    act_cpu = agent.act.to(device)
    act_cpu.eval()
    [setattr(param, 'requires_grad', False) for param in act_cpu.parameters()]
    del agent

    '''init Evaluator'''
    env_eval = deepcopy(env) if args.env_eval is None else args.env_eval
    evaluator = Evaluator(args.cwd, agent_id, device, env_eval,
                          args.eval_times1, args.eval_times2, args.eval_gap)
    evaluator.save_or_load_recoder(if_save=False)
    del env

    '''start training'''
    cwd = args.cwd
    break_step = args.break_step
    if_allow_break = args.if_allow_break
    del args

    if_train = True
    with torch.no_grad():
        while if_train:
            if_train = comm_eva.evaluate_and_save0(act_cpu, evaluator, if_allow_break, break_step, cwd)

    print(f'| UsedTime: {time.time() - evaluator.start_time:.0f} | SavedDir: {cwd}')
    evaluator.save_or_load_recoder(if_save=True)


class CommVecEnv:
    def __init__(self, args):
        self.env_num = args.env_num
        self.pipe_list = [mp.Pipe() for _ in range(self.env_num)]
        self.pipe0_list = [pipe[0] for pipe in self.pipe_list]

        from elegantrl.env import get_gym_env_info
        env = build_env(args.env)
        (self.env_name, self.state_dim, self.action_dim, self.action_max, self.max_step,
         self.if_discrete, self.target_return) = get_gym_env_info(env, if_print=False)
        del env

        self.process = list()
        for env_id in range(args.env_num):
            self.process.append(mp.Process(target=self.mp_env, args=(args, env_id)))
            args.random_seed += 1  # set different for each env
        # [p.start() for p in self.process]

    def reset(self):
        vec_state = [pipe0.recv() for pipe0 in self.pipe0_list]
        return vec_state

    def step(self, vec_action):  # pipe0_step
        for i in range(self.env_num):
            self.pipe0_list[i].send(vec_action[i])
        return [pipe0.recv() for pipe0 in self.pipe0_list]  # list of (state, reward, done)

    def mp_env(self, args, env_id):
        np.random.seed(args.random_seed)

        env = build_env(args.env, if_print=False)
        pipe1 = self.pipe_list[env_id][1]
        del args

        state = env.reset()
        pipe1.send(state)  # for reset

        while True:
            action = pipe1.recv()

            state, reward, done, _ = env.step(action)
            pipe1.send((env.reset() if done else state, reward, done))

    def close(self):
        [p.kill() for p in self.process]
    # def check(self):
    #     vec_state = self.reset()
    #     ten_state = np.array(vec_state)
    #     print(ten_state.shape)
    #
    #     vec_action = np.array(((0.0, 1.0, 0.0),
    #                            (0.0, 0.5, 0.0),
    #                            (0.0, 0.1, 0.0),))[:self.env_num]
    #     assert self.env_num <= 3
    #
    #     trajectory_list = list()
    #     for _ in range(8):
    #         s_r_d_list = self.step(vec_action)
    #         ten_state = np.array([s_r_d[0] for s_r_d in s_r_d_list])
    #         print(ten_state.shape)
    #         trajectory_list.append(s_r_d_list)
    #
    #     trajectory_list = list(map(list, zip(*trajectory_list)))  # 2D-list transpose
    #     print('| shape of trajectory_list:', len(trajectory_list), len(trajectory_list[0]))


def mp_worker(args, comm_exp, comm_env, worker_id, gpu_id=0):
    args.random_seed += gpu_id * args.worker_num + gpu_id
    args.init_before_training(if_main=False)

    '''init: Agent'''
    env = build_env(args.env, if_print=False)
    agent = args.agent
    agent.init(args.net_dim, env.state_dim, env.action_dim,
               args.learning_rate, args.if_per_or_gae, args.env_num)

    agent.save_or_load_agent(args.cwd, if_save=False)
    if_on_policy = agent.if_on_policy
    if args.env_num > 1:
        agent.explore_env = agent.explore_vec_env

    '''start training'''
    gamma = args.gamma
    target_step = args.target_step
    reward_scale = args.reward_scale
    del args

    if comm_env:
        env = comm_env
    if if_on_policy:
        agent.states = env.reset() if comm_env else [env.reset()]
    else:
        agent.states = comm_exp.pre_explore0(worker_id, env, target_step, reward_scale, gamma)

    with torch.no_grad():
        while True:
            comm_exp.explore0(worker_id, agent, env, target_step, reward_scale, gamma)


class CommExplore:
    def __init__(self, env_num, worker_num, if_on_policy):
        self.pipe_list = [mp.Pipe() for _ in range(worker_num)]

        self.env_num = env_num
        self.worker_num = worker_num

        if if_on_policy:
            self.explore1 = self.explore1_on_policy
            self.explore0 = self.explore0_on_policy
        else:
            self.explore1 = self.explore1_off_policy
            self.explore0 = self.explore0_off_policy

    def explore1_on_policy(self, agent, buffer):
        act_dict = agent.act.state_dict()
        for i in range(self.worker_num):
            self.pipe_list[i][1].send(act_dict)

        buffer_tuples = [pipe[1].recv() for pipe in self.pipe_list]
        buffer_tuples = list(map(list, zip(*buffer_tuples)))
        # buffer_tuple = (ary_state, ary_action, ary_noise, ary_reward, ary_mask)

        buffer[:] = [np.concatenate(arrays, axis=0) for arrays in buffer_tuples]
        steps = buffer[3].shape[0]  # buffer[3] = ary_reward
        r_exp = buffer[3].mean()  # buffer[3] = ary_reward
        return steps, r_exp

    def explore0_on_policy(self, worker_id, agent, env, target_step, reward_scale, gamma):
        act_dict = self.pipe_list[worker_id][0].recv()
        agent.act.load_state_dict(act_dict)

        trajectory = agent.explore_env(env, target_step)
        trajectory = self.convert_trajectory_on_policy(trajectory, reward_scale, gamma)
        self.pipe_list[worker_id][0].send(trajectory)

    def explore1_off_policy(self, agent, buffer_mp):
        act_dict = agent.act.state_dict()
        for env_i in range(self.worker_num):
            self.pipe_list[env_i][1].send(act_dict)

        trajectory_lists = [pipe[1].recv() for pipe in self.pipe_list]
        # trajectory_lists = (trajectory_list, ...) = ((ary_state, ary_other), ...)

        steps = 0
        r_exp = 0
        buffer_i = 0
        for env_i in range(self.worker_num):
            trajectory_list = trajectory_lists[env_i]
            for ary_state, ary_other in trajectory_list:
                steps += ary_other.shape[0]
                r_exp += ary_other[:, 0].sum()  # other = (reward, mask, *action)
                buffer_mp.buffers[buffer_i].extend_buffer(
                    torch.as_tensor(ary_state, device=agent.device),
                    torch.as_tensor(ary_other, device=agent.device), )

                buffer_i += 1

        r_exp /= steps
        return steps, r_exp

    def explore0_off_policy(self, worker_id, agent, env, target_step, reward_scale, gamma):
        act_dict = self.pipe_list[worker_id][0].recv()
        agent.act.load_state_dict(act_dict)

        trajectory_list = agent.explore_env(env, target_step)

        for env_i in range(self.env_num):  # plan to be elegant
            trajectory = trajectory_list[env_i]
            trajectory = self.convert_trajectory_off_policy(trajectory, reward_scale, gamma)
            trajectory_list[env_i] = trajectory

        self.pipe_list[worker_id][0].send(trajectory_list)

    def pre_explore1(self, agent, buffer_mp):
        buffer_i = 0
        for worker_i in range(self.worker_num):
            trajectory_list = self.pipe_list[worker_i][1].recv()
            for ary_state, ary_other in trajectory_list:
                buffer_mp.buffers[buffer_i].extend_buffer(
                    torch.as_tensor(ary_state, device=agent.device),
                    torch.as_tensor(ary_other, device=agent.device), )

                buffer_i += 1

    def pre_explore0(self, worker_id, env, target_step, reward_scale, gamma):
        explore_step = target_step // (self.worker_num * self.env_num)

        if self.env_num > 1:
            trajectory_list = explore_before_training_vec_env(env, explore_step)

            for env_i in range(self.env_num):  # plan to be elegant
                trajectory = trajectory_list[env_i]
                trajectory = self.convert_trajectory_off_policy(trajectory, reward_scale, gamma)

                trajectory_list[env_i] = trajectory
        else:
            trajectory = explore_before_training(env, explore_step)
            trajectory = self.convert_trajectory_off_policy(trajectory, reward_scale, gamma)

            trajectory_list = (trajectory,)

        self.pipe_list[worker_id][0].send(trajectory_list)

        last_states = [trajectory[0][-1]  # trajectory[0][-1] = ary_state[-1]
                       for trajectory in trajectory_list]
        return last_states

    @staticmethod
    def convert_trajectory_on_policy(trajectory, reward_scale, gamma):
        trajectory = list(map(list, zip(*trajectory)))  # 2D-list transpose

        ary_state = np.array(trajectory[0])
        ary_reward = np.array(trajectory[1], dtype=np.float32) * reward_scale
        ary_mask = (1 - np.array(trajectory[2], dtype=np.float32)) * gamma
        ary_action = np.array(trajectory[3])
        ary_noise = np.array(trajectory[4], dtype=np.float32)
        return ary_state, ary_action, ary_noise, ary_reward, ary_mask

    @staticmethod
    def convert_trajectory_off_policy(trajectory, reward_scale, gamma):
        ary_state = np.stack([item[0] for item in trajectory])
        ary_other = np.stack([item[1] for item in trajectory])
        ary_other[:, 0] = ary_other[:, 0] * reward_scale  # ary_reward
        ary_other[:, 1] = (1.0 - ary_other[:, 1]) * gamma  # ary_mask = (1.0 - ary_done) * gamma
        return ary_state, ary_other


def mp_learner(args, comm_eva, comm_exp):
    args.init_before_training(if_main=True)

    '''init: Agent'''
    env = build_env(args.env, if_print=False)
    agent = args.agent
    agent.init(args.net_dim, env.state_dim, env.action_dim,
               args.learning_rate, args.if_per_or_gae, args.env_num)
    agent.save_or_load_agent(args.cwd, if_save=False)
    if_on_policy = agent.if_on_policy
    if args.env_num > 1:
        agent.explore_env = agent.explore_vec_env

    '''init ReplayBuffer'''
    if if_on_policy:
        buffer = [list() for _ in range(args.worker_num)]
    else:
        buffer = ReplayBufferMP(max_len=args.max_memo, state_dim=env.state_dim,
                                action_dim=1 if env.if_discrete else env.action_dim,
                                if_use_per=args.if_per_or_gae,
                                worker_num=args.worker_num * args.env_num, gpu_id=0)

        if_load = buffer.save_or_load_history(args.cwd, if_save=False)
        if not if_load:
            comm_exp.pre_explore1(agent, buffer)

    '''start training'''
    cwd = args.cwd
    batch_size = args.batch_size
    repeat_times = args.repeat_times
    soft_update_tau = args.soft_update_tau
    del args

    if_train = True
    while if_train:
        steps, r_exp = comm_exp.explore1(agent, buffer)

        logging_tuple = agent.update_net(buffer, batch_size, repeat_times, soft_update_tau)

        if_train = comm_eva.evaluate_and_save1(agent.act, steps, r_exp, logging_tuple, if_train)

    agent.save_or_load_agent(cwd, if_save=True)
    buffer.save_or_load_history(cwd, if_save=True) if not if_on_policy else None


def train_and_evaluate_mp(args):
    process = list()

    comm_eva = CommEvaluate()
    process.append(mp.Process(target=mp_evaluator, args=(args, comm_eva)))

    comm_exp = CommExplore(args.env_num, args.worker_num, args.agent.if_on_policy)
    for worker_id in range(args.worker_num):
        comm_env = CommVecEnv(args) if args.env_num > 1 else None
        process.extend(comm_env.process if args.env_num > 1 else list())
        process.append(mp.Process(target=mp_worker, args=(args, comm_exp, comm_env, worker_id,)))

    process.append(mp.Process(target=mp_learner, args=(args, comm_eva, comm_exp)))

    [(p.start(), time.sleep(0.1)) for p in process]
    process[-1].join()
    process_safely_terminate(process)


"""Utils"""


def process_safely_terminate(process):
    for p in process:
        try:
            p.kill()
        except OSError as e:
            print(e)
            pass
