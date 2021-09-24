import os
import time
import shutil

import torch
import numpy as np
import numpy.random as rd
import multiprocessing as mp

from elegantrl.env import build_env
from elegantrl.replay import ReplayBuffer, ReplayBufferMP
from elegantrl.evaluator import Evaluator

"""[ElegantRL.2021.09.09](https://github.com/AI4Finance-LLC/ElegantRL)"""


class Arguments:
    def __init__(self, if_on_policy=False):
        self.env = None  # the environment for training
        self.agent = None  # Deep Reinforcement Learning algorithm

        '''Arguments for training'''
        self.gamma = 0.99  # discount factor of future rewards
        self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256
        self.learning_rate = 2 ** -15  # 2 ** -14 ~= 3e-5
        self.soft_update_tau = 2 ** -8  # 2 ** -8 ~= 5e-3

        self.if_on_policy = if_on_policy
        if self.if_on_policy:  # (on-policy)
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
            self.max_memo = 2 ** 21  # capacity of replay buffer
            self.if_per_or_gae = False  # PER for off-policy sparse reward: Prioritized Experience Replay.

        '''Arguments for device'''
        self.env_num = 1  # The Environment number for each worker. env_num == 1 means don't use VecEnv.
        self.worker_num = 2  # rollout workers number pre GPU (adjust it to get high GPU usage)
        self.thread_num = 8  # cpu_num for evaluate model, torch.set_num_threads(self.num_threads)
        self.visible_gpu = '0'  # for example: os.environ['CUDA_VISIBLE_DEVICES'] = '0, 2,'
        self.random_seed = 0  # initialize random seed in self.init_before_training()

        '''Arguments for evaluate and save'''
        self.cwd = None  # current work directory. None means set automatically
        self.if_remove = True  # remove the cwd folder? (True, False, None:ask me)
        self.break_step = 2 ** 20  # break training after 'total_step > break_step'
        self.if_allow_break = True  # allow break training when reach goal (early termination)

        self.eval_env = None  # the environment for evaluating. None means set automatically.
        self.eval_gap = 2 ** 7  # evaluate the agent per eval_gap seconds
        self.eval_times1 = 2 ** 3  # number of times that get episode return in first
        self.eval_times2 = 2 ** 4  # number of times that get episode return in second
        self.eval_device_id = -1  # -1 means use cpu, >=0 means use GPU

    def init_before_training(self, if_main):
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.set_num_threads(self.thread_num)
        torch.set_default_dtype(torch.float32)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.visible_gpu)

        '''env'''
        if self.env is None:
            raise RuntimeError(f'\n| Why env=None? For example:'
                               f'\n| args.env = XxxEnv()'
                               f'\n| args.env = str(env_name)'
                               f'\n| args.env = build_env(env_name), from elegantrl.env import build_env')
        if not (isinstance(self.env, str) or hasattr(self.env, 'env_name')):
            raise RuntimeError('\n| What is env.env_name? use env=PreprocessEnv(env).')

        '''agent'''
        if self.agent is None:
            raise RuntimeError(f'\n| Why agent=None? Assignment `args.agent = AgentXXX` please.')
        if not hasattr(self.agent, 'init'):
            raise RuntimeError(f"\n| why hasattr(self.agent, 'init') == False"
                               f'\n| Should be `agent=AgentXXX()` instead of `agent=AgentXXX`.')
        if self.agent.if_on_policy != self.if_on_policy:
            raise RuntimeError(f'\n| Why bool `if_on_policy` is not consistent?'
                               f'\n| self.if_on_policy: {self.if_on_policy}'
                               f'\n| self.agent.if_on_policy: {self.agent.if_on_policy}')

        '''cwd'''
        if self.cwd is None:
            agent_name = self.agent.__class__.__name__
            env_name = getattr(self.env, 'env_name', self.env)
            self.cwd = f'./{agent_name}_{env_name}_{self.visible_gpu}'
        if if_main:
            # remove history according to bool(if_remove)
            if self.if_remove is None:
                self.if_remove = bool(input(f"| PRESS 'y' to REMOVE: {self.cwd}? ") == 'y')
            elif self.if_remove:
                shutil.rmtree(self.cwd, ignore_errors=True)
                print(f"| Remove cwd: {self.cwd}")
            os.makedirs(self.cwd, exist_ok=True)


'''single processing training'''


def train_and_evaluate(args, agent_id=0):
    args.init_before_training(if_main=True)

    env = build_env(args.env, if_print=False)

    '''init: Agent'''
    agent = args.agent
    agent.init(args.net_dim, env.state_dim, env.action_dim, args.learning_rate, args.if_per_or_gae, args.env_num)
    agent.save_or_load_agent(args.cwd, if_save=False)
    if env.env_num == 1:
        agent.states = [env.reset(), ]
        assert isinstance(agent.states[0], np.ndarray)
        assert agent.states[0].shape == (env.state_dim,)
    else:
        agent.states = env.reset()
        assert isinstance(agent.states, torch.Tensor)
        assert agent.states.shape == (env.env_num, env.state_dim)

    '''init Evaluator'''
    eval_env = args.eval_env if args.eval_env else build_env(env)

    evaluator = Evaluator(args.cwd, agent_id, agent.device, eval_env,
                          args.eval_gap, args.eval_times1, args.eval_times2)
    evaluator.save_or_load_recoder(if_save=False)

    '''init ReplayBuffer'''
    if agent.if_on_policy:
        buffer = list()

        def update_buffer(_traj_list):
            buffer[:] = _traj_list[0]  # (ten_state, ten_reward, ten_mask, ten_action, ten_noise)

            _step, _r_exp = get_step_r_exp(ten_reward=buffer[1])
            return _step, _r_exp
    else:
        buffer = ReplayBuffer(max_len=args.max_memo, state_dim=env.state_dim,
                              action_dim=1 if env.if_discrete else env.action_dim,
                              if_use_per=args.if_per_or_gae)
        buffer.save_or_load_history(args.cwd, if_save=False)

        def update_buffer(_traj_list):
            ten_state, ten_other = _traj_list[0]
            buffer.extend_buffer(ten_state, ten_other)

            _steps, _r_exp = get_step_r_exp(ten_reward=ten_other[0])  # other = (reward, mask, action)
            return _steps, _r_exp

    """start training"""
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

    '''init ReplayBuffer after training start'''
    if not agent.if_on_policy:
        if_load = buffer.save_or_load_history(cwd, if_save=False)

        if not if_load:
            traj_list = agent.explore_env(env, target_step, reward_scale, gamma)
            steps, r_exp = update_buffer(traj_list)
            evaluator.total_step += steps

    '''start training loop'''
    if_train = True
    while if_train:
        with torch.no_grad():
            traj_list = agent.explore_env(env, target_step, reward_scale, gamma)
            steps, r_exp = update_buffer(traj_list)

        logging_tuple = agent.update_net(buffer, batch_size, repeat_times, soft_update_tau)
        with torch.no_grad():
            temp = evaluator.evaluate_and_save(agent.act, steps, r_exp, logging_tuple)
            if_reach_goal, if_save = temp
            if_train = not ((if_allow_break and if_reach_goal)
                            or evaluator.total_step > break_step
                            or os.path.exists(f'{cwd}/stop'))

    print(f'| UsedTime: {time.time() - evaluator.start_time:>7.0f} | SavedDir: {cwd}')

    env.close()
    agent.save_or_load_agent(cwd, if_save=True)
    buffer.save_or_load_history(cwd, if_save=True) if not agent.if_on_policy else None
    evaluator.save_or_load_recoder(if_save=True)


def get_step_r_exp(ten_reward):
    return len(ten_reward), ten_reward.mean().item()


'''multiple processing training'''


class PipeWorker:
    def __init__(self, env_num, worker_num):
        self.env_num = env_num
        self.worker_num = worker_num
        self.pipes = [mp.Pipe() for _ in range(worker_num)]
        self.pipe1s = [pipe[1] for pipe in self.pipes]

    def explore(self, agent):
        act_dict = agent.act.state_dict()
        for worker_id in range(self.worker_num):
            self.pipe1s[worker_id].send(act_dict)

        traj_lists = [pipe1.recv() for pipe1 in self.pipe1s]
        return traj_lists

    def run(self, args, comm_env, worker_id, learner_id):
        # print(f'| os.getpid()={os.getpid()} PipeExplore.run {learner_id}')
        args.init_before_training(if_main=False)

        '''init Agent'''
        env = build_env(args.env, if_print=False)
        agent = args.agent
        agent.init(args.net_dim, env.state_dim, env.action_dim,
                   args.learning_rate, args.if_per_or_gae, args.env_num, learner_id)

        '''loop'''
        gamma = args.gamma
        target_step = args.target_step
        reward_scale = args.reward_scale
        del args

        if comm_env:
            env = comm_env
            agent.states = env.reset()
        else:
            agent.states = [env.reset(), ]

        with torch.no_grad():
            while True:
                act_dict = self.pipes[worker_id][0].recv()
                agent.act.load_state_dict(act_dict)

                trajectory = agent.explore_env(env, target_step, reward_scale, gamma)
                self.pipes[worker_id][0].send(trajectory)


def get_comm_data(agent):
    act = list(agent.act.parameters())
    cri_optim = get_optim_parameters(agent.cri_optim)

    if agent.cri is agent.act:
        cri = None
        act_optim = None
    else:
        cri = list(agent.cri.parameters())
        act_optim = get_optim_parameters(agent.act_optim)

    act_target = list(agent.act_target.parameters()) if agent.if_use_act_target else None
    cri_target = list(agent.cri_target.parameters()) if agent.if_use_cri_target else None
    return act, act_optim, cri, cri_optim, act_target, cri_target  # data


class PipeLearner:
    def __init__(self, learner_num):
        self.learner_num = learner_num
        self.round_num = int(np.log2(learner_num))

        self.pipes = [mp.Pipe() for _ in range(learner_num)]
        pipes = [mp.Pipe() for _ in range(learner_num)]
        self.pipe0s = [pipe[0] for pipe in pipes]
        self.pipe1s = [pipe[1] for pipe in pipes]
        self.device_list = [torch.device(f'cuda:{i}') for i in range(learner_num)]

        if learner_num == 1:
            self.idx_l = None
        elif learner_num == 2:
            self.idx_l = [(1,), (0,), ]
        elif learner_num == 4:
            self.idx_l = [(1, 2), (0, 3),
                          (3, 0), (2, 1), ]
        elif learner_num == 8:
            self.idx_l = [(1, 2, 4), (0, 3, 5),
                          (3, 0, 6), (2, 1, 7),
                          (5, 6, 0), (4, 7, 1),
                          (7, 4, 2), (6, 5, 3), ]
        else:
            print(f"| LearnerPipe, ERROR: learner_num {learner_num} should in (1, 2, 4, 8)")
            exit()

    def comm_data(self, data, learner_id, round_id):
        if round_id == -1:
            learner_jd = self.idx_l[learner_id][round_id]
            self.pipes[learner_jd][0].send(data)
            return self.pipes[learner_id][1].recv()
        else:
            learner_jd = self.idx_l[learner_id][round_id]
            self.pipe0s[learner_jd].send(data)
            return self.pipe1s[learner_id].recv()

    def comm_network_optim(self, agent, learner_id):
        device = self.device_list[learner_id]

        for round_id in range(self.round_num):
            data = get_comm_data(agent)
            data = self.comm_data(data, learner_id, round_id)

            if data:
                avg_update_net(agent.act, data[0], device)
                avg_update_optim(agent.act_optim, data[1], device) if data[1] else None

                avg_update_net(agent.cri, data[2], device) if data[2] else None
                avg_update_optim(agent.cri_optim, data[3], device)

                avg_update_net(agent.act_target, data[4], device) if agent.if_use_act_target else None
                avg_update_net(agent.cri_target, data[5], device) if agent.if_use_cri_target else None

    def run(self, args, comm_eva, comm_exp, learner_id=0):
        # print(f'| os.getpid()={os.getpid()} PipeLearn.run, {learner_id}')
        args.init_before_training(if_main=learner_id == 0)

        env = build_env(args.env, if_print=False)
        if_on_policy = args.if_on_policy

        '''init Agent'''
        agent = args.agent
        agent.init(args.net_dim, env.state_dim, env.action_dim,
                   args.learning_rate, args.if_per_or_gae, args.env_num, learner_id)
        agent.save_or_load_agent(args.cwd, if_save=False)

        '''init ReplayBuffer'''
        if if_on_policy:
            buffer = list()

            def update_buffer(_traj_list):
                _traj_list = list(map(list, zip(*_traj_list)))
                _traj_list = [torch.cat(t, dim=0) for t in _traj_list]
                buffer[:] = _traj_list  # (ten_state, ten_reward, ten_mask, ten_action, ten_noise)
                _step, _r_exp = get_step_r_exp(ten_reward=buffer[1])
                return _step, _r_exp

        else:
            buffer_num = args.worker_num * args.env_num
            if self.learner_num > 1:
                buffer_num *= 2

            buffer = ReplayBufferMP(max_len=args.max_memo, state_dim=env.state_dim,
                                    action_dim=1 if env.if_discrete else env.action_dim,
                                    if_use_per=args.if_per_or_gae,
                                    buffer_num=buffer_num, gpu_id=learner_id)
            buffer.save_or_load_history(args.cwd, if_save=False)

            def update_buffer(_traj_list):
                step_sum = 0
                r_exp_sum = 0
                for buffer_i, (ten_state, ten_other) in enumerate(_traj_list):
                    buffer.buffers[buffer_i].extend_buffer(ten_state, ten_other)

                    step_r_exp = get_step_r_exp(ten_reward=ten_other[:, 0])  # other = (reward, mask, action)
                    step_sum += step_r_exp[0]
                    r_exp_sum += step_r_exp[1]
                return step_sum, r_exp_sum / len(_traj_list)

        '''start training'''
        cwd = args.cwd
        batch_size = args.batch_size
        repeat_times = args.repeat_times
        soft_update_tau = args.soft_update_tau
        del args

        if_train = True
        while if_train:
            traj_lists = comm_exp.explore(agent)
            if self.learner_num > 1:
                data = self.comm_data(traj_lists, learner_id, round_id=-1)
                traj_lists.extend(data)
            traj_list = sum(traj_lists, list())

            steps, r_exp = update_buffer(traj_list)
            del traj_lists
            logging_tuple = agent.update_net(buffer, batch_size, repeat_times, soft_update_tau)
            if self.learner_num > 1:
                self.comm_network_optim(agent, learner_id)

            if comm_eva:
                if_train, if_save = comm_eva.evaluate_and_save_mp(agent.act, steps, r_exp, logging_tuple)

        agent.save_or_load_agent(cwd, if_save=True)
        if not if_on_policy:
            print(f"| LearnerPipe.run: ReplayBuffer saving in {cwd}")
            buffer.save_or_load_history(cwd, if_save=True)


class PipeEvaluator:
    def __init__(self):
        super().__init__()
        self.pipe0, self.pipe1 = mp.Pipe()

    def evaluate_and_save_mp(self, agent_act, steps, r_exp, logging_tuple):
        if self.pipe1.poll():  # if_evaluator_idle
            if_train, if_save = self.pipe1.recv()
            act_cpu_dict = {k: v.cpu() for k, v in agent_act.state_dict().items()}
        else:
            if_train, if_save = True, False
            act_cpu_dict = None

        self.pipe1.send((act_cpu_dict, steps, r_exp, logging_tuple))
        return if_train, if_save

    def run(self, args, agent_id):
        # print(f'| os.getpid()={os.getpid()} PipeEvaluate.run {agent_id}')
        args.init_before_training(if_main=False)

        '''init: Agent'''
        env = build_env(args.env, if_print=False)
        agent = args.agent
        agent.init(args.net_dim, env.state_dim, env.action_dim, args.learning_rate,
                   args.if_per_or_gae, args.env_num, agent_id=args.eval_device_id)
        agent.save_or_load_agent(args.cwd, if_save=False)

        act_cpu = agent.act
        act_cpu.eval()
        [setattr(param, 'requires_grad', False) for param in act_cpu.parameters()]

        '''init Evaluator'''
        eval_env = args.eval_env if args.eval_env else build_env(env, if_print=False)
        evaluator = Evaluator(args.cwd, agent_id, agent.device, eval_env,
                              args.eval_gap, args.eval_times1, args.eval_times2)
        evaluator.save_or_load_recoder(if_save=False)
        del agent
        del env

        '''loop'''
        cwd = args.cwd
        break_step = args.break_step
        if_allow_break = args.if_allow_break
        del args

        if_save = False
        if_train = True
        if_reach_goal = False
        with torch.no_grad():
            while if_train:
                act_cpu_dict, steps, r_exp, logging_tuple = self.pipe0.recv()

                if act_cpu_dict:
                    act_cpu.load_state_dict(act_cpu_dict)
                    if_reach_goal, if_save = evaluator.evaluate_and_save(act_cpu, steps, r_exp, logging_tuple)
                else:
                    evaluator.total_step += steps

                if_train = not ((if_allow_break and if_reach_goal)
                                or evaluator.total_step > break_step
                                or os.path.exists(f'{cwd}/stop'))
                self.pipe0.send((if_train, if_save))

        print(f'| UsedTime: {time.time() - evaluator.start_time:>7.0f} | SavedDir: {cwd}')
        evaluator.save_or_load_recoder(if_save=True)


class PipeVectorEnv:
    def __init__(self, args):
        self.env_num = args.env_num
        self.pipes = [mp.Pipe() for _ in range(self.env_num)]
        self.pipe0s = [pipe[0] for pipe in self.pipes]

        env = build_env(args.eval_env)
        self.max_step = env.max_step
        self.env_name = env.env_name
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        self.action_max = env.action_max
        self.if_discrete = env.if_discrete
        self.target_return = env.target_return
        del env

        self.process = list()
        for env_id in range(args.env_num):
            self.process.append(mp.Process(target=self.run, args=(args, env_id)))
            args.random_seed += 1  # set different for each env
        # [p.start() for p in self.process]

    def reset(self):
        vec_state = [pipe0.recv() for pipe0 in self.pipe0s]
        return vec_state

    def step(self, vec_action):  # pipe0_step
        for i in range(self.env_num):
            self.pipe0s[i].send(vec_action[i])
        return [pipe0.recv() for pipe0 in self.pipe0s]  # list of (state, reward, done)

    def run(self, args, env_id):
        np.random.seed(args.random_seed)

        env = build_env(args.eval_env, if_print=False)
        pipe1 = self.pipes[env_id][1]
        del args

        state = env.reset()
        pipe1.send(state)

        while True:
            action = pipe1.recv()
            state, reward, done, _ = env.step(action)
            pipe1.send((env.reset() if done else state, reward, done))

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


def train_and_evaluate_mp(args, agent_id=0):
    process = list()
    mp.set_start_method(method='spawn', force=True)  # force all the multiprocessing to 'spawn' methods

    '''learner'''
    learner_num = get_num_learner(args.visible_gpu)
    learner_pipe = PipeLearner(learner_num)
    for learner_id in range(learner_num):
        '''evaluator'''
        if learner_id == learner_num - 1:
            evaluator_pipe = PipeEvaluator()
            process.append(mp.Process(target=evaluator_pipe.run, args=(args, agent_id)))
        else:
            evaluator_pipe = None

        '''explorer'''
        worker_pipe = PipeWorker(args.env_num, args.worker_num)
        for worker_id in range(args.worker_num):
            if args.env_num == 1:
                env_pipe = None
            else:
                env_pipe = PipeVectorEnv(args)
                process.extend(env_pipe.process)
            process.append(mp.Process(target=worker_pipe.run, args=(args, env_pipe, worker_id, learner_id)))

        process.append(mp.Process(target=learner_pipe.run, args=(args, evaluator_pipe, worker_pipe, learner_id)))

    [(p.start(), time.sleep(0.1)) for p in process]
    process[-1].join()
    process_safely_terminate(process)


"""Utils"""


def get_num_learner(visible_gpu):
    assert isinstance(visible_gpu, str)  # visible_gpu may in {'0', '1', '1,', '1,2', '1,2,'}
    visible_gpu = eval(visible_gpu)
    num_learner = 1 if isinstance(visible_gpu, int) else len(visible_gpu)
    return num_learner


def process_safely_terminate(process):
    for p in process:
        try:
            p.kill()
        except OSError as e:
            print(e)
            pass


def get_optim_parameters(optim):  # for avg_update_optim()
    params_list = list()
    for params_dict in optim.state_dict()['state'].values():
        params_list.extend([t for t in params_dict.values() if isinstance(t, torch.Tensor)])
    return params_list


def avg_update_optim(dst_optim, src_optim_param, device):
    for dst, src in zip(get_optim_parameters(dst_optim), src_optim_param):
        dst.data.copy_((dst.data + src.data.to(device)) * 0.5)
        # dst.data.copy_(src.data * tau + dst.data * (1 - tau))


def avg_update_net(dst_net, src_net_param, device):
    for dst, src in zip(dst_net.parameters(), src_net_param):
        dst.data.copy_((dst.data + src.data.to(device)) * 0.5)
