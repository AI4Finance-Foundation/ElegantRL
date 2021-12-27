import os
import time
import inspect
import multiprocessing as mp
from copy import deepcopy

import torch
import numpy as np

from elegantrl.train.config import Arguments
from elegantrl.train.evaluator import Evaluator
from elegantrl.train.replay_buffer import ReplayBuffer, ReplayBufferMP
from elegantrl.agents.AgentPPO import AgentPPO

'''env.py'''


def kwargs_filter(func, kwargs: dict):  # [ElegantRL.2021.12.12]
    """How does one ignore `unexpected keyword arguments passed to a function`?
    https://stackoverflow.com/a/67713935/9293137

    class ClassTest:
        def __init__(self, a, b=1):
            print(f'| ClassTest: a + b == {a + b}')

    old_kwargs = {'a': 1, 'b': 2, 'c': 3}
    new_kwargs = kwargs_filter(ClassTest.__init__, old_kwargs)
    assert new_kwargs == {'a': 1, 'b': 2}
    test = ClassTest(**new_kwargs)

    :param func: func(**kwargs)
    :param kwargs: the KeyWordArguments wait for
    :return: kwargs: [dict] filtered kwargs
    """

    sign = inspect.signature(func).parameters.values()
    sign = set([val.name for val in sign])

    common_args = sign.intersection(kwargs.keys())
    return {key: kwargs[key] for key in common_args}  # filtered kwargs


def build_env(env=None, env_func=None, env_args=None, device_id=-1):  # [ElegantRL.2021.12.12]
    if env is not None:
        env = deepcopy(env)
    else:
        env_args = deepcopy(env_args)
        env_args['device_id'] = device_id  # -1 means CPU, int >=1 means GPU id
        env_args = kwargs_filter(env_func.__init__, env_args)
        env = env_func(**env_args)
    return env


'''run.py'''


def train_and_evaluate(args):  # 2021.12.12
    args.init_before_training()  # necessary!
    learner_gpu = args.learner_gpus[0, 0]

    env = build_env(env=args.env, env_func=args.env_func, env_args=args.env_args)
    agent = init_agent(args, learner_gpu=learner_gpu, env=env)
    evaluator = init_evaluator(args)
    buffer, update_buffer = init_replay_buffer(args, learner_gpu, agent, env=env)

    """start training"""
    cwd = args.cwd
    break_step = args.break_step
    batch_size = args.batch_size
    target_step = args.target_step
    repeat_times = args.repeat_times
    if_allow_break = args.if_allow_break
    soft_update_tau = args.soft_update_tau
    del args

    '''start training loop'''
    if_train = True
    while if_train:
        traj_list = agent.explore_env(env, target_step)
        steps, r_exp = update_buffer(traj_list)

        torch.set_grad_enabled(True)
        logging_tuple = agent.update_net(buffer, batch_size, repeat_times, soft_update_tau)
        torch.set_grad_enabled(False)

        if_reach_goal, if_save = evaluator.evaluate_and_save(agent.act, steps, r_exp, logging_tuple)
        if_train = not ((if_allow_break and if_reach_goal)
                        or evaluator.total_step > break_step
                        or os.path.exists(f'{cwd}/stop'))

    print(f'| UsedTime: {time.time() - evaluator.start_time:>7.0f} | SavedDir: {cwd}')

    agent.save_or_load_agent(cwd, if_save=True)
    buffer.save_or_load_history(cwd, if_save=True) if agent.if_off_policy else None
    evaluator.save_or_load_recoder(if_save=True)


def init_agent(args, learner_gpu=0, env=None):
    agent = args.agent
    agent.init(net_dim=args.net_dim, state_dim=args.state_dim, action_dim=args.action_dim,
               gamma=args.gamma, reward_scale=args.reward_scale,
               learning_rate=args.learning_rate, if_per_or_gae=args.if_per_or_gae,
               env_num=args.env_num, gpu_id=learner_gpu, )
    agent.save_or_load_agent(args.cwd, if_save=False)

    if env is not None:
        '''init states'''
        if args.env_num == 1:
            states = [env.reset(), ]
            assert isinstance(states[0], np.ndarray)
            assert states[0].shape in {(args.state_dim,), args.state_dim}
        else:
            states = env.reset()
            assert isinstance(states, torch.Tensor)
            assert states.shape == (args.env_num, args.state_dim)
        agent.states = states

    return agent


def init_evaluator(args):
    eval_env = build_env(env=args.eval_env, device_id=args.eval_gpu_id,
                         env_func=args.eval_env_class, env_args=args.eval_env_args)
    evaluator = Evaluator(cwd=args.cwd, agent_id=0,
                          eval_env=eval_env, eval_gap=args.eval_gap,
                          eval_times1=args.eval_times1, eval_times2=args.eval_times2,
                          target_return=args.target_return, if_overwrite=args.if_overwrite)
    evaluator.save_or_load_recoder(if_save=False)
    return evaluator


def init_replay_buffer(args, learner_gpu, agent=None, env=None):
    def get_step_r_exp(ten_reward):
        return len(ten_reward), ten_reward.mean().item()

    if args.if_off_policy:
        buffer = ReplayBuffer(max_len=args.max_memo, state_dim=args.state_dim,
                              action_dim=1 if args.if_discrete else args.action_dim,
                              if_use_per=args.if_per_or_gae, gpu_id=learner_gpu)
        buffer.save_or_load_history(args.cwd, if_save=False)

        def update_buffer(traj_list):
            ten_state, ten_other = traj_list[0]
            buffer.extend_buffer(ten_state, ten_other)

            steps, r_exp = get_step_r_exp(ten_reward=ten_other)
            return steps, r_exp

        if_load = buffer.save_or_load_history(args.cwd, if_save=False)
        if (env is not None) and (not if_load):
            update_buffer(agent.explore_env(env, args.target_step))

    else:
        buffer = list()

        def update_buffer(traj_list):
            (ten_state, ten_reward, ten_mask, ten_action, ten_noise) = traj_list[0]
            buffer[:] = (ten_state.squeeze(1),
                         ten_reward,
                         ten_mask,
                         ten_action.squeeze(1),
                         ten_noise.squeeze(1))

            step, r_exp = get_step_r_exp(ten_reward=buffer[1])
            return step, r_exp
    return buffer, update_buffer


'''run_mp.py'''


def train_and_evaluate_mp(args, agent_id=0):
    args.init_before_training()  # necessary!

    process = list()
    mp.set_start_method(method='spawn', force=True)  # force all the multiprocessing to 'spawn' methods

    '''evaluator'''
    evaluator_pipe = PipeEvaluator()
    process.append(mp.Process(target=evaluator_pipe.run, args=(args, agent_id)))

    learner_num = args.learner_gpus.shape[1]
    learner_pipe = PipeLearner(learner_num)
    for learner_id in range(learner_num):
        '''explorer'''
        worker_pipe = PipeWorker(args.env_num, args.worker_num)
        process.extend([mp.Process(target=worker_pipe.run, args=(args, worker_id, learner_id))
                        for worker_id in range(args.worker_num)])

        '''learner'''
        evaluator_temp = evaluator_pipe if learner_id == 0 else None
        process.append(mp.Process(target=learner_pipe.run, args=(args, evaluator_temp, worker_pipe, learner_id)))

    [(p.start(), time.sleep(0.1)) for p in process]
    process[0].join()
    process_safely_terminate(process)


class PipeWorker:
    def __init__(self, env_num, worker_num):
        self.env_num = env_num
        self.worker_num = worker_num
        self.pipes = [mp.Pipe() for _ in range(worker_num)]
        self.pipe1s = [pipe[1] for pipe in self.pipes]

    def explore(self, agent):
        act_dict = agent.act.state_dict()

        if sys.platform == 'win32':  # Avoid CUDA runtime error (801)
            # Python3.9< multiprocessing can't send torch.tensor_gpu in WinOS. So I send torch.tensor_cpu
            for key, value in act_dict.items():
                act_dict[key] = value.to(torch.device('cpu'))

        for worker_id in range(self.worker_num):
            self.pipe1s[worker_id].send(act_dict)

        traj_lists = [pipe1.recv() for pipe1 in self.pipe1s]
        return traj_lists

    def run(self, args, worker_id, learner_id):
        learner_gpu = args.learner_gpus[learner_id]

        '''init Agent'''
        env = build_env(env=args.env, env_func=args.env_func, env_args=args.env_args)
        agent = init_agent(args, learner_gpu=learner_gpu, env=env)

        '''loop'''
        target_step = args.target_step
        del args

        torch.set_grad_enabled(False)
        while True:
            act_dict = self.pipes[worker_id][0].recv()
            change_act_dict_to_device(act_dict, agent.device) if sys.platform == 'win32' else None
            # WinOS Python<3.9, pipe can't send torch.tensor_gpu, but tensor_cpu can.

            agent.act.load_state_dict(act_dict)

            trajectory = agent.explore_env(env, target_step)
            change_trajectory_to_device(trajectory, torch.device('cpu')) if sys.platform == 'win32' else None
            # WinOS Python<3.9, pipe can't send torch.tensor_gpu, but tensor_cpu can.

            self.pipes[worker_id][0].send(trajectory)


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
            data = self.get_comm_data(agent)
            data = self.comm_data(data, learner_id, round_id)

            if data:
                self.average_param(agent.act.parameters(), data[0], device)
                self.average_param(agent.act_optim.parameters(), data[1], device) if data[1] else None
                # self.average_param(self.get_optim_param(agent.act_optim), data[1], device) if data[1] else None

                self.average_param(agent.cri.parameters(), data[2], device) if data[2] else None
                self.average_param(agent.cri_optim.parameters(), data[3], device)  # todo plan to be elegant
                # self.average_param(self.get_optim_param(agent.cri_optim), data[3], device)  # todo plan to be elegant

                self.average_param(agent.act_target.parameters(), data[4], device) if agent.if_use_act_target else None
                self.average_param(agent.cri_target.parameters(), data[5], device) if agent.if_use_cri_target else None

    def run(self, args, comm_eva, comm_exp, learner_id):
        learner_gpu = args.learner_gpus[learner_id]

        agent = init_agent(args, learner_gpu=learner_gpu, env=None)
        buffer, update_buffer = init_replay_buffer(args, learner_gpu, agent=None, env=None)

        '''start training'''
        cwd = args.cwd
        batch_size = args.batch_size
        repeat_times = args.repeat_times
        soft_update_tau = args.soft_update_tau
        del args

        '''add parameter() methods to object'''
        agent.act_optim.parameters = get_optim_param
        agent.cri_optim.parameters = get_optim_param

        if_train = True
        while if_train:
            traj_lists = comm_exp.explore(agent)
            if self.learner_num > 1:
                data = self.comm_data(traj_lists, learner_id, round_id=-1)
                traj_lists.extend(data)
            traj_list = sum(traj_lists, list())
            del traj_lists

            steps, r_exp = update_buffer(traj_list)

            logging_tuple = agent.update_net(buffer, batch_size, repeat_times, soft_update_tau)
            if self.learner_num > 1:
                self.comm_network_optim(agent, learner_id)

            if comm_eva:
                if_train, if_save = comm_eva.evaluate_and_save_mp(agent.act, steps, r_exp, logging_tuple)

        agent.save_or_load_agent(cwd, if_save=True)

        if hasattr(buffer, 'save_or_load_history'):
            print(f"| LearnerPipe.run: ReplayBuffer saving in {cwd}")
            buffer.save_or_load_history(cwd, if_save=True)

    def get_comm_data(self, agent):
        act = list(agent.act.parameters())
        cri_optim = self.get_optim_param(agent.cri_optim)

        if agent.cri is agent.act:
            cri = None
            act_optim = None
        else:
            cri = list(agent.cri.parameters())
            act_optim = self.get_optim_param(agent.act_optim)

        act_target = list(agent.act_target.parameters()) if agent.if_use_act_target else None
        cri_target = list(agent.cri_target.parameters()) if agent.if_use_cri_target else None
        return act, act_optim, cri, cri_optim, act_target, cri_target  # data

    @staticmethod
    def get_optim_param(optim):  # for avg_update_optim()
        params_list = list()
        for params_dict in optim.state_dict()['state'].values():
            params_list.extend([t for t in params_dict.values() if isinstance(t, torch.Tensor)])
        return params_list

    @staticmethod
    def average_param(dst_optim_param, src_optim_param, device):
        for dst, src in zip(dst_optim_param, src_optim_param):
            dst.data.copy_((dst.data + src.data.to(device)) * 0.5)
            # dst.data.copy_(src.data * tau + dst.data * (1 - tau))

def get_optim_param(self): # self = torch.optim.Adam(network_param, learning_rate)
    params_list = list()
    for params_dict in self.state_dict()['state'].values():
        params_list.extend([t for t in params_dict.values() if isinstance(t, torch.Tensor)])
    return params_list

class PipeEvaluator:  # [ElegantRL.10.21]
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
        gpu_id = args.eval_gpu_id
        pass

        agent = init_agent(args, learner_gpu=gpu_id, env=None)
        act = agent.act
        del agent

        evaluator = init_evaluator(args)

        '''loop'''
        cwd = args.cwd
        break_step = args.break_step
        if_allow_break = args.if_allow_break
        if_save = args.if_save
        if_reach_goal = args.if_reach_goal
        del args

        torch.set_grad_enabled(False)
        if_train = True
        while if_train:
            act_dict, steps, r_exp, logging_tuple = self.pipe0.recv()

            if act_dict:
                act.load_state_dict(act_dict)
                if_reach_goal, if_save = evaluator.evaluate_and_save(act, steps, r_exp, logging_tuple)
            else:
                evaluator.total_step += steps

            if_train = not ((if_allow_break and if_reach_goal)
                            or evaluator.total_step > break_step
                            or os.path.exists(f'{cwd}/stop'))
            self.pipe0.send((if_train, if_save))

        print(f'| UsedTime: {time.time() - evaluator.start_time:>7.0f} | SavedDir: {cwd}')
        evaluator.save_or_load_recoder(if_save=True)


def process_safely_terminate(process):
    for p in process:
        try:
            p.kill()
        except OSError as e:
            print(e)
            pass


def change_act_dict_to_device(act_dict, device):
    for key, value in act_dict.items():
        act_dict[key] = value.to(device)


def change_trajectory_to_device(trajectory, device):
    trajectory[:] = [[item.to(device)
                      for item in item_list]
                     for item_list in trajectory]


'''demo'''


def demo_continuous_action():
    from elegantrl.envs.Chasing import ChasingEnv
    dim = DIM
    env_func = ChasingEnv
    env_args = {
        # the parameter of env information
        'env_num': 1,
        'env_name': 'GoAfterEnv',
        'max_step': 1000,
        'state_dim': dim * 4,
        'action_dim': dim,
        'if_discrete': False,
        'target_return': 6.3,

        # the parameter for `env = env_func(env_args)`
        'dim': dim, }

    # args = Arguments(agent=AgentPPO(), env=build_env(env_func, env_args))
    args = Arguments(agent=AgentPPO(), env_func=env_func, env_args=env_args)

    args.net_dim = 2 ** 7
    args.batch_size = args.net_dim * 2
    args.target_step = args.max_step * 4

    args.learner_gpus = GPU

    if isinstance(args.learner_gpus, int):
        train_and_evaluate(args)
    else:
        train_and_evaluate_mp(args)


if __name__ == '__main__':
    import sys

    sys.argv.extend('0 2'.split(' '))
    GPU = eval(sys.argv[1])
    DIM = eval(sys.argv[2])
    demo_continuous_action()
