import os
import time
import torch
import numpy as np
import multiprocessing as mp

from elegantrl.agents.AgentBase import AgentBase
from elegantrl.train.config import build_env, Arguments
from elegantrl.train.evaluator import Evaluator
from elegantrl.train.replay_buffer import ReplayBuffer, ReplayBufferList

'''[ElegantRL.2022.05.05](github.com/AI4Fiance-Foundation/ElegantRL)'''


def train_and_evaluate(args: Arguments):
    """
    The training and evaluating loop.

    :param args: an object of ``Arguments`` class, which contains all hyper-parameters.
    """
    torch.set_grad_enabled(False)
    args.init_before_training()
    gpu_id = args.learner_gpus

    '''init'''
    env = build_env(args.env, args.env_func, args.env_args)

    agent = init_agent(args, gpu_id, env)
    buffer = init_buffer(args, gpu_id)
    evaluator = init_evaluator(args, gpu_id)

    agent.state = env.reset()
    if args.if_off_policy:
        trajectory = agent.explore_env(env, args.target_step)
        buffer.update_buffer([trajectory, ])

    '''start training'''
    cwd = args.cwd
    break_step = args.break_step
    target_step = args.target_step
    if_allow_break = args.if_allow_break
    del args

    if_train = True
    while if_train:
        trajectory = agent.explore_env(env, target_step)
        steps, r_exp = buffer.update_buffer([trajectory, ])

        torch.set_grad_enabled(True)
        logging_tuple = agent.update_net(buffer)
        torch.set_grad_enabled(False)

        (if_reach_goal, if_save) = evaluator.evaluate_save_and_plot(agent.act, steps, r_exp, logging_tuple)
        dont_break = not if_allow_break
        not_reached_goal = not if_reach_goal
        stop_dir_absent = not os.path.exists(f"{cwd}/stop")
        if_train = (
                (dont_break or not_reached_goal)
                and evaluator.total_step <= break_step
                and stop_dir_absent
        )
    print(f'| UsedTime: {time.time() - evaluator.start_time:.0f} | SavedDir: {cwd}')

    agent.save_or_load_agent(cwd, if_save=True)

    buffer.get_state_norm(
        cwd=cwd,
        state_avg=getattr(env, 'neg_state_avg', 0),
        state_std=getattr(env, 'div_state_std', 1),
    )
    if hasattr(buffer, 'save_or_load_history'):
        print(f"| LearnerPipe.run: ReplayBuffer saving in {cwd}")
        buffer.save_or_load_history(cwd, if_save=True)


def init_agent(args: Arguments, gpu_id: int, env=None) -> AgentBase:
    agent = args.agent_class(args.net_dim, args.state_dim, args.action_dim, gpu_id=gpu_id, args=args)
    agent.save_or_load_agent(args.cwd, if_save=False)

    if env is not None:
        '''assign `agent.states` for exploration'''
        if args.env_num == 1:
            state = env.reset()
            assert isinstance(state, np.ndarray) or isinstance(state, torch.Tensor)
            assert state.shape in {(args.state_dim,), args.state_dim}
            states = [state, ]
        else:
            states = env.reset()
            assert isinstance(states, torch.Tensor)
            assert states.shape == (args.env_num, args.state_dim)
        agent.states = states
    return agent


def init_buffer(args: Arguments, gpu_id: int) -> [ReplayBuffer or ReplayBufferList]:
    if args.if_off_policy:
        buffer = ReplayBuffer(gpu_id=gpu_id,
                              max_capacity=args.max_memo,
                              state_dim=args.state_dim,
                              action_dim=1 if args.if_discrete else args.action_dim, )
        buffer.save_or_load_history(args.cwd, if_save=False)

    else:
        buffer = ReplayBufferList()
    return buffer


def init_evaluator(args: Arguments, gpu_id: int) -> Evaluator:
    eval_func = args.eval_env_func if getattr(args, "eval_env_func") else args.env_func
    eval_args = args.eval_env_args if getattr(args, "eval_env_args") else args.env_args
    eval_env = build_env(args.env, eval_func, eval_args)
    evaluator = Evaluator(cwd=args.cwd, agent_id=gpu_id, eval_env=eval_env, args=args)
    return evaluator


'''train multiple process'''


def train_and_evaluate_mp(args: Arguments):
    args.init_before_training()

    process = list()
    mp.set_start_method(method='spawn', force=True)  # force all the multiprocessing to 'spawn' methods

    evaluator_pipe = PipeEvaluator()
    process.append(mp.Process(target=evaluator_pipe.run, args=(args,)))

    worker_pipe = PipeWorker(args.worker_num)
    process.extend([mp.Process(target=worker_pipe.run, args=(args, worker_id))
                    for worker_id in range(args.worker_num)])

    learner_pipe = PipeLearner()
    process.append(mp.Process(target=learner_pipe.run, args=(args, evaluator_pipe, worker_pipe)))

    for p in process:
        p.start()

    process[-1].join()  # waiting for learner
    process_safely_terminate(process)


class PipeWorker:
    def __init__(self, worker_num: int):
        self.worker_num = worker_num
        self.pipes = [mp.Pipe() for _ in range(worker_num)]
        self.pipe1s = [pipe[1] for pipe in self.pipes]

    def explore(self, agent: AgentBase):
        act_dict = agent.act.state_dict()

        for worker_id in range(self.worker_num):
            self.pipe1s[worker_id].send(act_dict)

        traj_lists = [pipe1.recv() for pipe1 in self.pipe1s]
        return traj_lists

    def run(self, args: Arguments, worker_id: int):
        torch.set_grad_enabled(False)
        gpu_id = args.learner_gpus

        '''init'''
        env = build_env(args.env, args.env_func, args.env_args)
        agent = init_agent(args, gpu_id, env)

        '''loop'''
        target_step = args.target_step
        if args.if_off_policy:
            trajectory = agent.explore_env(env, args.target_step)
            self.pipes[worker_id][0].send(trajectory)
        del args

        while True:
            act_dict = self.pipes[worker_id][0].recv()
            agent.act.load_state_dict(act_dict)
            trajectory = agent.explore_env(env, target_step)
            self.pipes[worker_id][0].send(trajectory)


class PipeLearner:
    def __init__(self):
        pass

    @staticmethod
    def run(args: Arguments, comm_eva: mp.Pipe, comm_exp: mp.Pipe):
        torch.set_grad_enabled(False)
        gpu_id = args.learner_gpus
        cwd = args.cwd

        '''init'''
        agent = init_agent(args, gpu_id)
        buffer = init_buffer(args, gpu_id)

        '''loop'''
        if_train = True
        while if_train:
            traj_list = comm_exp.explore(agent)
            steps, r_exp = buffer.update_buffer(traj_list)

            torch.set_grad_enabled(True)
            logging_tuple = agent.update_net(buffer)
            torch.set_grad_enabled(False)

            if_train, if_save = comm_eva.evaluate_and_save_mp(agent.act, steps, r_exp, logging_tuple)
        agent.save_or_load_agent(cwd, if_save=True)
        print(f'| Learner: Save in {cwd}')

        env = build_env(env_func=args.env_func, env_args=args.env_args)
        buffer.get_state_norm(
            cwd=cwd,
            state_avg=getattr(env, 'state_avg', 0.0),
            state_std=getattr(env, 'state_std', 1.0),
        )
        if hasattr(buffer, 'save_or_load_history'):
            print(f"| LearnerPipe.run: ReplayBuffer saving in {cwd}")
            buffer.save_or_load_history(cwd, if_save=True)


class PipeEvaluator:
    def __init__(self):
        self.pipe0, self.pipe1 = mp.Pipe()

    def evaluate_and_save_mp(self, act, steps: int, r_exp: float, logging_tuple: tuple) -> (bool, bool):
        if self.pipe1.poll():  # if_evaluator_idle
            if_train, if_save_agent = self.pipe1.recv()
            act_state_dict = act.state_dict().copy()  # deepcopy(act.state_dict())
        else:
            if_train = True
            if_save_agent = False
            act_state_dict = None

        self.pipe1.send((act_state_dict, steps, r_exp, logging_tuple))
        return if_train, if_save_agent

    def run(self, args: Arguments):
        torch.set_grad_enabled(False)
        gpu_id = args.learner_gpus

        '''init'''
        agent = init_agent(args, gpu_id)
        evaluator = init_evaluator(args, gpu_id)

        '''loop'''
        cwd = args.cwd
        act = agent.act
        break_step = args.break_step
        if_allow_break = args.if_allow_break
        save_gap = args.save_gap
        del args

        if_save = False
        if_train = True
        if_reach_goal = False
        save_counter = 0
        while if_train:
            act_dict, steps, r_exp, logging_tuple = self.pipe0.recv()

            if act_dict:
                act.load_state_dict(act_dict)
                if_reach_goal, if_save = evaluator.evaluate_save_and_plot(act, steps, r_exp, logging_tuple)

                save_counter += 1
                if save_counter == save_gap:
                    save_counter = 0
                    torch.save(act.state_dict(), f"{cwd}/actor_{evaluator.total_step:012}.pth")
            else:
                evaluator.total_step += steps

            if_train = not ((if_allow_break and if_reach_goal)
                            or evaluator.total_step > break_step
                            or os.path.exists(f'{cwd}/stop'))
            self.pipe0.send((if_train, if_save))

        print(f'| UsedTime: {time.time() - evaluator.start_time:>7.0f} | SavedDir: {cwd}')

        while True:  # wait for the forced stop from main process
            self.pipe0.recv()
            self.pipe0.send((False, False))


def process_safely_terminate(process: list):
    for p in process:
        try:
            p.kill()
        except OSError as e:
            print(e)
