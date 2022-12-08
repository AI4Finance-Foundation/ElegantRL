import multiprocessing as mp
import os
import time

import numpy as np
import torch

from elegantrl.agents.AgentBase import AgentBase
from elegantrl.train.config import Arguments
from elegantrl.train.config import build_env
from elegantrl.train.evaluator import Evaluator
from elegantrl.train.replay_buffer import ReplayBuffer, ReplayBufferList

"""vectorized env"""

from copy import deepcopy
from elegantrl.agents.AgentBase import AgentBaseVecEnv
from elegantrl.train.config import Config, build_vec_env
from elegantrl.train.replay_buffer import ReplayBufferVecEnv
from elegantrl.train.evaluator_vec_env import EvaluatorVecEnv


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


def init_agent_isaacgym(args, gpu_id: int, env=None):
    agent = args.agent(args.net_dim, args.state_dim, args.action_dim, gpu_id=gpu_id, args=args)
    agent.save_or_load_agent(args.cwd, if_save=False)

    if env is not None:
        '''assign `agent.states` for exploration'''
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


def train_and_evaluate(args):
    """
    The training and evaluating loop.

    :param args: an object of ``Arguments`` class, which contains all hyper-parameters.
    """
    torch.set_grad_enabled(False)
    args.init_before_training()
    gpu_id = args.learner_gpus

    '''init'''
    env = build_env(args.env, args.env_func, args.env_args)
    steps = 0

    agent = init_agent(args, gpu_id, env)
    buffer = init_buffer(args, gpu_id)
    evaluator = init_evaluator(args, gpu_id)
    agent.state = env.reset()
    if args.if_off_policy:
        trajectory = agent.explore_env(env, args.num_seed_steps * args.num_steps_per_episode)
        buffer.update_buffer(trajectory)

    '''start training'''
    cwd = args.cwd
    break_step = args.break_step
    horizon_len = args.horizon_len
    if_allow_break = args.if_allow_break
    if_off_policy = args.if_off_policy
    del args
    steps = 0
    if_train = True
    while if_train:
        trajectory = agent.explore_env(env, horizon_len)
        steps = horizon_len
        if if_off_policy:
            buffer.update_buffer(trajectory)
            torch.set_grad_enabled(True)
            logging_tuple = agent.update_net(buffer)
            torch.set_grad_enabled(False)
        else:
            r_exp = trajectory[3].mean().item()
            torch.set_grad_enabled(True)
            logging_tuple = agent.update_net(trajectory)
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


# import wandb
class PipeLearner:
    def __init__(self):
        # wandb.init(project="DDPG_H")
        pass

    @staticmethod
    def run(args: Arguments, comm_eva: mp.Pipe, comm_exp: mp.Pipe):
        torch.set_grad_enabled(False)
        gpu_id = args.learner_gpus
        cwd = args.cwd
        # wandb.init(project="DDPG_H")

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
            # wandb.log({"obj_cri": logging_tuple[0], "obj_act": logging_tuple[1]})
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


"""vectorized env"""

if os.name == 'nt':  # if is WindowOS (Windows NT)
    """Fix bug about Anaconda in WindowOS
    OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
    """
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def train_agent(args: Config):
    args.init_before_training()
    torch.set_grad_enabled(False)

    '''init environment'''
    env = build_vec_env(args.env_class, args.env_args, args.gpu_id)

    '''init agent'''
    agent = args.agent_class(args.net_dims, args.state_dim, args.action_dim, gpu_id=args.gpu_id, args=args)
    agent.save_or_load_agent(args.cwd, if_save=False)

    '''init agent.last_state'''
    state = env.reset()
    if args.num_envs == 1:
        assert state.shape == (args.state_dim,)
        assert isinstance(state, np.ndarray)
        state = torch.tensor(env.reset(), dtype=torch.float32, device=agent.device).unsqueeze(0)
    else:
        assert state.shape == (args.num_envs, args.state_dim)
        assert isinstance(state, torch.Tensor)
        state = env.reset().to(agent.device)
    assert state.shape == (args.num_envs, args.state_dim)
    assert isinstance(state, torch.Tensor)
    agent.last_state = state.detach()

    '''init buffer'''
    if args.if_off_policy:
        buffer = ReplayBufferVecEnv(gpu_id=args.gpu_id,
                                    num_envs=args.num_envs,
                                    max_size=args.buffer_size,
                                    state_dim=args.state_dim,
                                    action_dim=1 if args.if_discrete else args.action_dim)
        buffer_items = agent.explore_env(env, args.horizon_len * args.eval_times, if_random=True)
        buffer.update(buffer_items)  # warm up for ReplayBuffer
    else:
        buffer = []

    '''init evaluator'''
    eval_env_class = args.eval_env_class if args.eval_env_class else args.env_class
    eval_env_args = args.eval_env_args if args.eval_env_args else args.env_args
    eval_env = build_vec_env(eval_env_class, eval_env_args, args.gpu_id)
    evaluator = EvaluatorVecEnv(cwd=args.cwd, agent_id=0, eval_env=eval_env, args=args, if_tensorboard=False)

    '''train loop'''
    cwd = args.cwd
    break_step = args.break_step
    horizon_len = args.horizon_len
    if_off_policy = args.if_off_policy
    del args

    if_train = True
    while if_train:
        buffer_items = agent.explore_env(env, horizon_len)
        exp_r = buffer_items[2].mean().item()
        if if_off_policy:
            buffer.update(buffer_items)
        else:
            buffer[:] = buffer_items

        torch.set_grad_enabled(True)
        logging_tuple = agent.update_net(buffer)
        torch.set_grad_enabled(False)

        evaluator.evaluate_and_save(actor=agent.act, steps=horizon_len, exp_r=exp_r, logging_tuple=logging_tuple)
        if_train = (evaluator.total_step <= break_step) and (not os.path.exists(f"{cwd}/stop"))

    print(f'| UsedTime: {time.time() - evaluator.start_time:>7.0f} | SavedDir: {cwd}')
    evaluator.save_training_curve_jpg()
    agent.save_or_load_agent(cwd, if_save=True)
    buffer.save_or_load_history(cwd, if_save=True) if if_off_policy else None


def train_agent_multiprocessing(args: Config):
    args.init_before_training()

    mp.set_start_method(method='spawn', force=True)  # force all the multiprocessing to 'spawn' methods

    evaluator_pipe = PipeEvaluatorVecEnv()
    evaluator_proc = mp.Process(target=evaluator_pipe.run, args=(args,))

    worker_pipe = PipeWorkerVecEnv(args.num_workers)
    worker_procs = [mp.Process(target=worker_pipe.run, args=(args, worker_id)) for worker_id in
                    range(args.num_workers)]

    learner_pipe = PipeLearnerVecEnv()
    learner_proc = mp.Process(target=learner_pipe.run, args=(args, evaluator_pipe, worker_pipe))

    process = worker_procs + [learner_proc, evaluator_proc]
    [p.start() for p in process]
    [p.join() for p in process]


class PipeWorkerVecEnv:
    def __init__(self, worker_num: int):
        self.worker_num = worker_num
        pipes = [mp.Pipe() for _ in range(worker_num)]
        self.pipe0s = [pipe[0] for pipe in pipes]
        self.pipe1s = [pipe[1] for pipe in pipes]

    def explore(self, agent: AgentBaseVecEnv):
        actor = agent.act

        for worker_id in range(self.worker_num):
            self.pipe1s[worker_id].send(actor)

        recv_items = [pipe1.recv() for pipe1 in self.pipe1s]
        buffer_items, last_state = [item for item in zip(*recv_items)]

        buffer_items = [torch.cat(tensors, dim=1) for tensors in zip(*buffer_items)]
        """buffer_items
        states, actions, rewards, dones = buffer_items  # off-policy
        states, actions, logprobs, rewards, dones = buffer_items  # on-policy

        assert states.shape == (horizon_len, num_envs * worker_num, state_dim)
        assert actions.shape == (horizon_len, num_envs * worker_num, action_dim)
        assert logprobs.shape == (horizon_len, num_envs * worker_num, action_dim)
        assert rewards.shape == (horizon_len, num_envs * worker_num)
        assert dones.shape == (horizon_len, num_envs * worker_num)  
        """

        last_state = torch.cat(last_state, dim=0)
        """last_state
        assert last_state.shape == (num_envs * worker_num, state_dim)
        """
        return buffer_items, last_state

    def run(self, args: Config, worker_id: int):
        torch.set_grad_enabled(False)

        '''init environment'''
        env = build_env(args.env_class, args.env_args, args.gpu_id)

        '''init agent'''
        agent = args.agent_class(args.net_dims, args.state_dim, args.action_dim, gpu_id=args.gpu_id, args=args)
        agent.save_or_load_agent(args.cwd, if_save=False)

        '''init agent.last_state'''
        state = env.reset()
        if args.num_envs == 1:
            assert state.shape == (args.state_dim,)
            assert isinstance(state, np.ndarray)
            state = torch.tensor(env.reset(), dtype=torch.float32, device=agent.device).unsqueeze(0)
        else:
            assert state.shape == (args.num_envs, args.state_dim)
            assert isinstance(state, torch.Tensor)
            state = env.reset().to(agent.device)
        assert state.shape == (args.num_envs, args.state_dim)
        assert isinstance(state, torch.Tensor)
        agent.last_state = state.detach()

        '''init buffer'''
        horizon_len = args.horizon_len
        if args.if_off_policy:
            buffer_items = agent.explore_env(env, args.horizon_len * args.eval_times, if_random=True)
            self.pipe0s[worker_id].send((buffer_items, agent.last_state))
        del args

        '''loop'''
        while True:
            actor = self.pipe0s[worker_id].recv()
            if actor is None:
                break

            agent.act = actor
            buffer_items = agent.explore_env(env, horizon_len)
            self.pipe0s[worker_id].send((buffer_items, agent.last_state))

        '''close pipe1s'''
        while self.pipe1s[worker_id].poll():
            time.sleep(1)
            self.pipe1s[worker_id].recv()

        if hasattr(env, 'close'):
            env.close()


class PipeEvaluatorVecEnv:
    def __init__(self):
        self.pipe0, self.pipe1 = mp.Pipe()

    def evaluate_and_save(self, actor, steps: int, r_exp: float, logging_tuple: tuple) -> (bool, bool):
        if self.pipe1.poll():  # if_evaluator_idle
            if_train = self.pipe1.recv()
            actor = deepcopy(actor)
        else:
            if_train = True
            actor = None

        self.pipe1.send((actor, steps, r_exp, logging_tuple))
        return if_train

    def run(self, args: Config):
        torch.set_grad_enabled(False)
        # import wandb
        # wandb_project_name = "RL_training"
        # wandb.init(project=wandb_project_name)

        '''init evaluator'''
        eval_env_class = args.eval_env_class if args.eval_env_class else args.env_class
        eval_env_args = args.eval_env_args if args.eval_env_args else args.env_args
        eval_env = build_env(eval_env_class, eval_env_args, args.gpu_id)
        evaluator = EvaluatorVecEnv(cwd=args.cwd, agent_id=0, eval_env=eval_env, args=args, if_tensorboard=False)

        '''loop'''
        cwd = args.cwd
        break_step = args.break_step
        del args

        if_train = True
        while if_train:
            pipe0_recv = self.pipe0.recv()
            actor, steps, exp_r, logging_tuple = pipe0_recv
            # wandb.log({"obj_cri": logging_tuple[0], "obj_act": logging_tuple[1]})

            if actor is None:
                evaluator.total_step += steps  # update total_step but don't update recorder
                if_train = True
            else:
                evaluator.evaluate_and_save(actor, steps, exp_r, logging_tuple)
                if_train = (evaluator.total_step <= break_step) and (not os.path.exists(f"{cwd}/stop"))
            self.pipe0.send(if_train)

        evaluator.save_training_curve_jpg()
        print(f'| TrainingTime: {time.time() - evaluator.start_time:>7.0f} | SavedDir: {cwd}')

        if hasattr(eval_env, 'close'):
            eval_env.close()

        '''close pipe1'''
        while self.pipe1.poll():
            time.sleep(1)


class PipeLearnerVecEnv:
    def __init__(self):
        pass

    @staticmethod
    def run(args: Config, comm_eva: PipeEvaluatorVecEnv, comm_exp: PipeWorkerVecEnv):
        torch.set_grad_enabled(False)

        '''init agent'''
        agent = args.agent_class(args.net_dims, args.state_dim, args.action_dim, gpu_id=args.gpu_id, args=args)
        agent.save_or_load_agent(args.cwd, if_save=False)

        '''init buffer'''
        if args.if_off_policy:
            buffer = ReplayBufferVecEnv(gpu_id=args.gpu_id,
                                        num_envs=args.num_envs * args.num_workers,
                                        max_size=args.buffer_size,
                                        state_dim=args.state_dim,
                                        action_dim=1 if args.if_discrete else args.action_dim)
        else:
            buffer = []

        '''loop'''
        if_off_policy = args.if_off_policy
        if_save_buffer = args.if_save_buffer
        steps = args.horizon_len * args.num_workers
        cwd = args.cwd
        del args

        if_train = True
        while if_train:
            buffer_items, last_state = comm_exp.explore(agent)
            exp_r = buffer_items[2].mean().item()
            if if_off_policy:
                buffer.update(buffer_items)
            else:
                buffer[:] = buffer_items
            agent.last_state = last_state

            torch.set_grad_enabled(True)
            logging_tuple = agent.update_net(buffer)
            torch.set_grad_enabled(False)

            if_train = comm_eva.evaluate_and_save(agent.act, steps, exp_r, logging_tuple)

        agent.save_or_load_agent(cwd, if_save=True)
        # print(f'| Learner: Save in {cwd}')

        if if_save_buffer and hasattr(buffer, 'save_or_load_history'):
            print(f"| LearnerPipe.run: ReplayBuffer saving in {cwd}")
            buffer.save_or_load_history(cwd, if_save=True)
            print(f"| LearnerPipe.run: ReplayBuffer saved  in {cwd}")

        '''comm_exp: close pipe1'''
        for j, pipe1 in enumerate(comm_exp.pipe1s):
            while pipe1.poll():
                time.sleep(1)
                pipe1.recv()
        for j, pipe1 in enumerate(comm_exp.pipe1s):
            pipe1.send(None)

        '''comm_exp: waiting for closing pipe0'''
        for j, pipe0 in enumerate(comm_exp.pipe0s):
            while pipe0.poll():
                time.sleep(1)

        '''comm_eva: close pipe1'''
        while comm_eva.pipe1.poll():
            time.sleep(1)
            comm_eva.pipe1.recv()
