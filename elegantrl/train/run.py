import os
import time
import torch
import numpy as np
import multiprocessing as mp

from elegantrl.train.config import build_env
from elegantrl.train.evaluator import Evaluator
from elegantrl.train.replay_buffer import ReplayBuffer


def init_agent(args, gpu_id: int, env=None):
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


def init_buffer(args, gpu_id: int) -> [ReplayBuffer]:
    buffer = ReplayBuffer(gpu_id=gpu_id,
                          max_capacity=args.replay_buffer_size,
                          state_dim=args.state_dim,
                          action_dim=1 if args.if_discrete else args.action_dim,
                          if_use_per=args.if_use_per)
    return buffer


def init_evaluator(args, gpu_id: int) -> Evaluator:
    evaluator = Evaluator(cwd=args.cwd, agent_id=gpu_id, eval_env=args.env, args=args)
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
    env = args.env
    steps = 0

    agent = init_agent(args, gpu_id, env)
    buffer = init_buffer(args, gpu_id)
    evaluator = init_evaluator(args, gpu_id)

    agent.state = env.reset()
    if args.if_off_policy:
        trajectory, step = agent.explore_env(env, args.num_seed_steps * args.num_steps_per_episode, True)
        buffer.update_buffer(trajectory)
        steps += step

    '''start training'''
    cwd = args.cwd
    break_step = args.break_step
    horizon_len = args.horizon_len
    if_allow_break = args.if_allow_break
    if_off_policy = args.if_off_policy
    del args

    if_train = True
    while if_train:
        trajectory, step = agent.explore_env(env, horizon_len, False)
        steps += step
        if if_off_policy:
            buffer.update_buffer(trajectory)
            torch.set_grad_enabled(True)
            logging_tuple = agent.update_net(buffer)
            torch.set_grad_enabled(False)
        else:
            torch.set_grad_enabled(True)
            logging_tuple = agent.update_net(trajectory)
            torch.set_grad_enabled(False)

        r_exp = agent.reward_tracker.mean()
        step_exp = agent.step_tracker.mean()
        (if_reach_goal, if_save) = evaluator.evaluate_save_and_plot(agent.act, steps, r_exp, step_exp, logging_tuple)
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
