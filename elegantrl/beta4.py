import os
import time
import inspect
from copy import deepcopy

import torch
import numpy as np

from elegantrl.train.config import Arguments
from elegantrl.train.evaluator import Evaluator
from elegantrl.train.replay_buffer import ReplayBuffer
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

    '''init: Agent'''
    env = build_env(env=args.env, env_func=args.env_func, env_args=args.env_args)
    agent = args.agent
    agent.init(net_dim=args.net_dim, state_dim=args.state_dim, action_dim=args.action_dim,
               gamma=args.gamma, reward_scale=args.reward_scale,
               learning_rate=args.learning_rate, if_per_or_gae=args.if_per_or_gae,
               env_num=args.env_num, gpu_id=learner_gpu, )

    agent.save_or_load_agent(args.cwd, if_save=False)
    agent.states = get_states_for_init(args, env)

    '''init Evaluator'''
    eval_env = build_env(env=args.eval_env, device_id=args.eval_gpu_id,
                         env_func=args.eval_env_class, env_args=args.eval_env_args)
    evaluator = Evaluator(cwd=args.cwd, agent_id=0,
                          eval_env=eval_env, eval_gap=args.eval_gap,
                          eval_times1=args.eval_times1, eval_times2=args.eval_times2,
                          target_return=args.target_return, if_overwrite=args.if_overwrite)
    evaluator.save_or_load_recoder(if_save=False)

    '''init ReplayBuffer'''
    if args.if_off_policy:
        buffer = ReplayBuffer(max_len=args.max_memo, state_dim=args.state_dim,
                              action_dim=1 if args.if_discrete else args.action_dim,
                              if_use_per=args.if_per_or_gae, gpu_id=learner_gpu)
        buffer.save_or_load_history(args.cwd, if_save=False)

        def update_buffer(_traj_list):
            ten_state, ten_other = _traj_list[0]
            buffer.extend_buffer(ten_state, ten_other)

            _steps, _r_exp = get_step_r_exp(ten_reward=ten_other)
            return _steps, _r_exp
    else:
        buffer = list()

        def update_buffer(_traj_list):
            (ten_state, ten_reward, ten_mask, ten_action, ten_noise) = _traj_list[0]
            buffer[:] = (ten_state.squeeze(1),
                         ten_reward,
                         ten_mask,
                         ten_action.squeeze(1),
                         ten_noise.squeeze(1))

            _step, _r_exp = get_step_r_exp(ten_reward=buffer[1])
            return _step, _r_exp

    """start training"""
    cwd = args.cwd
    break_step = args.break_step
    batch_size = args.batch_size
    target_step = args.target_step
    repeat_times = args.repeat_times
    if_allow_break = args.if_allow_break
    soft_update_tau = args.soft_update_tau
    del args

    '''init ReplayBuffer after training start'''
    if agent.if_off_policy:
        if_load = buffer.save_or_load_history(cwd, if_save=False)

        if not if_load:
            traj_list = agent.explore_env(env, target_step)
            steps, r_exp = update_buffer(traj_list)
            evaluator.total_step += steps

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


def get_states_for_init(args, env):
    if args.env_num == 1:
        states = [env.reset(), ]
        assert isinstance(states[0], np.ndarray)
        assert states[0].shape in {(args.state_dim,), args.state_dim}
    else:
        states = env.reset()
        assert isinstance(states, torch.Tensor)
        assert states.shape == (args.env_num, args.state_dim)
    return states


def get_step_r_exp(ten_reward):
    return len(ten_reward), ten_reward.mean().item()


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

    args.learner_gpus = (GPU,)
    train_and_evaluate(args)


if __name__ == '__main__':
    import sys

    sys.argv.extend('0 2'.split(' '))
    GPU = eval(sys.argv[1])
    DIM = eval(sys.argv[2])
    demo_continuous_action()
