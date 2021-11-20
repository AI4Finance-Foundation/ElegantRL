import os
import time
import torch
import numpy as np
from elegantrl.envs.gym import build_env, build_eval_env
from elegantrl.replay_buffer import ReplayBuffer
from elegantrl.evaluator import Evaluator

"""[ElegantRL.2021.10.21](https://github.com/AI4Finance-Foundation/ElegantRL)"""

'''single processing training'''


def train_and_evaluate(args, learner_id=0):
    args.init_before_training()  # necessary!

    '''init: Agent'''
    agent = args.agent
    agent.init(net_dim=args.net_dim, gpu_id=args.learner_gpus[learner_id],
               state_dim=args.state_dim, action_dim=args.action_dim, env_num=args.env_num,
               learning_rate=args.learning_rate, if_per_or_gae=args.if_per_or_gae)
    agent.save_or_load_agent(args.cwd, if_save=False)

    env = build_env(env=args.env, if_print=False, device_id=args.eval_gpu_id, env_num=args.env_num)
    if env.env_num == 1:
        agent.states = [env.reset(), ]
        assert isinstance(agent.states[0], np.ndarray)
        assert agent.states[0].shape == (env.state_dim,)
    else:
        agent.states = env.reset()
        assert isinstance(agent.states, torch.Tensor)
        assert agent.states.shape == (env.env_num, env.state_dim)

    '''init Evaluator'''
    eval_env = build_eval_env(args.eval_env, args.env, args.eval_gpu_id, args.env_num)
    evaluator = Evaluator(cwd=args.cwd, agent_id=0,
                          eval_env=eval_env, eval_gap=args.eval_gap,
                          eval_times1=args.eval_times1, eval_times2=args.eval_times2,
                          target_return=args.target_return, if_overwrite=args.if_overwrite)
    evaluator.save_or_load_recoder(if_save=False)

    '''init ReplayBuffer'''
    if args.if_off_policy:
        buffer = ReplayBuffer(max_len=args.max_memo, state_dim=env.state_dim,
                              action_dim=1 if env.if_discrete else env.action_dim,
                              if_use_per=args.if_per_or_gae, gpu_id=args.learner_gpus[learner_id])
        buffer.save_or_load_history(args.cwd, if_save=False)

        def update_buffer(_traj_list):
            ten_state, ten_other = _traj_list[0]
            buffer.extend_buffer(ten_state, ten_other)

            _steps, _r_exp = get_step_r_exp(ten_reward=ten_other[0])  # other = (reward, mask, action)
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
    if agent.if_off_policy:
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

    agent.save_or_load_agent(cwd, if_save=True)
    buffer.save_or_load_history(cwd, if_save=True) if agent.if_off_policy else None
    evaluator.save_or_load_recoder(if_save=True)


def get_step_r_exp(ten_reward):
    return len(ten_reward), ten_reward.mean().item()
