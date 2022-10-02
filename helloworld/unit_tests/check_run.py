import shutil

import numpy as np

from run import *


def check_def_get_rewards_and_steps(net_dims=(64, 32)):
    pass

    """discrete env"""
    from env import gym
    env_args = {'env_name': 'CartPole-v1', 'state_dim': 4, 'action_dim': 2, 'if_discrete': True}
    env_class = gym.make
    env = build_env(env_class=env_class, env_args=env_args)

    '''discrete env, on-policy'''
    from net import QNet
    actor = QNet(dims=net_dims, state_dim=env.state_dim, action_dim=env.action_dim)
    cumulative_returns, episode_steps = get_rewards_and_steps(env=env, actor=actor)
    assert isinstance(cumulative_returns, float)
    assert isinstance(episode_steps, int)
    assert episode_steps >= 1

    """continuous env"""
    from env import PendulumEnv
    env_args = {'env_name': 'Pendulum-v1', 'state_dim': 3, 'action_dim': 1, 'if_discrete': False}
    env_class = PendulumEnv
    env = build_env(env_class=env_class, env_args=env_args)

    '''continuous env, off-policy'''
    from net import Actor
    actor = Actor(dims=net_dims, state_dim=env.state_dim, action_dim=env.action_dim)
    cumulative_returns, episode_steps = get_rewards_and_steps(env=env, actor=actor)
    assert isinstance(cumulative_returns, float)
    assert isinstance(episode_steps, int)
    assert episode_steps >= 1

    '''continuous env, on-policy'''
    from net import ActorPPO
    actor = ActorPPO(dims=net_dims, state_dim=env.state_dim, action_dim=env.action_dim)
    cumulative_returns, episode_steps = get_rewards_and_steps(env=env, actor=actor)
    assert isinstance(cumulative_returns, float)
    assert isinstance(episode_steps, int)
    assert episode_steps >= 1


def check_def_draw_learning_curve_using_recorder(cwd='./temp'):
    os.makedirs(cwd, exist_ok=True)
    recorder_path = f"{cwd}/recorder.npy"
    recorder_len = 8

    recorder = np.zeros((recorder_len, 3), dtype=np.float32)
    recorder[:, 0] = np.linspace(1, 100, num=recorder_len)  # total_step
    recorder[:, 1] = np.linspace(1, 200, num=recorder_len)  # used_time
    recorder[:, 2] = np.linspace(1, 300, num=recorder_len)  # average of cumulative rewards
    np.save(recorder_path, recorder)
    draw_learning_curve_using_recorder(cwd)
    assert os.path.exists(f"{cwd}/LearningCurve.jpg")
    shutil.rmtree(cwd)


def check_class_evaluator(net_dims=(64, 32), horizon_len=1024, eval_per_step=16, eval_times=2, cwd='./temp'):
    from env import PendulumEnv
    env_args = {'env_name': 'Pendulum-v1', 'state_dim': 3, 'action_dim': 1, 'if_discrete': False}
    env_class = PendulumEnv
    env = build_env(env_class, env_args)
    from net import Actor
    actor = Actor(dims=net_dims, state_dim=env.state_dim, action_dim=env.action_dim)

    os.makedirs(cwd, exist_ok=True)
    evaluator = Evaluator(eval_env=env, eval_per_step=eval_per_step, eval_times=eval_times, cwd=cwd)
    evaluator.evaluate_and_save(actor=actor, horizon_len=horizon_len, logging_tuple=(0.1, 0.2))
    evaluator.evaluate_and_save(actor=actor, horizon_len=horizon_len, logging_tuple=(0.3, 0.4))
    evaluator.close()
    assert os.path.exists(f"{evaluator.cwd}/recorder.npy")
    assert os.path.exists(f"{evaluator.cwd}/LearningCurve.jpg")
    shutil.rmtree(cwd)


if __name__ == '__main__':
    check_def_draw_learning_curve_using_recorder()
    check_def_get_rewards_and_steps()
    check_class_evaluator()
    print('| Finish checking.')

