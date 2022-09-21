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

"""
强化学习在智能体在与任务对应的环境中学习，根据实际表现去调整策略。
这种跟环境直接互动的特性，使其适合需要快速捕行情的变化的金融任务。

我们研发的金融强化学习训练框架，利用大规模采样技术去覆盖各种行情，
对策略空间进行彻底的搜索，找到使累积收益最大的交易策略。

这项技术能充分发挥GPU设备的并行特性，极大地缩短策略的迭代周期：
在行情变化后，我们将以往几天才完成的策略更新任务，压缩到分钟级别执行完成。
这些精准贴合行情的交易策略，将被快人一步地部署到市场获取更多的收益。

（其他机器学习算法依赖对市场进行建模，市场发生变化将导致算法失效。
况且金融数据的低信噪比，加大了建模的难度。
强化学习直接将交易智能体放到市场环境中学习，不依赖市场建模）

（从市场发生变化，到人类意识到变化发生、重新对市场建模，需要好几个工作日。
而我们研发的强化学习训练框架，因为大规模采样技术：
- 在仿真环境中模拟并覆盖多种行情 → 能搜索到更好的策略
- 设计了能利用GPU设备并刑加速的算法 → 能更快的搜索到策略
所以我们有以上优势，让别人有不得不用选择我们的理由。）
"""
