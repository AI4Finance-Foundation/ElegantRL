import os
import time
import numpy as np
import torch as th

from elegantrl2.traj_config import Config, build_env
from elegantrl2.traj_evaluator import Evaluator
from elegantrl2.traj_buffer import TrajBuffer

TEN = th.Tensor


def train_agent(args: Config):
    args.init_before_training()
    th.set_grad_enabled(False)

    '''init environment'''
    env = build_env(args.env_class, args.env_args, args.gpu_id)
    state, info_dict = env.reset()
    assert state.shape == (args.num_envs, args.state_dim)
    assert isinstance(state, TEN)

    '''init agent'''
    agent = args.agent_class(args.net_dims, args.state_dim, args.action_dim, gpu_id=args.gpu_id, args=args)
    agent.save_or_load_agent(args.cwd, if_save=False)
    agent.last_state = state.detach().to(agent.device)
    del state

    '''init buffer'''
    buffer = TrajBuffer(
        gpu_id=args.gpu_id,
        num_seqs=args.num_envs,
        max_size=args.buffer_size,
        state_dim=args.state_dim,
        action_dim=1 if args.if_discrete else args.action_dim,
        if_use_per=args.if_use_per,
        if_discrete=args.if_discrete,
        args=args,
    )


def train_ppo_for_lunar_lander_continuous(agent_class, gpu_id: int):
    num_envs = 8

    import gymnasium as gym
    env_class = gym.make  # run a custom env: PendulumEnv, which based on OpenAI pendulum
    env_args = {
        'env_name': 'LunarLanderContinuous-v2',
        'max_step': 1000,
        'state_dim': 8,
        'action_dim': 2,
        'if_discrete': False,

        'num_envs': num_envs,  # the number of sub envs in vectorized env
        'if_build_vec_env': True,
    }
    # get_gym_env_args(env=gym.make('LunarLanderContinuous-v2'), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `erl_config.py Arguments()` for hyperparameter explanation
    args.break_step = int(2e5)  # break training if 'total_step > break_step'
    args.net_dims = (256, 128, 64)  # the middle layer dimension of MultiLayer Perceptron
    args.batch_size = 512
    args.gamma = 0.99  # discount factor of future rewards
    args.horizon_len = args.max_step
    args.repeat_times = 64  # repeatedly update network using ReplayBuffer to keep critic's loss small
    args.reward_scale = 2 ** -1
    args.learning_rate = 2e-4
    args.state_value_tau = 0.1  # the tau of normalize for value and state `std = (1-std)*std + tau*std`
    args.lambda_gae_adv = 0.97
    args.lambda_entropy = 0.04

    args.eval_times = 32
    args.eval_per_step = 2e4

    args.gpu_id = gpu_id
    args.num_workers = 4
    train_agent(args=args)


if __name__ == '__main__':
    from elegantrl2.traj_agent_ppo import AgentPPO

    train_ppo_for_lunar_lander_continuous(agent_class=AgentPPO, gpu_id=0)
