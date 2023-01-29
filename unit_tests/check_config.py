import os
import gym
import torch
import numpy as np
from unittest.mock import patch
from torch import Tensor
from numpy import ndarray

from elegantrl.train.config import Config
from elegantrl.envs.CustomGymEnv import PendulumEnv
from elegantrl.envs.PointChasingEnv import PointChasingEnv
from elegantrl.agents.AgentDQN import AgentDQN
from elegantrl.agents.AgentSAC import AgentSAC
from elegantrl.agents.AgentPPO import AgentPPO

EnvArgsPendulum = {'env_name': 'Pendulum-v1', 'state_dim': 3, 'action_dim': 1, 'if_discrete': False}
EnvArgsCartPole = {'env_name': 'CartPole-v1', 'state_dim': 4, 'action_dim': 2, 'if_discrete': True}


def check_config():
    print("\n| check_config()")
    args = Config()  # check dummy Config
    assert args.get_if_off_policy() is True

    env_args = EnvArgsCartPole
    env_class = gym.make
    args = Config(agent_class=AgentDQN, env_class=env_class, env_args=env_args)
    assert args.get_if_off_policy() is True

    env_args = EnvArgsPendulum
    env_class = PendulumEnv
    args = Config(agent_class=AgentSAC, env_class=env_class, env_args=env_args)
    assert args.get_if_off_policy() is True

    env_args = EnvArgsPendulum
    env_class = PendulumEnv
    args = Config(agent_class=AgentPPO, env_class=env_class, env_args=env_args)
    assert args.get_if_off_policy() is False

    args.if_remove = False
    args.init_before_training()  # os.path.exists(args.cwd) == False
    args.init_before_training()  # os.path.exists(args.cwd) == True
    assert os.path.exists(args.cwd)
    os.rmdir(args.cwd)

    args.if_remove = True
    args.init_before_training()  # os.path.exists(args.cwd) == False
    args.init_before_training()  # os.path.exists(args.cwd) == True
    assert os.path.exists(args.cwd)
    os.rmdir(args.cwd)


@patch('builtins.input', lambda *args: 'input_str')
def _tutorial_unittest_mock_patch():
    print('Print_input():', input())


@patch('builtins.input', lambda *args: 'y')
def _config_init_before_training_yes():
    print("\n| check_config_init_before_training_yes()")
    env_args = EnvArgsPendulum
    env_class = gym.make
    args = Config(agent_class=AgentSAC, env_class=env_class, env_args=env_args)
    args.if_remove = None
    args.init_before_training()
    assert os.path.exists(args.cwd)
    os.rmdir(args.cwd)


@patch('builtins.input', lambda *args: 'n')
def _config_init_before_training_no():
    print("\n| check_config_init_before_training_no()")
    env_args = EnvArgsPendulum
    env_class = PendulumEnv
    args = Config(agent_class=AgentSAC, env_class=env_class, env_args=env_args)
    args.if_remove = None
    args.init_before_training()
    assert os.path.exists(args.cwd)
    os.rmdir(args.cwd)


def check_config_init_before_training():
    print("\n| check_config_init_before_training()")

    _tutorial_unittest_mock_patch()
    _config_init_before_training_yes()
    _config_init_before_training_no()


def check_kwargs_filter():
    print("\n| check_kwargs_filter()")
    from elegantrl.train.config import kwargs_filter

    dim = 2
    env_args = {
        'env_name': 'PointChasingEnv',
        'state_dim': 2 * dim,
        'action_dim': dim,
        'if_discrete': False,

        'dim': dim
    }
    env_class = PointChasingEnv
    env = env_class(**kwargs_filter(env_class.__init__, env_args.copy()))
    assert hasattr(env, 'reset')
    assert hasattr(env, 'step')


def check_build_env():
    print("\n| check_build_env()")
    from elegantrl.train.config import build_env

    '''check single env '''
    env_args_env_class_list = (
        (EnvArgsCartPole, gym.make),  # discrete action space
        (EnvArgsPendulum, PendulumEnv),  # continuous action space
    )
    for env_args, env_class in env_args_env_class_list:
        env_name = env_args['env_name']
        state_dim = env_args['state_dim']
        action_dim = env_args['action_dim']
        if_discrete = env_args['if_discrete']
        print(f"  env_name = {env_name}")

        env = build_env(env_class=env_class, env_args=env_args)
        assert isinstance(env.env_name, str)
        assert isinstance(env.state_dim, int)
        assert isinstance(env.action_dim, int)
        assert isinstance(env.if_discrete, bool)

        state = env.reset()
        assert isinstance(state, ndarray)
        assert state.shape == (state_dim,)

        for _ in range(4):
            if if_discrete:
                action = np.random.randint(action_dim)
            else:
                action = np.random.rand(action_dim) * 2. - 1.
            state, reward, done, info_dict = env.step(action)
            assert isinstance(state, ndarray)
            assert state.shape == (state_dim,)
            assert isinstance(reward, float)
            assert isinstance(done, bool)
            assert not done

    '''check vectorized env (if_build_vec_env=True)'''
    gpu_id = -1
    num_envs = 4
    env_args_env_class_list = (
        (EnvArgsCartPole, gym.make),  # discrete action space
        (EnvArgsPendulum, PendulumEnv),  # continuous action space
    )
    for env_args, env_class in env_args_env_class_list:
        _env_args = env_args.copy()
        _env_args['num_envs'] = num_envs
        _env_args['if_build_vec_env'] = True

        env_name = _env_args['env_name']
        state_dim = _env_args['state_dim']
        action_dim = _env_args['action_dim']
        if_discrete = _env_args['if_discrete']
        print(f"  env_name = {env_name}  if_build_vec_env = True")

        env = build_env(env_class=env_class, env_args=_env_args, gpu_id=gpu_id)
        assert isinstance(env.env_name, str)
        assert isinstance(env.state_dim, int)
        assert isinstance(env.action_dim, int)
        assert isinstance(env.if_discrete, bool)

        states = env.reset()
        assert isinstance(states, Tensor)
        assert states.shape == (num_envs, state_dim)

        for _ in range(4):
            if if_discrete:
                action = torch.randint(action_dim, size=(num_envs, 1))
            else:
                action = torch.rand(num_envs, action_dim)
            state, reward, done, info_dict = env.step(action)

            assert isinstance(state, Tensor)
            assert state.dtype is torch.float
            assert state.shape == (num_envs, state_dim,)

            assert isinstance(reward, Tensor)
            assert reward.dtype is torch.float
            assert reward.shape == (num_envs,)

            assert isinstance(done, Tensor)
            assert done.dtype is torch.bool
            assert done.shape == (num_envs,)
        env.close()


def check_get_gym_env_args():
    print("\n| check_get_gym_env_args()")
    from elegantrl.train.config import build_env
    from elegantrl.train.config import get_gym_env_args

    env_args = EnvArgsCartPole
    env_class = gym.make
    env = build_env(env_class=env_class, env_args=env_args)
    env_args = get_gym_env_args(env, if_print=True)
    assert isinstance(env_args['env_name'], str)
    assert isinstance(env_args['state_dim'], int)
    assert isinstance(env_args['action_dim'], int)
    assert isinstance(env_args['if_discrete'], bool)

    env_args = EnvArgsPendulum
    env_class = PendulumEnv
    env = build_env(env_class=env_class, env_args=env_args)
    env_args = get_gym_env_args(env, if_print=True)
    assert isinstance(env_args['env_name'], str)
    assert isinstance(env_args['state_dim'], int)
    assert isinstance(env_args['action_dim'], int)
    assert isinstance(env_args['if_discrete'], bool)


def check_sub_env():
    print("\n| check_sub_env()")
    from elegantrl.train.config import SubEnv
    from multiprocessing import Pipe
    sub_pipe0, sub_pipe1 = Pipe(duplex=False)  # recv, send
    vec_pipe0, vec_pipe1 = Pipe(duplex=False)  # recv, send

    env_args = EnvArgsPendulum
    env_class = PendulumEnv
    env_id = 0

    state_dim = env_args['state_dim']
    action_dim = env_args['action_dim']
    if_discrete = env_args['if_discrete']

    '''build sub_env'''
    sub_env = SubEnv(sub_pipe0=sub_pipe0, vec_pipe1=vec_pipe1,
                     env_class=env_class, env_args=env_args, env_id=env_id)
    sub_env.start()

    '''check reset'''
    for i in range(2):
        print(f"  check_sub_env() loop:{i}")
        sub_pipe1.send(None)  # reset
        _env_id, state = vec_pipe0.recv()
        assert _env_id == env_id
        assert isinstance(state, ndarray)
        assert state.shape == (state_dim,)

        '''check step loop'''
        for _ in range(2):
            action = torch.ones(action_dim, dtype=torch.float32).detach().numpy()
            if if_discrete:
                action = action.squeeze(1)
            sub_pipe1.send(action)

            _env_id, state, reward, done, info_dict = vec_pipe0.recv()
            assert _env_id == env_id
            assert isinstance(state, ndarray)
            assert state.shape == (state_dim,)
            assert isinstance(reward, float)
            assert isinstance(done, bool)
            assert not done

    sub_env.terminate()


def check_vec_env():
    print("\n| check_vec_env()")
    from elegantrl.train.config import VecEnv

    '''check for elegantrl.train.config build_env()'''
    gpu_id = -1
    num_envs = 4
    env_args_env_class_list = (
        (EnvArgsCartPole, gym.make),  # discrete action space
        (EnvArgsPendulum, PendulumEnv),  # continuous action space
    )
    for env_args, env_class in env_args_env_class_list:
        _env_args = env_args.copy()
        _env_args['num_envs'] = num_envs
        _env_args['if_build_vec_env'] = True

        env_name = _env_args['env_name']
        state_dim = _env_args['state_dim']
        action_dim = _env_args['action_dim']
        if_discrete = _env_args['if_discrete']
        print(f"  env_name = {env_name}  if_build_vec_env = True")

        # env = build_env(env_class=env_class, env_args=_env_args, gpu_id=gpu_id)
        env = VecEnv(env_class=env_class, env_args=_env_args, num_envs=num_envs, gpu_id=gpu_id)
        assert isinstance(env.env_name, str)
        assert isinstance(env.state_dim, int)
        assert isinstance(env.action_dim, int)
        assert isinstance(env.if_discrete, bool)

        states = env.reset()
        assert isinstance(states, Tensor)
        assert states.shape == (num_envs, state_dim)

        for _ in range(4):
            if if_discrete:
                action = torch.randint(action_dim, size=(num_envs, 1))
            else:
                action = torch.rand(num_envs, action_dim)
            state, reward, done, info_dict = env.step(action)

            assert isinstance(state, Tensor)
            assert state.dtype is torch.float
            assert state.shape == (num_envs, state_dim,)

            assert isinstance(reward, Tensor)
            assert reward.dtype is torch.float
            assert reward.shape == (num_envs,)

            assert isinstance(done, Tensor)
            assert done.dtype is torch.bool
            assert done.shape == (num_envs,)
        env.close()


if __name__ == '__main__':
    print("\n| check_config.py")
    check_config()
    check_config_init_before_training()

    check_build_env()
    check_kwargs_filter()
    check_get_gym_env_args()

    check_sub_env()
    check_vec_env()
    print('| Finish checking.')
