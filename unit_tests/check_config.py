import os
import gym
from unittest.mock import patch

from elegantrl.train.config import Config
from elegantrl.envs.CustomGymEnv import PendulumEnv
from elegantrl.envs.PointChasingEnv import PointChasingEnv
from elegantrl.agents.AgentDQN import AgentDQN
from elegantrl.agents.AgentSAC import AgentSAC
from elegantrl.agents.AgentPPO import AgentPPO


def check_config():
    print("\n| check_config()")
    args = Config()  # check dummy Config
    assert args.get_if_off_policy() is True

    env_args = {'env_name': 'CartPole-v1', 'state_dim': 4, 'action_dim': 2, 'if_discrete': True}
    env_class = gym.make
    args = Config(agent_class=AgentDQN, env_class=env_class, env_args=env_args)
    assert args.get_if_off_policy() is True

    env_args = {'env_name': 'Pendulum', 'state_dim': 3, 'action_dim': 1, 'if_discrete': False}
    env_class = PendulumEnv
    args = Config(agent_class=AgentSAC, env_class=env_class, env_args=env_args)
    assert args.get_if_off_policy() is True

    env_args = {'env_name': 'Pendulum', 'state_dim': 3, 'action_dim': 1, 'if_discrete': False}
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
def tutorial_unittest_mock_patch():
    print('Print_input():', input())


@patch('builtins.input', lambda *args: 'y')
def _config_init_before_training_yes():
    print("\n| check_config_init_before_training_yes()")
    env_args = {'env_name': 'Pendulum-v1', 'state_dim': 3, 'action_dim': 1, 'if_discrete': False}
    env_class = gym.make
    args = Config(agent_class=AgentSAC, env_class=env_class, env_args=env_args)
    args.if_remove = None
    args.init_before_training()
    assert os.path.exists(args.cwd)
    os.rmdir(args.cwd)


@patch('builtins.input', lambda *args: 'n')
def _config_init_before_training_no():
    print("\n| check_config_init_before_training_no()")
    env_args = {'env_name': 'Pendulum', 'state_dim': 3, 'action_dim': 1, 'if_discrete': False}
    env_class = PendulumEnv
    args = Config(agent_class=AgentSAC, env_class=env_class, env_args=env_args)
    args.if_remove = None
    args.init_before_training()
    assert os.path.exists(args.cwd)
    os.rmdir(args.cwd)


def check_config_init_before_training():
    print("\n| check_config_init_before_training()")

    tutorial_unittest_mock_patch()
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
    env_args = {'env_name': 'CartPole-v1', 'state_dim': 4, 'action_dim': 2, 'if_discrete': True}
    env_class = gym.make
    env = build_env(env_class=env_class, env_args=env_args)
    assert isinstance(env.env_name, str)
    assert isinstance(env.state_dim, int)
    assert isinstance(env.action_dim, int)
    assert isinstance(env.if_discrete, bool)

    env_args = {'env_name': 'Pendulum-v1', 'state_dim': 3, 'action_dim': 1, 'if_discrete': False}
    env_class = PendulumEnv
    env = build_env(env_class=env_class, env_args=env_args)
    assert isinstance(env.env_name, str)
    assert isinstance(env.state_dim, int)
    assert isinstance(env.action_dim, int)
    assert isinstance(env.if_discrete, bool)


def check_get_gym_env_args():
    print("\n| check_get_gym_env_args()")
    from elegantrl.train.config import build_env
    from elegantrl.train.config import get_gym_env_args

    env_args = {'env_name': 'CartPole-v1', 'state_dim': 4, 'action_dim': 2, 'if_discrete': True}
    env_class = gym.make
    env = build_env(env_class=env_class, env_args=env_args)
    env_args = get_gym_env_args(env, if_print=True)
    assert isinstance(env_args['env_name'], str)
    assert isinstance(env_args['state_dim'], int)
    assert isinstance(env_args['action_dim'], int)
    assert isinstance(env_args['if_discrete'], bool)

    env_args = {'env_name': 'Pendulum-v1', 'state_dim': 3, 'action_dim': 1, 'if_discrete': False}
    env_class = PendulumEnv
    env = build_env(env_class=env_class, env_args=env_args)
    env_args = get_gym_env_args(env, if_print=True)
    assert isinstance(env_args['env_name'], str)
    assert isinstance(env_args['state_dim'], int)
    assert isinstance(env_args['action_dim'], int)
    assert isinstance(env_args['if_discrete'], bool)


if __name__ == '__main__':
    check_config()
    check_config_init_before_training()
    check_kwargs_filter()
    check_build_env()
    check_get_gym_env_args()
    print('| Finish checking.')
