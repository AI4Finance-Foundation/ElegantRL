from config import *
from env import PendulumEnv
from unittest.mock import patch


def check_config():
    args = Config()  # check dummy Config
    assert args.get_if_off_policy() is True

    from agent import AgentDQN
    env_args = {'env_name': 'CartPole-v1', 'state_dim': 4, 'action_dim': 2, 'if_discrete': True}
    env_class = gym.make
    args = Config(agent_class=AgentDQN, env_class=env_class, env_args=env_args)
    assert args.get_if_off_policy() is True

    from agent import AgentDDPG
    env_args = {'env_name': 'Pendulum', 'state_dim': 3, 'action_dim': 1, 'if_discrete': False}
    env_class = PendulumEnv
    args = Config(agent_class=AgentDDPG, env_class=env_class, env_args=env_args)
    assert args.get_if_off_policy() is True

    from agent import AgentPPO
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


@patch('builtins.input', lambda *args: 'y')
def check_def_config_init_before_training_yes():
    from agent import AgentDDPG
    env_args = {'env_name': 'Pendulum-v1', 'state_dim': 3, 'action_dim': 1, 'if_discrete': False}
    env_class = gym.make
    args = Config(agent_class=AgentDDPG, env_class=env_class, env_args=env_args)
    args.if_remove = None
    args.init_before_training()
    assert os.path.exists(args.cwd)
    os.rmdir(args.cwd)


@patch('builtins.input', lambda *args: 'n')
def check_def_config_init_before_training_no():
    from agent import AgentDDPG
    env_args = {'env_name': 'Pendulum', 'state_dim': 3, 'action_dim': 1, 'if_discrete': False}
    env_class = PendulumEnv
    args = Config(agent_class=AgentDDPG, env_class=env_class, env_args=env_args)
    args.if_remove = None
    args.init_before_training()
    assert os.path.exists(args.cwd)
    os.rmdir(args.cwd)


@patch('builtins.input', lambda *args: 'input_str')
def tutorial_unittest_mock_patch():
    print('Print_input():', input())


def check_def_kwargs_filter():
    env_args = {'env_name': 'Pendulum-v1', 'state_dim': 3, 'action_dim': 1, 'if_discrete': False}
    env_class = PendulumEnv
    env = env_class(**kwargs_filter(env_class.__init__, env_args.copy()))
    assert hasattr(env, 'reset')
    assert hasattr(env, 'step')


def check_def_build_env():
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


def check_def_get_gym_env_args():
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
    check_def_config_init_before_training_no()
    check_def_config_init_before_training_yes()
    tutorial_unittest_mock_patch()

    check_def_kwargs_filter()
    check_def_build_env()
    check_def_get_gym_env_args()
    print('| Finish checking.')
