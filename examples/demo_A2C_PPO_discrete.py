import sys
from argparse import ArgumentParser

try:
    from ..elegantrl import Config
    from ..elegantrl import train_agent
    from ..elegantrl import get_gym_env_args
    from ..elegantrl.agents import AgentDiscretePPO, AgentDiscreteA2C
except ImportError or ModuleNotFoundError:
    sys.path.append("..")
    from elegantrl import Config
    from elegantrl import train_agent
    from elegantrl import get_gym_env_args
    from elegantrl.agents import AgentDiscretePPO, AgentDiscreteA2C


def train_discrete_ppo_a2c_for_cartpole(agent_class, gpu_id: int):
    assert agent_class in {AgentDiscretePPO, AgentDiscreteA2C}  # DRL algorithm name

    import gymnasium as gym
    env_class = gym.make  # run a custom env: PendulumEnv, which based on OpenAI pendulum
    env_args = {
        'env_name': 'CartPole-v1',
        'max_step': 500,
        'state_dim': 4,
        'action_dim': 2,
        'if_discrete': True,
    }
    get_gym_env_args(env=gym.make('CartPole-v1'), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `erl_config.py Arguments()` for hyperparameter explanation
    args.break_step = int(1e5)  # break training if 'total_step > break_step'
    args.net_dims = (256, 128)  # the middle layer dimension of MultiLayer Perceptron
    args.batch_size = 512
    args.gamma = 0.99  # discount factor of future rewards
    args.horizon_len = args.max_step * 2
    args.repeat_times = 16  # repeatedly update network using ReplayBuffer to keep critic's loss small
    args.reward_scale = 2 ** -2
    args.learning_rate = 2e-5
    args.state_value_tau = 0.1  # the tau of normalize for value and state `std = (1-std)*std + tau*std`

    args.eval_times = 32
    args.eval_per_step = 1e4

    args.gpu_id = gpu_id
    args.num_workers = 4
    train_agent(args=args, if_single_process=False)

    """
0 < 5 < 400 < 500
################################################################################
ID     Step    Time |    avgR   stdR   avgS  stdS |    expR   objC   etc.
0  1.00e+03       1 |    9.41    0.7      9     1 |   -0.69   1.56  -0.01   0.00
0  1.10e+04      12 |   61.00   33.7     61    34 |   -0.69   1.14   0.02   0.00
0  2.10e+04      23 |  152.88   93.4    153    93 |   -0.66   1.49   0.01   0.00
0  3.10e+04      36 |  299.69   76.8    300    77 |   -0.62   1.69   0.01   0.00
0  4.10e+04      48 |  201.50   33.7    202    34 |   -0.61   0.97   0.02   0.00
0  5.10e+04      62 |  406.38   81.1    406    81 |   -0.59   1.20   0.02   0.00
0  6.10e+04      76 |  392.88   80.0    393    80 |   -0.58   0.65   0.00   0.00
0  7.10e+04      89 |  230.25   26.5    230    26 |   -0.56   0.99   0.01   0.00
0  8.10e+04     102 |  500.00    0.0    500     0 |   -0.54   1.03   0.00   0.00
0  9.10e+04     116 |  487.31   23.1    487    23 |   -0.55   0.44   0.01   0.00
0  1.01e+05     129 |  500.00    0.0    500     0 |   -0.54   0.84  -0.00   0.00
| UsedTime:     129 | SavedDir: ./CartPole-v1_DiscreteVecPPO_0
    """


def train_discrete_ppo_a2c_for_cartpole_vec_env(agent_class, gpu_id: int):
    assert agent_class in {AgentDiscretePPO, AgentDiscreteA2C}  # DRL algorithm name
    num_envs = 8

    import gymnasium as gym
    env_class = gym.make  # run a custom env: PendulumEnv, which based on OpenAI pendulum
    env_args = {
        'env_name': 'CartPole-v1',
        'max_step': 500,
        'state_dim': 4,
        'action_dim': 2,
        'if_discrete': True,

        'num_envs': num_envs,  # the number of sub envs in vectorized env
        'if_build_vec_env': True,
    }
    get_gym_env_args(env=gym.make('CartPole-v1'), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `erl_config.py Arguments()` for hyperparameter explanation
    args.break_step = int(1e5)  # break training if 'total_step > break_step'
    args.net_dims = (256, 128)  # the middle layer dimension of MultiLayer Perceptron
    args.batch_size = 512
    args.gamma = 0.99  # discount factor of future rewards
    args.horizon_len = args.max_step * 2
    args.repeat_times = 16  # repeatedly update network using ReplayBuffer to keep critic's loss small
    args.reward_scale = 2 ** -2
    args.learning_rate = 1e-4
    args.state_value_tau = 0.01  # the tau of normalize for value and state `std = (1-std)*std + tau*std`

    args.eval_times = 32
    args.eval_per_step = 1e4

    args.gpu_id = gpu_id
    args.num_workers = 4
    train_agent(args=args, if_single_process=False)

    """
0 < 5 < 400 < 500
################################################################################
ID     Step    Time |    avgR   stdR   avgS  stdS |    expR   objC   etc.
0  8.00e+03      18 |   56.69   23.5     57    24 |   -0.69   1.44   0.02   0.00
0  2.40e+04      27 |  326.74   82.4    327    82 |   -0.64   1.84   0.03   0.00
0  3.60e+04      36 |  288.28   73.7    288    74 |   -0.61   2.17   0.02   0.00
0  4.80e+04      45 |  344.19   95.4    344    95 |   -0.58   2.11   0.00   0.00
0  6.00e+04      54 |  368.11   76.7    368    77 |   -0.57   1.88   0.03   0.00
0  7.20e+04      64 |  404.28   54.9    404    55 |   -0.56   1.35   0.02   0.00
0  8.40e+04      73 |  425.89   78.2    426    78 |   -0.55   0.85   0.02   0.00
0  9.60e+04      82 |  447.61   65.2    448    65 |   -0.55   0.87   0.02   0.00
| TrainingTime:      83 | SavedDir: ./CartPole-v1_DiscreteVecPPO_0
    """


def train_discrete_ppo_a2c_for_lunar_lander(agent_class, gpu_id: int):
    assert agent_class in {AgentDiscretePPO, AgentDiscreteA2C}  # DRL algorithm name

    import gymnasium as gym
    env_class = gym.make  # run a custom env: PendulumEnv, which based on OpenAI pendulum
    env_args = {
        'env_name': 'LunarLander-v2',
        'max_step': 1000,
        'state_dim': 8,
        'action_dim': 2,
        'if_discrete': True
    }
    get_gym_env_args(env=gym.make('LunarLander-v2'), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `erl_config.py Arguments()` for hyperparameter explanation
    args.break_step = int(4e6)  # break training if 'total_step > break_step'
    args.net_dims = (256, 128)  # the middle layer dimension of MultiLayer Perceptron
    args.batch_size = 512
    args.gamma = 0.99  # discount factor of future rewards
    args.horizon_len = args.max_step * 4
    args.repeat_times = 32  # repeatedly update network using ReplayBuffer to keep critic's loss small
    args.reward_scale = 2 ** -1
    args.learning_rate = 2e-5
    args.state_value_tau = 0.01  # the tau of normalize for value and state `std = (1-std)*std + tau*std`
    args.lambda_gae_adv = 0.97
    args.lambda_entropy = 0.1
    # args.if_use_v_trace = True

    args.eval_times = 32
    args.eval_per_step = 5e4

    args.gpu_id = gpu_id
    args.num_workers = 4
    train_agent(args=args, if_single_process=False)

    """
-1500 < -200 < 200 < 290
################################################################################
ID     Step    Time |    avgR   stdR   avgS  stdS |    expR   objC   etc.
0  1.60e+04      20 | -138.39   24.0     70    13 |   -2.87  10.25   0.13   0.01
0  7.20e+04      74 | -169.52   42.6    352   214 |   -2.92   4.08   0.12   0.04
0  1.28e+05     151 |  148.34   96.1    628   128 |   -2.96   1.73   0.15   0.07
0  1.84e+05     179 |  212.45   44.2    460   154 |   -2.99   0.73   0.17   0.09
0  2.40e+05     218 |  238.36   19.4    377    80 |   -3.05   0.86   0.15   0.11
0  2.96e+05     262 |  239.83   35.4    390   119 |   -3.09   0.80   0.25   0.13
0  3.52e+05     300 |  269.49   32.6    304   146 |   -3.14   0.58   0.21   0.16
0  4.08e+05     340 |  254.45   58.6    239    53 |   -3.21   1.00   0.24   0.19
| TrainingTime:     340 | SavedDir: ./LunarLanderContinuous-v2_VecPPO_0
    """


def train_discrete_ppo_a2c_for_lunar_lander_vec_env(agent_class, gpu_id: int):
    assert agent_class in {AgentDiscretePPO, AgentDiscreteA2C}  # DRL algorithm name
    num_envs = 8

    import gymnasium as gym
    env_class = gym.make  # run a custom env: PendulumEnv, which based on OpenAI pendulum
    env_args = {
        'env_name': 'LunarLander-v2',
        'max_step': 1000,
        'state_dim': 8,
        'action_dim': 2,
        'if_discrete': True,

        'num_envs': num_envs,  # the number of sub envs in vectorized env
        'if_build_vec_env': True,
    }
    get_gym_env_args(env=gym.make('LunarLander-v2'), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `erl_config.py Arguments()` for hyperparameter explanation
    args.break_step = int(4e6)  # break training if 'total_step > break_step'
    args.net_dims = (256, 128)  # the middle layer dimension of MultiLayer Perceptron
    args.batch_size = 512
    args.gamma = 0.99  # discount factor of future rewards
    args.horizon_len = args.max_step * 2
    args.repeat_times = 32  # repeatedly update network using ReplayBuffer to keep critic's loss small
    args.reward_scale = 2 ** -3
    args.learning_rate = 2e-5
    args.state_value_tau = 0.01  # the tau of normalize for value and state `std = (1-std)*std + tau*std`
    args.lambda_gae_adv = 0.97
    args.lambda_entropy = 0.1
    # args.if_use_v_trace = True

    args.eval_times = 32
    args.eval_per_step = 2e4

    args.gpu_id = gpu_id
    args.num_workers = 4
    train_agent(args=args, if_single_process=False)

    """
-1500 < -200 < 200 < 290
################################################################################
ID     Step    Time |    avgR   stdR   avgS  stdS |    expR   objC   etc.
0  8.00e+03      18 |   62.42   25.6     62    26 |   -0.69   8.03   0.01   0.00
0  2.80e+04      29 |  105.77   42.9    106    43 |   -0.67   9.55   0.02   0.00
0  4.00e+04      38 |  259.23   76.2    259    76 |   -0.64  10.98   0.02   0.00
0  5.20e+04      46 |  377.11   48.2    377    48 |   -0.61  12.39   0.01   0.00
0  6.40e+04      55 |  421.39   87.8    421    88 |   -0.60  12.93   0.03   0.00
0  7.60e+04      64 |  230.57   56.1    231    56 |   -0.58  13.37   0.03   0.00
0  8.80e+04      72 |  365.26  114.2    365   114 |   -0.58  13.32   0.02   0.00
0  1.00e+05      81 |  394.84  107.5    395   107 |   -0.58  13.09   0.02   0.00
| TrainingTime:      82 | SavedDir: ./CartPole-v1_DiscreteVecPPO_0
    """


if __name__ == '__main__':
    Parser = ArgumentParser(description='ArgumentParser for ElegantRL')
    Parser.add_argument('--gpu', type=int, default=0, help='GPU device ID for training')
    Parser.add_argument('--drl', type=int, default=0, help='RL algorithms ID for training')
    Parser.add_argument('--env', type=str, default='1', help='the environment ID for training')

    Args = Parser.parse_args()
    GPU_ID = Args.gpu
    DRL_ID = Args.drl
    ENV_ID = Args.env

    AgentClassList = [AgentDiscretePPO, AgentDiscreteA2C]
    AgentClass = AgentClassList[DRL_ID]  # DRL algorithm name
    if ENV_ID in {'0', 'cartpole'}:
        train_discrete_ppo_a2c_for_cartpole(agent_class=AgentClass, gpu_id=GPU_ID)
    elif ENV_ID in {'1', 'cartpole_vec'}:
        train_discrete_ppo_a2c_for_cartpole_vec_env(agent_class=AgentClass, gpu_id=GPU_ID)
    elif ENV_ID in {'2', 'lunar_lander'}:
        train_discrete_ppo_a2c_for_lunar_lander(agent_class=AgentClass, gpu_id=GPU_ID)
    elif ENV_ID in {'3', 'lunar_lander_vec'}:
        train_discrete_ppo_a2c_for_lunar_lander_vec_env(agent_class=AgentClass, gpu_id=GPU_ID)
    else:
        print('ENV_ID not match')
