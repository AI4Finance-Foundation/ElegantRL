import sys
from argparse import ArgumentParser

sys.path.append("..")
if True:  # write after `sys.path.append("..")`
    from elegantrl import train_agent, train_agent_multiprocessing
    from elegantrl import Config, get_gym_env_args
    from elegantrl.agents import AgentDQN, AgentDuelingDQN
    from elegantrl.agents import AgentDoubleDQN, AgentD3QN


def train_dqn_for_cartpole():
    import gym

    agent_class = [AgentD3QN, AgentDoubleDQN, AgentDuelingDQN, AgentDQN][DRL_ID]  # DRL algorithm name
    env_class = gym.make  # run a custom env: PendulumEnv, which based on OpenAI pendulum
    env_args = {'env_name': 'CartPole-v1',
                'max_step': 500,
                'state_dim': 4,
                'action_dim': 2,
                'if_discrete': True, }
    get_gym_env_args(env=gym.make('CartPole-v1'), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `erl_config.py Arguments()` for hyperparameter explanation
    args.break_step = int(1e5)  # break training if 'total_step > break_step'
    args.net_dims = (128, 64)  # the middle layer dimension of MultiLayer Perceptron
    args.batch_size = 512
    args.gamma = 0.98  # discount factor of future rewards
    args.horizon_len = args.max_step * 2
    args.buffer_size = int(4e4)
    args.repeat_times = 1.0  # repeatedly update network using ReplayBuffer to keep critic's loss small
    args.reward_scale = 2 ** 0
    args.learning_rate = 1e-4

    args.eval_times = 32
    args.eval_per_step = 1e4

    args.gpu_id = GPU_ID
    args.num_workers = 4
    if_single_process = False
    if if_single_process:
        train_agent(args)
    else:
        train_agent_multiprocessing(args)
    """
0 < 9 < 400 < 500
################################################################################
ID     Step    Time |    avgR   stdR   avgS  stdS |    expR   objC   objA   etc.
1  8.00e+03      21 |  262.53   11.1    263    11 |    1.00   0.46   6.13
1  2.00e+04      34 |  149.19   18.4    149    18 |    1.00   0.62  16.77
1  3.20e+04      51 |  170.72   25.4    171    25 |    1.00   0.74  26.38
1  4.40e+04      69 |  309.00   53.1    309    53 |    1.00   0.74  33.06
1  5.60e+04      86 |  316.09   34.2    316    34 |    1.00   0.67  38.54
1  6.80e+04     104 |  330.19   21.5    330    21 |    1.00   0.61  42.25
1  8.00e+04     122 |  414.72   28.8    415    29 |    1.00   0.54  44.59
1  9.20e+04     139 |  421.78   62.7    422    63 |    1.00   0.49  46.19
1  1.04e+05     157 |  482.03   70.7    482    71 |    1.00   0.47  47.32
| TrainingTime:     158 | SavedDir: ./CartPole-v1_D3QN_0
    """


def train_dqn_for_cartpole_vec_env():
    import gym
    num_envs = 16

    agent_class = [AgentD3QN, AgentDoubleDQN, AgentDuelingDQN, AgentDQN][DRL_ID]  # DRL algorithm name
    env_class = gym.make
    env_args = {
        'env_name': 'CartPole-v1',
        'max_step': 500,
        'state_dim': 4,
        'action_dim': 2,
        'if_discrete': True,

        'num_envs': num_envs,
        'if_build_vec_env': True,
    }
    get_gym_env_args(env=gym.make('CartPole-v1'), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `erl_config.py Arguments()` for hyperparameter explanation
    args.break_step = int(1e5)  # break training if 'total_step > break_step'
    args.net_dims = (128, 64)  # the middle layer dimension of MultiLayer Perceptron
    args.batch_size = 512
    args.gamma = 0.98  # discount factor of future rewards
    args.horizon_len = args.max_step * 2
    args.buffer_size = int(4e4)
    args.repeat_times = 1.0  # repeatedly update network using ReplayBuffer to keep critic's loss small
    args.reward_scale = 2 ** 0
    args.learning_rate = 1e-4

    args.eval_times = 32
    args.eval_per_step = 1e4

    args.gpu_id = GPU_ID
    args.random_seed = GPU_ID
    args.num_workers = 2
    if_single_process = False
    if if_single_process:
        train_agent(args)
    else:
        train_agent_multiprocessing(args)
    """
0 < 9 < 400 < 500
################################################################################
ID     Step    Time |    avgR   stdR   avgS  stdS |    expR   objC   objA   etc.
2  4.00e+03      18 |   69.95   22.8     70    23 |    1.00   0.48   6.17
2  1.40e+04      42 |  178.10   40.3    178    40 |    1.00   0.77  23.37
2  2.40e+04      68 |  161.90   39.8    162    40 |    1.00   0.76  34.73
2  3.40e+04      94 |  143.69   44.8    144    45 |    1.00   0.71  40.83
2  4.40e+04     120 |  235.07   35.1    235    35 |    1.00   0.56  43.29
2  5.40e+04     146 |  234.13   15.4    234    15 |    1.00   0.53  45.33
2  6.40e+04     172 |  227.72    6.4    228     6 |    1.00   0.54  47.13
2  7.40e+04     198 |  500.00    0.0    500     0 |    1.00   0.54  48.35
2  8.40e+04     224 |  500.00    0.0    500     0 |    1.00   0.42  49.44
2  9.40e+04     249 |  500.00    0.0    500     0 |    1.00   0.36  50.39
    """


def train_dqn_for_lunar_lander():
    import gym

    agent_class = [AgentD3QN, AgentDoubleDQN, AgentDuelingDQN, AgentDQN][DRL_ID]  # DRL algorithm name
    env_class = gym.make  # run a custom env: PendulumEnv, which based on OpenAI pendulum
    env_args = {'env_name': 'LunarLander-v2',
                'max_step': 1000,
                'state_dim': 8,
                'action_dim': 4,
                'if_discrete': True, }

    get_gym_env_args(env=gym.make('LunarLander-v2'), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `erl_config.py Arguments()` for hyperparameter explanation
    args.break_step = int(5e5)  # break training if 'total_step > break_step'
    args.net_dims = (256, 128)  # the middle layer dimension of MultiLayer Perceptron
    args.batch_size = 512
    args.gamma = 0.99  # discount factor of future rewards
    args.horizon_len = args.max_step * 2
    args.buffer_size = int(2e5)
    args.repeat_times = 1.0  # GPU 2 repeatedly update network using ReplayBuffer to keep critic's loss small
    args.reward_scale = 2 ** -1
    args.learning_rate = 1e-4

    args.eval_times = 32
    args.eval_per_step = 1e4

    args.gpu_id = GPU_ID
    args.num_workers = 4
    if_single_process = False
    if if_single_process:
        train_agent(args)
    else:
        train_agent_multiprocessing(args)
    """
-1500 < -140 < 300 < 340
################################################################################
ID     Step    Time |    avgR   stdR   avgS  stdS |    expR   objC   objA   etc.
2  1.60e+04      51 | -179.07   44.4    549   218 |   -1.16   1.90   0.98
2  4.00e+04     111 |    5.54   20.0   1000     0 |    0.00   2.78   1.77
2  5.60e+04     161 |  -72.86   47.4    964   135 |    0.01   2.64   3.94
2  8.00e+04     217 |  -13.52   17.7   1000     0 |   -0.01   2.40   8.39
2  1.20e+05     241 |  203.85   26.5    605   104 |    0.03   2.31   8.76
2  1.36e+05     263 |  197.63   56.9    559   145 |    0.09   2.34   8.50
2  1.60e+05     278 |  232.98   16.3    437    76 |    0.13   2.19   8.40
2  1.76e+05     294 |  202.22   97.2    472   182 |    0.13   2.09   8.64
2  1.92e+05     320 |   28.22  215.9    643   216 |    0.10   1.95   9.03
    """


def train_dqn_for_lunar_lander_vec_env():
    import gym
    num_envs = 16

    agent_class = [AgentD3QN, AgentDoubleDQN, AgentDuelingDQN, AgentDQN][DRL_ID]  # DRL algorithm name
    env_class = gym.make  # run a custom env: PendulumEnv, which based on OpenAI pendulum
    env_args = {
        'env_name': 'LunarLander-v2',
        'max_step': 1000,
        'state_dim': 8,
        'action_dim': 4,
        'if_discrete': True,

        'num_envs': num_envs,
        'if_build_vec_env': True,
    }
    get_gym_env_args(env=gym.make('LunarLander-v2'), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `erl_config.py Arguments()` for hyperparameter explanation
    args.break_step = int(2e5)  # break training if 'total_step > break_step'
    args.net_dims = (256, 128)  # the middle layer dimension of MultiLayer Perceptron
    args.batch_size = 512
    args.gamma = 0.99  # discount factor of future rewards
    args.horizon_len = args.max_step * 1
    args.buffer_size = int(2e5)
    args.repeat_times = 1.0  # GPU 2 repeatedly update network using ReplayBuffer to keep critic's loss small
    args.reward_scale = 2 ** -1
    args.learning_rate = 1e-4

    args.eval_times = 32
    args.eval_per_step = 1e4

    args.gpu_id = GPU_ID
    args.num_workers = 2
    args.random_seed = GPU_ID
    if_single_process = False
    if if_single_process:
        train_agent(args)
    else:
        train_agent_multiprocessing(args)
    """
-1500 < -140 < 300 < 340

################################################################################
ID     Step    Time |    avgR   stdR   avgS  stdS |    expR   objC   objA   etc.
1  4.00e+03      28 | -248.06  108.9    820   164 |   -0.78   1.39   0.32
1  1.40e+04      55 |  -68.66   21.6   1000     0 |    0.03   1.30   4.16
1  2.40e+04      86 |  -46.58   40.3    924   212 |   -0.01   1.11   7.00
1  3.40e+04     116 |  -40.73   45.5    961   157 |   -0.02   1.02   8.21
1  4.40e+04     147 |  -14.67   20.1   1000     0 |   -0.01   0.96   8.65
1  5.40e+04     179 |  -13.01   19.8   1000     0 |    0.00   0.83   8.52
1  6.40e+04     210 |  -14.20   21.6   1000     0 |   -0.00   0.82   8.36
1  7.40e+04     241 |  -11.80   17.8   1000     0 |    0.03   0.88   8.24
1  8.40e+04     271 |   -2.28   21.3   1000     0 |    0.06   0.81   7.83
1  9.40e+04     302 |  -17.59   26.5   1000     0 |   -0.00   0.79   7.37
1  1.04e+05     332 |   95.49  106.9    775   199 |    0.07   0.81   7.03
1  1.14e+05     361 |  185.96   29.1    670   108 |    0.09   0.81   6.73
1  1.24e+05     389 |  199.56   77.0    452    80 |    0.12   0.81   6.64
1  1.34e+05     418 |  220.45   73.0    419   198 |    0.14   0.78   6.75
1  1.44e+05     449 |  237.53   64.3    332   108 |    0.14   0.78   6.95
1  1.54e+05     479 |  239.72   71.2    331   101 |    0.18   0.76   7.57
1  1.64e+05     509 |  192.97  107.0    260    96 |    0.16   0.71   8.37
1  1.74e+05     538 |  224.81   84.8    276    93 |    0.16   0.72   9.32
1  1.84e+05     567 |  177.06  119.4    271    97 |    0.15   0.65  10.48
1  1.94e+05     598 |  243.42   68.4    287   138 |    0.17   0.59  11.38
1  2.04e+05     627 |  251.35   65.8    294   178 |    0.14   0.57  12.08
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

    if ENV_ID in {'0', 'cartpole'}:
        train_dqn_for_cartpole()
    elif ENV_ID in {'1', 'cartpole_vec'}:
        train_dqn_for_cartpole_vec_env()
    elif ENV_ID in {'2', 'lunar_lander'}:
        train_dqn_for_lunar_lander()
    elif ENV_ID in {'3', 'lunar_lander_vec'}:
        train_dqn_for_lunar_lander_vec_env()
    else:
        print('ENV_ID not match')
