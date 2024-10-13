from argparse import ArgumentParser

try:
    from ..elegantrl import Config
    from ..elegantrl import train_agent
    from ..elegantrl import get_gym_env_args
    from ..elegantrl.agents import AgentEmbedDQN, AgentEnsembleDQN
except ImportError or ModuleNotFoundError:
    from elegantrl import Config
    from elegantrl import train_agent
    from elegantrl import get_gym_env_args
    from elegantrl.agents import AgentEmbedDQN, AgentEnsembleDQN


def train_dqn_for_cartpole(agent_class, gpu_id: int):
    assert agent_class in {AgentEnsembleDQN, AgentEmbedDQN}

    import gymnasium as gym
    env_class = gym.make  # run a custom env: PendulumEnv, which based on OpenAI pendulum
    env_args = {'env_name': 'CartPole-v1',
                'max_step': 500,
                'state_dim': 4,
                'action_dim': 2,
                'if_discrete': True, }
    get_gym_env_args(env=gym.make('CartPole-v1'), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `erl_config.py Arguments()` for hyperparameter explanation
    args.break_step = int(6e5)  # break training if 'total_step > break_step'
    args.net_dims = [64, 64]  # the middle layer dimension of MultiLayer Perceptron
    args.batch_size = 256
    args.gamma = 0.98  # discount factor of future rewards
    args.horizon_len = args.max_step // 16
    args.buffer_size = int(4e5)
    args.repeat_times = 1.0  # repeatedly update network using ReplayBuffer to keep critic's loss small
    args.reward_scale = 2 ** 0
    args.learning_rate = 5e-4
    args.state_value_tau = 1e-4
    args.soft_update_tau = 1e-3

    args.explore_rate = 0.05

    args.eval_times = 32
    args.eval_per_step = 2e4

    args.gpu_id = gpu_id
    args.num_workers = 8
    train_agent(args=args, if_single_process=False)

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

################################################################################
ID     Step    Time |    avgR   stdR   avgS  stdS |    expR   objC   objA   etc.
0  4.96e+02       1 |    9.25    0.8      9     1 |    1.00   0.00   0.00
0  2.06e+04       9 |    9.31    0.5      9     1 |    1.00   0.01   0.99
0  4.07e+04      23 |    9.22    0.7      9     1 |    1.00   0.01   1.63
0  6.08e+04      43 |    9.34    0.7      9     1 |    1.00   0.02   2.88
0  8.08e+04      72 |  498.75    7.1    499     7 |    1.00   0.03   4.63
0  1.01e+05     105 |  500.00    0.0    500     0 |    1.00   0.04   6.89
0  1.21e+05     135 |  500.00    0.0    500     0 |    1.00   0.05   9.93
0  1.41e+05     178 |  500.00    0.0    500     0 |    1.00   0.06  13.51
0  1.61e+05     226 |  498.34    5.6    498     6 |    1.00   0.10  17.56
0  1.81e+05     272 |  137.75   41.0    138    41 |    1.00   0.08  21.63
0  2.01e+05     321 |   89.28    2.6     89     3 |    1.00   0.16  26.06
0  2.21e+05     375 |   92.19    4.5     92     5 |    1.00   0.27  30.10
0  2.42e+05     433 |   94.44   30.9     94    31 |    1.00   0.30  33.44
0  2.62e+05     495 |  116.12    8.9    116     9 |    1.00   0.38  36.29
0  2.82e+05     563 |  215.03  129.8    215   130 |    1.00   0.40  39.06
0  3.02e+05     633 |   98.81    5.5     99     5 |    1.00   0.44  41.83
0  3.22e+05     707 |   94.34    1.5     94     1 |    1.00   0.47  44.19
0  3.42e+05     786 |   95.94    1.6     96     2 |    1.00   0.43  46.33
0  3.62e+05     871 |  175.06   15.3    175    15 |    1.00   0.52  48.68
0  3.82e+05     960 |   97.12   12.6     97    13 |    1.00   0.56  50.98
0  4.02e+05    1051 |   33.19   85.2     33    85 |    1.00   0.73  52.14
0  4.22e+05    1151 |  178.47    7.7    178     8 |    1.00   0.72  54.49
0  4.42e+05    1249 |  120.03   71.4    120    71 |    1.00   0.79  56.32
0  4.63e+05    1349 |   97.38   10.4     97    10 |    1.00   0.74  56.99
0  4.83e+05    1404 |   15.91    1.6     16     2 |    1.00   0.75  56.51
0  5.03e+05    1463 |   53.56  117.1     54   117 |    1.00   0.91  54.61
0  5.23e+05    1529 |  500.00    0.0    500     0 |    1.00   0.75  53.02
0  5.43e+05    1595 |  500.00    0.0    500     0 |    1.00   0.76  51.91
0  5.63e+05    1664 |  500.00    0.0    500     0 |    1.00   0.68  51.57
0  5.83e+05    1760 |  477.69   87.8    478    88 |    1.00   0.66  50.98
| UsedTime:    1816 | SavedDir: ./CartPole-v1_DoubleDQN_0
    """


def train_dqn_for_cartpole_vec_env(agent_class, gpu_id: int):
    assert agent_class in {AgentEnsembleDQN, AgentEmbedDQN}

    import gymnasium as gym
    num_envs = 8

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
    args.break_step = int(4e5)  # break training if 'total_step > break_step'
    args.net_dims = [128, 128]  # the middle layer dimension of MultiLayer Perceptron
    args.batch_size = 512
    args.gamma = 0.98  # discount factor of future rewards
    args.horizon_len = args.max_step // 8
    args.buffer_size = int(2e5)
    args.repeat_times = 1.0  # repeatedly update network using ReplayBuffer to keep critic's loss small
    args.reward_scale = 2 ** 0
    args.learning_rate = 4e-4
    args.state_value_tau = 1e-5
    args.soft_update_tau = 5e-3

    args.explore_rate = 0.05
    args.if_eval_target = True

    args.eval_times = 32
    args.eval_per_step = 2e4

    args.gpu_id = gpu_id
    args.num_workers = 4
    train_agent(args=args, if_single_process=False)

    """
0 < 9 < 400 < 500
################################################################################
ID     Step    Time |    avgR   stdR   avgS  stdS |    expR   objC   objA   etc.
0  4.96e+02      10 |    9.33    0.7      9     1 |    1.00   0.00   0.00
0  2.06e+04      19 |    9.37    0.8      9     1 |    1.00   0.02   1.30
0  4.07e+04      33 |  188.41   67.4    188    67 |    1.00   0.10   5.03
0  6.08e+04      49 |  229.47   18.9    229    19 |    1.00   0.17  11.70
0  8.08e+04      68 |  365.56  102.3    366   102 |    1.00   0.47  22.62
0  1.01e+05      89 |  179.62  114.9    180   115 |    1.00   0.51  32.25
0  1.21e+05     111 |  244.90   70.8    245    71 |    1.00   0.39  38.29
0  1.41e+05     136 |  500.00    0.0    500     0 |    1.00   0.33  42.19
0  1.61e+05     162 |  491.50   48.1    492    48 |    1.00   0.31  44.78
0  1.81e+05     190 |  500.00    0.0    500     0 |    1.00   0.34  47.38
0  2.01e+05     221 |  439.24  119.0    439   119 |    1.00   0.32  49.95
0  2.21e+05     253 |  118.22   80.8    118    81 |    1.00   0.40  50.68
0  2.42e+05     287 |  500.00    0.0    500     0 |    1.00   0.58  52.59
0  2.62e+05     323 |   88.53    7.4     89     7 |    1.00   0.67  55.28
0  2.82e+05     361 |   47.34   33.0     47    33 |    1.00   0.76  57.15
0  3.02e+05     401 |  118.74    2.8    119     3 |    1.00   0.76  56.01
0  3.22e+05     443 |  292.28    5.4    292     5 |    1.00   0.67  53.46
0  3.42e+05     486 |  203.16   33.2    203    33 |    1.00   0.63  51.05
0  3.62e+05     532 |  500.00    0.0    500     0 |    1.00   0.55  50.00
0  3.82e+05     579 |  500.00    0.0    500     0 |    1.00   0.50  49.51
| UsedTime:     622 | SavedDir: ./CartPole-v1_DoubleDQN_0
    """


def train_dqn_for_lunar_lander(agent_class, gpu_id: int):
    assert agent_class in {AgentEnsembleDQN, AgentEmbedDQN}

    import gymnasium as gym
    env_class = gym.make  # run a custom env: PendulumEnv, which based on OpenAI pendulum
    env_args = {'env_name': 'LunarLander-v2',
                'max_step': 1000,
                'state_dim': 8,
                'action_dim': 4,
                'if_discrete': True, }

    get_gym_env_args(env=gym.make('LunarLander-v2'), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `erl_config.py Arguments()` for hyperparameter explanation
    args.break_step = int(8e5)  # break training if 'total_step > break_step'
    args.net_dims = [256, 128]  # the middle layer dimension of MultiLayer Perceptron
    args.batch_size = 512
    args.gamma = 0.985  # discount factor of future rewards
    args.horizon_len = args.max_step // 4
    args.buffer_size = int(4e5)
    args.repeat_times = 8.0  # GPU 2 repeatedly update network using ReplayBuffer to keep critic's loss small
    args.reward_scale = 2 ** -1
    args.learning_rate = 5e-4

    args.explore_rate = 0.1

    args.eval_times = 32
    args.eval_per_step = 4e4

    args.gpu_id = gpu_id
    args.num_workers = 8
    train_agent(args=args, if_single_process=False)

    """
-1500 < -140 < 300 < 340
################################################################################
ID     Step    Time |    avgR   stdR   avgS  stdS |    expR   objC   objA   etc.
0  4.00e+03       6 | -702.29  108.6    112    21 |   -2.79   0.00   0.00
0  4.40e+04      24 | -385.69  108.2    283   243 |   -0.35   1.26  -2.52
0  8.40e+04      57 | -136.70   22.9    987    76 |   -0.03   0.89  -1.79
0  1.24e+05      95 | -126.09   24.2   1000     0 |   -0.03   0.84   1.55
0  1.64e+05     136 |  -96.64   25.9    756   397 |    0.01   0.81   3.26
0  2.04e+05     184 |  -97.15   26.6   1000     0 |    0.01   0.63   2.65
0  2.44e+05     237 |  -75.03   17.0   1000     0 |    0.00   0.70   3.81
0  2.84e+05     295 |  -52.66   78.9    965    95 |    0.00   0.68   7.26
0  3.24e+05     352 |  -99.60   73.5    433   156 |   -0.08   0.76   8.93
0  3.64e+05     421 |   -7.50   25.0   1000     0 |    0.01   0.69   9.77
0  4.04e+05     494 |   51.89  126.8    793   244 |    0.06   0.67  11.60
0  4.44e+05     574 |    3.61   62.5    980    61 |    0.10   0.56   8.73
0  4.84e+05     652 |  184.48   45.6    597   170 |    0.15   0.55   7.31
0  5.24e+05     738 |  171.02   97.4    555   166 |    0.20   0.57   7.03
0  5.64e+05     830 |  131.65  122.2    586   218 |    0.14   0.55   7.56
0  6.04e+05    1000 |  226.24   42.6    453   137 |    0.32   0.55   8.44
0  6.44e+05    1200 |  197.76   67.4    534   183 |    0.18   0.47   7.97
0  6.84e+05    1408 |  170.30  117.8    509   279 |    0.31   0.46   9.87
0  7.24e+05    1624 |  192.84   87.6    372   255 |    0.28   0.46   9.38
0  7.64e+05    1868 |  208.07   72.1    452   288 |    0.32   0.46  10.61
0  8.04e+05    2088 |  177.43   97.4    451   284 |    0.17   0.46   9.35
    """


def train_dqn_for_lunar_lander_vec_env(agent_class, gpu_id: int):
    assert agent_class in {AgentEnsembleDQN, AgentEmbedDQN}
    num_envs = 8

    import gymnasium as gym
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
    args.break_step = int(4e5)  # break training if 'total_step > break_step'
    args.net_dims = [256, 128]  # the middle layer dimension of MultiLayer Perceptron
    args.batch_size = 512
    args.gamma = 0.985  # discount factor of future rewards
    args.horizon_len = args.max_step // 4
    args.buffer_size = int(2e5)
    args.repeat_times = 4.0  # GPU 2 repeatedly update network using ReplayBuffer to keep critic's loss small
    args.reward_scale = 2 ** -1
    args.learning_rate = 1e-3
    if gpu_id == 4:
        args.lambda_fit_cum_r = 0.
    if gpu_id == 5:
        args.lambda_fit_cum_r = 0.1
    if gpu_id == 6:
        args.lambda_fit_cum_r = 0.5
    if gpu_id == 7:
        args.lambda_fit_cum_r = 1.0

    args.explore_rate = 0.1

    args.eval_times = 32
    args.eval_per_step = 2e4

    args.gpu_id = gpu_id
    args.num_workers = 2
    train_agent(args=args, if_single_process=False)

    """
-1500 < -140 < 300 < 340
################################################################################
ID     Step    Time |    avgR   stdR   avgS  stdS |    expR   objC   objA   etc.
0  2.50e+02       4 | -131.52   44.3     70    13 |   -0.78   0.00   0.00
0  2.02e+04      45 | -157.77   47.2    912   157 |   -0.07   0.82   2.40
0  4.02e+04     126 |   54.89  119.3    855   176 |    0.03   0.52   7.73
0  6.02e+04     246 |  234.89   39.8    432   115 |    0.34   0.48   8.74
0  8.02e+04     408 |  237.73   29.0    406   128 |    0.12   0.40   9.13
0  1.00e+05     611 |  225.22   75.6    403   255 |    0.12   0.35  10.16
0  1.20e+05     856 |  198.26   87.1    320   194 |    0.30   0.39  11.06
0  1.40e+05    1141 |  256.00   47.5    374   270 |    0.21   0.42   9.89
0  1.60e+05    1470 |  262.14   32.1    244   117 |    0.35   0.40  11.03
0  1.80e+05    1836 |  209.33   87.3    428   350 |    0.25   0.41  11.53
0  2.00e+05    2246 |  261.16   35.9    264   167 |    0.36   0.41  11.76

lambda_fit_cum_r = 0.1
################################################################################
ID     Step    Time |    avgR   stdR   avgS  stdS |    expR   objC   objA   etc.
0  1.00e+03      13 | -815.41  554.0    119    53 |   -3.04  14.41  -0.53
0  2.10e+04      34 |  -82.07   24.2   1000     0 |   -0.04   2.41  -2.00
0  4.10e+04      73 | -102.24   38.1    592   394 |   -0.05   1.42  -1.59
0  6.10e+04     124 |  127.47  128.0    457   232 |    0.12   1.00  -1.16
0  8.10e+04     186 |  -25.05   61.2    916   197 |   -0.01   1.45   6.39
0  1.01e+05     261 |  146.39  122.0    557   318 |    0.07   1.04   4.53
0  1.21e+05     348 |   58.56  135.3    642   390 |   -0.04   0.85   3.50
0  1.41e+05     445 |  179.57  113.3    468   361 |    0.33   0.56   1.89
0  1.61e+05     553 |  253.64   40.7    282   143 |    0.32   0.54   2.73
0  1.81e+05     674 |  239.29   52.9    313   233 |    0.33   0.48   3.11
0  2.01e+05     804 |  239.34   64.5    301   214 |    0.05   0.44   3.35
    """


if __name__ == '__main__':
    Parser = ArgumentParser(description='ArgumentParser for ElegantRL')
    Parser.add_argument('--gpu', type=int, default=0, help='GPU device ID for training')
    Parser.add_argument('--drl', type=int, default=0, help='RL algorithms ID for training')
    Parser.add_argument('--env', type=str, default='3', help='the environment ID for training')

    Args = Parser.parse_args()
    GPU_ID = Args.gpu
    DRL_ID = Args.drl
    ENV_ID = Args.env

    AgentClassList = [AgentEnsembleDQN, AgentEmbedDQN]
    AgentClass = AgentClassList[DRL_ID]  # DRL algorithm name

    if ENV_ID in {'0', 'cartpole'}:
        train_dqn_for_cartpole(agent_class=AgentClass, gpu_id=GPU_ID)
    elif ENV_ID in {'1', 'cartpole_vec'}:
        train_dqn_for_cartpole_vec_env(agent_class=AgentClass, gpu_id=GPU_ID)
    elif ENV_ID in {'2', 'lunar_lander'}:
        train_dqn_for_lunar_lander(agent_class=AgentClass, gpu_id=GPU_ID)
    elif ENV_ID in {'3', 'lunar_lander_vec'}:
        train_dqn_for_lunar_lander_vec_env(agent_class=AgentClass, gpu_id=GPU_ID)
    else:
        print('ENV_ID not match')
