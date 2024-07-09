import sys
from argparse import ArgumentParser

sys.path.append("..")
if True:  # write after `sys.path.append("..")`
    from elegantrl import train_agent, train_agent_multiprocessing
    from elegantrl import Config, get_gym_env_args
    from elegantrl.agents import AgentPPO, AgentDiscretePPO
    from elegantrl.agents import AgentA2C, AgentDiscreteA2C

"""continuous action"""


def train_ppo_a2c_for_pendulum():
    from elegantrl.envs.CustomGymEnv import PendulumEnv

    agent_class = [AgentPPO, AgentA2C][DRL_ID]  # DRL algorithm name
    env_class = PendulumEnv  # run a custom env: PendulumEnv, which based on OpenAI pendulum
    env_args = {
        'env_name': 'Pendulum',  # Apply torque on the free end to swing a pendulum into an upright position
        'max_step': 200,  # the max step number of an episode.
        'state_dim': 3,  # the x-y coordinates of the pendulum's free end and its angular velocity.
        'action_dim': 1,  # the torque applied to free end of the pendulum
        'if_discrete': False  # continuous action space, symbols → direction, value → force
    }
    get_gym_env_args(env=PendulumEnv(), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `erl_config.py Arguments()` for hyperparameter explanation
    args.break_step = int(8e4)  # break training if 'total_step > break_step'
    args.net_dims = (128, 64)  # the middle layer dimension of MultiLayer Perceptron
    args.gamma = 0.97  # discount factor of future rewards
    args.horizon_len = args.max_step * 4

    args.repeat_times = 32  # repeatedly update network using ReplayBuffer to keep critic's loss small
    args.learning_rate = 2e-4
    args.state_value_tau = 0.1  # the tau of normalize for value and state `std = (1-std)*std + tau*std`

    args.gpu_id = GPU_ID
    args.num_workers = 4
    if_single_process = True
    if if_single_process:
        train_agent(args)
    else:
        train_agent_multiprocessing(args)  # train_agent(args)
    """
-2000 < -1200 < -200 < -80
################################################################################
ID     Step    Time |    avgR   stdR   avgS  stdS |    expR   objC   etc.
0  8.00e+02       2 |-1219.07  279.3    200     0 |   -1.41  49.69   0.02  -0.01
0  2.08e+04      46 | -162.10   74.0    200     0 |   -1.25   9.47   0.01  -0.13
0  4.08e+04      91 | -162.31  185.5    200     0 |   -1.14   0.95   0.01  -0.29
0  6.08e+04     136 |  -81.47   70.3    200     0 |   -1.00   0.17   0.02  -0.45
0  8.08e+04     201 |  -84.41   70.0    200     0 |   -0.84   2.62   0.01  -0.53
| UsedTime:     202 | SavedDir: ./Pendulum_VecPPO_0
    """


def train_ppo_a2c_for_pendulum_vec_env():
    from elegantrl.envs.CustomGymEnv import PendulumEnv

    agent_class = [AgentPPO, AgentA2C][DRL_ID]  # DRL algorithm name
    env_class = PendulumEnv  # run a custom env: PendulumEnv, which based on OpenAI pendulum
    env_args = {
        'env_name': 'Pendulum',  # Apply torque on the free end to swing a pendulum into an upright position
        'max_step': 200,  # the max step number in an episode for evaluation
        'state_dim': 3,  # the x-y coordinates of the pendulum's free end and its angular velocity.
        'action_dim': 1,  # the torque applied to free end of the pendulum
        'if_discrete': False,  # continuous action space, symbols → direction, value → force

        'num_envs': 4,  # the number of sub envs in vectorized env
        'if_build_vec_env': True,
    }
    get_gym_env_args(env=PendulumEnv(), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `erl_config.py Arguments()` for hyperparameter explanation
    args.break_step = int(8e4)
    args.net_dims = (128, 64)  # the middle layer dimension of MultiLayer Perceptron
    args.gamma = 0.97  # discount factor of future rewards
    args.reward_scale = 2 ** -2

    args.horizon_len = args.max_step * 1
    args.repeat_times = 16  # repeatedly update network using ReplayBuffer to keep critic's loss small
    args.learning_rate = 4e-4
    args.state_value_tau = 0.2  # the tau of normalize for value and state `std = (1-std)*std + tau*std`

    args.gpu_id = GPU_ID
    args.num_workers = 4
    train_agent_multiprocessing(args)  # train_agent(args)
    """
-2000 < -1200 < -200 < -80
################################################################################
ID     Step    Time |    avgR   stdR   avgS  stdS |    expR   objC   etc.
0  1.60e+03       9 |-1065.59  245.6    200     0 |   -1.41  10.00  -0.04  -0.00
0  2.16e+04      31 |-1152.15   11.0    200     0 |   -1.43   2.95  -0.04   0.02
0  4.16e+04      52 | -954.16   52.4    200     0 |   -1.42   3.21   0.00   0.01
0  6.16e+04      73 | -237.63  183.1    200     0 |   -1.34   0.53   0.05  -0.07
| TrainingTime:      92 | SavedDir: ./Pendulum_VecPPO_0
    """


def train_ppo_a2c_for_lunar_lander_continuous():
    import gym

    agent_class = [AgentPPO, AgentA2C][DRL_ID]  # DRL algorithm name
    env_class = gym.make  # run a custom env: PendulumEnv, which based on OpenAI pendulum
    env_args = {'env_name': 'LunarLanderContinuous-v2',
                'num_envs': 1,
                'max_step': 1000,
                'state_dim': 8,
                'action_dim': 2,
                'if_discrete': False}
    get_gym_env_args(env=gym.make('LunarLanderContinuous-v2'), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `erl_config.py Arguments()` for hyperparameter explanation
    args.break_step = int(4e5)  # break training if 'total_step > break_step'
    args.net_dims = (256, 128)  # the middle layer dimension of MultiLayer Perceptron
    args.batch_size = 512
    args.gamma = 0.99  # discount factor of future rewards
    args.horizon_len = args.max_step * 2
    args.repeat_times = 16  # repeatedly update network using ReplayBuffer to keep critic's loss small
    args.reward_scale = 2 ** -1
    args.learning_rate = 2e-4
    args.state_value_tau = 0.1  # the tau of normalize for value and state `std = (1-std)*std + tau*std`
    args.lambda_gae_adv = 0.97
    args.lambda_entropy = 0.04

    args.eval_times = 32
    args.eval_per_step = 5e4

    args.gpu_id = GPU_ID
    args.num_workers = 4
    train_agent_multiprocessing(args)  # train_agent(args)
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


def train_ppo_a2c_for_lunar_lander_continuous_vec_env():
    import gym

    agent_class = [AgentPPO, AgentA2C][DRL_ID]  # DRL algorithm name
    env_class = gym.make  # run a custom env: PendulumEnv, which based on OpenAI pendulum
    env_args = {
        'env_name': 'LunarLanderContinuous-v2',
        'max_step': 1000,
        'state_dim': 8,
        'action_dim': 2,
        'if_discrete': False,

        'num_envs': 4,  # the number of sub envs in vectorized env
        'if_build_vec_env': True,
    }
    get_gym_env_args(env=gym.make('LunarLanderContinuous-v2'), if_print=True)  # return env_args

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

    args.gpu_id = GPU_ID
    args.num_workers = 4
    train_agent_multiprocessing(args)  # train_agent(args)
    """
-1500 < -200 < 200 < 290
################################################################################
ID     Step    Time |    avgR   stdR   avgS  stdS |    expR   objC   etc.
0  8.00e+03      35 | -109.92   74.8     81    14 |   -2.85   9.17   0.15   0.02
0  2.80e+04      92 |  -79.63  119.7    460   258 |   -2.91   3.15   0.13   0.04
0  5.60e+04     132 |  239.43   36.7    402    70 |   -2.96   0.78   0.17   0.06
0  7.60e+04     159 |  251.94   61.9    273    44 |   -2.94   0.53   0.26   0.06
0  9.60e+04     187 |  276.30   18.2    221    23 |   -2.94   0.87   0.49   0.05
0  1.16e+05     218 |  273.28   19.6    220    17 |   -2.96   0.28   0.24   0.07
0  1.36e+05     248 |  275.14   17.7    215    35 |   -2.98   0.15   0.12   0.07
0  1.56e+05     280 |  272.89   22.4    223    45 |   -3.03   0.28   0.18   0.10
0  1.76e+05     310 |  275.35   16.8    219    78 |   -3.09   0.28   0.19   0.13
0  1.96e+05     339 |  275.55   16.5    219    77 |   -3.13   0.20   0.37   0.15
| TrainingTime:     340 | SavedDir: ./LunarLanderContinuous-v2_VecPPO_0
    """


def train_ppo_a2c_for_bipedal_walker():
    import gym

    agent_class = [AgentPPO, AgentA2C][DRL_ID]  # DRL algorithm name
    env_class = gym.make  # run a custom env: PendulumEnv, which based on OpenAI pendulum
    env_args = {
        'env_name': 'BipedalWalker-v3',
        'num_envs': 1,
        'max_step': 1600,
        'state_dim': 24,
        'action_dim': 4,
        'if_discrete': False,
    }
    get_gym_env_args(env=gym.make('BipedalWalker-v3'), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `erl_config.py Arguments()` for hyperparameter explanation
    args.break_step = int(8e5)  # break training if 'total_step > break_step'
    args.net_dims = (256, 128, 128)  # the middle layer dimension of MultiLayer Perceptron
    args.batch_size = 512
    args.gamma = 0.97  # discount factor of future rewards
    args.horizon_len = args.max_step * 3
    args.repeat_times = 32  # repeatedly update network using ReplayBuffer to keep critic's loss small
    args.learning_rate = 1e-4
    args.state_value_tau = 0.01  # the tau of normalize for value and state `std = (1-std)*std + tau*std`
    args.lambda_gae_adv = 0.93
    args.lambda_entropy = 0.02
    args.clip_ratio = 0.4

    args.eval_times = 16
    args.eval_per_step = 8e4
    args.if_keep_save = False  # keeping save the checkpoint. False means save until stop training.

    args.gpu_id = GPU_ID
    args.random_seed = GPU_ID
    args.num_workers = 2
    train_agent_multiprocessing(args)  # train_agent(args)
    """
-200 < -150 < 300 < 330
################################################################################
ID     Step    Time |    avgR   stdR   avgS  stdS |    expR   objC   etc.
0  1.92e+04      29 | -107.14   21.4    231   365 |   -5.75   0.60   0.14   0.02
0  1.06e+05     136 |  -58.44    5.8   1600     0 |   -5.97   0.22   0.45   0.07
0  1.92e+05     228 |  -65.31   16.3   1332   576 |   -6.00   0.06   0.15   0.08
0  2.78e+05     325 |   63.46    8.0   1600     0 |   -5.82   0.03   0.13   0.03
0  3.65e+05     419 |  192.51   49.7   1561   158 |   -5.55   0.10   0.26  -0.04
0  4.51e+05     490 | -107.56    3.5     88     8 |   -5.55   0.21   0.25  -0.04
0  5.38e+05     588 |  147.98  162.6    864   471 |   -5.57   0.36   0.09  -0.02
0  6.24e+05     681 |  256.13   81.9   1136   221 |   -5.70   0.50   0.13   0.00
0  7.10e+05     769 |  264.97   59.3   1079   131 |   -5.72   0.20   0.16   0.01
0  7.97e+05     857 |  279.37    1.3   1065    18 |   -5.77   0.11   0.13   0.02
| TrainingTime:     857 | SavedDir: ./BipedalWalker-v3_VecPPO_2
    """


def train_ppo_a2c_for_bipedal_walker_vec_env():
    import gym

    agent_class = [AgentPPO, AgentA2C][DRL_ID]  # DRL algorithm name
    env_class = gym.make  # run a custom env: PendulumEnv, which based on OpenAI pendulum
    env_args = {
        'env_name': 'BipedalWalker-v3',
        'max_step': 1600,
        'state_dim': 24,
        'action_dim': 4,
        'if_discrete': False,

        'num_envs': 4,  # the number of sub envs in vectorized env
        'if_build_vec_env': True,
    }
    get_gym_env_args(env=gym.make('BipedalWalker-v3'), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `erl_config.py Arguments()` for hyperparameter explanation
    args.break_step = int(8e5)  # break training if 'total_step > break_step'
    args.net_dims = (256, 128, 128)  # the middle layer dimension of MultiLayer Perceptron
    args.batch_size = 512
    args.gamma = 0.98
    args.horizon_len = args.max_step // 1
    args.repeat_times = 32  # repeatedly update network using ReplayBuffer to keep critic's loss small
    args.learning_rate = 2e-4
    args.state_value_tau = 0.01  # the tau of normalize for value and state `std = (1-std)*std + tau*std`
    args.lambda_gae_adv = 0.93
    args.lambda_entropy = 0.02

    args.eval_times = 16
    args.eval_per_step = 5e4
    args.if_keep_save = False  # keeping save the checkpoint. False means save until stop training.

    args.gpu_id = GPU_ID
    args.random_seed = GPU_ID
    args.num_workers = 2
    train_agent_multiprocessing(args)  # train_agent(args)
    """
    -200 < -150 < 300 < 330
    ################################################################################
    ID     Step    Time |    avgR   stdR   avgS  stdS |    expR   objC   etc.
    0  6.40e+03      33 | -107.05    5.9    169    30 |   -5.67   1.30   0.69  -0.01
    0  6.40e+03      33 | -107.05
    0  5.76e+04     113 |  -37.95    2.0   1600     0 |   -5.70   0.05   0.12  -0.00
    0  5.76e+04     113 |  -37.95
    0  1.09e+05     196 |  163.69   76.5   1497   287 |   -5.39   0.07   0.24  -0.08
    0  1.09e+05     196 |  163.69
    0  1.60e+05     280 |   28.24  120.4    690   434 |   -5.33   0.46   0.17  -0.08
    0  2.11e+05     364 |   97.72  147.8    801   396 |   -5.32   0.28   0.18  -0.09
    0  2.62e+05     447 |  254.85   78.5   1071   165 |   -5.37   0.29   0.16  -0.08
    0  2.62e+05     447 |  254.85
    0  3.14e+05     530 |  274.90   61.5   1001   123 |   -5.48   0.34   0.15  -0.04
    0  3.14e+05     530 |  274.90
    0  3.65e+05     611 |  196.47  121.1    806   220 |   -5.60   0.35   0.18  -0.01
    0  4.16e+05     689 |  250.12   89.0    890   143 |   -5.78   0.32   0.18   0.03
    0  4.67e+05     768 |  282.29   25.5    909    17 |   -5.94   0.47   0.17   0.07
    0  4.67e+05     768 |  282.29
    0  5.18e+05     848 |  289.36    1.4    897    14 |   -6.07   0.26   0.16   0.10
    0  5.18e+05     848 |  289.36
    0  5.70e+05     929 |  283.14   33.8    874    35 |   -6.29   0.27   0.13   0.16
    0  6.21e+05    1007 |  288.53    1.1    870    13 |   -6.52   0.22   0.15   0.21
    0  6.72e+05    1087 |  288.50    0.9    856    13 |   -6.68   0.40   0.15   0.25
    0  7.23e+05    1167 |  286.92    1.3    842    16 |   -6.86   0.40   0.15   0.30
    0  7.74e+05    1246 |  264.75   74.0    790   122 |   -7.10   0.42   0.18   0.36
    | TrainingTime:    1278 | SavedDir: ./BipedalWalker-v3_PPO_5
    """


def train_ppo_a2c_for_stock_trading():
    from elegantrl.envs.StockTradingEnv import StockTradingEnv
    id0 = 0
    id1 = int(1113 * 0.8)
    id2 = 1113
    gamma = 0.99

    agent_class = [AgentPPO, AgentA2C][DRL_ID]  # DRL algorithm name
    env_class = StockTradingEnv
    env_args = {'env_name': 'StockTradingEnv-v2',
                'num_envs': 1,
                'max_step': id2 - id1 - 1,
                'state_dim': 151,
                'action_dim': 15,
                'if_discrete': False,

                'gamma': gamma,
                'beg_idx': id0,
                'end_idx': id1, }
    # get_gym_vec_env_args(env=StockTradingEnv(), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `erl_config.py Arguments()` for hyperparameter explanation
    args.break_step = int(2e5)  # break training if 'total_step > break_step'
    args.net_dims = (128, 64)  # the middle layer dimension of MultiLayer Perceptron
    args.gamma = gamma  # discount factor of future rewards
    args.horizon_len = args.max_step

    args.repeat_times = 16  # repeatedly update network using ReplayBuffer to keep critic's loss small
    args.learning_rate = 1e-4
    args.state_value_tau = 0.1  # the tau of normalize for value and state `std = (1-std)*std + tau*std`

    args.eval_times = 2 ** 5
    args.eval_per_step = int(2e4)
    args.eval_env_class = StockTradingEnv
    args.eval_env_args = {'env_name': 'StockTradingEnv-v2',
                          'num_envs': 1,
                          'max_step': id2 - id1 - 1,
                          'state_dim': 151,
                          'action_dim': 15,
                          'if_discrete': False,

                          'beg_idx': id1,
                          'end_idx': id2, }

    args.gpu_id = GPU_ID
    args.num_workers = 4
    train_agent_multiprocessing(args)  # train_agent(args)
    """
RewardRange: 0.0 < 1.0 < 1.5 < 2.0
################################################################################
ID     Step    Time |    avgR   stdR   avgS  stdS |    expR   objC   etc.
0  7.12e+03       8 |    1.08    0.1    222     0 |  -21.40   4.36   0.23   0.00
0  2.85e+04      21 |    1.64    0.1    222     0 |  -21.36   6.70   0.22   0.01
0  4.98e+04      34 |    1.58    0.1    222     0 |  -21.47   4.98   0.22   0.01
0  7.12e+04      47 |    1.53    0.1    222     0 |  -21.47   3.99   0.24   0.01
0  9.26e+04      60 |    1.52    0.1    222     0 |  -21.55   3.80   0.25   0.02
0  1.14e+05      73 |    1.51    0.1    222     0 |  -21.61   3.16   0.26   0.02
0  1.35e+05      86 |    1.53    0.1    222     0 |  -21.63   3.48   0.18   0.02
0  1.57e+05     100 |    1.50    0.1    222     0 |  -21.67   2.68   0.22   0.02
0  1.78e+05     114 |    1.51    0.1    222     0 |  -21.80   2.18   0.22   0.03
0  1.99e+05     129 |    1.50    0.1    222     0 |  -21.76   2.10   0.24   0.03
| TrainingTime:     130 | SavedDir: ./StockTradingEnv-v2_PPO_0
    """


def train_ppo_a2c_for_stock_trading_vec_env():
    from elegantrl.envs.StockTradingEnv import StockTradingVecEnv
    id0 = 0
    id1 = int(1113 * 0.8)
    id2 = 1113
    num_envs = 2 ** 11
    gamma = 0.99

    agent_class = [AgentPPO, AgentA2C][DRL_ID]  # DRL algorithm name
    env_class = StockTradingVecEnv
    env_args = {'env_name': 'StockTradingVecEnv-v2',
                'num_envs': num_envs,
                'max_step': id2 - id1 - 1,
                'state_dim': 151,
                'action_dim': 15,
                'if_discrete': False,

                'gamma': gamma,
                'beg_idx': id0,
                'end_idx': id1, }
    # get_gym_vec_env_args(env=StockTradingVecEnv(), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `erl_config.py Arguments()` for hyperparameter explanation
    args.break_step = int(1e5)  # break training if 'total_step > break_step'
    args.net_dims = (128, 64)  # the middle layer dimension of MultiLayer Perceptron
    args.gamma = gamma  # discount factor of future rewards
    args.horizon_len = args.max_step

    args.repeat_times = 16  # repeatedly update network using ReplayBuffer to keep critic's loss small
    args.learning_rate = 2e-4
    args.state_value_tau = 0.1  # the tau of normalize for value and state `std = (1-std)*std + tau*std`

    args.eval_times = 2 ** 14
    args.eval_per_step = int(2e4)
    args.eval_env_class = StockTradingVecEnv
    args.eval_env_args = {'env_name': 'StockTradingVecEnv-v2',
                          'num_envs': num_envs,
                          'max_step': id2 - id1 - 1,
                          'state_dim': 151,
                          'action_dim': 15,
                          'if_discrete': False,

                          'beg_idx': id1,
                          'end_idx': id2, }

    args.gpu_id = GPU_ID
    args.random_seed = GPU_ID
    args.num_workers = 2
    train_agent_multiprocessing(args)  # train_agent(args)
    """
0.0 < 1.0 < 1.5 < 2.0
################################################################################
ID     Step    Time |    avgR   stdR   avgS  stdS |    expR   objC   etc.
0  8.88e+02      30 |    1.52    0.2    222     0 |  -21.29  19.51   0.19   0.00
0  2.13e+04     180 |    1.52    0.2    222     0 |  -21.58   1.74   0.23   0.02
0  4.17e+04     333 |    1.52    0.2    222     0 |  -21.85   0.81   0.24   0.04
0  6.22e+04     485 |    1.52    0.2    222     0 |  -22.16   0.56   0.24   0.06
0  8.26e+04     635 |    1.52    0.2    222     0 |  -22.45   0.50   0.21   0.08
| TrainingTime:     746 | SavedDir: ./StockTradingVecEnv-v2_PPO_0
    """


"""discrete action"""


def train_discrete_ppo_a2c_for_cartpole():
    import gym

    agent_class = [AgentDiscretePPO, AgentDiscreteA2C][DRL_ID]  # DRL algorithm name
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

    args.gpu_id = GPU_ID
    args.num_workers = 4
    # train_agent_multiprocessing(args)
    train_agent(args)
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


def train_discrete_ppo_a2c_for_cartpole_vec_env():
    import gym

    agent_class = [AgentDiscretePPO, AgentDiscreteA2C][DRL_ID]  # DRL algorithm name
    env_class = gym.make  # run a custom env: PendulumEnv, which based on OpenAI pendulum
    env_args = {
        'env_name': 'CartPole-v1',
        'max_step': 500,
        'state_dim': 4,
        'action_dim': 2,
        'if_discrete': True,

        'num_envs': 4,  # the number of sub envs in vectorized env
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

    args.gpu_id = GPU_ID
    args.num_workers = 4
    train_agent_multiprocessing(args)  # train_agent(args)
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


def train_discrete_ppo_a2c_for_lunar_lander():
    import gym

    agent_class = [AgentDiscretePPO, AgentDiscreteA2C][DRL_ID]  # DRL algorithm name
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

    args.gpu_id = GPU_ID
    args.num_workers = 4
    train_agent_multiprocessing(args)  # train_agent(args)
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


def train_discrete_ppo_a2c_for_lunar_lander_vec_env():
    import gym

    agent_class = [AgentDiscretePPO, AgentDiscreteA2C][DRL_ID]  # DRL algorithm name
    env_class = gym.make  # run a custom env: PendulumEnv, which based on OpenAI pendulum
    env_args = {
        'env_name': 'LunarLander-v2',
        'max_step': 1000,
        'state_dim': 8,
        'action_dim': 2,
        'if_discrete': True,

        'num_envs': 4,  # the number of sub envs in vectorized env
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

    args.gpu_id = GPU_ID
    args.num_workers = 4
    train_agent_multiprocessing(args)  # train_agent(args)
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


'''utils'''


def demo_load_pendulum_and_render():
    import torch

    gpu_id = 0  # >=0 means GPU ID, -1 means CPU
    device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

    from elegantrl.envs.CustomGymEnv import PendulumEnv
    env_class = PendulumEnv  # run a custom env: PendulumEnv, which based on OpenAI pendulum
    env_args = {
        'env_name': 'Pendulum',  # Apply torque on the free end to swing a pendulum into an upright position
        # Reward: r = -(theta + 0.1 * theta_dt + 0.001 * torque)
        'num_envs': 1,  # the number of sub envs in vectorized env. `num_envs=1` in single env.
        'state_dim': 3,  # the x-y coordinates of the pendulum's free end and its angular velocity.
        'action_dim': 1,  # the torque applied to free end of the pendulum
        'if_discrete': False  # continuous action space, symbols → direction, value → force
    }

    '''init'''
    from elegantrl.train.config import build_env
    env = build_env(env_class=env_class, env_args=env_args)
    act = torch.load(f"./Pendulum_PPO_0/act.pt", map_location=device)

    '''evaluate'''
    eval_times = 2 ** 7
    from elegantrl.train.evaluator import get_cumulative_rewards_and_steps
    rewards_step_list = [get_cumulative_rewards_and_steps(env, act) for _ in range(eval_times)]
    rewards_step_ten = torch.tensor(rewards_step_list)
    print(f"\n| average cumulative_returns {rewards_step_ten[:, 0].mean().item():9.3f}"
          f"\n| average      episode steps {rewards_step_ten[:, 1].mean().item():9.3f}")

    '''render'''
    if_discrete = env.if_discrete
    device = next(act.parameters()).device  # net.parameters() is a Python generator.

    state = env.reset()
    steps = None
    returns = 0.0  # sum of rewards in an episode
    for steps in range(12345):
        s_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        a_tensor = act(s_tensor).argmax(dim=1) if if_discrete else act(s_tensor)
        action = a_tensor.detach().cpu().numpy()[0]  # not need detach(), because using torch.no_grad() outside
        state, reward, done, _ = env.step(action)
        returns += reward
        env.render()

        if done:
            break
    returns = getattr(env, 'cumulative_rewards', returns)
    steps += 1

    print(f"\n| cumulative_returns {returns}"
          f"\n|      episode steps {steps}")


def demo_load_pendulum_vectorized_env():
    import torch

    gpu_id = 0  # >=0 means GPU ID, -1 means CPU
    device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

    from elegantrl.envs.CustomGymEnv import PendulumEnv
    env_class = PendulumEnv  # run a custom env: PendulumEnv, which based on OpenAI pendulum
    num_envs = 4
    env_args = {
        'env_name': 'Pendulum',  # Apply torque on the free end to swing a pendulum into an upright position
        'max_step': 200,  # the max step number in an episode for evaluation
        'state_dim': 3,  # the x-y coordinates of the pendulum's free end and its angular velocity.
        'action_dim': 1,  # the torque applied to free end of the pendulum
        'if_discrete': False,  # continuous action space, symbols → direction, value → force

        'num_envs': num_envs,  # the number of sub envs in vectorized env
        'if_build_vec_env': True,
    }

    '''init'''
    from elegantrl.train.config import build_env
    env = build_env(env_class=env_class, env_args=env_args)
    act = torch.load(f"./Pendulum_PPO_0/act.pt", map_location=device)

    '''evaluate'''
    eval_times = 2 ** 7
    from elegantrl.train.evaluator import get_cumulative_rewards_and_step_from_vec_env
    rewards_step_list = []
    [rewards_step_list.extend(get_cumulative_rewards_and_step_from_vec_env(env, act)) for _ in range(eval_times // num_envs)]
    rewards_step_ten = torch.tensor(rewards_step_list)
    print(f"\n| average cumulative_returns {rewards_step_ten[:, 0].mean().item():9.3f}"
          f"\n| average      episode steps {rewards_step_ten[:, 1].mean().item():9.3f}")


if __name__ == '__main__':
    Parser = ArgumentParser(description='ArgumentParser for ElegantRL')
    Parser.add_argument('--gpu', type=int, default=0, help='GPU device ID for training')
    Parser.add_argument('--drl', type=int, default=0, help='RL algorithms ID for training')
    Parser.add_argument('--env', type=str, default='0', help='the environment ID for training')

    Args = Parser.parse_args()
    GPU_ID = Args.gpu
    DRL_ID = Args.drl
    ENV_ID = Args.env

    if ENV_ID in {'0', 'pendulum'}:
        train_ppo_a2c_for_pendulum()
    elif ENV_ID in {'1', 'pendulum_vec'}:
        train_ppo_a2c_for_pendulum_vec_env()
    elif ENV_ID in {'2', 'lunar_lander_continuous'}:
        train_ppo_a2c_for_lunar_lander_continuous()
    elif ENV_ID in {'3', 'lunar_lander_continuous_vec'}:
        train_ppo_a2c_for_lunar_lander_continuous_vec_env()
    elif ENV_ID in {'4', 'bipedal_walker'}:
        train_ppo_a2c_for_bipedal_walker()
    elif ENV_ID in {'5', 'bipedal_walker_vec'}:
        train_ppo_a2c_for_bipedal_walker_vec_env()

    elif ENV_ID in {'6', 'cartpole'}:
        train_discrete_ppo_a2c_for_cartpole()
    elif ENV_ID in {'7', 'cartpole_vec'}:
        train_discrete_ppo_a2c_for_cartpole_vec_env()
    elif ENV_ID in {'8', 'lunar_lander'}:
        train_discrete_ppo_a2c_for_lunar_lander()
    elif ENV_ID in {'9', 'lunar_lander_vec'}:
        train_discrete_ppo_a2c_for_lunar_lander_vec_env()
    else:
        print('ENV_ID not match')
