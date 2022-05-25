from elegantrl_helloworld.config import Arguments
from elegantrl_helloworld.run import train_agent, evaluate_agent
from elegantrl_helloworld.env import get_gym_env_args, PendulumEnv


def train_dqn_in_cartpole(gpu_id=0):  # DQN is a simple but low sample efficiency.
    from elegantrl_helloworld.agent import AgentDQN
    agent_class = AgentDQN
    env_name = "CartPole-v0"

    import gym
    gym.logger.set_level(40)  # Block warning
    env = gym.make(env_name)
    env_func = gym.make
    env_args = get_gym_env_args(env, if_print=True)

    args = Arguments(agent_class, env_func, env_args)

    '''reward shaping'''
    args.reward_scale = 2 ** 0
    args.gamma = 0.97

    '''network update'''
    args.target_step = args.max_step * 2
    args.net_dim = 2 ** 7
    args.num_layer = 3
    args.batch_size = 2 ** 7
    args.repeat_times = 2 ** 0
    args.explore_rate = 0.25

    '''evaluate'''
    args.eval_gap = 2 ** 5
    args.eval_times = 2 ** 3
    args.break_step = int(8e4)

    args.learner_gpus = gpu_id
    train_agent(args)
    evaluate_agent(args)
    print('| The cumulative returns of CartPole-v0  is ∈ (0, (0, 195), 200)')

    """
    | `Steps` denotes the number of samples, or the total training step, or the running times of `env.step()`.
    | `ExpR` denotes average rewards during exploration. The agent gets this rewards with noisy action.
    | `ObjC` denotes the objective of Critic network. Or call it loss function of critic network.
    | `ObjA` denotes the objective of Actor network. It is the average Q value of the critic network.

    | Steps 4.06e+02  ExpR     1.00| ObjC     0.05  ObjA     1.86
    | Steps 1.40e+04  ExpR     1.00| ObjC     0.31  ObjA    44.92
    | Steps 2.33e+04  ExpR     1.00| ObjC     2.66  ObjA    32.86
    | Steps 3.07e+04  ExpR     1.00| ObjC     1.36  ObjA    31.46
    | Steps 3.71e+04  ExpR     1.00| ObjC     0.29  ObjA    32.31
    | Steps 4.22e+04  ExpR     1.00| ObjC     0.18  ObjA    30.42
    | Steps 4.68e+04  ExpR     1.00| ObjC     2.10  ObjA    30.12
    | Steps 5.12e+04  ExpR     1.00| ObjC     1.90  ObjA    32.92
    | Steps 5.49e+04  ExpR     1.00| ObjC     0.38  ObjA    28.36
    | Steps 5.88e+04  ExpR     1.00| ObjC     0.55  ObjA    29.84
    | Steps 6.28e+04  ExpR     1.00| ObjC     0.02  ObjA    31.68
    | Steps 6.61e+04  ExpR     1.00| ObjC     2.35  ObjA    28.81
    | Steps 6.98e+04  ExpR     1.00| ObjC     0.27  ObjA    28.38
    | Steps 7.28e+04  ExpR     1.00| ObjC     0.07  ObjA    31.23
    | UsedTime: 452 | SavedDir: ./CartPole-v0_DQN_6

    | The cumulative returns of CartPole-v0  is ∈ (0, (0, 195), 200)
    | `Steps` denotes the number of samples, or the total training step, or the running times of `env.step()`.
    | `avgR` denotes average value of cumulative rewards, which is the sum of rewards in an episode.
    | `stdR` denotes standard dev of cumulative rewards, which is the sum of rewards in an episode.
    | `avgS` denotes the average number of steps in an episode.
    
    | Steps 4.06e+02  | avgR    10.000  stdR     0.866| avgS      9
    | Steps 1.40e+04  | avgR     9.750  stdR     0.968| avgS      9
    | Steps 2.33e+04  | avgR   145.125  stdR    25.746| avgS    144
    | Steps 3.07e+04  | avgR   185.125  stdR    19.934| avgS    184
    | Steps 3.71e+04  | avgR   114.625  stdR     9.578| avgS    114
    | Steps 4.22e+04  | avgR   153.250  stdR    25.820| avgS    152
    | Steps 4.68e+04  | avgR   172.375  stdR    14.186| avgS    171
    | Steps 5.12e+04  | avgR   181.750  stdR    29.329| avgS    181
    | Steps 5.49e+04  | avgR    68.750  stdR    16.902| avgS     68
    | Steps 5.88e+04  | avgR    92.125  stdR    40.563| avgS     91
    | Steps 6.28e+04  | avgR   200.000  stdR     0.000| avgS    199
    | Steps 6.61e+04  | avgR   200.000  stdR     0.000| avgS    199
    | Steps 6.98e+04  | avgR   119.000  stdR     5.362| avgS    118
    | Steps 7.28e+04  | avgR    94.250  stdR     3.192| avgS     93    
    """


def train_dqn_in_lunar_lander(gpu_id=0):  # DQN is a simple but low sample efficiency.
    from elegantrl_helloworld.agent import AgentDQN
    agent_class = AgentDQN
    env_name = "LunarLander-v2"

    import gym
    gym.logger.set_level(40)  # Block warning
    env = gym.make(env_name)
    env_func = gym.make
    env_args = get_gym_env_args(env, if_print=True)

    args = Arguments(agent_class, env_func, env_args)

    '''reward shaping'''
    args.reward_scale = 2 ** 0
    args.gamma = 0.99

    '''network update'''
    args.target_step = args.max_step
    args.net_dim = 2 ** 7
    args.num_layer = 3

    args.batch_size = 2 ** 6

    args.repeat_times = 2 ** 0
    args.explore_noise = 0.125

    '''evaluate'''
    args.eval_gap = 2 ** 7
    args.eval_times = 2 ** 4
    args.break_step = int(4e5)  # LunarLander needs a larger `break_step`

    args.learner_gpus = gpu_id
    train_agent(args)
    evaluate_agent(args)
    print('| The cumulative returns of LunarLander-v2 is ∈ (-1800, (-600, 200), 340)')

    """    ========args.batch_size = 2 ** 7================
    | Arguments Remove cwd: ./LunarLander-v2_DQN_4
    
    | `Steps` denotes the number of samples, or the total training step, or the running times of `env.step()`.
    | `ExpR` denotes average rewards during exploration. The agent gets this rewards with noisy action.
    | `ObjC` denotes the objective of Critic network. Or call it loss function of critic network.
    | `ObjA` denotes the objective of Actor network. It is the average Q value of the critic network.
    
    | Steps 1.07e+03  ExpR    -3.11| ObjC     1.39  ObjA    -3.31
    | Steps 5.70e+04  ExpR    -0.32| ObjC     7.38  ObjA    -2.51
    | Steps 9.65e+04  ExpR    -0.13| ObjC     2.56  ObjA     9.18
    | Steps 1.35e+05  ExpR    -0.22| ObjC     3.60  ObjA    43.27
    | Steps 1.71e+05  ExpR    -0.03| ObjC     1.39  ObjA    53.53
    | Steps 2.05e+05  ExpR     0.02| ObjC     2.18  ObjA    17.03
    | Steps 2.38e+05  ExpR     0.29| ObjC    12.58  ObjA    -2.42
    | Steps 2.63e+05  ExpR    -0.10| ObjC    38.59  ObjA   -27.72
    | Steps 2.89e+05  ExpR    -0.10| ObjC     2.39  ObjA    39.32
    | Steps 3.15e+05  ExpR    -0.48| ObjC     1.41  ObjA    38.57
    | Steps 3.39e+05  ExpR    -0.21| ObjC     0.95  ObjA    65.95
    | Steps 3.65e+05  ExpR     0.02| ObjC     0.73  ObjA    60.34
    | Steps 3.88e+05  ExpR     0.24| ObjC     0.87  ObjA    58.65
    | UsedTime: 1636 | SavedDir: ./LunarLander-v2_DQN_4
    | Arguments Keep cwd: ./LunarLander-v2_DQN_4

    | The cumulative returns of LunarLander-v2 is ∈ (-1800, (-600, 200), 340)

    | `Steps` denotes the number of samples, or the total training step, or the running times of `env.step()`.
    | `avgR` denotes average value of cumulative rewards, which is the sum of rewards in an episode.
    | `stdR` denotes standard dev of cumulative rewards, which is the sum of rewards in an episode.
    | `avgS` denotes the average number of steps in an episode.
    
    | Steps 1.07e+03  | avgR  -690.080  stdR    80.464| avgS    107
    | Steps 5.70e+04  | avgR  -127.956  stdR    15.654| avgS    371
    | Steps 9.65e+04  | avgR  -121.114  stdR    30.467| avgS   1000
    | Steps 1.35e+05  | avgR     3.904  stdR   131.609| avgS    622
    | Steps 1.71e+05  | avgR    80.979  stdR   109.091| avgS    304
    | Steps 2.05e+05  | avgR    -2.671  stdR   168.546| avgS    322
    | Steps 2.38e+05  | avgR    45.882  stdR   103.641| avgS    273
    | Steps 2.63e+05  | avgR    27.995  stdR   112.317| avgS    428
    | Steps 2.89e+05  | avgR    33.219  stdR    94.456| avgS    131
    | Steps 3.15e+05  | avgR   -77.332  stdR   132.200| avgS    525
    | Steps 3.39e+05  | avgR   -71.449  stdR   121.398| avgS    154
    | Steps 3.65e+05  | avgR   -62.966  stdR   182.249| avgS    346
    | Steps 3.88e+05  | avgR    18.644  stdR    62.837| avgS    161
    """

    """    ========args.batch_size = 2 ** 6================
    | Arguments Remove cwd: ./LunarLander-v2_DQN_5
    
    | `Steps` denotes the number of samples, or the total training step, or the running times of `env.step()`.
    | `ExpR` denotes average rewards during exploration. The agent gets this rewards with noisy action.
    | `ObjC` denotes the objective of Critic network. Or call it loss function of critic network.
    | `ObjA` denotes the objective of Actor network. It is the average Q value of the critic network.
    
    | Steps 1.01e+03  ExpR    -2.43| ObjC     1.01  ObjA    -3.96
    | Steps 3.34e+04  ExpR    -0.10| ObjC     0.93  ObjA    18.87
    | Steps 5.75e+04  ExpR    -0.01| ObjC     8.59  ObjA   -46.80
    | Steps 8.02e+04  ExpR    -0.05| ObjC    19.03  ObjA    14.51
    | Steps 1.01e+05  ExpR     0.18| ObjC     1.26  ObjA    34.26
    | Steps 1.24e+05  ExpR    -0.03| ObjC     0.97  ObjA    29.73
    | Steps 1.45e+05  ExpR     0.10| ObjC     2.19  ObjA    23.51
    | Steps 1.64e+05  ExpR    -0.69| ObjC     3.20  ObjA    21.15
    | Steps 1.82e+05  ExpR    -0.06| ObjC     1.85  ObjA    -1.69
    | Steps 1.96e+05  ExpR    -0.00| ObjC     1.29  ObjA     9.82
    | Steps 2.10e+05  ExpR     0.09| ObjC     1.64  ObjA    40.96
    | Steps 2.25e+05  ExpR    -0.05| ObjC    19.29  ObjA   -24.78
    | Steps 2.38e+05  ExpR     0.17| ObjC     0.96  ObjA    47.89
    | Steps 2.55e+05  ExpR     0.18| ObjC     1.27  ObjA    82.55
    | Steps 2.70e+05  ExpR     0.26| ObjC     0.80  ObjA    46.55
    | Steps 2.85e+05  ExpR     0.07| ObjC     1.31  ObjA    44.45
    | Steps 3.00e+05  ExpR     0.17| ObjC     7.35  ObjA    44.21
    | Steps 3.15e+05  ExpR     0.05| ObjC     2.05  ObjA    -0.03
    | Steps 3.30e+05  ExpR     0.02| ObjC     0.96  ObjA    48.78
    | Steps 3.43e+05  ExpR     0.01| ObjC     1.51  ObjA    42.02
    | Steps 3.56e+05  ExpR     0.10| ObjC     3.74  ObjA    17.00
    | Steps 3.69e+05  ExpR    -0.07| ObjC     4.76  ObjA    -0.25
    | Steps 3.82e+05  ExpR     0.14| ObjC     0.57  ObjA    32.75
    | Steps 3.94e+05  ExpR     0.06| ObjC     2.33  ObjA    44.47
    | UsedTime: 3145 | SavedDir: ./LunarLander-v2_DQN_5
    | Arguments Keep cwd: ./LunarLander-v2_DQN_5
    
    | `Steps` denotes the number of samples, or the total training step, or the running times of `env.step()`.
    | `avgR` denotes average value of cumulative rewards, which is the sum of rewards in an episode.
    | `stdR` denotes standard dev of cumulative rewards, which is the sum of rewards in an episode.
    | `avgS` denotes the average number of steps in an episode.
    
    | Steps 1.01e+03  | avgR  -623.288  stdR   140.350| avgS    109
    | Steps 3.34e+04  | avgR  -136.745  stdR    17.630| avgS    952
    | Steps 5.75e+04  | avgR   -81.121  stdR    26.961| avgS   1000
    | Steps 8.02e+04  | avgR   -34.619  stdR    31.289| avgS   1000
    | Steps 1.01e+05  | avgR    35.350  stdR   114.496| avgS    521
    | Steps 1.24e+05  | avgR    -7.270  stdR   118.275| avgS    216
    | Steps 1.45e+05  | avgR    50.880  stdR    80.105| avgS    228
    | Steps 1.64e+05  | avgR    15.081  stdR    91.996| avgS    264
    | Steps 1.82e+05  | avgR   -72.091  stdR    18.038| avgS   1000
    | Steps 1.96e+05  | avgR   -16.467  stdR    94.893| avgS    930
    | Steps 2.10e+05  | avgR   -26.394  stdR   110.414| avgS    931
    | Steps 2.25e+05  | avgR   114.066  stdR   117.302| avgS    623
    | Steps 2.38e+05  | avgR   133.624  stdR   106.857| avgS    399
    | Steps 2.55e+05  | avgR    71.769  stdR   123.068| avgS    164
    | Steps 2.70e+05  | avgR   147.773  stdR   138.012| avgS    340
    | Steps 2.85e+05  | avgR    93.060  stdR   136.614| avgS    333
    | Steps 3.00e+05  | avgR   116.197  stdR    88.235| avgS    494
    | Steps 3.15e+05  | avgR   -10.208  stdR    37.468| avgS    121
    | Steps 3.30e+05  | avgR   126.338  stdR   115.467| avgS    416
    | Steps 3.43e+05  | avgR    38.537  stdR   173.269| avgS    693
    | Steps 3.56e+05  | avgR   205.933  stdR    44.255| avgS    540
    | Steps 3.69e+05  | avgR   174.410  stdR    67.452| avgS    729
    | Steps 3.82e+05  | avgR    -2.781  stdR    80.126| avgS    923
    | Steps 3.94e+05  | avgR   -65.562  stdR   122.434| avgS    599
    """


def train_ddpg_in_pendulum(gpu_id=0):  # DDPG is a simple but low sample efficiency and unstable.
    from elegantrl_helloworld.agent import AgentDDPG
    agent_class = AgentDDPG

    env = PendulumEnv()
    env_func = PendulumEnv
    env_args = get_gym_env_args(env, if_print=True)

    args = Arguments(agent_class, env_func, env_args)

    '''reward shaping'''
    args.reward_scale = 2 ** -1  # RewardRange: -1800 < -200 < -50 < 0
    args.gamma = 0.97

    '''network update'''
    args.target_step = args.max_step * 2
    args.net_dim = 2 ** 7
    args.batch_size = 2 ** 7
    args.repeat_times = 2 ** 0
    args.explore_noise = 0.1

    '''evaluate'''
    args.eval_gap = 2 ** 6
    args.eval_times = 2 ** 3
    args.break_step = int(1e5)

    args.learner_gpus = gpu_id
    train_agent(args)
    evaluate_agent(args)
    print('| The cumulative returns of Pendulum-v1 is ∈ (-1600, (-1400, -200), 0)')

    """
    | Arguments Remove cwd: ./Pendulum-v1_DDPG_4
    
    | `Steps` denotes the number of samples, or the total training step, or the running times of `env.step()`.
    | `ExpR` denotes average rewards during exploration. The agent gets this rewards with noisy action.
    | `ObjC` denotes the objective of Critic network. Or call it loss function of critic network.
    | `ObjA` denotes the objective of Actor network. It is the average Q value of the critic network.
    
    | Steps 4.00e+02  ExpR    -3.32| ObjC     0.94  ObjA    -0.94
    | Steps 2.08e+04  ExpR    -0.61| ObjC     0.51  ObjA   -38.80
    | Steps 3.60e+04  ExpR    -0.33| ObjC     0.23  ObjA   -27.02
    | Steps 4.88e+04  ExpR    -0.31| ObjC     0.12  ObjA   -23.40
    | Steps 6.00e+04  ExpR    -0.59| ObjC     0.13  ObjA   -18.43
    | Steps 7.00e+04  ExpR    -0.63| ObjC     0.46  ObjA   -21.12
    | Steps 7.92e+04  ExpR    -0.16| ObjC     0.12  ObjA   -20.51
    | Steps 8.76e+04  ExpR    -0.31| ObjC     0.08  ObjA   -17.31
    | Steps 9.56e+04  ExpR    -0.31| ObjC     0.19  ObjA   -17.15
    | UsedTime: 566 | SavedDir: ./Pendulum-v1_DDPG_4
    | Arguments Keep cwd: ./Pendulum-v1_DDPG_4
    
    | The cumulative returns of Pendulum-v1 is ∈ (-1600, (-1400, -200), 0)
    
    | `Steps` denotes the number of samples, or the total training step, or the running times of `env.step()`.
    | `avgR` denotes average value of cumulative rewards, which is the sum of rewards in an episode.
    | `stdR` denotes standard dev of cumulative rewards, which is the sum of rewards in an episode.
    | `avgS` denotes the average number of steps in an episode.
    
    | Steps 4.00e+02  | avgR -1412.013  stdR   312.738| avgS    200
    | Steps 2.08e+04  | avgR  -202.588  stdR   121.154| avgS    200
    | Steps 3.60e+04  | avgR  -109.563  stdR    71.815| avgS    200
    | Steps 4.88e+04  | avgR  -148.049  stdR    77.257| avgS    200
    | Steps 6.00e+04  | avgR  -111.263  stdR    77.877| avgS    200
    | Steps 7.00e+04  | avgR  -200.349  stdR    62.672| avgS    200
    | Steps 7.92e+04  | avgR  -122.915  stdR    82.007| avgS    200
    | Steps 8.76e+04  | avgR  -173.456  stdR    92.135| avgS    200
    | Steps 9.56e+04  | avgR  -196.744  stdR    96.240| avgS    200

    """


def train_ddpg_in_lunar_lander_or_bipedal_walker(gpu_id=0):  # DDPG is a simple but low sample efficiency and unstable.
    from elegantrl_helloworld.agent import AgentDDPG
    agent_class = AgentDDPG
    env_name = ["LunarLanderContinuous-v2", "BipedalWalker-v3"][1]

    if env_name == "LunarLanderContinuous-v2":
        """
        | Arguments Remove cwd: ./LunarLanderContinuous-v2_DDPG_5
        | Step 1.04e+03  ExpR    -1.08  | ObjC     3.69  ObjA    -1.16
        | Step 5.50e+04  ExpR    -0.13  | ObjC    10.65  ObjA   -93.88
        | Step 9.44e+04  ExpR    -0.17  | ObjC     7.94  ObjA   112.21
        | Step 1.29e+05  ExpR    -6.02  | ObjC    13.47  ObjA   250.81
        | Step 1.64e+05  ExpR     0.03  | ObjC     1.76  ObjA   152.00
        | Step 1.96e+05  ExpR    -0.24  | ObjC     1.58  ObjA    93.80
        | Step 2.25e+05  ExpR    -0.00  | ObjC     1.23  ObjA   111.35
        | Step 2.51e+05  ExpR    -0.00  | ObjC     2.52  ObjA    87.91
        | Step 2.76e+05  ExpR     0.25  | ObjC     1.04  ObjA    84.15
        | Step 2.97e+05  ExpR    -0.05  | ObjC     3.77  ObjA    67.31
        | Step 3.20e+05  ExpR     0.48  | ObjC     2.09  ObjA     5.80
        | Step 3.41e+05  ExpR     0.15  | ObjC     7.45  ObjA    79.38
        | Step 3.63e+05  ExpR    -0.27  | ObjC     5.83  ObjA    77.82
        | Step 3.85e+05  ExpR     0.12  | ObjC     2.78  ObjA    99.50
        | Step 3.95e+05  ExpR     0.01  | ObjC     1.59  ObjA    97.64
        | UsedTime: 3674 | SavedDir: ./LunarLanderContinuous-v2_DDPG_5

        | Arguments Keep cwd: ./LunarLanderContinuous-v2_DDPG_5
        | Steps 1.04e+03  | Returns avg  -918.185  std   836.075
        | Steps 5.50e+04  | Returns avg  -160.366  std    63.918
        | Steps 9.44e+04  | Returns avg  -177.795  std    43.567
        | Steps 1.29e+05  | Returns avg  -416.993  std   110.154
        | Steps 1.64e+05  | Returns avg   -92.654  std    72.691
        | Steps 1.96e+05  | Returns avg   -60.600  std    59.443
        | Steps 2.25e+05  | Returns avg   -37.197  std    36.825
        | Steps 2.39e+05  | Returns avg    -5.715  std    28.418
        | Steps 2.63e+05  | Returns avg    10.495  std    84.063
        | Steps 2.76e+05  | Returns avg   125.594  std    57.873
        | Steps 2.97e+05  | Returns avg   -84.772  std    53.977
        | Steps 3.20e+05  | Returns avg    73.619  std   157.157
        | Steps 3.31e+05  | Returns avg   132.285  std    86.783
        | Steps 3.41e+05  | Returns avg   219.191  std   117.352
        | Steps 3.52e+05  | Returns avg   -77.722  std    79.539
        | Steps 3.85e+05  | Returns avg    50.904  std   144.877
        | Steps 3.95e+05  | Returns avg    67.412  std   108.368
        """
        import gym
        env = gym.make(env_name)
        env_func = gym.make
        env_args = get_gym_env_args(env, if_print=True)

        args = Arguments(agent_class, env_func, env_args)

        '''reward shaping'''
        args.reward_scale = 2 ** 0
        args.gamma = 0.99

        '''network update'''
        args.target_step = args.max_step // 2
        args.net_dim = 2 ** 7
        args.batch_size = 2 ** 7
        args.repeat_times = 2 ** 0
        args.explore_noise = 0.1

        '''evaluate'''
        args.eval_gap = 2 ** 7
        args.eval_times = 2 ** 4
        args.break_step = int(4e5)
    elif env_name == "BipedalWalker-v3":
        import gym
        env = gym.make(env_name)
        env_func = gym.make
        env_args = get_gym_env_args(env, if_print=True)

        args = Arguments(agent_class, env_func, env_args)

        '''reward shaping'''
        args.reward_scale = 2 ** -1
        args.gamma = 0.99

        '''network update'''
        args.target_step = args.max_step // 2
        args.net_dim = 2 ** 8
        args.num_layer = 3
        args.batch_size = 2 ** 7
        args.repeat_times = 2 ** 0
        args.explore_noise = 0.1

        '''evaluate'''
        args.eval_gap = 2 ** 7
        args.eval_times = 2 ** 3
        args.break_step = int(1e6)
    else:
        raise ValueError("env_name:", env_name)

    args.learner_gpus = gpu_id
    train_agent(args)
    evaluate_agent(args)


def train_ppo_in_pendulum(gpu_id=0):
    from elegantrl_helloworld.agent import AgentPPO
    agent_class = AgentPPO

    env = PendulumEnv()
    env_func = PendulumEnv
    env_args = get_gym_env_args(env, if_print=True)

    args = Arguments(agent_class, env_func, env_args)

    '''reward shaping'''
    args.reward_scale = 2 ** -1  # RewardRange: -1800 < -200 < -50 < 0
    args.gamma = 0.97

    '''network update'''
    args.target_step = args.max_step * 8
    args.net_dim = 2 ** 7
    args.num_layer = 2
    args.batch_size = 2 ** 8
    args.repeat_times = 2 ** 5

    '''evaluate'''
    args.eval_gap = 2 ** 6
    args.eval_times = 2 ** 3
    args.break_step = int(8e5)

    args.learner_gpus = gpu_id
    train_agent(args)
    evaluate_agent(args)
    print('| The cumulative returns of Pendulum-v1 is ∈ (-1600, (-1400, -200), 0)')

    """
    | Arguments Remove cwd: ./Pendulum-v1_PPO_2

    | `Steps` denotes the number of samples, or the total training step, or the running times of `env.step()`.
    | `ExpR` denotes average rewards during exploration. The agent gets this rewards with noisy action.
    | `ObjC` denotes the objective of Critic network. Or call it loss function of critic network.
    | `ObjA` denotes the objective of Actor network. It is the average Q value of the critic network.
    
    | Steps 1.60e+03  ExpR    -3.66  | ObjC   104.02  ObjA     0.02
    | Steps 1.12e+05  ExpR    -3.48  | ObjC    26.91  ObjA     0.03
    | Steps 2.22e+05  ExpR    -1.92  | ObjC    14.93  ObjA    -0.04
    | Steps 3.33e+05  ExpR    -1.60  | ObjC     9.75  ObjA     0.02
    | Steps 4.45e+05  ExpR    -0.50  | ObjC     2.34  ObjA     0.06
    | Steps 5.55e+05  ExpR    -0.58  | ObjC     3.69  ObjA     0.02
    | Steps 6.64e+05  ExpR    -0.56  | ObjC     1.65  ObjA     0.06
    | Steps 7.73e+05  ExpR    -0.44  | ObjC     0.77  ObjA    -0.08
    | UsedTime: 470 | SavedDir: ./Pendulum-v1_PPO_2
    | Arguments Keep cwd: ./Pendulum-v1_PPO_2
    
    | The cumulative returns of Pendulum-v1 is ∈ (-1600, (-1400, -200), 0)

    | `Steps` denotes the number of samples, or the total training step, or the running times of `env.step()`.
    | `avgR` denotes average value of cumulative rewards, which is the sum of rewards in an episode.
    | `stdR` denotes standard dev of cumulative rewards, which is the sum of rewards in an episode.
    | `avgS` denotes the average number of steps in an episode.
    
    | Steps 1.60e+03  | avgR -1473.298  stdR   110.238  | avgS    200
    | Steps 1.12e+05  | avgR -1306.492  stdR    56.596  | avgS    200
    | Steps 2.22e+05  | avgR  -718.661  stdR    90.832  | avgS    200
    | Steps 3.33e+05  | avgR  -514.451  stdR   133.747  | avgS    200
    | Steps 4.45e+05  | avgR  -276.975  stdR   102.758  | avgS    200
    | Steps 5.55e+05  | avgR  -300.184  stdR   195.712  | avgS    200
    | Steps 6.64e+05  | avgR  -228.195  stdR   106.305  | avgS    200
    | Steps 7.73e+05  | avgR  -193.420  stdR   103.970  | avgS    200
    | Save learning curve in ./Pendulum-v1_PPO_2/LearningCurve_Pendulum-v1_AgentPPO.jpg
    """


def train_ppo_in_lunar_lander_or_bipedal_walker(gpu_id=0):
    from elegantrl_helloworld.agent import AgentPPO
    agent_class = AgentPPO
    env_name = ["LunarLanderContinuous-v2", "BipedalWalker-v3"][1]

    if env_name == "LunarLanderContinuous-v2":
        import gym
        env = gym.make(env_name)
        env_func = gym.make
        env_args = get_gym_env_args(env, if_print=True)

        args = Arguments(agent_class, env_func, env_args)

        '''reward shaping'''
        args.gamma = 0.99
        args.reward_scale = 2 ** -1

        '''network update'''
        args.target_step = args.max_step * 8
        args.num_layer = 3
        args.batch_size = 2 ** 7
        args.repeat_times = 2 ** 4
        args.lambda_entropy = 0.04

        '''evaluate'''
        args.eval_gap = 2 ** 6
        args.eval_times = 2 ** 5
        args.break_step = int(4e5)

        args.learner_gpus = gpu_id
        train_agent(args)
        evaluate_agent(args)
        print('| The cumulative returns of LunarLanderContinuous-v2 is ∈ (-1800, (-300, 200), 310+)')

        """
        | Arguments Keep cwd: ./LunarLanderContinuous-v2_PPO_4

        | `Steps` denotes the number of samples, or the total training step, or the running times of `env.step()`.
        | `ExpR` denotes average rewards during exploration. The agent gets this rewards with noisy action.
        | `ObjC` denotes the objective of Critic network. Or call it loss function of critic network.
        | `ObjA` denotes the objective of Actor network. It is the average Q value of the critic network.
        
        | Steps 8.04e+03  ExpR    -0.95  | ObjC    24.03  ObjA     0.14
        | Steps 5.62e+04  ExpR    -0.11  | ObjC     7.07  ObjA    -0.00
        | Steps 9.65e+04  ExpR     0.01  | ObjC     4.59  ObjA    -0.06
        | Steps 1.22e+05  ExpR     0.04  | ObjC     5.88  ObjA    -0.00
        | Steps 1.48e+05  ExpR     0.02  | ObjC     4.17  ObjA    -0.01
        | Steps 1.73e+05  ExpR     0.04  | ObjC     4.48  ObjA     0.09
        | Steps 1.99e+05  ExpR     0.02  | ObjC     3.10  ObjA     0.13
        | Steps 2.24e+05  ExpR     0.04  | ObjC     2.40  ObjA    -0.09
        | Steps 2.56e+05  ExpR     0.06  | ObjC     1.10  ObjA     0.05
        | Steps 2.88e+05  ExpR     0.06  | ObjC     0.68  ObjA     0.05
        | Steps 3.21e+05  ExpR     0.08  | ObjC     0.48  ObjA    -0.14
        | Steps 3.53e+05  ExpR     0.08  | ObjC     0.45  ObjA    -0.02
        | Steps 3.87e+05  ExpR     0.08  | ObjC     0.36  ObjA    -0.05
        | Steps 4.20e+05  ExpR     0.09  | ObjC     0.36  ObjA    -0.14
        | Steps 4.53e+05  ExpR     0.08  | ObjC     0.37  ObjA    -0.09
        | Steps 4.85e+05  ExpR     0.08  | ObjC     0.37  ObjA     0.09
        | Steps 5.18e+05  ExpR     0.10  | ObjC     1.39  ObjA    -0.07
        | Steps 5.60e+05  ExpR     0.10  | ObjC     1.66  ObjA     0.10
        | Steps 5.94e+05  ExpR     0.11  | ObjC     2.40  ObjA    -0.00
        | UsedTime: 1284 | SavedDir: ./LunarLanderContinuous-v2_PPO_4
        | Arguments Keep cwd: ./LunarLanderContinuous-v2_PPO_4
        
        | The cumulative returns of LunarLanderContinuous-v2 is ∈ (-1800, (-300, 200), 310+)
        
        | `Steps` denotes the number of samples, or the total training step, or the running times of `env.step()`.
        | `avgR` denotes average value of cumulative rewards, which is the sum of rewards in an episode.
        | `stdR` denotes standard dev of cumulative rewards, which is the sum of rewards in an episode.
        | `avgS` denotes the average number of steps in an episode.
        
        | Steps 8.04e+03  | avgR  -299.022  stdR    88.357  | avgS    125
        | Steps 5.62e+04  | avgR  -214.367  stdR    47.372  | avgS    161
        | Steps 9.65e+04  | avgR  -141.571  stdR    81.062  | avgS    260
        | Steps 1.22e+05  | avgR   -59.750  stdR   120.134  | avgS    343
        | Steps 1.48e+05  | avgR     6.748  stdR   159.249  | avgS    489
        | Steps 1.73e+05  | avgR   118.458  stdR   173.501  | avgS    373
        | Steps 1.99e+05  | avgR   113.434  stdR   143.486  | avgS    424
        | Steps 2.24e+05  | avgR   197.959  stdR   140.750  | avgS    309
        | Steps 2.56e+05  | avgR   230.408  stdR    62.407  | avgS    287
        | Steps 2.88e+05  | avgR   217.490  stdR    80.702  | avgS    318
        | Steps 3.21e+05  | avgR   211.975  stdR   121.265  | avgS    294
        | Steps 3.53e+05  | avgR   237.058  stdR    89.797  | avgS    276
        | Steps 3.87e+05  | avgR   239.979  stdR    59.422  | avgS    269
        | Steps 4.20e+05  | avgR   170.393  stdR   109.753  | avgS    317
        | Steps 4.53e+05  | avgR   255.573  stdR    53.690  | avgS    248
        | Steps 4.85e+05  | avgR   251.967  stdR    51.781  | avgS    312
        | Steps 5.18e+05  | avgR   220.062  stdR    65.878  | avgS    279
        | Steps 5.60e+05  | avgR   261.734  stdR    43.785  | avgS    255
        | Steps 5.94e+05  | avgR   230.703  stdR    87.323  | avgS    290
        | Save learning curve in ./LunarLanderContinuous-v2_PPO_4/LearningCurve_LunarLanderContinuous-v2_AgentPPO.jpg
        """

    elif env_name == "BipedalWalker-v3":
        import gym
        env = gym.make(env_name)
        env_func = gym.make
        env_args = get_gym_env_args(env, if_print=True)

        args = Arguments(agent_class, env_func, env_args)

        '''reward shaping'''
        args.reward_scale = 2 ** -1
        args.gamma = 0.98

        '''network update'''
        args.target_step = args.max_step
        args.net_dim = 2 ** 8
        args.num_layer = 3
        args.batch_size = 2 ** 8
        args.repeat_times = 2 ** 4

        '''evaluate'''
        args.eval_gap = 2 ** 6
        args.eval_times = 2 ** 4
        args.break_step = int(1e6)

        args.learner_gpus = gpu_id
        args.random_seed += gpu_id
        train_agent(args)
        evaluate_agent(args)
        print('| The cumulative returns of BipedalWalker-v3 is ∈ (-150, (-100, 280), 320+)')

        """
        | Arguments Remove cwd: ./BipedalWalker-v3_PPO_3
        
        | `Steps` denotes the number of samples, or the total training step, or the running times of `env.step()`.
        | `ExpR` denotes average rewards during exploration. The agent gets this rewards with noisy action.
        | `ObjC` denotes the objective of Critic network. Or call it loss function of critic network.
        | `ObjA` denotes the objective of Actor network. It is the average Q value of the critic network.
        
        | Steps 1.76e+03  ExpR    -0.08  | ObjC     1.91  ObjA     0.07
        | Steps 8.37e+04  ExpR    -0.02  | ObjC     0.01  ObjA    -0.02
        | Steps 1.64e+05  ExpR    -0.01  | ObjC     0.01  ObjA     0.03
        | Steps 2.45e+05  ExpR    -0.01  | ObjC     0.87  ObjA     0.05
        | Steps 3.29e+05  ExpR    -0.06  | ObjC     2.20  ObjA     0.06
        | Steps 4.12e+05  ExpR     0.04  | ObjC     0.26  ObjA     0.01
        | Steps 4.94e+05  ExpR    -0.03  | ObjC     1.61  ObjA     0.03
        | Steps 5.77e+05  ExpR     0.04  | ObjC     0.61  ObjA     0.00
        | Steps 6.62e+05  ExpR     0.02  | ObjC     0.78  ObjA     0.10
        | Steps 7.46e+05  ExpR     0.07  | ObjC     0.09  ObjA     0.07
        | Steps 8.29e+05  ExpR     0.02  | ObjC     0.58  ObjA     0.04
        | Steps 9.13e+05  ExpR     0.02  | ObjC     0.36  ObjA     0.10
        | Steps 9.97e+05  ExpR     0.05  | ObjC     0.48  ObjA     0.11
        | Steps 1.08e+06  ExpR     0.10  | ObjC     0.19  ObjA    -0.04
        | Steps 1.16e+06  ExpR     0.07  | ObjC     0.61  ObjA     0.06
        | Steps 1.25e+06  ExpR     0.12  | ObjC     0.18  ObjA     0.11
        | Steps 1.33e+06  ExpR     0.11  | ObjC     0.24  ObjA    -0.00
        | Steps 1.42e+06  ExpR     0.08  | ObjC     0.29  ObjA     0.07
        | Steps 1.50e+06  ExpR     0.09  | ObjC     0.38  ObjA    -0.01
        | Steps 1.58e+06  ExpR     0.09  | ObjC     0.79  ObjA    -0.01
        | Steps 1.66e+06  ExpR     0.10  | ObjC     0.60  ObjA     0.05
        | Steps 1.75e+06  ExpR     0.12  | ObjC     0.21  ObjA     0.02
        | Steps 1.83e+06  ExpR     0.09  | ObjC     0.71  ObjA    -0.05
        | Steps 1.91e+06  ExpR     0.10  | ObjC     0.82  ObjA     0.06
        | Steps 2.00e+06  ExpR     0.13  | ObjC     0.28  ObjA     0.08
        | UsedTime: 1562 | SavedDir: ./BipedalWalker-v3_PPO_3
        | The cumulative returns of BipedalWalker-v3 is ∈ (-150, (-100, 280), 320+)
        | Arguments Keep cwd: ./BipedalWalker-v3_PPO_3
        
        | `Steps` denotes the number of samples, or the total training step, or the running times of `env.step()`.
        | `avgR` denotes average value of cumulative rewards, which is the sum of rewards in an episode.
        | `stdR` denotes standard dev of cumulative rewards, which is the sum of rewards in an episode.
        | `avgS` denotes the average number of steps in an episode.
        
        | Steps 1.76e+03  | avgR  -115.003  stdR     0.116  | avgS     77
        | Steps 8.37e+04  | avgR   -35.847  stdR    22.081  | avgS   1507
        | Steps 1.64e+05  | avgR   -18.735  stdR     9.770  | avgS   1600
        | Steps 2.45e+05  | avgR    67.252  stdR    93.453  | avgS   1241
        | Steps 3.29e+05  | avgR   -29.998  stdR    86.421  | avgS    526
        | Steps 4.12e+05  | avgR   119.324  stdR   143.967  | avgS   1056
        | Steps 4.94e+05  | avgR   270.148  stdR    48.794  | avgS   1541
        | Steps 5.77e+05  | avgR   156.052  stdR   134.798  | avgS   1151
        | Steps 6.62e+05  | avgR   198.212  stdR   117.652  | avgS   1397
        | Steps 7.46e+05  | avgR   -40.360  stdR   120.715  | avgS    311
        | Steps 8.29e+05  | avgR   -27.157  stdR   100.747  | avgS    456
        | Steps 9.13e+05  | avgR   214.932  stdR    97.118  | avgS   1360
        | Steps 9.97e+05  | avgR    43.986  stdR   134.294  | avgS    671
        | Steps 1.08e+06  | avgR   290.690  stdR     1.815  | avgS   1277
        | Steps 1.16e+06  | avgR   263.563  stdR    84.633  | avgS   1126
        | Steps 1.25e+06  | avgR   286.703  stdR    35.514  | avgS   1098
        | Steps 1.33e+06  | avgR   296.509  stdR     1.452  | avgS   1154
        | Steps 1.42e+06  | avgR   270.809  stdR    69.728  | avgS   1075
        | Steps 1.50e+06  | avgR   299.834  stdR     1.090  | avgS   1105
        | Steps 1.58e+06  | avgR   277.359  stdR    83.360  | avgS   1054
        | Steps 1.66e+06  | avgR   290.936  stdR    27.699  | avgS   1131
        | Steps 1.75e+06  | avgR   298.836  stdR     2.494  | avgS   1122
        | Steps 1.83e+06  | avgR   298.921  stdR     1.822  | avgS   1075
        | Steps 1.91e+06  | avgR   285.068  stdR    54.862  | avgS   1028
        | Steps 2.00e+06  | avgR   283.990  stdR    60.790  | avgS   1023
        | Save learning curve in ./BipedalWalker-v3_PPO_3/LearningCurve_BipedalWalker-v3_AgentPPO.jpg
        """


if __name__ == "__main__":
    train_dqn_in_cartpole()
    train_dqn_in_lunar_lander()

    train_ddpg_in_pendulum()
    train_ppo_in_pendulum()

    train_ddpg_in_lunar_lander_or_bipedal_walker()
    train_ppo_in_lunar_lander_or_bipedal_walker()
