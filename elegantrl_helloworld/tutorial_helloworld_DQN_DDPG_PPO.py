from elegantrl_helloworld.run import train_agent, evaluate_agent, Arguments
from elegantrl_helloworld.env import get_gym_env_args, PendulumEnv


def train_dqn(gpu_id=0):  # DQN is a simple but low sample efficiency.
    from elegantrl_helloworld.agent import AgentDQN
    agent_class = AgentDQN
    env_name = ["CartPole-v0", "LunarLander-v2"][0]

    if env_name == "CartPole-v0":
        """
        | Arguments Remove cwd: ./CartPole-v0_DQN_6
        | Step 4.51e+02  ExpR     1.00  | ObjC     0.26  ObjA     0.29
        | Step 4.14e+04  ExpR     1.00  | ObjC     0.25  ObjA    31.08
        | Step 6.81e+04  ExpR     1.00  | ObjC     0.71  ObjA    33.63
        | Step 8.92e+04  ExpR     1.00  | ObjC     0.39  ObjA    33.55
        | UsedTime: 234 | SavedDir: ./CartPole-v0_DQN_6

        | Arguments Keep cwd: ./CartPole-v0_DQN_6
        | Steps 4.51e+02  | Returns avg    89.125  std    13.364
        | Steps 4.14e+04  | Returns avg   194.625  std    11.124
        | Steps 6.81e+04  | Returns avg   199.125  std     2.315
        | Steps 8.92e+04  | Returns avg   200.000  std     0.000
        """
        import gym
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
        args.batch_size = 2 ** 7
        args.repeat_times = 2 ** 0
        args.explore_rate = 0.25

        '''evaluate'''
        args.eval_gap = 2 ** 5
        args.eval_times = 2 ** 3
        args.break_step = int(1e5)
    elif env_name == "LunarLander-v2":
        """
        | Arguments Remove cwd: ./LunarLander-v2_DQN_6
        | Step 1.02e+03  ExpR    -1.78  | ObjC     1.28  ObjA     0.41
        | Step 4.89e+04  ExpR    -0.26  | ObjC     0.72  ObjA    31.29
        | Step 8.30e+04  ExpR    -0.02  | ObjC     0.54  ObjA    33.75
        | Step 1.16e+05  ExpR    -0.05  | ObjC     0.73  ObjA    38.20
        | Step 1.46e+05  ExpR    -0.00  | ObjC     3.99  ObjA    30.10
        | Step 1.74e+05  ExpR     0.27  | ObjC     0.96  ObjA    35.75
        | Step 1.99e+05  ExpR     0.00  | ObjC     0.92  ObjA    34.67
        | Step 2.24e+05  ExpR     0.13  | ObjC     0.36  ObjA    34.80
        | Step 2.50e+05  ExpR     0.24  | ObjC     0.52  ObjA    35.56
        | Step 2.78e+05  ExpR     0.19  | ObjC     1.58  ObjA    35.60
        | Step 3.01e+05  ExpR     0.10  | ObjC     1.14  ObjA    35.82
        | Step 3.27e+05  ExpR     0.11  | ObjC     0.46  ObjA    46.97
        | Step 3.52e+05  ExpR     0.22  | ObjC     0.78  ObjA    43.40
        | Step 3.76e+05  ExpR     0.11  | ObjC     0.52  ObjA    31.03
        | Step 3.97e+05  ExpR     0.22  | ObjC     0.69  ObjA    32.60
        | UsedTime: 1852 | SavedDir: ./LunarLander-v2_DQN_6

        | Arguments Keep cwd: ./LunarLander-v2_DQN_6
        | Steps 1.02e+03  | Returns avg  -658.350  std   270.733
        | Steps 4.89e+04  | Returns avg  -251.114  std    68.343
        | Steps 8.30e+04  | Returns avg   -22.167  std    24.154
        | Steps 1.16e+05  | Returns avg   -18.969  std    26.477
        | Steps 1.46e+05  | Returns avg   -14.664  std    21.717
        | Steps 1.74e+05  | Returns avg    38.013  std    78.876
        | Steps 1.99e+05  | Returns avg    -9.254  std    14.049
        | Steps 2.24e+05  | Returns avg    10.651  std    34.788
        | Steps 2.50e+05  | Returns avg   211.407  std    28.154
        | Steps 2.78e+05  | Returns avg   184.687  std    99.236
        | Steps 3.01e+05  | Returns avg   200.371  std    80.144
        | Steps 3.27e+05  | Returns avg   235.982  std    43.354
        | Steps 3.52e+05  | Returns avg   249.160  std    18.904
        | Steps 3.76e+05  | Returns avg   206.413  std    68.381
        | Steps 3.97e+05  | Returns avg   213.074  std    28.583
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
        args.target_step = args.max_step
        args.net_dim = 2 ** 7
        args.batch_size = 2 ** 7
        args.repeat_times = 2 ** 0
        args.explore_noise = 0.125

        '''evaluate'''
        args.eval_gap = 2 ** 7
        args.eval_times = 2 ** 4
        args.break_step = int(4e5)
    else:
        raise ValueError("env_name:", env_name)

    args.learner_gpus = gpu_id
    train_agent(args)
    evaluate_agent(args)


def train_ddpg(gpu_id=0):  # DDPG is a simple but low sample efficiency and unstable.
    from elegantrl_helloworld.agent import AgentDDPG
    agent_class = AgentDDPG
    env_name = ["Pendulum-v1", "LunarLanderContinuous-v2", "BipedalWalker-v3"][0]

    if env_name == "Pendulum-v1":
        """
        | Arguments Remove cwd: ./Pendulum-v1_DDPG_6
        |Step 4.00e+02  ExpR    -3.60  |ObjC     3.22  ObjA     0.15
        |Step 4.00e+04  ExpR    -1.98  |ObjC     0.86  ObjA   -82.97
        |Step 5.88e+04  ExpR    -1.58  |ObjC     0.82  ObjA   -66.20
        |Step 7.32e+04  ExpR    -0.60  |ObjC     0.62  ObjA   -45.66
        |Step 8.52e+04  ExpR    -0.47  |ObjC     0.36  ObjA   -33.60
        |Step 9.56e+04  ExpR    -0.33  |ObjC     0.35  ObjA   -28.88
        | UsedTime: 357 | SavedDir: ./Pendulum-v1_DDPG_6

        | Arguments Keep cwd: ./Pendulum-v1_DDPG_6
        |Steps          400  |Returns avg -1391.019  std   272.423
        |Steps        40000  |Returns avg  -822.530  std    77.746
        |Steps        58800  |Returns avg  -583.974  std    54.622
        |Steps        73200  |Returns avg  -199.278  std    83.178
        |Steps        85200  |Returns avg  -163.388  std    82.727
        |Steps        95600  |Returns avg  -211.675  std    72.861
        """
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
    elif env_name == "LunarLanderContinuous-v2":
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
        args.gamma = 0.98

        '''network update'''
        args.target_step = args.max_step // 2
        args.net_dim = 2 ** 8
        args.batch_size = 2 ** 8
        args.repeat_times = 2 ** 0
        args.explore_noise = 0.05

        '''evaluate'''
        args.eval_gap = 2 ** 7
        args.eval_times = 2 ** 3
        args.break_step = int(3e5)
    else:
        raise ValueError("env_name:", env_name)

    args.learner_gpus = gpu_id
    train_agent(args)
    evaluate_agent(args)


def train_ppo(gpu_id=0):
    from elegantrl_helloworld.agent import AgentPPO
    agent_class = AgentPPO
    env_name = ["Pendulum-v1", "LunarLanderContinuous-v2", "BipedalWalker-v3"][0]

    if env_name == "Pendulum-v1":
        """
        | Arguments Remove cwd: ./Pendulum-v1_PPO_6
        | Step 1.60e+03  ExpR    -2.92  | ObjC    83.17  ObjA    -0.05
        | Step 1.09e+05  ExpR    -3.04  | ObjC    25.93  ObjA     0.08
        | Step 1.86e+05  ExpR    -2.59  | ObjC    10.67  ObjA     0.01
        | Step 2.64e+05  ExpR    -2.31  | ObjC    11.20  ObjA     0.03
        | Step 3.89e+05  ExpR    -1.71  | ObjC     9.37  ObjA     0.11
        | Step 5.14e+05  ExpR    -0.74  | ObjC     3.89  ObjA     0.03
        | Step 6.35e+05  ExpR    -0.38  | ObjC     2.15  ObjA     0.06
        | Step 7.30e+05  ExpR    -1.89  | ObjC     9.54  ObjA    -0.09
        | Step 8.02e+05  ExpR    -0.52  | ObjC     2.75  ObjA     0.04
        | UsedTime: 517 | SavedDir: ./Pendulum-v1_PPO_6

        | Arguments Keep cwd: ./Pendulum-v1_PPO_6
        | Steps         1600  | Returns avg -1204.508  std   371.570
        | Steps       108800  | Returns avg -1199.940  std   197.476
        | Steps       185600  | Returns avg  -777.957  std    69.074
        | Steps       264000  | Returns avg  -686.187  std   106.359
        | Steps       388800  | Returns avg  -531.859  std    88.013
        | Steps       513600  | Returns avg  -273.945  std   212.517
        | Steps       635200  | Returns avg  -501.901  std   447.703
        | Steps       729600  | Returns avg  -441.479  std   386.449
        | Steps       801600  | Returns avg  -157.741  std    55.348
        """
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
        args.batch_size = 2 ** 8
        args.repeat_times = 2 ** 4

        '''evaluate'''
        args.eval_gap = 2 ** 6
        args.eval_times = 2 ** 3
        args.break_step = int(8e5)
    elif env_name == "LunarLanderContinuous-v2":
        import gym
        env = gym.make(env_name)
        env_func = gym.make
        env_args = get_gym_env_args(env, if_print=True)

        args = Arguments(agent_class, env_func, env_args)

        '''reward shaping'''
        args.reward_scale = 2 ** -2
        args.gamma = 0.99

        '''network update'''
        args.target_step = args.max_step * 2
        args.net_dim = 2 ** 7
        args.batch_size = 2 ** 8
        args.repeat_times = 2 ** 5

        '''evaluate'''
        args.eval_gap = 2 ** 6
        args.eval_times = 2 ** 5
        args.break_step = int(6e5)
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
        args.batch_size = 2 ** 9
        args.repeat_times = 2 ** 4

        '''evaluate'''
        args.eval_gap = 2 ** 6
        args.eval_times = 2 ** 4
        args.break_step = int(6e5)
    else:
        raise ValueError("env_name:", env_name)

    args.learner_gpus = gpu_id
    train_agent(args)
    evaluate_agent(args)


if __name__ == "__main__":
    train_dqn()
    # train_ddpg()
    # train_ppo()
