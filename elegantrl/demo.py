import sys

from elegantrl.run import *

'''custom env'''


class PendulumEnv(gym.Wrapper):  # [ElegantRL.2021.11.11]
    def __init__(self, gym_env_id='Pendulum-v1', target_return=-200):
        # Pendulum-v0 gym.__version__ == 0.17.0
        # Pendulum-v1 gym.__version__ == 0.21.0
        gym.logger.set_level(40)  # Block warning
        super(PendulumEnv, self).__init__(env=gym.make(gym_env_id))

        # from elegantrl.envs.Gym import get_gym_env_info
        # get_gym_env_info(env, if_print=True)  # use this function to print the env information
        self.env_num = 1  # the env number of VectorEnv is greater than 1
        self.env_name = gym_env_id  # the name of this env.
        self.max_step = 200  # the max step of each episode
        self.state_dim = 3  # feature number of state
        self.action_dim = 1  # feature number of action
        self.if_discrete = False  # discrete action or continuous action
        self.target_return = target_return  # episode return is between (-1600, 0)

    def reset(self):
        return self.env.reset().astype(np.float32)

    def step(self, action: np.ndarray):
        # PendulumEnv set its action space as (-2, +2). It is bad.  # https://github.com/openai/gym/wiki/Pendulum-v0
        # I suggest to set action space as (-1, +1) when you design your own env.
        state, reward, done, info_dict = self.env.step(action * 2)  # state, reward, done, info_dict
        return state.astype(np.float32), reward, done, info_dict


'''demo'''


def demo_continuous_action_off_policy_redq():
    env_name = ['Pendulum-v0',
                'Pendulum-v1',
                'LunarLanderContinuous-v2',
                'BipedalWalker-v3', ][ENV_ID]
    agent = [AgentREDqSAC, AgentSyncREDqSAC][-1]

    gpu_id = GPU_ID  # >=0 means GPU ID, -1 means CPU

    if env_name in {'Pendulum-v0', 'Pendulum-v1'}:
        """
        ID     Step    maxR |    avgR   stdR   avgS  stdS |    expR   objC   etc.
        1  1.60e+03 -979.95 | -979.95  165.5    200     0 |   -2.01   1.56   0.57   1.00
        1  5.44e+04 -827.21 | -900.53  214.7    200     0 |   -2.28   0.50 -50.04   0.81
        1  6.72e+04 -242.52 | -242.52  155.3    200     0 |   -0.79   1.10 -59.92   0.72
        1  7.76e+04 -238.89 | -238.89  107.1    200     0 |   -0.73   1.38 -66.98   0.66
        1  8.72e+04 -238.53 | -238.53   98.6    200     0 |   -0.83   1.22 -69.88   0.63
        1  9.60e+04 -149.37 | -149.37  136.2    200     0 |   -0.31   1.81 -69.55   0.64
        | UsedTime:     798 |

        ID     Step    maxR |    avgR   stdR   avgS  stdS |    expR   objC   etc.
        3  1.60e+03-1199.01 |-1199.01  252.2    200     0 |   -2.83   2.03   0.59   1.00
        3  4.96e+04-1024.38 |-1024.38  172.7    200     0 |   -2.32   0.39 -45.36   0.84
        3  6.16e+04 -327.84 | -327.84  127.8    200     0 |   -1.61   0.63 -58.63   0.76
        3  7.20e+04 -198.66 | -198.66   82.0    200     0 |   -0.86   0.92 -65.27   0.70
        3  8.88e+04 -156.18 | -169.78   86.5    200     0 |   -0.45   1.09 -70.35   0.62
        3  1.03e+05 -139.14 | -139.14   74.5    200     0 |   -0.46   1.29 -75.14   0.65
        | UsedTime:    1070 |
        
        ID     Step    maxR |    avgR   stdR   avgS  stdS |    expR   objC   etc.
        3  1.60e+03-1391.83 |-1391.83  241.9    200     0 |   -3.19   2.26   0.50   1.00
        3  5.04e+04-1045.01 |-1104.32   85.4    200     0 |   -2.66   0.90 -42.83   0.83
        3  6.16e+04 -218.75 | -218.75  151.5    200     0 |   -1.63   0.59 -53.07   0.76
        3  7.12e+04 -201.54 | -201.54  134.3    200     0 |   -0.88   0.83 -59.02   0.70
        3  8.80e+04 -182.13 | -193.92  106.1    200     0 |   -0.60   1.84 -63.65   0.65
        3  9.52e+04 -167.25 | -167.25   84.2    200     0 |   -0.31   0.86 -66.83   0.68
        3  1.02e+05 -139.24 | -139.24   73.5    200     0 |   -0.83   1.29 -61.21   0.72
        | UsedTime:    1068 |
        """

        env = PendulumEnv(env_name, target_return=-150)
        "TotalStep: 1e5, TargetReward: -200, UsedTime: 600s"
        args = Arguments(agent, env)
        args.reward_scale = 2 ** -1  # RewardRange: -1800 < -200 < -50 < 0
        args.gamma = 0.97
        args.target_step = args.max_step * 2
        args.eval_times = 2 ** 3
    elif env_name == 'LunarLanderContinuous-v2':
        """
        
        | Arguments Remove cwd: ./LunarLanderContinuous-v2_SyncREDqSAC_0
        ################################################################################
        ID     Step    maxR |    avgR   stdR   avgS  stdS |    expR   objC   etc.
        0  4.22e+03 -254.23 | -254.23  120.0    134    54 |   -0.57   0.78   0.10   0.15
        0  6.40e+04 -100.91 | -100.91   54.9    976    56 |    0.00   0.24   0.50   0.16
        0  8.87e+04  -55.83 |  -55.83   87.8    813   280 |    0.00   0.30   2.94   0.18
        0  1.10e+05  -43.72 |  -43.72   96.1    478   313 |    0.01   0.36   4.41   0.19
        0  1.26e+05  112.65 |  112.65  125.1    598   274 |   -0.00   0.37   4.63   0.19
        0  1.59e+05  147.64 |  101.66   97.5    826   143 |    0.02   0.37   7.11   0.18
        0  2.29e+05  166.62 |  150.04   75.7    707   185 |    0.02   0.54  14.27   0.19
        0  2.39e+05  187.70 |  187.70   76.0    591   175 |    0.01   0.59  12.77   0.19
        0  2.84e+05  195.30 |  166.09   89.9    599   247 |    0.03   0.68  14.24   0.17
        0  2.92e+05  195.30 |  166.12   82.5    586   252 |    0.03   0.55  12.08   0.17
        0  3.00e+05  217.07 |  217.07   30.2    448   125 |    0.01   0.59  14.13   0.17
        | UsedTime:    2852 | 
        """

        # env = gym.make('LunarLanderContinuous-v2')
        # get_gym_env_args(env=env, if_print=True)
        env_func = gym.make
        env_args = {'env_num': 1,
                    'env_name': 'LunarLanderContinuous-v2',
                    'max_step': 1000,
                    'state_dim': 8,
                    'action_dim': 2,
                    'if_discrete': False,
                    'target_return': 200,

                    'id': 'LunarLanderContinuous-v2'}
        args = Arguments(agent, env_func=env_func, env_args=env_args)

        args.target_step = args.max_step
        args.gamma = 0.99
        args.reward_scale = 2 ** -2
        args.eval_times = 2 ** 5
    elif env_name == 'BipedalWalker-v3':
        """
        ID     Step    maxR |    avgR   stdR   avgS  stdS |    expR   objC   etc.
        0  7.28e+03 -111.85 | -111.85    0.2    109     3 |   -0.26   0.69   0.04   0.02
        0  1.24e+05  -96.80 | -103.65    0.2     50     0 |   -0.42   0.19  -0.20   0.02
        0  1.75e+05  -84.13 |  -84.13   15.8    236   352 |   -1.18   0.20  -2.09   0.02
        0  1.96e+05  -53.51 |  -53.51   10.9   1600     0 |   -0.03   0.16  -4.75   0.03
        0  2.30e+05  -34.98 |  -37.49    3.0   1600     0 |   -0.02   0.24  -5.33   0.03
        0  2.89e+05  -26.53 |  -26.53   12.0   1600     0 |   -0.03   0.17  -4.17   0.05
        0  3.03e+05  -24.50 |  -24.50   18.1   1600     0 |   -0.02   0.16  -1.19   0.05
        0  3.18e+05    3.44 |    3.44   53.4   1299   466 |    0.00   0.16  -1.71   0.04
        0  3.30e+05   29.50 |   29.50   75.0   1263   515 |    0.03   0.16  -1.08   0.04
        0  3.40e+05  107.37 |  107.37   86.7   1365   412 |    0.05   0.16   0.24   0.04
        0  3.59e+05  206.41 |  137.65   99.3   1540   233 |    0.08   0.14   0.62   0.03
        0  3.89e+05  269.83 |  269.83   42.6   1581    74 |    0.16   0.10  -0.25   0.04
        0  4.01e+05  281.13 |  281.13   76.3   1503   246 |    0.12   0.09  -0.27   0.04
        0  4.12e+05  301.77 |  301.77    1.0   1467    28 |    0.17   0.09   0.28   0.04
        | UsedTime:    3294 | 
        
        
        ID     Step    maxR |    avgR   stdR   avgS  stdS |    expR   objC   etc.
        0  6.68e+03 -111.72 | -111.72    0.1    103     3 |   -0.10   0.69   0.05   0.02
        0  1.06e+05  -93.81 |  -93.81    0.1    116     7 |   -0.16   0.18  -0.09   0.02
        0  1.31e+05  -58.96 |  -58.96   25.0   1125   704 |   -0.11   0.13  -1.04   0.02
        0  1.74e+05  -32.93 |  -91.37    0.4    113     2 |   -0.15   0.11  -1.13   0.02
        0  4.20e+05  -32.93 |  -33.38   15.8   1600     0 |   -0.01   0.04  -0.37   0.02
        0  4.39e+05   16.15 |   16.15   26.3   1600     0 |    0.01   0.04  -1.40   0.02
        0  4.52e+05   40.90 |   10.57   42.8   1600     0 |   -0.02   0.04  -1.03   0.02
        0  4.84e+05   63.49 |   63.49   40.0   1600     0 |    0.02   0.04  -0.72   0.02
        0  4.90e+05   63.49 |   53.72   56.5   1490   294 |    0.03   0.04  -0.55   0.02
        0  5.04e+05   71.35 |   43.56   50.1   1600     0 |    0.03   0.04  -1.07   0.02
        0  5.23e+05   93.06 |   93.06   90.7   1535   253 |    0.03   0.03   0.21   0.02
        0  5.36e+05  173.26 |  121.82  123.4   1279   557 |    0.10   0.03  -0.62   0.02
        0  5.49e+05  253.63 |  253.63   43.9   1582    68 |    0.08   0.03   0.70   0.02
        0  5.63e+05  266.36 |  163.69  154.3   1030   476 |    0.17   0.04   0.68   0.03
        0  5.85e+05  302.20 |  302.20    0.9   1318    18 |    0.17   0.04   1.30   0.03
        | UsedTime:    8100 |


        """
        env_func = gym.make
        env_args = {'env_num': 1,
                    'env_name': 'BipedalWalker-v3',
                    'max_step': 1600,
                    'state_dim': 24,
                    'action_dim': 4,
                    'if_discrete': False,
                    'target_return': 300,

                    'id': 'BipedalWalker-v3', }
        args = Arguments(agent, env_func=env_func, env_args=env_args)
        args.target_step = args.max_step
        args.gamma = 0.98
        args.eval_times = 2 ** 4
    else:
        raise ValueError('env_name:', env_name)

    args.learner_gpus = gpu_id
    args.random_seed += gpu_id

    if_check = 0
    if if_check:
        train_and_evaluate(args)
    else:
        train_and_evaluate_mp(args)


def demo_continuous_action_off_policy():
    env_name = ['Pendulum-v0',
                'Pendulum-v1',
                'LunarLanderContinuous-v2',
                'BipedalWalker-v3',
                ''][ENV_ID]
    gpu_id = GPU_ID  # >=0 means GPU ID, -1 means CPU

    if env_name in {'Pendulum-v0', 'Pendulum-v1'}:
        env = PendulumEnv(env_name, target_return=-500)
        """
        ID     Step    maxR |    avgR   stdR   avgS  stdS |    expR   objC   etc.
        2  1.60e+03-1147.49 |-1147.49  179.2    200     0 |   -2.61   0.90   0.55   1.00
        2  5.84e+04 -121.61 | -121.61   59.0    200     0 |   -0.81   0.33 -40.64   0.79
        | UsedTime:     132 |
        
        ID     Step    maxR |    avgR   stdR   avgS  stdS |    expR   objC   etc.
        1  1.60e+03-1267.96 |-1267.96  329.7    200     0 |   -2.67   0.88   0.56   1.00
        1  8.48e+04 -171.79 | -182.24   63.3    200     0 |   -0.30   0.32 -30.75   0.64
        1  1.19e+05 -171.79 | -178.25  116.8    200     0 |   -0.31   0.16 -22.52   0.43
        1  1.34e+05 -164.56 | -164.56   99.1    200     0 |   -0.31   0.15 -18.09   0.35
        1  1.47e+05 -135.20 | -135.20   92.1    200     0 |   -0.31   0.14 -15.65   0.29
        | UsedTime:     783 | 
        """
        args = Arguments(AgentModSAC, env)
        args.reward_scale = 2 ** -1  # RewardRange: -1800 < -200 < -50 < 0
        args.gamma = 0.97
        args.target_step = args.max_step * 2
        args.eval_times = 2 ** 3
    elif env_name == 'LunarLanderContinuous-v2':
        """
        ID     Step    maxR |    avgR   stdR   avgS  stdS |    expR   objC   etc.
        2  4.25e+03 -143.93 | -143.93   29.6     69    12 |   -2.47   1.06   0.13   0.15
        2  1.05e+05  170.35 |  170.35   57.9    645   177 |    0.06   1.59  15.93   0.20
        2  1.59e+05  170.35 |   80.46  125.0    775   285 |    0.07   1.14  29.92   0.29
        2  1.95e+05  221.39 |  221.39   19.7    449   127 |    0.12   1.09  32.16   0.40
        | UsedTime:     421 |
        
        ID     Step    maxR |    avgR   stdR   avgS  stdS |    expR   objC   etc.
        1  4.26e+03 -139.77 | -139.77   36.7     67    12 |   -2.16  11.20   0.12   0.15
        1  1.11e+05 -105.09 | -105.09   84.3    821   244 |   -0.14  27.60   1.04   0.21
        1  2.03e+05  -15.21 |  -15.21   22.7   1000     0 |   -0.01  17.96  36.95   0.45
        1  3.87e+05   59.39 |   54.09  160.7    756   223 |    0.00  16.57  88.99   0.73
        1  4.03e+05   59.39 |   56.16  103.5    908   120 |    0.06  16.47  84.27   0.71
        1  5.10e+05  186.59 |  186.59  103.6    547   257 |   -0.02  12.72  67.97   0.57
        1  5.89e+05  226.93 |  226.93   20.0    486   154 |    0.13   9.27  68.29   0.51
        | UsedTime:    3407 |

        ID     Step    maxR |    avgR   stdR   avgS  stdS |    expR   objC   etc.
        1  4.15e+03 -169.01 | -169.01   87.9    110    59 |   -2.18  11.86   0.10   0.15
        1  1.09e+05  -84.47 |  -84.47   80.1    465   293 |   -0.30  30.64  -6.29   0.20
        1  4.25e+05   -8.33 |   -8.33   48.4    994    26 |    0.07  13.51  76.99   0.62
        1  4.39e+05   87.29 |   87.29   86.9    892   141 |    0.04  12.76  70.37   0.61
        1  5.57e+05  159.17 |  159.17   65.7    721   159 |    0.10  10.31  59.90   0.51
        1  5.87e+05  190.09 |  190.09   71.7    577   175 |    0.09   9.45  61.74   0.48
        1  6.20e+05  206.74 |  206.74   29.1    497   108 |    0.09   9.21  62.06   0.47
        | UsedTime:    4433 |
        """
        # env = gym.make('LunarLanderContinuous-v2')
        # get_gym_env_args(env=env, if_print=True)
        env_func = gym.make
        env_args = {'env_num': 1,
                    'env_name': 'LunarLanderContinuous-v2',
                    'max_step': 1000,
                    'state_dim': 8,
                    'action_dim': 2,
                    'if_discrete': False,
                    'target_return': 200,

                    'id': 'LunarLanderContinuous-v2'}
        args = Arguments(AgentModSAC, env_func=env_func, env_args=env_args)

        args.target_step = args.max_step
        args.gamma = 0.99
        args.eval_times = 2 ** 5
    elif env_name == 'BipedalWalker-v3':
        """
        ID     Step    maxR |    avgR   stdR   avgS  stdS |    expR   objC   etc.
        3  7.51e+03 -111.59 | -111.59    0.2     97     7 |   -0.18   4.23  -0.03   0.02
        3  1.48e+05 -110.19 | -110.19    1.6     84    30 |   -0.59   2.46   3.18   0.03
        3  5.02e+05  -31.84 | -102.27   54.0   1359   335 |   -0.06   0.85   2.84   0.04
        3  1.00e+06   -7.94 |   -7.94   73.2    411   276 |   -0.17   0.72   1.96   0.03
        3  1.04e+06  131.50 |  131.50  168.3    990   627 |    0.06   0.46   1.69   0.04
        3  1.11e+06  214.12 |  214.12  146.6   1029   405 |    0.09   0.50   1.63   0.04
        3  1.20e+06  308.34 |  308.34    0.7   1106    20 |    0.29   0.72   4.56   0.05
        | UsedTime:    8611 |

        ID     Step    maxR |    avgR   stdR   avgS  stdS |    expR   objC   etc.
        3  6.75e+03  -92.44 |  -92.44    0.2    120     3 |   -0.18   1.94  -0.00   0.02
        3  3.95e+05  -37.16 |  -37.16    9.2   1600     0 |   -0.06   1.90   4.20   0.07
        3  6.79e+05  -23.32 |  -42.54   90.0   1197   599 |   -0.02   0.91   1.57   0.04
        3  6.93e+05   46.92 |   46.92   96.9    808   395 |   -0.04   0.57   1.34   0.04
        3  8.38e+05  118.86 |  118.86  154.5    999   538 |    0.14   1.44   0.75   0.05
        3  1.00e+06  225.56 |  225.56  124.1   1207   382 |    0.13   0.72   4.75   0.06
        3  1.02e+06  283.37 |  283.37   86.3   1259   245 |    0.14   0.80   3.96   0.06
        3  1.19e+06  313.36 |  313.36    0.9   1097    20 |    0.21   0.78   6.80   0.06
        | UsedTime:    9354 | SavedDir: ./BipedalWalker-v3_ModSAC_3
        
        ID     Step    maxR |    avgR   stdR   avgS  stdS |    expR   objC   etc.
        3  6.55e+03 -109.86 | -109.86    4.5    156    30 |   -0.06   0.71  -0.01   0.02
        3  1.24e+05  -88.28 |  -88.28   26.2    475   650 |   -0.15   0.15   0.04   0.02
        3  3.01e+05  -47.89 |  -56.76   21.7   1341   540 |   -0.03   0.19  -2.76   0.05
        3  3.82e+05   80.89 |   53.79  140.1    983   596 |   -0.01   0.18   0.46   0.05
        3  4.35e+05  137.70 |   28.54  104.7    936   581 |   -0.01   0.21   0.63   0.06
        3  4.80e+05  158.71 |   25.54  114.7    524   338 |    0.18   0.17   6.17   0.06
        3  5.31e+05  205.81 |  203.27  143.9   1048   388 |    0.14   0.15   4.00   0.06
        3  6.93e+05  254.40 |  252.74  121.1    992   280 |    0.21   0.12   7.34   0.06
        3  7.11e+05  304.79 |  304.79   73.4   1015   151 |    0.21   0.12   5.69   0.06
        | UsedTime:    3215 | 
        """
        env_func = gym.make
        env_args = {'env_num': 1,
                    'env_name': 'BipedalWalker-v3',
                    'max_step': 1600,
                    'state_dim': 24,
                    'action_dim': 4,
                    'if_discrete': False,
                    'target_return': 300,

                    'id': 'BipedalWalker-v3', }
        args = Arguments(AgentModSAC, env_func=env_func, env_args=env_args)
        args.target_step = args.max_step
        args.gamma = 0.98
        args.eval_times = 2 ** 4
    else:
        raise ValueError('env_name:', env_name)

    args.learner_gpus = gpu_id
    args.random_seed += gpu_id

    if_check = 0
    if if_check:
        train_and_evaluate(args)
    else:
        train_and_evaluate_mp(args)


def demo_continuous_action_on_policy():
    env_name = ['Pendulum-v0',
                'Pendulum-v1',
                'LunarLanderContinuous-v2',
                'BipedalWalker-v3'][ENV_ID]
    gpu_id = GPU_ID  # >=0 means GPU ID, -1 means CPU

    if env_name in {'Pendulum-v0', 'Pendulum-v1'}:
        env = PendulumEnv(env_name, target_return=-500)
        "TotalStep: 1e5, TargetReward: -200, UsedTime: 600s"
        args = Arguments(AgentPPO, env)
        args.reward_scale = 2 ** -1  # RewardRange: -1800 < -200 < -50 < 0
        args.gamma = 0.97
        args.target_step = args.max_step * 8
        args.eval_times = 2 ** 3
    elif env_name == 'LunarLanderContinuous-v2':
        """
        ID     Step    maxR |    avgR   stdR   avgS  stdS |    expR   objC   etc.
        2  8.40e+03 -167.99 | -167.99  119.9     96    13 |   -1.408795.41   0.02  -0.50
        2  1.27e+05 -167.99 | -185.92   44.3    187    77 |    0.07 396.60   0.02  -0.51
        2  2.27e+05  191.79 |  191.79   83.7    401    96 |    0.16  39.93   0.06  -0.52
        2  3.40e+05  220.93 |  220.93   87.7    375    99 |    0.19 121.32  -0.01  -0.53
        | UsedTime:     418 |

        ID     Step    maxR |    avgR   stdR   avgS  stdS |    expR   objC   etc.
        2  8.31e+03  -90.85 |  -90.85   49.2     72    12 |   -1.295778.93   0.01  -0.50
        2  1.16e+05  -90.85 | -126.58   92.2    312   271 |    0.03 215.40  -0.01  -0.50
        2  1.96e+05  133.57 |  133.57  156.4    380   108 |    0.04 227.81   0.04  -0.51
        2  3.85e+05  195.56 |  195.56   78.4    393    87 |    0.14  26.79  -0.05  -0.54
        2  4.97e+05  212.20 |  212.20   90.5    383    72 |    0.18 357.67  -0.01  -0.55
        | UsedTime:     681 |
        """
        # env = gym.make('LunarLanderContinuous-v2')
        # get_gym_env_args(env=env, if_print=True)
        env_func = gym.make
        env_args = {'env_num': 1,
                    'env_name': 'LunarLanderContinuous-v2',
                    'max_step': 1000,
                    'state_dim': 8,
                    'action_dim': 2,
                    'if_discrete': False,
                    'target_return': 200,

                    'id': 'LunarLanderContinuous-v2'}
        args = Arguments(AgentPPO, env_func=env_func, env_args=env_args)

        args.target_step = args.max_step * 2
        args.gamma = 0.99
        args.eval_times = 2 ** 5
    elif env_name == 'BipedalWalker-v3':
        """
        ID     Step    maxR |    avgR   stdR   avgS  stdS |    expR   objC   etc.
        0  2.72e+04  -38.64 |  -38.64   43.7   1236   630 |   -0.11  83.06  -0.03  -0.50
        0  4.32e+05  -30.57 |  -30.57    4.7   1600     0 |   -0.01   0.33  -0.06  -0.53
        0  6.38e+05  179.12 |  179.12    5.2   1600     0 |    0.06   4.16   0.01  -0.57
        0  1.06e+06  274.76 |  274.76    4.5   1600     0 |    0.12   1.11   0.03  -0.61
        0  2.11e+06  287.37 |  287.37   46.9   1308   104 |    0.17   5.40   0.03  -0.72
        0  2.33e+06  296.76 |  296.76   29.9   1191    30 |    0.20   2.86   0.00  -0.74
        0  2.54e+06  307.66 |  307.66    1.9   1163    34 |    0.19   5.40   0.02  -0.75
        | UsedTime:    1641 |

        ID     Step    maxR |    avgR   stdR   avgS  stdS |    expR   objC   etc.
        4  2.88e+04 -112.06 | -112.06    0.1    128     8 |   -0.12 120.04   0.03  -0.50
        4  4.41e+05  -36.22 |  -36.22    4.0   1600     0 |   -0.03   0.20  -0.01  -0.53
        4  6.58e+05  127.33 |  127.33    6.2   1600     0 |    0.03   0.35   0.04  -0.58
        4  8.76e+05  150.14 |  150.14    4.8   1600     0 |    0.07   0.32   0.07  -0.62
        4  1.10e+06  233.32 |  233.32    3.9   1600     0 |    0.10   0.74  -0.01  -0.66
        4  1.97e+06  269.85 |  269.85   11.1   1600     0 |    0.14   1.40   0.01  -0.77
        4  2.40e+06  293.55 |  293.55    1.7   1485    32 |    0.16   3.38  -0.00  -0.82
        4  3.31e+06  300.05 |  300.05    1.6   1290    29 |    0.20   2.43   0.04  -0.90
        | UsedTime:    2036 |

        1  1.81e+06  301.14 |  301.14    0.7   1067    11 |    0.11   3.73  -0.02  -0.68 | UsedTime:    1099 |
        0  2.54e+06  307.66 |  307.66    1.9   1163    34 |    0.19   5.40   0.02  -0.75 | UsedTime:    1641 |
        """
        env_func = gym.make
        env_args = {'env_num': 1,
                    'env_name': 'BipedalWalker-v3',
                    'max_step': 1600,
                    'state_dim': 24,
                    'action_dim': 4,
                    'if_discrete': False,
                    'target_return': 300,

                    'id': 'BipedalWalker-v3', }
        args = Arguments(AgentPPO, env_func=env_func, env_args=env_args)
        args.target_step = args.max_step * 4
        args.reward_scale = 2 ** -1
        args.gamma = 0.98
        args.eval_times = 2 ** 4
    else:
        raise ValueError('env_name:', env_name)

    args.learner_gpus = gpu_id
    args.random_seed += gpu_id

    if_check = 0
    if if_check:
        train_and_evaluate(args)
    else:
        train_and_evaluate_mp(args)


def demo_discrete_action_off_policy():
    env_name = ['CartPole-v0',
                'LunarLander-v2', ][ENV_ID]
    gpu_id = GPU_ID  # >=0 means GPU ID, -1 means CPU

    if env_name == 'CartPole-v0':
        """
        ID     Step    maxR |    avgR   stdR   avgS  stdS |    expR   objC   etc.
        1  2.15e+02    9.00 |    9.00    0.7      9     1 |    1.00   1.00   0.02
        1  1.89e+04  200.00 |  200.00    0.0    200     0 |    1.00   5.59  28.07
        | UsedTime: 17 |
        """
        # env = gym.make(env_name)
        # get_gym_env_args(env=env, if_print=True)
        env_func = gym.make
        env_args = {
            'env_num': 1,
            'env_name': 'CartPole-v0',
            'max_step': 200,
            'state_dim': 4,
            'action_dim': 2,
            'if_discrete': True,
            'target_return': 195.0,
        }
        args = Arguments(AgentDQN, env_func=env_func, env_args=env_args)

        args.target_step = args.max_step
        args.net_dim = 2 ** 7
        args.batch_size = args.net_dim

        args.gamma = 0.97
        args.eval_times = 2 ** 3
        args.eval_gap = 2 ** 4
    elif env_name == 'LunarLander-v2':
        """
        ID     Step    maxR |    avgR   stdR   avgS  stdS |    expR   objC   etc.
        1  4.17e+03 -571.17 | -571.17  134.8     68     9 |   -1.84  25.04  -0.18
        1  1.30e+05 -231.33 | -231.33   30.9    415   145 |   -0.15  15.28   0.45
        1  1.80e+05  -64.84 |  -64.84   23.1   1000     0 |   -0.02   2.75  13.27
        1  3.66e+05  -46.99 |  -50.78   25.6   1000     0 |   -0.01   4.91   6.55
        1  3.89e+05   37.80 |   37.80  103.2    804   295 |   -0.01   0.30  10.48
        1  5.20e+05   82.97 |   82.97  116.4    773   236 |   -0.01   0.16   8.98
        1  6.00e+05   83.73 |   20.15   44.8    990    39 |    0.03   2.15   5.51
        1  6.50e+05  193.18 |  138.69   46.5    880   224 |    0.05   0.35   6.63
        1  6.64e+05  236.45 |  236.45   26.6    396   137 |    0.10   0.38  10.10
        | UsedTime:    3149 |
        """
        env_func = gym.make
        env_args = {'env_num': 1,
                    'env_name': 'LunarLander-v2',
                    'max_step': 1000,
                    'state_dim': 8,
                    'action_dim': 4,
                    'if_discrete': True,
                    'target_return': 200, }
        args = Arguments(AgentD3QN, env_func=env_func, env_args=env_args)
        args.target_step = args.max_step
        args.reward_scale = 2 ** -2
        args.gamma = 0.99
        args.eval_times = 2 ** 4
    else:
        raise ValueError('env_name:', env_name)

    args.learner_gpus = gpu_id
    args.random_seed += gpu_id

    if_check = 0
    if if_check:
        train_and_evaluate(args)
    else:
        train_and_evaluate_mp(args)


def demo_discrete_action_on_policy():
    env_name = ['CartPole-v0',
                'LunarLander-v2',
                ][ENV_ID]
    gpu_id = GPU_ID  # >=0 means GPU ID, -1 means CPU

    if env_name == 'CartPole-v0':
        """
        ID     Step    maxR |    avgR   stdR   avgS  stdS |    expR   objC   etc.
        2  3.22e+03   77.00 |   77.00   23.4     77    23 |    1.00 142.04   0.01   0.00
        2  3.59e+04  200.00 |  200.00    0.0    200     0 |    1.00  38.12  -0.03   0.00
        | UsedTime: 19 | SavedDir: ./CartPole-v0_DiscretePPO_2
        """
        # env = gym.make(env_name)
        # get_gym_env_args(env=env, if_print=True)
        env_func = gym.make
        env_args = {'env_num': 1,
                    'env_name': 'CartPole-v0',
                    'max_step': 200,
                    'state_dim': 4,
                    'action_dim': 2,
                    'if_discrete': True,
                    'target_return': 195.0, }
        args = Arguments(AgentDiscretePPO, env_func=env_func, env_args=env_args)

        args.target_step = args.max_step * 2
        args.net_dim = 2 ** 6
        args.batch_size = args.net_dim * 2

        args.gamma = 0.97
        args.eval_times = 2 ** 3
        args.eval_gap = 2 ** 4
    elif env_name == 'LunarLander-v2':
        """ Hyper-parameter will influence training used time.

        args.repeat_times = 2 ** 5
        args.reward_scale = 2 ** 0 
        args.if_cri_target = False 
        ID     Step    maxR |    avgR   stdR   avgS  stdS |    expR   objC   etc.
        1  1.61e+04 -155.46 | -155.46   53.9    695   180 |   -1.56 828.09   0.01   0.00
        1  1.51e+05  -48.06 |  -48.06   23.7   1000     0 |    0.05 250.08  -0.04   0.00
        1  2.57e+05  176.18 |  176.18   47.6    749   229 |    0.16  64.25   0.02   0.00
        1  4.83e+05  198.96 |  173.47   45.7    669   263 |    0.21 266.75   0.01   0.00
        1  6.04e+05  198.96 |  170.54   69.2    585   291 |    0.24 310.07  -0.08   0.00
        1  7.20e+05  203.56 |  203.56   53.4    592   235 |    0.37 299.88   0.01   0.00
        | UsedTime:     817 |
        ID     Step    maxR |    avgR   stdR   avgS  stdS |    expR   objC   etc.
        2  1.69e+04 -162.68 | -162.68   54.1     66    12 |   -1.451490.59  -0.01   0.00
        2  1.53e+05 -162.68 | -504.17   53.1    536   120 |    0.43 403.69  -0.00   0.00
        2  2.87e+05  244.23 |  244.23   24.2    403   131 |    0.81  71.60  -0.02   0.00
        | UsedTime:     282 |

        args.repeat_times = 2 ** 4
        args.reward_scale = 2 ** 0 
        args.if_cri_target = True 
        ID     Step    maxR |    avgR   stdR   avgS  stdS |    expR   objC   etc.
        2  1.63e+04  143.77 |  143.77   74.0    801   204 |   -1.322105.17  -0.04   0.00
        2  1.62e+05  143.77 | -129.86  125.4    316   197 |    0.251364.32  -0.01   0.00
        2  3.46e+05  194.14 |  194.14  114.6    339    35 |    0.84 730.46  -0.01   0.00
        2  5.53e+05  252.13 |  252.13   26.9    299    41 |    1.07  43.28  -0.04   0.00
        | UsedTime:     409 |

        args.repeat_times = 2 ** 4
        args.reward_scale = 2 ** -2  # slow
        args.if_cri_target = False 
        ID     Step    maxR |    avgR   stdR   avgS  stdS |    expR   objC   etc.
        2  1.63e+04  -28.75 |  -28.75   39.2     82    12 |   -0.37  88.73  -0.03   0.00
        2  3.11e+05   17.93 |   17.93  104.1    879   114 |    0.04  30.11  -0.06   0.00
        2  1.00e+06   57.35 |  -22.35   52.9    988    46 |    0.00   2.17  -0.01   0.00
        2  4.08e+06   57.35 |  -26.53   93.2    970   118 |   -0.01   0.31  -0.06   0.00
        2  6.28e+06  107.97 |   58.17   76.0    906   197 |    0.02   4.73   0.01   0.00
        2  1.00e+07  160.05 |  133.85   64.0    870   154 |    0.05   9.15   0.02   0.00
        2  1.62e+07  176.17 |  176.17   39.0    735   105 |    0.03  10.39  -0.00   0.00
        2  1.97e+07  210.55 |  210.55   43.8    558   130 |    0.08   4.46   0.01   0.00
        | UsedTime:   22758 |
        """
        env_func = gym.make
        env_args = {'env_num': 1,
                    'env_name': 'LunarLander-v2',
                    'max_step': 1000,
                    'state_dim': 8,
                    'action_dim': 4,
                    'if_discrete': True,
                    'target_return': 200, }
        args = Arguments(AgentDiscretePPO, env_func=env_func, env_args=env_args)
        args.target_step = args.max_step * 4
        args.repeat_times = 2 ** 5
        args.gamma = 0.99
        args.eval_times = 2 ** 4
    else:
        raise ValueError('env_name:', env_name)

    args.learner_gpus = gpu_id
    args.random_seed += gpu_id

    if_check = 0
    if if_check:
        train_and_evaluate(args)
    else:
        train_and_evaluate_mp(args)


if __name__ == '__main__':
    GPU_ID = int(sys.argv[1])
    ENV_ID = int(sys.argv[2])

    # demo_continuous_action_off_policy()
    # demo_continuous_action_on_policy()
    # demo_discrete_action_off_policy()
    # demo_discrete_action_on_policy()
