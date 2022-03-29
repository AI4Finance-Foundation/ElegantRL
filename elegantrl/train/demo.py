import sys
import gym
from elegantrl.train.run import *
from elegantrl.agents import *
from elegantrl.train.config import Arguments

"""custom env"""


class PendulumEnv(gym.Wrapper):  # [ElegantRL.2021.11.11]
    def __init__(self, gym_env_id="Pendulum-v1", target_return=-200):
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
        state, reward, done, info_dict = self.env.step(
            action * 2
        )  # state, reward, done, info_dict
        return state.astype(np.float32), reward, done, info_dict


class HumanoidEnv(gym.Wrapper):  # [ElegantRL.2021.11.11]
    def __init__(self, gym_env_id="Humanoid-v3", target_return=3000):
        gym.logger.set_level(40)  # Block warning
        super(HumanoidEnv, self).__init__(env=gym.make(gym_env_id))

        # from elegantrl.envs.Gym import get_gym_env_info
        # get_gym_env_info(env, if_print=True)  # use this function to print the env information
        self.env_num = 1  # the env number of VectorEnv is greater than 1
        self.env_name = gym_env_id  # the name of this env.
        self.max_step = 1000  # the max step of each episode
        self.state_dim = 376  # feature number of state
        self.action_dim = 17  # feature number of action
        self.if_discrete = False  # discrete action or continuous action
        self.target_return = target_return  # episode return is between (-1600, 0)

    def reset(self):
        return self.env.reset()

    def step(self, action: np.ndarray):
        # PendulumEnv set its action space as (-2, +2). It is bad.  # https://github.com/openai/gym/wiki/Pendulum-v0
        # I suggest to set action space as (-1, +1) when you design your own env.
        # action_space.high = 0.4
        # action_space.low = -0.4
        state, reward, done, info_dict = self.env.step(
            action * 2.5
        )  # state, reward, done, info_dict
        return state.astype(np.float32), reward, done, info_dict


"""demo"""


def demo_continuous_action_off_policy():  # 2022.02.02
    env_name = [
        "Pendulum-v0",
        "Pendulum-v1",
        "LunarLanderContinuous-v2",
        "BipedalWalker-v3",
        "",
    ][ENV_ID]
    agent = [AgentTD3, AgentSAC, AgentModSAC][2]
    gpu_id = GPU_ID  # >=0 means GPU ID, -1 means CPU

    if env_name in {"Pendulum-v0", "Pendulum-v1"}:
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
        args = Arguments(agent, env)
        args.reward_scale = 2**-1  # RewardRange: -1800 < -200 < -50 < 0
        args.gamma = 0.97
        args.target_step = args.max_step * 2
        args.eval_times = 2**3
    elif env_name == "LunarLanderContinuous-v2":
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
        env_args = {
            "env_num": 1,
            "env_name": "LunarLanderContinuous-v2",
            "max_step": 1000,
            "state_dim": 8,
            "action_dim": 2,
            "if_discrete": False,
            "target_return": 200,
            "id": "LunarLanderContinuous-v2",
        }
        args = Arguments(agent, env_func=env_func, env_args=env_args)

        args.target_step = args.max_step
        args.gamma = 0.99
        args.eval_times = 2**5
    elif env_name == "BipedalWalker-v3":
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

        ID     Step    maxR |    avgR   stdR   avgS  stdS |    expR   objC   etc.
        1  7.08e+03 -106.48 | -106.48    6.0    170    17 |   -0.14   0.70   0.03   0.02
        1  2.38e+05  -89.62 |  -89.62   29.8    775   728 |   -0.30   0.31 -13.44   0.04
        1  4.12e+05  -33.40 |  -34.50   27.6   1342   516 |   -0.01   0.20   1.34   0.06
        1  5.05e+05    2.54 |  -47.29   20.9   1342   516 |    0.02   0.17   0.24   0.05
        1  5.43e+05   52.93 |   52.93  107.6   1084   540 |   -0.21   0.15   0.32   0.05
        1  5.80e+05  138.30 |  136.60   77.6   1460   176 |    0.10   0.16   2.14   0.05
        1  6.16e+05  188.98 |  171.72   99.2   1386   305 |    0.12   0.16  -0.40   0.05
        1  7.06e+05  250.72 |  231.97  142.9   1247   448 |    0.12   0.13   2.81   0.05
        1  8.06e+05  287.28 |  -68.06    5.9    211    19 |   -0.08   0.12   7.83   0.06
        1  8.56e+05  291.10 |  286.19   56.0   1181    63 |    0.17   0.13   6.37   0.06
        1  8.83e+05  314.54 |  314.54    1.0   1252    19 |    0.11   0.12   7.23   0.06
        | UsedTime:    5008 |
        """
        env_func = gym.make
        env_args = {
            "env_num": 1,
            "env_name": "BipedalWalker-v3",
            "max_step": 1600,
            "state_dim": 24,
            "action_dim": 4,
            "if_discrete": False,
            "target_return": 300,
            "id": "BipedalWalker-v3",
        }
        args = Arguments(agent, env_func=env_func, env_args=env_args)
        args.target_step = args.max_step
        args.gamma = 0.98
        args.eval_times = 2**4
    else:
        raise ValueError("env_name:", env_name)

    args.learner_gpus = gpu_id
    args.random_seed += gpu_id

    if_check = 0
    if if_check:
        train_and_evaluate(args)
    else:
        train_and_evaluate_mp(args)


def demo_continuous_action_on_policy():
    gpu_id = (
        int(sys.argv[1]) if len(sys.argv) > 1 else 0
    )  # >=0 means GPU ID, -1 means CPU
    drl_id = 0  # int(sys.argv[2])
    env_id = 4  # int(sys.argv[3])

    env_name = [
        "Pendulum-v0",
        "Pendulum-v1",
        "LunarLanderContinuous-v2",
        "BipedalWalker-v3",
        "Hopper-v2",
        "Humanoid-v3",
    ][env_id]
    agent = [AgentPPO, AgentHtermPPO][drl_id]

    print("agent", agent.__name__)
    print("gpu_id", gpu_id)
    print("env_name", env_name)

    if env_name in {"Pendulum-v0", "Pendulum-v1"}:
        env = PendulumEnv(env_name, target_return=-500)
        "TotalStep: 1e5, TargetReward: -200, UsedTime: 600s"
        args = Arguments(agent, env=env)
        args.reward_scale = 2**-1  # RewardRange: -1800 < -200 < -50 < 0
        args.gamma = 0.97
        args.target_step = args.max_step * 8
        args.eval_times = 2**3
    elif env_name == "LunarLanderContinuous-v2":
        """
        ID     Step    maxR |    avgR   stdR   avgS  stdS |    expR   objC   etc.
        1  8.12e+03 -133.73 | -133.73   51.5     74    12 |   -0.59   9.57   0.02  -0.50
        1  1.86e+05   80.11 |  -23.68  161.1    214   111 |    0.05   1.99   0.04  -0.63
        1  9.88e+05  245.42 |  240.23   62.3    267   132 |    0.51   9.22   0.15  -1.13
        1  1.77e+06  271.45 |  246.60   79.0    208    39 |    0.72   2.41   0.11  -1.03
        1  3.55e+06  286.92 |  281.84   25.7    185    72 |    0.88   1.19   0.01  -1.00
        1  3.87e+06  286.98 |  286.98   17.9    160    14 |    0.77   1.05  -0.00  -1.02
        | UsedTime:    3900 |

        ID     Step    maxR |    avgR   stdR   avgS  stdS |    expR   objC   etc.
        1  8.17e+03 -236.66 | -236.66  109.5    115    32 |   -0.81  15.13   0.15  -0.50
        1  9.54e+04 -128.74 | -128.74   94.7    335   243 |    0.02   7.12  -0.01  -0.57
        1  1.71e+05  110.53 |  110.53  166.0    345   121 |    0.06   0.92   0.05  -0.64
        1  5.07e+05  218.86 |  210.64  111.2    297   112 |    0.47   7.01   0.02  -1.08
        1  6.43e+05  268.47 |  268.47   26.2    239   104 |    0.42   6.52  -0.05  -1.13
        1  1.06e+06  284.71 |  275.28   19.1    180    22 |    0.77   1.64  -0.06  -1.16
        1  1.33e+06  290.29 |  271.59   25.3    216   116 |    0.61   2.17   0.03  -1.17
        1  3.89e+06  293.44 |  282.32   19.0    160    13 |    0.91   2.07   0.09  -1.02
        | UsedTime:    3742 |
        """
        env_func = gym.make
        env_args = {
            "env_num": 1,
            "env_name": "LunarLanderContinuous-v2",
            "max_step": 1000,
            "state_dim": 8,
            "action_dim": 2,
            "if_discrete": False,
            "target_return": 200,
            "id": "LunarLanderContinuous-v2",
        }
        args = Arguments(agent, env_func=env_func, env_args=env_args)

        args.target_step = args.max_step * 2
        args.reward_scale = 2**-1
        args.gamma = 0.99

        args.net_dim = 2**7
        args.layer_num = 3
        args.batch_size = int(args.net_dim * 2)
        args.repeat_times = 2**4

        args.eval_times = 2**5

        args.lambda_h_term = 2**-5
    elif env_name == "BipedalWalker-v3":
        """
        ID     Step    maxR |    avgR   stdR   avgS  stdS |    expR   objC   etc.
        0  2.86e+04  -99.74 |  -99.74    1.4    112     5 |   -0.09   0.92   0.05  -0.53
        0  1.71e+05  -23.46 |  -23.46    4.6   1600     0 |   -0.01   0.03   0.06  -0.87
        0  3.12e+05   91.61 |   91.61   72.6   1600     0 |    0.04   0.07   0.07  -1.19
        0  4.59e+05  207.14 |  207.14  105.8   1405   363 |    0.06   0.48  -0.08  -1.28
        0  7.43e+05  317.28 |   31.39   58.2    487   185 |    0.08   0.73   0.07  -1.21
        0  2.24e+06  326.64 |  325.26    1.0    693    10 |    0.20   0.95   0.04  -1.14
        0  3.90e+06  329.56 |  327.68    0.4    625     3 |    0.22   1.07   0.01  -1.12
        | UsedTime:    3835 |

        ID     Step    maxR |    avgR   stdR   avgS  stdS |    expR   objC   etc.
        1  2.74e+04  -33.96 |  -33.96    1.3   1600     0 |   -0.05   0.29   0.01  -0.53
        1  1.69e+05  102.58 |  102.58    2.3   1600     0 |    0.01   0.03   0.04  -0.88
        1  4.50e+05  206.61 |  206.61    4.0   1600     0 |    0.05   0.21  -0.06  -1.33
        1  8.83e+05  256.48 |  245.42  126.5   1320   351 |    0.08   0.23   0.13  -1.29
        1  1.03e+06  315.47 |  315.47    1.2   1204    22 |    0.11   0.28   0.08  -1.30
        1  2.79e+06  320.63 |  319.06    1.2    899    19 |    0.16   0.40   0.11  -1.21
        1  3.98e+06  320.63 |  319.25    0.2    740     7 |    0.18   0.96  -0.01  -1.19
        | UsedTime:    3704 |
        """
        env_func = gym.make
        env_args = {
            "env_num": 1,
            "env_name": "BipedalWalker-v3",
            "max_step": 1600,
            "state_dim": 24,
            "action_dim": 4,
            "if_discrete": False,
            "target_return": 300,
        }
        args = Arguments(agent, env_func=env_func, env_args=env_args)

        args.gamma = 0.98
        args.eval_times = 2**4
        args.reward_scale = 2**-1

        args.target_step = args.max_step * 4
        args.worker_num = 2
        args.net_dim = 2**7
        args.layer_num = 3
        args.batch_size = int(args.net_dim * 2)
        args.repeat_times = 2**4
        args.ratio_clip = 0.25
        args.lambda_gae_adv = 0.96
        args.lambda_entropy = 0.02
        args.if_use_gae = True

        args.lambda_h_term = 2**-5
    elif env_name == "Hopper-v2":
        """
        ID     Step    maxR |    avgR   stdR   avgS  stdS |    expR   objC   etc.
        5  1.61e+04  131.99 |  131.99    3.6     81     2 |    0.03   0.09   0.03  -0.54
        5  2.20e+05  391.44 |  391.44    0.3    158     0 |    0.08   0.01  -0.06  -0.75
        5  4.25e+05  860.96 |  860.96   11.9    280     5 |    0.09   0.11   0.12  -0.84
        5  6.27e+05 3001.43 | 3001.43    7.9   1000     0 |    0.10   0.78  -0.01  -0.85
        5  1.64e+06 3203.09 | 3103.14    0.0   1000     0 |    0.10   1.82  -0.06  -0.76
        5  2.86e+06 3256.43 | 3152.72    0.0   1000     0 |    0.10   0.75   0.01  -0.67
        5  3.88e+06 3256.43 | 1549.69    0.0    512     0 |    0.10   0.86   0.00  -0.71
        | UsedTime:    2565 |

        ID     Step    maxR |    avgR   stdR   avgS  stdS |    expR   objC   etc.
        2  1.60e+04  328.68 |  328.68    6.2    262     6 |    0.02   0.01  -0.02  -0.54
        2  2.16e+05 2460.57 | 2460.57   14.3   1000     0 |    0.09   0.86   0.20  -0.74
        2  6.22e+05 2789.97 | 2788.28   30.9   1000     0 |    0.10   0.40  -0.11  -1.04
        2  1.23e+06 3263.16 | 3216.96    0.0   1000     0 |    0.10   1.06   0.12  -1.05
        2  2.46e+06 3378.50 | 3364.02    0.0   1000     0 |    0.11   0.87   0.02  -0.92
        2  3.90e+06 3397.88 | 3302.80    0.0   1000     0 |    0.11   0.46   0.01  -0.93
        | UsedTime:    2557 |

        ID     Step    maxR |    avgR   stdR   avgS  stdS |    expR   objC   etc.
        4  2.41e+04  222.39 |  222.39    1.5    120     1 |    0.94   8.45   0.05  -0.55
        4  5.34e+05  344.58 |  344.58    0.4    142     0 |    2.41   1.91   0.02  -0.94
        4  8.74e+05  540.69 |  540.69   20.1    180     4 |    2.96   5.82   0.00  -1.10
        4  1.39e+06  989.51 |  989.51    2.2    308     2 |    3.20  16.75   0.07  -1.08
        4  1.73e+06 3161.60 | 3149.35    0.0   1000     0 |    3.26  43.84  -0.02  -1.08
        4  2.06e+06 3367.27 | 3105.77    0.0   1000     0 |    3.32  44.14   0.00  -1.13
        4  3.92e+06 3604.42 | 3565.39    0.0   1000     0 |    3.44  30.54   0.04  -1.04
        4  5.76e+06 3717.06 | 3607.94    0.0   1000     0 |    3.40  51.92   0.07  -0.95
        4  6.26e+06 3840.95 | 3409.25    0.0   1000     0 |    3.32  66.48  -0.02  -0.94
        | UsedTime:    6251 |
        """
        env_func = gym.make
        env_args = {
            "env_num": 1,
            "env_name": "Hopper-v2",
            "max_step": 1000,
            "state_dim": 11,
            "action_dim": 3,
            "if_discrete": False,
            "target_return": 3800.0,
        }
        args = Arguments(agent, env_func=env_func, env_args=env_args)
        args.eval_times = 2**2
        args.reward_scale = 2**-4

        args.target_step = args.max_step * 4  # 6
        args.worker_num = 2

        args.net_dim = 2**7
        args.layer_num = 3
        args.batch_size = int(args.net_dim * 2)
        args.repeat_times = 2**4
        args.ratio_clip = 0.25
        args.gamma = 0.993
        args.lambda_entropy = 0.02
        args.lambda_h_term = 2**-5

        args.if_allow_break = False
        args.break_step = int(8e6)
    elif env_name == "Humanoid-v3":
        env_func = HumanoidEnv
        env_args = {
            "env_num": 1,
            "env_name": "Humanoid-v3",
            "max_step": 1000,
            "state_dim": 376,
            "action_dim": 17,
            "if_discrete": False,
            "target_return": 3000.0,
        }
        args = Arguments(agent, env_func=env_func, env_args=env_args)

        args.target_step = args.max_step * 16
        args.if_cri_target = False
        args.repeat_times = 2**6
        args.lambda_entropy = 2**-6

        args.worker_num = 2
        args.batch_size = args.net_dim * 8
        args.gamma = 0.995
        args.eval_times = 2**4
        args.max_step = int(8e7)
    else:
        raise ValueError("env_name:", env_name)

    args.learner_gpus = gpu_id
    args.random_seed += gpu_id

    if_check = 0
    if if_check:
        train_and_evaluate(args)
    else:
        train_and_evaluate_mp(args)


def demo_discrete_action_off_policy():
    env_name = [
        "CartPole-v0",
        "LunarLander-v2",
    ][ENV_ID]
    gpu_id = GPU_ID  # >=0 means GPU ID, -1 means CPU

    if env_name == "CartPole-v0":
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
            "env_num": 1,
            "env_name": "CartPole-v0",
            "max_step": 200,
            "state_dim": 4,
            "action_dim": 2,
            "if_discrete": True,
            "target_return": 195.0,
        }
        args = Arguments(AgentDQN, env_func=env_func, env_args=env_args)

        args.target_step = args.max_step
        args.net_dim = 2**7
        args.batch_size = args.net_dim

        args.gamma = 0.97
        args.eval_times = 2**3
        args.eval_gap = 2**4
    elif env_name == "LunarLander-v2":
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
        env_args = {
            "env_num": 1,
            "env_name": "LunarLander-v2",
            "max_step": 1000,
            "state_dim": 8,
            "action_dim": 4,
            "if_discrete": True,
            "target_return": 200,
        }
        args = Arguments(AgentD3QN, env_func=env_func, env_args=env_args)
        args.target_step = args.max_step
        args.reward_scale = 2**-2
        args.gamma = 0.99
        args.eval_times = 2**4
    else:
        raise ValueError("env_name:", env_name)

    args.learner_gpus = gpu_id
    args.random_seed += gpu_id

    if_check = 0
    if if_check:
        train_and_evaluate(args)
    else:
        train_and_evaluate_mp(args)


def demo_discrete_action_on_policy():
    env_name = [
        "CartPole-v0",
        "LunarLander-v2",
    ][ENV_ID]
    gpu_id = GPU_ID  # >=0 means GPU ID, -1 means CPU

    if env_name == "CartPole-v0":
        """
        ID     Step    maxR |    avgR   stdR   avgS  stdS |    expR   objC   etc.
        2  3.22e+03   77.00 |   77.00   23.4     77    23 |    1.00 142.04   0.01   0.00
        2  3.59e+04  200.00 |  200.00    0.0    200     0 |    1.00  38.12  -0.03   0.00
        | UsedTime: 19 | SavedDir: ./CartPole-v0_DiscretePPO_2
        """
        # env = gym.make(env_name)
        # get_gym_env_args(env=env, if_print=True)
        env_func = gym.make
        env_args = {
            "env_num": 1,
            "env_name": "CartPole-v0",
            "max_step": 200,
            "state_dim": 4,
            "action_dim": 2,
            "if_discrete": True,
            "target_return": 195.0,
        }
        args = Arguments(AgentDiscretePPO, env_func=env_func, env_args=env_args)

        args.target_step = args.max_step * 2
        args.net_dim = 2**6
        args.batch_size = args.net_dim * 2

        args.gamma = 0.97
        args.eval_times = 2**3
        args.eval_gap = 2**4
    elif env_name == "LunarLander-v2":
        """Hyper-parameter will influence training used time.

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
        env_args = {
            "env_num": 1,
            "env_name": "LunarLander-v2",
            "max_step": 1000,
            "state_dim": 8,
            "action_dim": 4,
            "if_discrete": True,
            "target_return": 200,
        }
        args = Arguments(AgentDiscretePPO, env_func=env_func, env_args=env_args)
        args.target_step = args.max_step * 4
        args.repeat_times = 2**5
        args.gamma = 0.99
        args.eval_times = 2**4
    else:
        raise ValueError("env_name:", env_name)

    args.learner_gpus = gpu_id
    args.random_seed += gpu_id

    if_check = 0
    if if_check:
        train_and_evaluate(args)
    else:
        train_and_evaluate_mp(args)


if __name__ == "__main__":
    GPU_ID = int(sys.argv[1])
    ENV_ID = int(sys.argv[2])

    # demo_continuous_action_off_policy()
    demo_continuous_action_on_policy()
    # demo_discrete_action_off_policy()
    # demo_discrete_action_on_policy()
