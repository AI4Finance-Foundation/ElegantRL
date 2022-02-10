from elegantrl_helloworld.run import *

"""custom env"""


class PendulumEnv(gym.Wrapper):
    def __init__(self, gym_env_id="Pendulum-v1", target_return=-200):
        # Pendulum-v0 gym.__version__ == 0.17.0
        # Pendulum-v1 gym.__version__ == 0.21.0
        gym.logger.set_level(40)  # Block warning
        super().__init__(env=gym.make(gym_env_id))

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


"""demo"""


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
        args = Arguments(AgentDQN, env_func=env_func, env_args=env_args)
        args.target_step = args.max_step
        args.reward_scale = 2**-2
        args.gamma = 0.99
        args.eval_times = 2**4
    else:
        raise ValueError("env_name:", env_name)

    args.learner_gpus = gpu_id
    args.random_seed += gpu_id

    train_and_evaluate(args)


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
        """ """
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
        args.reward_scale = 2**-2
        args.gamma = 0.99
        args.eval_times = 2**4
    else:
        raise ValueError("env_name:", env_name)

    args.learner_gpus = gpu_id
    args.random_seed += gpu_id

    train_and_evaluate(args)


def demo_continuous_action_off_policy():
    env_name = [
        "Pendulum-v0",
        "Pendulum-v1",
        "LunarLanderContinuous-v2",
        "BipedalWalker-v3",
        "",
    ][ENV_ID]
    gpu_id = GPU_ID  # >=0 means GPU ID, -1 means CPU

    if env_name in {"Pendulum-v0", "Pendulum-v1"}:
        env = PendulumEnv(env_name, target_return=-500)
        "TotalStep: 1e5, TargetReward: -200, UsedTime: 600s"
        args = Arguments(AgentSAC, env)
        args.reward_scale = 2**-1  # RewardRange: -1800 < -200 < -50 < 0
        args.gamma = 0.97
        args.target_step = args.max_step * 2
        args.eval_times = 2**3
    elif env_name == "LunarLanderContinuous-v2":
        """
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
        args = Arguments(AgentSAC, env_func=env_func, env_args=env_args)

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
        args = Arguments(AgentSAC, env_func=env_func, env_args=env_args)
        args.target_step = args.max_step
        args.gamma = 0.98
        args.eval_times = 2**4
    else:
        raise ValueError("env_name:", env_name)

    args.learner_gpus = gpu_id
    args.random_seed += gpu_id

    train_and_evaluate(args)


def demo_continuous_action_on_policy():
    env_name = [
        "Pendulum-v0",
        "Pendulum-v1",
        "LunarLanderContinuous-v2",
        "BipedalWalker-v3",
    ][ENV_ID]
    gpu_id = GPU_ID  # >=0 means GPU ID, -1 means CPU

    if env_name in {"Pendulum-v0", "Pendulum-v1"}:
        env = PendulumEnv(env_name, target_return=-500)
        "TotalStep: 1e5, TargetReward: -200, UsedTime: 600s"
        args = Arguments(AgentPPO, env)
        args.reward_scale = 2**-1  # RewardRange: -1800 < -200 < -50 < 0
        args.gamma = 0.97
        args.target_step = args.max_step * 8
        args.eval_times = 2**3
    elif env_name == "LunarLanderContinuous-v2":
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
        args = Arguments(AgentPPO, env_func=env_func, env_args=env_args)

        args.target_step = args.max_step * 2
        args.gamma = 0.99
        args.eval_times = 2**5
    elif env_name == "BipedalWalker-v3":
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
        args = Arguments(AgentPPO, env_func=env_func, env_args=env_args)
        args.target_step = args.max_step * 4
        args.reward_scale = 2**-1
        args.gamma = 0.98
        args.eval_times = 2**4
    else:
        raise ValueError("env_name:", env_name)

    args.learner_gpus = gpu_id
    args.random_seed += gpu_id

    train_and_evaluate(args)


if __name__ == "__main__":
    GPU_ID = 1  # int(sys.argv[1])
    ENV_ID = 3  # int(sys.argv[2])

    # demo_continuous_action_off_policy()
    demo_continuous_action_on_policy()
    # demo_discrete_action_off_policy()
    # demo_discrete_action_on_policy()
