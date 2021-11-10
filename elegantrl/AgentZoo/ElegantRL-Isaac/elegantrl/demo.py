from elegantrl.agent import *
from elegantrl.env import build_env
from elegantrl.run import Arguments, train_and_evaluate, train_and_evaluate_mp

"""[ElegantRL.2021.10.10](https://github.com/AI4Finance-Foundation/ElegantRL)"""

'''train'''


def demo_continuous_action_off_policy():  # [ElegantRL.2021.10.10]
    env_name = ['Pendulum-v1', 'LunarLanderContinuous-v2',
                'BipedalWalker-v3', 'BipedalWalkerHardcore-v3'][0]
    agent_class = [AgentModSAC, AgentSAC,
                   AgentTD3, AgentDDPG][0]
    args = Arguments(env=build_env(env_name), agent=agent_class())

    if env_name in {'Pendulum-v1', 'Pendulum-v0'}:
        """EpisodeReturn: (-1800) -1000 ~ -200 (-50)
        Step 2e5,  Reward -200,  UsedTime 200s ModSAC
        """
        # args = Arguments(env=build_env(env_name), agent=agent_class())  # One way to build env
        # args = Arguments(env=env_name, agent=agent_class())  # Another way to build env
        # args.env_num = 1
        # args.max_step = 200
        # args.state_dim = 3
        # args.action_dim = 1
        # args.if_discrete = False
        # args.target_return = -200

        args.gamma = 0.97
        args.net_dim = 2 ** 7
        args.worker_num = 2
        args.reward_scale = 2 ** -2
        args.target_step = 200 * 4  # max_step = 200
    if env_name in {'LunarLanderContinuous-v2', 'LunarLanderContinuous-v1'}:
        """EpisodeReturn: (-800) -200 ~ 200 (302)
        Step 4e5,  Reward 200,  UsedTime  900s, TD3
        Step 5e5,  Reward 200,  UsedTime 1500s, ModSAC
        """
        args.eval_times1 = 2 ** 4
        args.eval_times2 = 2 ** 6

        args.target_step = args.env.max_step
    if env_name in {'BipedalWalker-v3', 'BipedalWalker-v2'}:
        """EpisodeReturn: (-200) -140 ~ 300 (341)
        Step 08e5,  Reward 300,  UsedTime 1800s TD3
        Step 11e5,  Reward 329,  UsedTime 6000s TD3
        Step  4e5,  Reward 300,  UsedTime 2000s ModSAC
        Step  8e5,  Reward 330,  UsedTime 5000s ModSAC
        """
        args.eval_times1 = 2 ** 3
        args.eval_times2 = 2 ** 5

        args.gamma = 0.98
        args.target_step = args.env.max_step
    if env_name in {'BipedalWalkerHardcore-v3', 'BipedalWalkerHardcore-v2'}:
        '''EpisodeReturn: (-200) -150 ~ 300 (334)
        TotalStep (2e6) 4e6
        
        Step 12e5,  Reward  20
        Step 18e5,  Reward 135
        Step 25e5,  Reward 202
        Step 43e5,  Reward 309, UsedTime 68ks,  ModSAC, worker_num=4
        
        Step 14e5,  Reward  15
        Step 18e5,  Reward 117
        Step 28e5,  Reward 212
        Step 45e5,  Reward 306,  UsedTime 67ks,  ModSAC, worker_num=4
        
        Step  8e5,  Reward  13
        Step 16e5,  Reward 136
        Step 23e5,  Reward 219
        Step 38e5,  Reward 302
        UsedTime 99ks  ModSAC, worker_num=2
        '''
        args.gamma = 0.98
        args.net_dim = 2 ** 8
        args.max_memo = 2 ** 22
        args.break_step = int(80e6)
        args.batch_size = args.net_dim * 2
        args.repeat_times = 1.5
        args.learning_rate = 2 ** -15

        args.eval_gap = 2 ** 9
        args.eval_times1 = 2 ** 2
        args.eval_times2 = 2 ** 5

        args.worker_num = 4
        args.target_step = args.env.max_step * 1
    # args.learner_gpus = (0, )  # single GPU
    # args.learner_gpus = (0, 1)  # multiple GPUs
    # train_and_evaluate(args)  # single process
    train_and_evaluate_mp(args)  # multiple process


def demo_continuous_action_on_policy():  # [ElegantRL.2021.10.13]
    env_name = ['Pendulum-v1', 'LunarLanderContinuous-v2',
                'BipedalWalker-v3', 'BipedalWalkerHardcore-v3'][ENV_ID]
    agent_class = [AgentPPO, AgentA2C][0]
    args = Arguments(env=build_env(env_name), agent=agent_class())
    # args.if_per_or_gae = True

    if env_name in {'Pendulum-v1', 'Pendulum-v0'}:
        """
        Step 45e4,  Reward -138,  UsedTime 373s PPO
        Step 40e4,  Reward -200,  UsedTime 400s PPO
        Step 46e4,  Reward -213,  UsedTime 300s PPO
        """
        # args = Arguments(env=build_env(env_name), agent=agent_class())  # One way to build env
        # args = Arguments(env=env_name, agent=agent_class())  # Another way to build env
        # args.env_num = 1
        # args.max_step = 200
        # args.state_dim = 3
        # args.action_dim = 1
        # args.if_discrete = False
        # args.target_return = -200

        args.gamma = 0.97
        args.net_dim = 2 ** 8
        args.worker_num = 2
        args.reward_scale = 2 ** -2
        args.target_step = 200 * 16  # max_step = 200

        args.eval_gap = 2 ** 5
    if env_name in {'LunarLanderContinuous-v2', 'LunarLanderContinuous-v1'}:
        """
        Step  9e5,  Reward 210,  UsedTime 1127s PPO
        Step 13e5,  Reward 223,  UsedTime 1416s PPO
        Step 15e5,  Reward 250,  UsedTime 1648s PPO
        Step 19e5,  Reward 201,  UsedTime 1880s PPO
        Step 43e5,  Reward 224,  UsedTime 3738s PPO
        Step 14e5,  Reward 213,  UsedTime 1654s PPO GAE
        Step 12e5,  Reward 216,  UsedTime 1710s PPO GAE
        """
        args.eval_times1 = 2 ** 4
        args.eval_times2 = 2 ** 6

        args.target_step = args.env.max_step * 8
    if env_name in {'BipedalWalker-v3', 'BipedalWalker-v2'}:
        """
        Step 51e5,  Reward 300,  UsedTime 2827s PPO
        Step 78e5,  Reward 304,  UsedTime 4747s PPO
        Step 61e5,  Reward 300,  UsedTime 3977s PPO GAE
        Step 95e5,  Reward 291,  UsedTime 6193s PPO GAE
        """
        args.eval_times1 = 2 ** 3
        args.eval_times2 = 2 ** 5

        args.gamma = 0.98
        args.target_step = args.env.max_step * 16
    if env_name in {'BipedalWalkerHardcore-v3', 'BipedalWalkerHardcore-v2'}:
        """
        Step 57e5,  Reward 295,  UsedTime 17ks PPO
        Step 70e5,  Reward 300,  UsedTime 21ks PPO
        """
        args.gamma = 0.98
        args.net_dim = 2 ** 8
        args.max_memo = 2 ** 22
        args.batch_size = args.net_dim * 4
        args.repeat_times = 2 ** 4
        args.learning_rate = 2 ** -16

        args.eval_gap = 2 ** 8
        args.eval_times1 = 2 ** 2
        args.eval_times2 = 2 ** 5
        # args.break_step = int(80e5)

        args.worker_num = 4
        args.target_step = args.env.max_step * 16

    args.learner_gpus = (GPU_ID,)  # single GPU
    # args.learner_gpus = (0, 1)  # multiple GPUs
    # train_and_evaluate(args)  # single process
    train_and_evaluate_mp(args)  # multiple process


def demo_discrete_action_off_policy():  # [ElegantRL.2021.10.10]
    env_name = ['CartPole-v0', 'LunarLander-v2',
                'SlimeVolley-v0', ][0]
    agent_class = [AgentDoubleDQN, AgentDQN][0]
    args = Arguments(env=build_env(env_name), agent=agent_class())
    args.agent.if_use_dueling = True  # DuelingDQN

    if env_name in {'CartPole-v0', }:
        "Step 1e5,  Reward 200,  UsedTime 40s, AgentD3QN"
        args.target_return = 195

        args.reward_scale = 2 ** -1
        args.target_step = args.env.max_step * 4

        args.eval_gap = 2 ** 5
    if env_name in {'LunarLander-v2', }:
        "Step 29e4,  Reward 222,  UsedTime 5811s D3QN"
        args.max_memo = 2 ** 19

        args.reward_scale = 2 ** -1
        args.target_step = args.env.max_step

    # args.learner_gpus = (0, )  # single GPU
    # args.learner_gpus = (0, 1)  # multiple GPUs
    # train_and_evaluate(args)  # single process
    train_and_evaluate_mp(args)  # multiple process


def demo_discrete_action_on_policy():  # [ElegantRL.2021.10.12]
    env_name = ['CartPole-v0', 'LunarLander-v2',
                'SlimeVolley-v0', ][0]
    agent_class = [AgentDiscretePPO, AgentDiscreteA2C][0]
    args = Arguments(env=build_env(env_name), agent=agent_class())

    if env_name in {'CartPole-v0', }:
        "Step 1e5,  Reward 200,  UsedTime 40s, DiscretePPO"
        args.target_return = 195

        args.reward_scale = 2 ** -1
        args.target_step = args.env.max_step * 8

        args.eval_gap = 2 ** 5
        train_and_evaluate(args)  # single process
    if env_name in {'LunarLander-v2', }:
        '''
        Step 70e5,  Reward 110,  UsedTime 9961s  DiscretePPO, repeat_times = 2 ** 4
        Step 10e5,  Reward 218,  UsedTime 1336s  DiscretePPO, repeat_times = 2 ** 5
        '''
        args.reward_scale = 2 ** -1
        args.repeat_times = 2 ** 5

        args.worker_num = 2
        args.target_step = args.env.max_step * 4
        train_and_evaluate_mp(args)  # multiple process


def demo_pixel_level_on_policy():  # 2021-09-07
    env_name = ['CarRacingFix', ][ENV_ID]
    agent_class = [AgentPPO, AgentSharePPO, AgentShareA2C][0]
    # args = Arguments(env=build_env(env_name, if_print=True), agent=agent_class())
    args = Arguments(env=env_name, agent=agent_class())

    if env_name == 'CarRacingFix':
        args.state_dim = (112, 112, 6)
        args.action_dim = 6
        args.max_step = 512
        args.if_discrete = False
        args.target_return = 950

        "Step 12e5,  Reward 300,  UsedTime 10ks PPO"
        "Step 20e5,  Reward 700,  UsedTime 25ks PPO"
        "Step 40e5,  Reward 800,  UsedTime 50ks PPO"
        args.agent.ratio_clip = 0.5
        args.agent.explore_rate = 0.75
        args.agent.if_use_cri_target = True

        args.gamma = 0.98
        args.net_dim = 2 ** 8
        args.repeat_times = 2 ** 4
        args.learning_rate = 2 ** -17
        args.soft_update_tau = 2 ** -11
        args.batch_size = args.net_dim * 4
        args.if_per_or_gae = True
        args.agent.lambda_gae_adv = 0.96

        args.eval_gap = 2 ** 9
        args.eval_times1 = 2 ** 2
        args.eval_times1 = 2 ** 4
        args.if_allow_break = False
        args.break_step = int(2 ** 22)

        # args.worker_num = 6  # about 96 cores
        args.worker_num = 2  # about 32 cores
        args.target_step = int(args.max_step * 12 / args.worker_num)

    args.learner_gpus = (GPU_ID,)  # single GPU
    args.eval_gpu_id = GPU_ID
    train_and_evaluate_mp(args)


def demo_isaac_gym_on_policy():
    env_name = ['IsaacVecEnvAnt', 'IsaacVecEnvHumanoid'][0]
    args = Arguments(env=env_name, agent=AgentPPO())
    args.learner_gpus = (0,)
    args.eval_gpu_id = 1

    if env_name in {'IsaacVecEnvAnt', 'IsaacOneEnvAnt'}:
        '''
        Step  21e7, Reward  8350, UsedTime  35ks
        Step 484e7, Reward 16206, UsedTime 960ks  PPO
        Step  20e7, Reward  9196, UsedTime  35ks
        Step 471e7, Reward 15021, UsedTime 960ks  PPO, if_use_cri_target = True
        Step  23e7, Reward  7111, UsedTime  12ks  PPO
        Step  22e7, Reward  5412, UsedTime  12ks  PPO, max_step * 2
        '''
        args.eval_env = 'IsaacOneEnvAnt'
        args.env = f'IsaacVecEnvAnt'
        args.env_num = 4096
        args.max_step = 1000
        args.state_dim = 60
        args.action_dim = 8
        args.if_discrete = False
        args.target_return = 8000

        args.agent.lambda_entropy = 0.05
        args.agent.lambda_gae_adv = 0.97
        args.agent.if_use_cri_target = False

        args.if_per_or_gae = True
        args.learning_rate = 2 ** -14

        args.net_dim = int(2 ** 8 * 1.5)
        args.batch_size = args.net_dim * 2 ** 4
        args.target_step = args.max_step * 1
        args.repeat_times = 2 ** 4
        args.reward_scale = 2 ** -2  # (-50) 0 ~ 2500 (3340)

        args.break_step = int(8e14)
        args.if_allow_break = False
        args.eval_times1 = 2 ** 1
        args.eval_times1 = 2 ** 3

    if env_name in {'IsaacVecEnvHumanoid', 'IsaacOneEnvHumanoid'}:
        '''
        Step 126e7, Reward  8021
        Step 216e7, Reward  9517
        Step 283e7, Reward  9998
        Step 438e7, Reward 10749, UsedTime 960ks  PPO, env_num = 4096
        Step  71e7, Reward  7800
        Step 215e7, Reward  9794, UsedTime 465ks  PPO, env_num = 2048
        Step   1e7, Reward   117
        Step  16e7, Reward   538
        Step  21e7, Reward  3044
        Step  38e7, Reward  5015
        Step  65e7, Reward  6010
        Step  72e7, Reward  6257, UsedTime 129ks  PPO, if_use_cri_target = True
        Step  77e7, Reward  5399, UsedTime 143ks  PPO
        Step  86e7, Reward  5822, UsedTime 157ks  PPO, max_step * 2
        '''
        args.eval_env = 'IsaacOneEnvHumanoid'
        args.env = f'IsaacVecEnvHumanoid'
        args.env_num = 4096
        args.max_step = 1000
        args.state_dim = 108
        args.action_dim = 21
        args.if_discrete = False
        args.target_return = 7000

        args.agent.lambda_entropy = 0.05
        args.agent.lambda_gae_adv = 0.97
        args.agent.if_use_cri_target = True

        args.net_dim = int(2 ** 8 * 1.5)
        args.batch_size = args.net_dim * 2 ** 5
        args.target_step = args.max_step * 1
        args.repeat_times = 2 ** 4
        args.reward_scale = 2 ** -2  # (-50) 0 ~ 2500 (3340)
        args.if_per_or_gae = True
        args.learning_rate = 2 ** -15

        args.break_step = int(8e14)
        args.if_allow_break = False
        args.eval_times1 = 2 ** 1
        args.eval_times1 = 2 ** 3

    args.worker_num = 1
    args.workers_gpus = args.learner_gpus
    train_and_evaluate_mp(args)  # train_and_evaluate(args)


def demo_pybullet_off_policy():
    env_name = ['AntBulletEnv-v0', 'HumanoidBulletEnv-v0',
                'ReacherBulletEnv-v0', 'MinitaurBulletEnv-v0', ][0]
    agent_class = [AgentModSAC, AgentTD3,
                   AgentShareSAC, AgentShareAC][0]
    args = Arguments(env=build_env(env_name, if_print=True), agent=agent_class())

    if env_name == 'AntBulletEnv-v0':
        """EpisodeReturn (-50) 0 ~ 2500 (3340)
        TotalStep (8e5) 10e5
        0  4.29e+06 2446.47 |  431.34   82.1    999     0 |    0.08   1.65-275.32   0.26 | UsedTime   14393 |
        0  1.41e+07 3499.37 | 3317.42    5.9    999     0 |    0.24   0.06 -49.94   0.03 | UsedTime   70020 |
        0  3.54e+06 2875.30 |  888.67    4.7    999     0 |    0.19   0.11 -69.10   0.05 | UsedTime   54701 |
        0  2.00e+07 2960.38 |  698.58   42.5    999     0 |    0.08   0.05 -39.44   0.03 | UsedTime   53545 |
        """
        args.agent.if_use_act_target = False

        args.net_dim = 2 ** 9
        args.max_memo = 2 ** 22
        args.repeat_times = 2 ** 1
        args.reward_scale = 2 ** -2
        args.batch_size = args.net_dim * 2
        args.target_step = args.env.max_step * 2

        args.eval_gap = 2 ** 8
        args.eval_times1 = 2 ** 1
        args.eval_times2 = 2 ** 4
        args.break_step = int(8e7)
        args.if_allow_break = False
    if env_name == 'HumanoidBulletEnv-v0':
        """
        0  1.50e+07 2571.46 |   53.63   66.8    128    58 |    0.04   0.96-153.29   0.06 | UsedTime    74470 |
        0  1.51e+07 2822.93 |   -1.51   27.1     99    36 |    0.03   0.58 -96.48   0.04 | UsedTime    74480 |
        0  1.09e+06   66.96 |   58.69    8.2     58    12 |    0.22   0.28 -22.92   0.00
        0  3.01e+06  129.69 |  101.39   40.6     96    33 |    0.14   0.28 -20.16   0.03
        0  5.02e+06  263.13 |  208.69  122.6    195    59 |    0.11   0.29 -32.71   0.03
        0  6.03e+06  791.89 |  527.79  282.7    360   144 |    0.21   0.26 -36.51   0.03
        0  8.00e+06 2432.21 |   35.78   49.3    113    54 |   -0.08   1.30-168.28   0.05
        0  8.13e+06 2432.21 |  907.28  644.9    606   374 |    0.11   0.72-134.01   0.05
        0  8.29e+06 2432.21 | 2341.30   39.4    999     0 |    0.41   0.41 -96.96   0.03
        0  1.09e+07 2936.10 | 2936.10   24.8    999     0 |    0.60   0.13 -68.74   0.02
        0  2.83e+07 2968.08 | 2737.18   15.9    999     0 |    0.57   0.21 -81.07   0.03 | UsedTime    74512 |
        """
        args.net_dim = 2 ** 9
        args.reward_scale = 2 ** -2
        args.batch_size = args.net_dim * 2
        args.target_step = args.env.max_step * 2

        args.break_step = int(8e7)
        args.if_allow_break = False
    if env_name == 'ReacherBulletEnv-v0':
        """EpisodeReturn (-37) 0 ~ 18 (29) 
        TotalStep: (4e4) 5e4  # low eval_times
        """
        args.explore_rate = 0.9
        args.learning_rate = 2 ** -15

        args.gamma = 0.99
        args.net_dim = 2 ** 8
        args.break_step = int(4e7)
        args.batch_size = args.net_dim * 2
        args.repeat_times = 2 ** 0
        args.reward_scale = 2 ** 2

        args.target_step = args.env.max_step * 4

        args.eval_gap = 2 ** 8
        args.eval_times1 = 2 ** 3
        args.eval_times2 = 2 ** 5
    if env_name == 'MinitaurBulletEnv-v0':
        """EpisodeReturn (-2) 0 ~ 16 (20)
        TotalStep (2e6) 4e6
        0  1.00e+06    0.46 |    0.24    0.0     98    37 |    0.06   0.06  -7.64   0.02
        0  1.26e+06    1.36 |    1.36    0.7    731   398 |    0.10   0.08 -10.40   0.02
        0  1.30e+06    3.18 |    3.18    0.8    999     0 |    0.13   0.08 -10.99   0.02
        0  2.00e+06    3.18 |    0.04    0.0     28     0 |    0.13   0.09 -16.02   0.02
        0  4.04e+06    7.11 |    6.68    0.6    999     0 |    0.17   0.08 -19.67   0.02
        0  5.72e+06    9.79 |    9.28    0.1    999     0 |    0.22   0.03 -23.89   0.01
        0  6.01e+06   10.69 |   10.09    0.8    999     0 |    0.22   0.03 -24.98   0.01
        """

        args.net_dim = 2 ** 9
        args.reward_scale = 2 ** 5  # (-2) 0 ~ 16 (20)
        args.learning_rate = 2 ** -16
        args.batch_size = args.net_dim * 2
        args.target_step = args.env.max_step * 2

        args.eval_gap = 2 ** 8
        args.eval_times1 = 2 ** 2
        args.eval_times2 = 2 ** 4
        args.break_step = int(8e7)
        args.if_allow_break = False

    args.worker_num = 4
    args.learner_gpus = (0,)
    train_and_evaluate_mp(args)


def demo_pybullet_on_policy():
    env_name = ['AntBulletEnv-v0', 'HumanoidBulletEnv-v0',
                'ReacherBulletEnv-v0', 'MinitaurBulletEnv-v0', ][0]
    agent_class = [AgentPPO, AgentSharePPO][0]
    args = Arguments(env=build_env(env_name, if_print=True), agent=agent_class())

    if env_name == 'AntBulletEnv-v0':
        """
        0  1.98e+07 3322.16 | 3322.16   48.7    999     0 |    0.78   0.48  -0.01  -0.80 | UsedTime 12380 PPO
        0  1.99e+07 3104.05 | 3071.44   14.5    999     0 |    0.74   0.47   0.01  -0.79 | UsedTime 12976
        0  1.98e+07 3246.79 | 3245.98   25.3    999     0 |    0.75   0.48  -0.02  -0.81 | UsedTime 13170
        0  1.97e+07 3345.48 | 3345.48   29.0    999     0 |    0.80   0.49  -0.01  -0.81 | UsedTime 8169  PPO 2GPU
        0  1.98e+07 3028.69 | 3004.67   10.3    999     0 |    0.72   0.48   0.05  -0.82 | UsedTime 8734  PPO 2GPU
        """
        args.agent.lambda_entropy = 0.05
        args.agent.lambda_gae_adv = 0.97

        args.net_dim = 2 ** 9
        args.reward_scale = 2 ** -2  # (-50) 0 ~ 2500 (3340)
        args.repeat_times = 2 ** 3
        args.learning_rate = 2 ** -15
        args.if_per_or_gae = True
        args.batch_size = args.net_dim * 2 ** 3
        args.target_step = args.env.max_step * 2

        args.break_step = int(8e7)
        args.if_allow_break = False
    if env_name == 'HumanoidBulletEnv-v0':
        """
        0  2.00e+07 2049.87 | 1905.57  686.5    883   308 |    0.93   0.42  -0.02  -1.14 | UsedTime 15292
        0  3.99e+07 2977.80 | 2611.64  979.6    879   317 |    1.29   0.46  -0.01  -1.16 | UsedTime 19685
        0  7.99e+07 3047.88 | 3041.95   41.1    999     0 |    1.37   0.46  -0.04  -1.15 | UsedTime 38693
        """
        args.agent.lambda_entropy = 0.02
        args.agent.lambda_gae_adv = 0.97

        args.net_dim = 2 ** 9
        args.batch_size = args.net_dim * 2 ** 3
        args.target_step = args.env.max_step * 4
        args.reward_scale = 2 ** -1
        args.repeat_times = 2 ** 3
        args.if_per_or_gae = True
        args.learning_rate = 2 ** -14

        args.break_step = int(8e7)
        args.if_allow_break = False
    if env_name == 'ReacherBulletEnv-v0':
        '''eval_times = 4
        Step 1e5, Return: 18,  UsedTime  3ks, PPO eval_times =< 4
        Step 1e6, Return: 18,  UsedTime 30ks, PPO eval_times =< 4

        eval_times = 64
        The probability of the following results is only 25%.      
        0  5.00e+05    3.23 |    3.23   12.6    149     0 |   -0.03   0.64  -0.03  -0.51
        0  3.55e+06    7.69 |    7.69   10.3    149     0 |   -0.19   0.56  -0.04  -0.59
        0  5.07e+06    9.72 |    7.89    7.6    149     0 |    0.27   0.24   0.02  -0.71
        0  6.85e+06   11.89 |    6.52   12.3    149     0 |    0.22   0.18  -0.06  -0.85
        0  7.87e+06   18.59 |   18.59    9.4    149     0 |    0.39   0.18  -0.01  -0.94
        0  1.01e+06   -2.19 |   -7.30   10.9    149     0 |   -0.05   0.70   0.03  -0.52
        0  4.05e+06    9.29 |   -1.86   15.0    149     0 |    0.08   0.28  -0.05  -0.65
        0  4.82e+06   11.12 |   11.12   12.4    149     0 |    0.13   0.26  -0.07  -0.71
        0  6.07e+06   15.66 |   15.66   11.1    149     0 |    0.16   0.14   0.00  -0.81
        0  9.46e+06   18.58 |   18.58    8.2    149     0 |    0.19   0.10  -0.06  -1.09
        0  2.20e+06    3.63 |    3.26    7.3    149     0 |   -0.05   0.43  -0.01  -0.55
        0  4.19e+06    5.24 |    4.60    9.2    149     0 |    0.04   0.24   0.00  -0.66
        0  6.16e+06    5.24 |    4.80    9.2    149     0 |    0.03   0.15  -0.00  -0.81
        0  7.40e+06   12.99 |   12.99   13.2    149     0 |    0.07   0.19  -0.03  -0.91
        0  1.01e+07   18.09 |   18.09    7.6    149     0 |    0.18   0.16  -0.00  -1.09
        0  1.06e+06    3.25 |    3.25    7.6    149     0 |   -0.21   0.72  -0.05  -0.51 
        0  2.13e+06    3.56 |    0.94    6.1    149     0 |    0.08   0.54   0.02  -0.56
        0  5.85e+06   11.61 |   11.61   11.0    149     0 |    0.13   0.22  -0.05  -0.78
        0  9.04e+06   14.07 |   13.57   10.5    149     0 |    0.27   0.17   0.01  -1.05
        0  1.01e+07   16.16 |   16.16   10.8    149     0 |    0.29   0.19  -0.08  -1.13
        0  1.14e+07   21.33 |   21.33    7.8    149     0 |    0.21   0.24  -0.06  -1.21
        0  1.02e+07    4.06 |   -3.27   11.1    149     0 |   -0.01   0.34  -0.03  -0.88
        0  2.00e+07    9.23 |   -1.57    7.6    149     0 |    0.06   0.12  -0.08  -1.26
        0  3.00e+07   11.78 |   11.78    7.6    149     0 |    0.05   0.08  -0.04  -1.40
        0  4.01e+07   13.20 |   12.35    7.8    149     0 |    0.14   0.08   0.01  -1.42
        0  5.02e+07   14.13 |   11.53    6.5    149     0 |    0.10   0.03   0.03  -1.42
        0  6.00e+07   15.75 |    6.33    6.1    149     0 |    0.18   0.13  -0.03  -1.43
        0  7.29e+07   20.71 |   20.71    8.1    149     0 |    0.16   0.03  -0.00  -1.41
        '''
        args.agent.ratio_clip = 0.5
        args.agent.lambda_gae_adv = 0.97
        args.agent.if_use_cri_target = True

        args.gamma = 0.99
        args.reward_scale = 2 ** 1
        args.if_per_or_gae = True
        args.break_step = int(8e7)
        args.explore_rate = 0.9
        args.learning_rate = 2 ** -16

        args.net_dim = 2 ** 8
        args.batch_size = args.net_dim * 4
        args.repeat_times = 2 ** 4
        args.target_step = args.env.max_step * 4

        args.eval_gap = 2 ** 8
        args.eval_times1 = 2 ** 3
        args.eval_times2 = 2 ** 5
    if env_name == 'MinitaurBulletEnv-v0':
        """EpisodeReturn (-2) 0 ~ 16 (PPO 34)
        TotalStep (PPO 3e5) 5e5
        0  5.91e+05   10.59 |   10.59    3.9    727   282 |    0.27   0.69  -0.03  -0.52
        0  1.15e+06   14.91 |   12.48    2.2    860   158 |    0.40   0.65  -0.02  -0.55
        0  2.27e+06   25.38 |   22.54    4.7    968    54 |    0.75   0.61  -0.06  -0.60
        0  4.13e+06   29.05 |   28.33    1.0    999     0 |    0.89   0.51  -0.07  -0.65
        0  8.07e+06   32.66 |   32.17    0.9    999     0 |    0.97   0.45  -0.06  -0.73
        0  1.10e+07   32.66 |   32.33    1.3    999     0 |    0.94   0.40  -0.07  -0.80 | UsedTime   20208 |

        0  5.91e+05    5.48 |    5.48    1.5    781   219 |    0.24   0.66  -0.04  -0.52
        0  1.01e+06   12.35 |    9.77    2.9    754   253 |    0.34   0.74  -0.05  -0.54
        0  2.10e+06   12.35 |   12.21    4.8    588   285 |    0.60   0.65  -0.01  -0.58
        0  4.09e+06   28.31 |   22.88   12.6    776   385 |    0.88   0.51  -0.03  -0.66
        0  8.03e+06   30.96 |   28.32    6.8    905   163 |    0.93   0.52  -0.05  -0.76
        0  1.09e+07   32.07 |   31.29    0.9    999     0 |    0.95   0.47  -0.07  -0.82 | UsedTime   20238 |
        """
        args.agent.lambda_entropy = 0.05
        args.agent.lambda_gae_adv = 0.97

        args.net_dim = 2 ** 9
        args.reward_scale = 2 ** 5  # (-2) 0 ~ 16 (20)
        args.repeat_times = 2 ** 4
        args.batch_size = args.net_dim * 4
        args.target_step = args.env.max_step * 2
        args.if_per_or_gae = True
        args.learning_rate = 2 ** -15

        args.break_step = int(8e7)
        args.if_allow_break = False

        args.eval_gap = 2 ** 8
        args.eval_times1 = 2 ** 2
        args.eval_times2 = 2 ** 4
    args.worker_num = 4
    args.learner_gpus = (0,)
    train_and_evaluate_mp(args)


def demo_step1_off_policy():
    env_name = ['DownLinkEnv-v0', 'DownLinkEnv-v1'][ENV_ID]
    agent_class = [AgentStep1AC, AgentShareStep1AC][1]
    args = Arguments(env=build_env(env_name), agent=agent_class())
    args.random_seed += GPU_ID

    args.net_dim = 2 ** 8
    args.batch_size = int(args.net_dim * 2 ** -1)

    args.max_memo = 2 ** 17
    args.target_step = int(args.max_memo * 2 ** -4)
    args.repeat_times = 0.75
    args.reward_scale = 2 ** 2
    args.agent.exploration_noise = 2 ** -5

    args.eval_gpu_id = GPU_ID
    args.eval_gap = 2 ** 9
    args.eval_times1 = 2 ** 0
    args.eval_times2 = 2 ** 1

    args.learner_gpus = (GPU_ID,)

    if_use_single_process = 0
    if if_use_single_process:
        train_and_evaluate(args, )
    else:
        args.worker_num = 4
        train_and_evaluate_mp(args, )


'''train and watch'''

if __name__ == '__main__':
    GPU_ID = 0  # eval(sys.argv[1])
    ENV_ID = 0  # eval(sys.argv[2])
    # demo_continuous_action_off_policy()
    # demo_continuous_action_on_policy()
    # demo_discrete_action_off_policy()
    # demo_discrete_action_on_policy()
    # demo_pixel_level_on_policy()
    # demo_pybullet_off_policy()
    # demo_pybullet_on_policy()
    # demo_step1_off_policy()
    pass
