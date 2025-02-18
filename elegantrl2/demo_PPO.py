from argparse import ArgumentParser

try:
    from ..elegantrl2.traj_config import Config, get_gym_env_args
    from ..elegantrl2.traj_run import train_agent
    from ..elegantrl2.traj_agent_ppo import AgentPPO
except ImportError or ModuleNotFoundError:
    from elegantrl2.traj_config import Config, get_gym_env_args
    from elegantrl2.traj_run import train_agent
    from elegantrl2.traj_agent_ppo import AgentPPO

'''demo'''


def train_ppo_for_pendulum(agent_class, gpu_id: int):
    num_envs = 8

    from elegantrl.envs.CustomGymEnv import PendulumEnv
    env_class = PendulumEnv  # run a custom env: PendulumEnv, which based on OpenAI pendulum
    env_args = {
        'env_name': 'Pendulum',  # Apply torque on the free end to swing a pendulum into an upright position
        'max_step': 200,  # the max step number in an episode for evaluation
        'state_dim': 3,  # the x-y coordinates of the pendulum's free end and its angular velocity.
        'action_dim': 1,  # the torque applied to free end of the pendulum
        'if_discrete': False,  # continuous action space, symbols → direction, value → force

        'num_envs': num_envs,  # the number of sub envs in vectorized env
        'if_build_vec_env': True,
    }
    get_gym_env_args(env=PendulumEnv(), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `erl_config.py Arguments()` for hyperparameter explanation
    args.break_step = int(8e4)
    args.net_dims = [128, 64]  # the middle layer dimension of MultiLayer Perceptron
    args.gamma = 0.97  # discount factor of future rewards
    args.reward_scale = 2 ** -2

    args.horizon_len = args.max_step * 1
    args.repeat_times = 16  # repeatedly update network using ReplayBuffer to keep critic's loss small
    args.learning_rate = 4e-4
    args.state_value_tau = 0.2  # the tau of normalize for value and state `std = (1-std)*std + tau*std`

    args.gpu_id = gpu_id
    args.num_workers = 4
    train_agent(args=args, if_single_process=False)

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


def train_ppo_for_lunar_lander_continuous(agent_class, gpu_id: int):
    num_envs = 8

    import gymnasium as gym
    env_class = gym.make  # run a custom env: PendulumEnv, which based on OpenAI pendulum
    env_args = {
        'env_name': 'LunarLanderContinuous-v2',
        'max_step': 1000,
        'state_dim': 8,
        'action_dim': 2,
        'if_discrete': False,

        'num_envs': num_envs,  # the number of sub envs in vectorized env
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

    args.gpu_id = gpu_id
    args.num_workers = 4
    train_agent(args=args)

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


def train_ppo_for_bipedal_walker(agent_class, gpu_id: int):
    num_envs = 8

    import gymnasium as gym
    env_class = gym.make  # run a custom env: PendulumEnv, which based on OpenAI pendulum
    env_args = {
        'env_name': 'BipedalWalker-v3',
        'max_step': 1600,
        'state_dim': 24,
        'action_dim': 4,
        'if_discrete': False,

        'num_envs': num_envs,  # the number of sub envs in vectorized env
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

    args.gpu_id = gpu_id
    args.random_seed = GPU_ID
    args.num_workers = 2
    train_agent(args=args, if_single_process=False)

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


def demo_load_pendulum_and_render():
    import torch as th

    gpu_id = 0  # >=0 means GPU ID, -1 means CPU
    device = th.device(f"cuda:{gpu_id}" if (th.cuda.is_available() and (gpu_id >= 0)) else "cpu")

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

    agent = AgentPPO(net_dims=[128, 64], state_dim=env_args['state_dim'], action_dim=env_args['action_dim'],
                     gpu_id=gpu_id)
    act = agent.act
    act.load_state_dict(
        th.load(f"./Pendulum_PPO_0/act.pt", map_location=lambda storage, loc: storage, weights_only=True))
    # act = th.load(f"./Pendulum_PPO_0/act.pt", map_location=device)

    '''evaluate'''
    eval_times = 2 ** 7
    from elegantrl.train.evaluator import get_rewards_and_steps
    rewards_step_list = [get_rewards_and_steps(env, act) for _ in range(eval_times)]
    rewards_step_ten = th.tensor(rewards_step_list)
    print(f"\n| average cumulative_returns {rewards_step_ten[:, 0].mean().item():9.3f}"
          f"\n| average      episode steps {rewards_step_ten[:, 1].mean().item():9.3f}", flush=True)

    '''render'''
    if_discrete = env.if_discrete
    device = next(act.parameters()).device  # net.parameters() is a Python generator.

    state = env.reset()
    steps = None
    returns = 0.0  # sum of rewards in an episode
    for steps in range(12345):
        s_tensor = th.as_tensor(state, dtype=th.float32, device=device).unsqueeze(0)
        a_tensor = act(s_tensor).argmax(dim=1) if if_discrete else act(s_tensor)
        action = a_tensor.detach().cpu().numpy()[0]  # not need detach(), because using th.no_grad() outside
        state, reward, done, _ = env.step(action)
        returns += reward
        env.render()

        if done:
            break
    returns = getattr(env, 'cumulative_rewards', returns)
    steps += 1

    print(f"\n| cumulative_returns {returns}"
          f"\n|      episode steps {steps}", flush=True)


if __name__ == '__main__':
    Parser = ArgumentParser(description='ArgumentParser for ElegantRL')
    Parser.add_argument('--gpu', type=int, default=0, help='GPU device ID for training')
    Parser.add_argument('--env', type=str, default='1', help='the environment ID for training')
    Parser.add_argument('--drl', type=int, default=0, help='RL algorithms ID for training')

    Args = Parser.parse_args()
    GPU_ID = Args.gpu
    DRL_ID = Args.drl
    ENV_ID = Args.env

    AgentClassList = [AgentPPO, AgentA2C]
    AgentClass = AgentClassList[DRL_ID]  # DRL algorithm name

    if ENV_ID in {'0', 'pendulum'}:
        train_ppo_for_pendulum(agent_class=AgentClass, gpu_id=GPU_ID)
    elif ENV_ID in {'1', 'lunar_lander_continuous'}:
        train_ppo_for_lunar_lander_continuous(agent_class=AgentClass, gpu_id=GPU_ID)
    elif ENV_ID in {'2', 'bipedal_walker'}:
        train_ppo_for_bipedal_walker(agent_class=AgentClass, gpu_id=GPU_ID)
    else:
        print('ENV_ID not match', flush=True)
