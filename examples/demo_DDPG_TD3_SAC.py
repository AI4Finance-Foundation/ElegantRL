import sys
from argparse import ArgumentParser

sys.path.append("..")
if True:  # write after `sys.path.append("..")`
    from elegantrl.train.run import train_agent, train_agent_multiprocessing
    from elegantrl.train.config import Config, get_gym_env_args
    from elegantrl.agents.AgentTD3 import AgentTD3
    from elegantrl.agents.AgentSAC import AgentSAC, AgentModSAC


def train_ddpg_for_pendulum():
    from elegantrl.envs.CustomGymEnv import PendulumEnv

    agent_class = [AgentTD3, AgentSAC, AgentModSAC][DRL_ID]  # DRL algorithm name
    env_class = PendulumEnv  # run a custom env: PendulumEnv, which based on OpenAI pendulum
    env_args = {
        'env_name': 'Pendulum',  # Apply torque on the free end to swing a pendulum into an upright position
        'max_step': 200,  # the max step number of an episode.
        'state_dim': 3,  # the x-y coordinates of the pendulum's free end and its angular velocity.
        'action_dim': 1,  # the torque applied to free end of the pendulum
        'if_discrete': False  # continuous action space, symbols → direction, value → force
    }
    get_gym_env_args(env=PendulumEnv(), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `config.py Arguments()` for hyperparameter explanation
    args.break_step = int(8e4)  # break training if 'total_step > break_step'
    args.net_dims = (128, 64)  # the middle layer dimension of MultiLayer Perceptron
    args.gamma = 0.97  # discount factor of future rewards
    args.horizon_len = args.max_step * 2

    args.repeat_times = 1.0  # repeatedly update network using ReplayBuffer to keep critic's loss small
    args.learning_rate = 1e-4
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
ID     Step    Time |    avgR   stdR   avgS  stdS |    expR   objC   objA   etc.
0  4.00e+02       1 |-1211.60    4.7    200     0 |   -8.24   7.17  -5.75
0  2.04e+04      58 | -207.35  138.9    200     0 |   -0.91   2.25 -45.49
0  4.04e+04     117 |  -85.54   71.5    200     0 |   -0.95   1.04 -17.13
| UsedTime:     146 | SavedDir: ./Pendulum_TD3_0
    """


def train_ddpg_for_pendulum_vec_env():
    from elegantrl.envs.CustomGymEnv import PendulumEnv

    agent_class = [AgentTD3, AgentSAC, AgentModSAC][DRL_ID]  # DRL algorithm name
    env_class = PendulumEnv  # run a custom env: PendulumEnv, which based on OpenAI pendulum
    env_args = {
        'env_name': 'Pendulum',  # Apply torque on the free end to swing a pendulum into an upright position
        'max_step': 200,  # the max step number of an episode.
        'state_dim': 3,  # the x-y coordinates of the pendulum's free end and its angular velocity.
        'action_dim': 1,  # the torque applied to free end of the pendulum
        'if_discrete': False,  # continuous action space, symbols → direction, value → force

        'num_envs': 8,  # the number of sub envs in vectorized env
        'if_build_vec_env': True,
    }
    get_gym_env_args(env=PendulumEnv(), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `config.py Arguments()` for hyperparameter explanation
    args.break_step = int(2e4)  # break training if 'total_step > break_step'
    # args.net_dims = (128, 64)  # the middle layer dimension of MultiLayer Perceptron
    args.batch_size = 512  # vectorized env need a larger batch_size
    args.gamma = 0.97  # discount factor of future rewards
    args.horizon_len = args.max_step

    args.repeat_times = 1.0  # repeatedly update network using ReplayBuffer to keep critic's loss small
    args.learning_rate = 1e-4
    args.state_value_tau = 0.2  # the tau of normalize for value and state `std = (1-std)*std + tau*std`

    args.gpu_id = GPU_ID
    args.eval_per_step = int(4e3)
    args.num_workers = 4
    if_single_process = False
    if if_single_process:
        train_agent(args)
    else:
        train_agent_multiprocessing(args)  # train_agent(args)
    """
-2000 < -1200 < -200 < -80
################################################################################
ID     Step    Time |    avgR   stdR   avgS  stdS |    expR   objC   objA   etc.
1  1.60e+03      10 |-1444.53  275.0    200     0 |   -6.58   5.49 -18.76
1  2.16e+04      64 | -200.44  115.7    200     0 |   -1.18   1.87 -47.91
1  4.16e+04     120 | -153.00   84.5    200     0 |   -0.75   0.92 -18.70
| UsedTime:     120 | SavedDir: ./Pendulum_TD3_0
################################################################################
ID     Step    Time |    avgR   stdR   avgS  stdS |    expR   objC   objA   etc.
1  1.60e+03      10 |-1387.47  167.9    200     0 |   -6.46   5.67 -28.75   0.33
1  5.60e+03      17 |-1513.77   83.9    200     0 |   -5.79   0.72 -62.95   0.27
1  9.60e+03      26 |-1054.83   41.8    200     0 |   -5.73   0.61 -79.48   0.23
1  1.36e+04      35 | -734.88   81.0    200     0 |   -4.38   0.82 -89.61   0.26
1  1.76e+04      45 | -140.39   77.9    200     0 |   -2.53   1.21 -91.22   0.34
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

    if ENV_ID in {'0', 'pendulum'}:
        train_ddpg_for_pendulum()
    elif ENV_ID in {'1', 'pendulum_vec'}:
        train_ddpg_for_pendulum_vec_env()
    else:
        print('ENV_ID not match')
