import sys
import torch as th
import gymnasium as gym
from argparse import ArgumentParser

sys.path.append("..")
if True:  # write after `sys.path.append("..")`
    from elegantrl import train_agent, train_agent_multiprocessing
    from elegantrl import Config, get_gym_env_args
    from elegantrl.agents import AgentPPO
    from elegantrl.agents import AgentA2C


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


def build_env(env_name: str):
    def build_func():
        return gym.make(env_name)

    return build_func


'''unit tests'''


def check_gym_single():
    env_name = 'LunarLanderContinuous-v2'
    env = gym.make(env_name)

    max_step = 2 ** 10

    state, info = env.reset()
    cumulative_rewards = 0.0
    for i in range(max_step):
        action = env.action_space.sample()

        next_state, reward, terminated, truncated, info = env.step(action)
        cumulative_rewards += reward

        if terminated or truncated:
            break
    print(f"cumulative_rewards: {cumulative_rewards:9.2f}")
    env.close()


def check_gym_vector():
    env_name = 'LunarLanderContinuous-v2'
    num_envs = 8
    # env = gym.make(env_name)
    envs = gym.vector.SyncVectorEnv([build_env(env_name) for _ in range(num_envs)])

    max_step = 2 ** 10

    state, info = envs.reset()
    cumulative_rewards = th.zeros(num_envs, dtype=th.float32).numpy()
    for i in range(max_step):
        action = envs.action_space.sample()

        next_state, reward, terminated, truncated, info = envs.step(action)

        state = next_state
        cumulative_rewards += reward

    print(f"cumulative_rewards: {cumulative_rewards.mean():9.2f}")
    envs.close()


def check_get_gym_env_args():
    env_name = 'LunarLanderContinuous-v2'
    num_envs = 8
    # env = gym.make(env_name)
    envs = gym.vector.SyncVectorEnv([build_env(env_name) for _ in range(num_envs)])
    env = envs.envs[0]

    env_args = get_gym_env_args(env, if_print=True)


if __name__ == '__main__':
    check_gym_single()
    check_gym_vector()
    check_get_gym_env_args()

    # Parser = ArgumentParser(description='ArgumentParser for ElegantRL')
    # Parser.add_argument('--gpu', type=int, default=0, help='GPU device ID for training')
    # Parser.add_argument('--drl', type=int, default=0, help='RL algorithms ID for training')
    # Parser.add_argument('--env', type=str, default='0', help='the environment ID for training')
    #
    # Args = Parser.parse_args()
    # GPU_ID = Args.gpu
    # DRL_ID = Args.drl
    # ENV_ID = Args.env
    #
    # if ENV_ID in {'0', 'pendulum'}:
    #     train_ppo_a2c_for_pendulum()
    # elif ENV_ID in {'1', 'pendulum_vec'}:
    #     train_ppo_a2c_for_pendulum_vec_env()
    # else:
    #     print('ENV_ID not match')
