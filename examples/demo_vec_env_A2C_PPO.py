import sys
from argparse import ArgumentParser

from elegantrl.train.run import train_agent, train_agent_multiprocessing
from elegantrl.train.config import Config, get_gym_env_args
from elegantrl.agents.AgentPPO import AgentVecPPO
from elegantrl.agents.AgentA2C import AgentVecA2C

sys.path.append("../")


def train_ppo_a2c_for_pendulum():
    from elegantrl.envs.CustomGymEnv import PendulumEnv

    agent_class = [AgentVecPPO, AgentVecA2C][0]  # DRL algorithm name
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
    0  6.40e+03      14 |-1192.19  199.4    200     0 |   -1.44  32.65   0.02   0.01
    0  6.40e+03      14 |-1192.19
    0  2.88e+04      38 | -952.89   70.4    200     0 |   -1.39  13.91   0.02  -0.03
    0  2.88e+04      38 | -952.89
    0  5.12e+04      65 | -421.47   72.3    200     0 |   -1.38  12.35   0.00  -0.06
    0  5.12e+04      65 | -421.47
    0  7.36e+04      91 | -168.78   74.8    200     0 |   -1.28   4.49   0.04  -0.16
    0  7.36e+04      91 | -168.78
    | TrainingTime:     103 | SavedDir: ./Pendulum_PPO_0
    """


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
    from elegantrl.train.config import build_vec_env
    env = build_vec_env(env_class=env_class, env_args=env_args)
    act = torch.load(f"./Pendulum_PPO_0/act.pt", map_location=device)

    '''evaluate'''
    eval_times = 2 ** 7
    from elegantrl.train.evaluator_vec_env import get_rewards_and_step
    rewards_step_list = [get_rewards_and_step(env, act) for _ in range(eval_times)]
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

    from elegantrl.envs.CustomGymEnv import PendulumVecEnv
    env_class = PendulumVecEnv  # run a custom env: PendulumEnv, which based on OpenAI pendulum
    num_envs = 4
    env_args = {
        'env_name': 'Pendulum',  # Apply torque on the free end to swing a pendulum into an upright position
        'num_envs': num_envs,  # the number of sub envs in vectorized env
        'max_step': 200,  # the max step number in an episode for evaluation
        'state_dim': 3,  # the x-y coordinates of the pendulum's free end and its angular velocity.
        'action_dim': 1,  # the torque applied to free end of the pendulum
        'if_discrete': False  # continuous action space, symbols → direction, value → force
    }

    '''init'''
    from elegantrl.train.config import build_vec_env
    env = build_vec_env(env_class=env_class, env_args=env_args)
    act = torch.load(f"./Pendulum_PPO_0/act.pt", map_location=device)

    '''evaluate'''
    eval_times = 2 ** 7
    from elegantrl.train.evaluator_vec_env import get_rewards_and_step_from_vec_env
    rewards_step_list = []
    [rewards_step_list.extend(get_rewards_and_step_from_vec_env(env, act)) for _ in range(eval_times // num_envs)]
    rewards_step_ten = torch.tensor(rewards_step_list)
    print(f"\n| average cumulative_returns {rewards_step_ten[:, 0].mean().item():9.3f}"
          f"\n| average      episode steps {rewards_step_ten[:, 1].mean().item():9.3f}")


if __name__ == '__main__':
    Parser = ArgumentParser(description='ArgumentParser for ElegantRL')
    Parser.add_argument('--gpu', type=int, default=0, help='GPU device ID for training')
    Parser.add_argument('--drl', type=int, default=0, help='RL algorithms ID for training')
    Parser.add_argument('--env', type=int, default=0, help='the environment ID for training')

    Args = Parser.parse_args()
    GPU_ID = Args.gpu
    DRL_ID = Args.drl
    ENV_ID = Args.env

    train_ppo_a2c_for_pendulum()
