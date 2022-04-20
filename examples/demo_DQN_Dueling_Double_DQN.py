import sys
import gym

from elegantrl.train.run import train_and_evaluate, train_and_evaluate_mp
from elegantrl.train.config import Arguments
from elegantrl.agents.AgentDQN import AgentDQN
from elegantrl.agents.AgentDuelingDQN import AgentDuelingDQN
from elegantrl.agents.AgentDoubleDQN import AgentDoubleDQN
from elegantrl.agents.AgentDuelingDoubleDQN import AgentDuelingDoubleDQN


def demo_discrete_action_off_policy(gpu_id):
    env_name = ['CartPole-v0',
                'LunarLander-v2', ][1]
    agent_class = [AgentDQN, AgentDuelingDQN, AgentDoubleDQN, AgentDuelingDoubleDQN][3]

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
        args = Arguments(agent_class, env_func=env_func, env_args=env_args)

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
        
        CPU Win10 AgentDuelingDoubleDQN
        ################################################################################
        ID     Step    maxR |    avgR   stdR   avgS  stdS |    expR   objC   etc.
        0  1.01e+03 -149.38 |
        0  1.01e+03 -149.38 | -149.38   51.4     73    11 |   -1.38   1.69  -0.40
        0  5.38e+04  -50.17 |
        0  5.38e+04  -50.17 |  -50.17   23.9   1000     0 |   -0.00   0.98   0.50
        0  7.37e+04  -48.50 |
        0  7.37e+04  -48.50 |  -48.50   89.6    896   275 |   -0.02   0.38   5.07
        0  8.84e+04  -42.61 |
        0  8.84e+04  -42.61 |  -42.61   22.0   1000     0 |   -0.02   0.16   5.03
        0  1.02e+05  -42.61 |  -42.97   13.6   1000     0 |   -0.01   0.22   2.60
        0  1.15e+05  -42.61 |  -65.57   20.2    959    81 |    0.06   0.25   7.48
        0  1.29e+05  221.02 |
        0  1.29e+05  221.02 |  221.02   77.9    410    55 |    0.02   0.61   5.04
        | UsedTime: 796 | SavedDir: ./LunarLander-v2_DuelingDoubleDQN_0
        | ReplayBuffer save in: ./LunarLander-v2_DuelingDoubleDQN_0/replay_0.npz
        """
        env_func = gym.make
        env_args = {'env_num': 1,
                    'env_name': 'LunarLander-v2',
                    'max_step': 1000,
                    'state_dim': 8,
                    'action_dim': 4,
                    'if_discrete': True,
                    'target_return': 200, }
        args = Arguments(agent_class, env_func=env_func, env_args=env_args)
        args.target_step = args.max_step
        args.reward_scale = 2 ** -2
        args.gamma = 0.99
        args.eval_times = 2 ** 4
    else:
        raise ValueError('env_name:', env_name)

    args.learner_gpus = gpu_id
    args.random_seed += gpu_id

    if_check = 1
    if if_check:
        train_and_evaluate(args)
    else:
        train_and_evaluate_mp(args)


if __name__ == '__main__':
    GPU_ID = int(sys.argv[1] if len(sys.argv) > 1 else 0)

    demo_discrete_action_off_policy(GPU_ID)
