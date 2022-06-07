from elegantrl_helloworld.config import Arguments
from elegantrl_helloworld.run import train_agent, evaluate_agent
from elegantrl_helloworld.env import get_gym_env_args, PendulumEnv


def train_ddpg_in_pendulum(gpu_id=0):  # DDPG is a simple but low sample efficiency and unstable.
    from elegantrl_helloworld.agent import AgentDDPG
    agent_class = AgentDDPG

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

    args.learner_gpus = gpu_id
    train_agent(args)
    evaluate_agent(args)
    print('| The cumulative returns of Pendulum-v1 is ∈ (-1600, (-1400, -200), 0)')
    '''
    | `Steps` denotes the number of samples, or the total training step, or the running times of `env.step()`.
    | `ExpR` denotes average rewards during exploration. The agent gets this rewards with noisy action.
    | `ObjC` denotes the objective of Critic network. Or call it loss function of critic network.
    | `ObjA` denotes the objective of Actor network. It is the average Q value of the critic network.
    '''

def train_ddpg_in_lunar_lander_or_bipedal_walker(gpu_id=0):  # DDPG is a simple but low sample efficiency and unstable.
    from elegantrl_helloworld.agent import AgentDDPG
    agent_class = AgentDDPG
    env_name = ["LunarLanderContinuous-v2", "BipedalWalker-v3"][1]

    if env_name == "LunarLanderContinuous-v2":
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
        args.gamma = 0.99

        '''network update'''
        args.target_step = args.max_step // 2
        args.net_dim = 2 ** 8
        args.num_layer = 3
        args.batch_size = 2 ** 7
        args.repeat_times = 2 ** 0
        args.explore_noise = 0.1

        '''evaluate'''
        args.eval_gap = 2 ** 7
        args.eval_times = 2 ** 3
        args.break_step = int(1e6)
    else:
        raise ValueError("env_name:", env_name)

    args.learner_gpus = gpu_id
    train_agent(args)
    evaluate_agent(args)

if __name__ == "__main__":
    train_ddpg_in_pendulum()
    train_ddpg_in_lunar_lander_or_bipedal_walker()

