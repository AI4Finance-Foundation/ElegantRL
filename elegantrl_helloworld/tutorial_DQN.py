from ElegantRL.elegantrl_helloworld.run import train_and_evaluate
from elegantrl_helloworld.config import Arguments
from elegantrl_helloworld.run import train_agent, evaluate_agent
from elegantrl_helloworld.env import get_gym_env_args, PendulumEnv
import gym
import yaml
gym.logger.set_level(40)  # Block warning

def train_dqn_in_cartpole(gpu_id=0):  # DQN is a simple but low sample efficiency.
    
    env_name = "CartPole-v0"
    with open("config.yml", 'r') as f:
        hyp = yaml.safe_load(f)
    env = gym.make(env_name)
    env_func = gym.make
    env_args = get_gym_env_args(env, if_print=True)

    args = Arguments(env_func, env_args, hyp)

   
    train_and_evaluate(args)
    
def train_dqn_in_lunar_lander(gpu_id=0):  # DQN is a simple but low sample efficiency.
    from elegantrl_helloworld.agent import AgentDQN
    agent_class = AgentDQN
    env_name = "LunarLander-v2"

    import gym
    gym.logger.set_level(40)  # Block warning
    env = gym.make(env_name)
    env_func = gym.make
    env_args = get_gym_env_args(env, if_print=True)

    args = Arguments(agent_class, env_func, env_args)

    '''reward shaping'''
    args.reward_scale = 2 ** 0
    args.gamma = 0.99

    '''network update'''
    args.target_step = args.max_step
    args.net_dim = 2 ** 7
    args.num_layer = 3

    args.batch_size = 2 ** 6

    args.repeat_times = 2 ** 0
    args.explore_noise = 0.125

    '''evaluate'''
    args.eval_gap = 2 ** 7
    args.eval_times = 2 ** 4
    args.break_step = int(4e5)  # LunarLander needs a larger `break_step`

    args.learner_gpus = gpu_id
    train_agent(args)
    evaluate_agent(args)
    print('| The cumulative returns of LunarLander-v2 is ∈ (-1800, (-600, 200), 340)')


if __name__ == "__main__":
    train_dqn_in_cartpole()
    train_dqn_in_lunar_lander()
