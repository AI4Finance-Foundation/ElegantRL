from elegantrl_helloworld.config import Arguments
from elegantrl_helloworld.run import train_and_evaluate
from elegantrl_helloworld.env import get_gym_env_args
import gym
import yaml
gym.logger.set_level(40)  # Block warning

def train_dqn_in_cartpole(gpu_id=0):  # DQN is a simple but low sample efficiency.
    env_name = "CartPole-v0"
    alg = "DQN"
    with open("config.yml", 'r') as f:
        hyp = yaml.safe_load(f)[alg][env_name]
    env = gym.make(env_name)
    env_func = gym.make
    env_args = get_gym_env_args(env, if_print=True)

    args = Arguments(env_func, env_args, hyp)

   
    train_and_evaluate(args)
    
def train_dqn_in_lunar_lander(gpu_id=0):  # DQN is a simple but low sample efficiency.
    env_name = "LunarLander-v2"
    alg = "DQN"
    with open("config.yml", 'r') as f:
        hyp = yaml.safe_load(f)[alg][env_name]
    env = gym.make(env_name)
    env_func = gym.make
    env_args = get_gym_env_args(env, if_print=True)

    args = Arguments(env_func, env_args, hyp)

    train_and_evaluate(args)


if __name__ == "__main__":
    train_dqn_in_cartpole()
    train_dqn_in_lunar_lander()
