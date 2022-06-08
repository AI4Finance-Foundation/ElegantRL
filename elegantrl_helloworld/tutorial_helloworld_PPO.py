from elegantrl_helloworld.config import Arguments
from elegantrl_helloworld.run import train_and_evaluate
from elegantrl_helloworld.env import get_gym_env_args, PendulumEnv
import yaml
import gym

def train_ppo_in_pendulum():
    env_name = "Pendulum"
    alg = "PPO"
    with open("config.yml", 'r') as f:
       hyp = yaml.safe_load(f)[alg][env_name]
    env = PendulumEnv()
    env_func = PendulumEnv
    env_args = get_gym_env_args(env, if_print=True)

    args = Arguments(env_func, env_args, hyp)
    
    train_and_evaluate(args)
    
def train_ppo_in_lunar_lander():
    env_name = "LunarLanderContinuous-v2"
    alg = "PPO"
    with open("config.yml", 'r') as f:
       hyp = yaml.safe_load(f)[alg][env_name]
    env = gym.make(env_name)
    env_func = gym.make
    env_args = get_gym_env_args(env, if_print=True)

    args = Arguments(env_func, env_args, hyp)

    
    train_and_evaluate(args)


def train_ppo_in_bipedal_walker(gpu_id = 0):

    env_name = "BipedalWalker-v3"
    alg = "PPO"
    with open("config.yml", 'r') as f:
       hyp = yaml.safe_load(f)[alg][env_name]
    env = gym.make(env_name)
    env_func = gym.make
    env_args = get_gym_env_args(env, if_print=True)

    args = Arguments(env_func, env_args, hyp)
    train_and_evaluate(args)

if __name__ == "__main__":
    train_ppo_in_pendulum()
    train_ppo_in_lunar_lander()
    train_ppo_in_bipedal_walker()
