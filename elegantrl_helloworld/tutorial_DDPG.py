from elegantrl_helloworld.config import Arguments
from elegantrl_helloworld.run import train_and_evaluate
from elegantrl_helloworld.env import get_gym_env_args, PendulumEnv
import yaml
import gym


def train_ddpg_in_pendulum(gpu_id=0):  # DDPG is a simple but low sample efficiency and unstable.
    env_name = "Pendulum"
    alg = "DDPG"
    with open("config.yml", 'r') as f:
        hyp = yaml.safe_load(f)[alg][env_name]

    env = PendulumEnv()
    env_func = PendulumEnv
    env_args = get_gym_env_args(env, if_print=True)

    args = Arguments( env_func, env_args, hyp)

    train_and_evaluate(args)

def train_ddpg_in_lunar_lander(gpu_id=0):  # DDPG is a simple but low sample efficiency and unstable.
    env_name = "LunarLanderContinuous-v2"
    alg = "DDPG"
    
    with open("config.yml", 'r') as f:
        hyp = yaml.safe_load(f)[alg][env_name]
        
    env = gym.make(env_name)
    env_func = gym.make
    env_args = get_gym_env_args(env, if_print=True)

    args = Arguments(env_func, env_args, hyp)
    
    train_and_evaluate(args)


def train_ddpg_in_bipedal_walker(gpu_id=0):  # DDPG is a simple but low sample efficiency and unstable.
    env_name = "BipedalWalker-v3"
    alg = "DDPG"
    
    with open("config.yml", 'r') as f:
       hyp = yaml.safe_load(f)[alg][env_name]
       
    env = gym.make(env_name)
    env_func = gym.make
    env_args = get_gym_env_args(env, if_print=True)

    args = Arguments(env_func, env_args, hyp)

    train_and_evaluate(args)




if __name__ == "__main__":
    train_ddpg_in_pendulum()
    train_ddpg_in_lunar_lander()
    train_ddpg_in_bipedal_walker()

