import gym
from elegantrl.agent import AgentModSAC
from elegantrl.config import get_gym_env_args, Arguments
from elegantrl.run import *

gym.logger.set_level(40)  # Block warning

get_gym_env_args(gym.make("LunarLanderContinuous-v2"), if_print=False)

env_func = gym.make
env_args = {
    "env_num": 1,
    "env_name": "LunarLanderContinuous-v2",
    "max_step": 1000,
    "state_dim": 8,
    "action_dim": 2,
    "if_discrete": False,
    "target_return": 200,
    "id": "LunarLanderContinuous-v2",
}
args = Arguments(AgentModSAC, env_func=env_func, env_args=env_args)

args.target_step = args.max_step
args.gamma = 0.99
args.eval_times = 2**5
args.random_seed = 2022

train_and_evaluate(args)
