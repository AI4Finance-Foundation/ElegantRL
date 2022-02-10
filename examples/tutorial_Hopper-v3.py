import gym
from elegantrl.agent import AgentPPO
from elegantrl.config import get_gym_env_args, Arguments
from elegantrl.run import *

# set environment name here (e.g. 'Hopper-v3', 'LunarLanderContinuous-v2', 'BipedalWalker-v3')
env_name = "Hopper-v3"

# retrieve appropriate training arguments for this environment
env_args = get_gym_env_args(gym.make(env_name), if_print=False)
args = Arguments(AgentPPO, env_func=gym.make, env_args=env_args)

# set/modify any arguments you'd like to here
args.batch_size = 2**16
args.eval_times = 2**4
args.max_memo = 2**16
args.target_step = 2**16

# print out arguments in an easy-to-read format to show you what you're about to train...
args.print()

# ...and go!
train_and_evaluate(args)
