from elegantrl.train.run_tutorial import *
from elegantrl.train.config import Arguments
from elegantrl.agents.AgentTD3 import AgentTD3
from elegantrl.envs.Gym import build_env
import gym

gym.logger.set_level(40)  # Block warning


# demo for continuous action space + off policy algorithms
agent = AgentTD3()
env = build_env('BipedalWalker-v3')
args = Arguments(env, agent)

args.eval_times1 = 2 ** 3
args.eval_times2 = 2 ** 5

args.gamma = 0.98
args.target_step = args.env.max_step

train_and_evaluate(args)
