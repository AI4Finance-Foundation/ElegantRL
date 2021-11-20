from elegantrl.train.run_tutorial import *
from elegantrl.train.config import Arguments
from elegantrl.agents.AgentDoubleDQN import AgentDoubleDQN
from elegantrl.envs.Gym import build_env
import gym

gym.logger.set_level(40)  # Block warning

agent = AgentDoubleDQN()
env = build_env('CartPole-v0')
args = Arguments(env, agent)

args.target_return = 195

args.reward_scale = 2 ** -1
args.target_step = args.env.max_step * 4

args.eval_gap = 2 ** 5

train_and_evaluate(args)  # the training process will terminate once it reaches the target reward.
