from elegantrl.train.run_tutorial import *
from elegantrl.train.config import Arguments
from elegantrl.agents.AgentDoubleDQN import AgentDoubleDQN
from elegantrl.envs.Gym import build_env
import gym

gym.logger.set_level(40)  # Block warning

agent = AgentDoubleDQN()
env = build_env('LunarLander-v2')
args = Arguments(env, agent)

args.max_memo = 2 ** 19
args.if_use_cri_target = True
args.agent.if_use_dueling = True  # using Dueling DQN trick
args.reward_scale = 2 ** -1
args.target_step = args.env.max_step

train_and_evaluate(args)  # the training process will terminate once it reaches the target reward.
