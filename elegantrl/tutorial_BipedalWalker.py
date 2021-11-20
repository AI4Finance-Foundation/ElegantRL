from elegantrl.train.run_tutorial import *
from elegantrl.train.config import Arguments
from elegantrl.agents.AgentPPO import AgentPPO
from elegantrl.envs.gym import PreprocessEnv
import gym

gym.logger.set_level(40)  # Block warning

agent = AgentPPO()  # AgentSAC(), AgentTD3(), AgentDDPG()
env = PreprocessEnv(env=gym.make('BipedalWalker-v3'))
args = Arguments(agent, env)

args.reward_scale = 2 ** -1  # RewardRange: -200 < -150 < 300 < 334
args.gamma = 0.95
args.rollout_num = 2  # the number of rollout workers (larger is not always faster)

train_and_evaluate(args)
