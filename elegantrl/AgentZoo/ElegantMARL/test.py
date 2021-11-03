from run import *
from agent import  AgentMADDPG
from env import PreprocessEnv
import gym
gym.logger.set_level(40) # Block warning    
env = mpe_make_env('simple_spread')
args = Arguments(if_on_policy=False)  # AgentSAC(), AgentTD3(), AgentDDPG()
args.agent = AgentMADDPG()
args.env = PreprocessEnv(env)
args.reward_scale = 2 ** -1  # RewardRange: -200 < -150 < 300 < 334
args.gamma = 0.95
args.marl=True
args.max_step = 100
args.n_agents = 3
args.rollout_num = 2# the number of rollout workers (larger is not always faster)
train_and_evaluate(args) # the training process will terminate once it reaches the target reward.
