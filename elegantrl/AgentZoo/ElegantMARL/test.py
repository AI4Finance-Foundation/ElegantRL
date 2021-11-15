from run import *
from agent import  AgentMADDPG, AgentREDQ
from env import PreprocessEnv
import gym
gym.logger.set_level(40) # Block warning    
#env = mpe_make_env('simple_spread')
env = gym.make("Hopper-v2")
args = Arguments(if_off_policy=True)  # AgentSAC(), AgentTD3(), AgentDDPG()
args.agent = AgentREDQ()
args.env = PreprocessEnv(env)
args.reward_scale = 2 ** -1  # RewardRange: -200 < -150 < 300 < 334
args.gamma = 0.95
args.marl=False
args.max_step = 100
args.n_agents = 3
args.rollout_num = 2# the number of rollout workers (larger is not always faster)
train_and_evaluate(args) # the training process will terminate once it reaches the target reward.
