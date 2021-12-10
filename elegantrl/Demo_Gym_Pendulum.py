from elegantrl.train.run_tutorial import *
from elegantrl.train.config import Arguments
from elegantrl.agents.AgentPPO import AgentPPO
from elegantrl.envs.Gym import build_env
import gym

gym.logger.set_level(40)  # Block warning

# demo for continuous action space + on policy algorithms
agent = AgentPPO()
env = build_env('Pendulum-v1')
args = Arguments(env, agent)

args.gamma = 0.97
args.net_dim = 2 ** 8
args.worker_num = 2
args.reward_scale = 2 ** -2
args.target_step = 200 * 16  # max_step = 200

args.eval_gap = 2 ** 5

train_and_evaluate(args)  # the training process will terminate once it reaches the target reward.
