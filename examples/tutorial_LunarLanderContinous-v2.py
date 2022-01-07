import gym

from elegantrl.agents.AgentSAC import AgentSAC
from elegantrl.envs.Gym import get_gym_env_args
from elegantrl.train.config import Arguments
from elegantrl.train.run import train_and_evaluate

get_gym_env_args(gym.make('LunarLanderContinuous-v2'), if_print=True)

env_func = gym.make
env_args = {
    'env_num': 1,
    'env_name': 'LunarLanderContinuous-v2',
    'max_step': 1000,
    'state_dim': 8,
    'action_dim': 4,
    'if_discrete': True,
    'target_return': 200,
    'id': 'LunarLanderContinuous-v2'
}

args = Arguments(agent=AgentSAC(), env_func=env_func, env_args=env_args)

args.net_dim = 2 ** 9
args.max_memo = 2 ** 22
args.repeat_times = 2 ** 1
args.reward_scale = 2 ** -2
args.batch_size = args.net_dim * 2
args.target_step = 2 * env_args['max_step']

args.eval_gap = 2 ** 8
args.eval_times1 = 2 ** 1
args.eval_times2 = 2 ** 4
args.break_step = int(8e7)
args.if_allow_break = False
args.worker_num = 1
args.learner_gpus = -1  # no GPU usage

train_and_evaluate(args)
