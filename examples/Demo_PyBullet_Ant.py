from elegantrl.agents.AgentSAC import AgentModSAC
from elegantrl.train.run import train_and_evaluate
from elegantrl.train.config import Arguments
import gym
import pybullet_envs
from elegantrl.envs.Gym import get_gym_env_args

dir(pybullet_envs)
get_gym_env_args(gym.make('AntBulletEnv-v0'), if_print=True)

env_func = gym.make
env_args = {
    'env_num': 1,
    'env_name': 'AntBulletEnv-v0',
    'max_step': 1000,
    'state_dim': 28,
    'action_dim': 8,
    'if_discrete': False,
    'target_return': 2500,
    'id': 'AntBulletEnv-v0',
}

args = Arguments(agent=AgentModSAC(), env_func=env_func, env_args=env_args)

args.agent.if_use_act_target = False
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

train_and_evaluate(args)  # multiple processing
