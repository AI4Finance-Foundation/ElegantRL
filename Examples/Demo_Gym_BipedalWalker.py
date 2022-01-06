from elegantrl.agents.AgentPPO import AgentPPO
from elegantrl.train.run import train_and_evaluate_mp
from elegantrl.train.config import Arguments
import gym

from elegantrl.envs.Gym import get_gym_env_args
get_gym_env_args(gym.make('BipedalWalker-v3'), if_print=True)

env_func = gym.make
env_args = {
    'env_num': 1,
    'env_name': 'BipedalWalker-v3',
    'max_step': 1600,
    'state_dim': 24,
    'action_dim': 4,
    'if_discrete': False,
    'target_return': 300,

    'id': 'BipedalWalker-v3',
}

args = Arguments(agent=AgentPPO(), env_func=env_func, env_args=env_args)

args.net_dim = 2 ** 8
args.batch_size = args.net_dim * 2
args.target_step = args.max_step * 2
args.worker_num = 4

args.save_gap = 2 ** 9
args.eval_gap = 2 ** 8
args.eval_times1 = 2 ** 4
args.eval_times2 = 2 ** 5
args.if_allow_break = False


args.learner_gpus = [i for i in range(4)]  # multi-GPU

train_and_evaluate_mp(args)  # multiple processing
