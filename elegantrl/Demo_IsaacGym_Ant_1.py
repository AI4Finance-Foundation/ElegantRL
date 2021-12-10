from elegantrl.train.run_parallel import *
from elegantrl.train.config import Arguments
from elegantrl.agents.AgentPPO import AgentPPO
from elegantrl.envs.Gym import build_env

agent = AgentPPO()
env = build_env('IsaacVecEnvAnt', env_num=4069, device_id=0)

args = Arguments(env, agent)
GPU_ID = (0,)

args.learner_gpus = GPU_ID
args.workers_gpus = args.learner_gpus
args.worker_num = 1
args.max_step = 1000
args.state_dim = 60
args.action_dim = 8
args.if_discrete = False
args.target_return = 4000

args.agent.lambda_entropy = 0.05
args.agent.lambda_gae_adv = 0.97
args.learning_rate = 2 ** -15
args.if_per_or_gae = True

args.reward_scale = 2 ** -2  # (-50) 0 ~ 2500 (3340)
args.repeat_times = 2 ** 3
args.net_dim = 2 ** 9
args.batch_size = args.net_dim * 2 ** 3
args.target_step = 2 ** 10

args.break_step = int(2e7)
args.if_allow_break = False

train_and_evaluate_mp(args)
