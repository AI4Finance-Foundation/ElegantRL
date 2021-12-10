from elegantrl.train.run_tutorial import *
from elegantrl.train.config import Arguments
from elegantrl.agents.AgentPPO import AgentPPO
from elegantrl.envs.Gym import build_env


# demo for MuJoCo environments, using 1 GPU
agent = AgentPPO()
env = build_env('Ant-v3')

args = Arguments(env, agent)
GPU_ID = 0

args.learner_gpus = (GPU_ID, )
args.eval_times1 = 2 ** 3
args.eval_times2 = 2 ** 5
args.target_step = args.env.max_step * 4

train_and_evaluate(args)
