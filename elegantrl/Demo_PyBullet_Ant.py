from elegantrl.train.run_tutorial import *
from elegantrl.train.config import Arguments
from elegantrl.agents.AgentSAC import AgentModSAC

# demo for MuJoCo environment, using 1 GPU
args = Arguments(env=build_env('AntBulletEnv-v0', if_print=True), agent=AgentModSAC())
GPU_ID = 0

args.learner_gpus = (GPU_ID, )
args.agent.if_use_act_target = False
args.net_dim = 2 ** 9
args.max_memo = 2 ** 22
args.repeat_times = 2 ** 1
args.reward_scale = 2 ** -2
args.batch_size = args.net_dim * 2
args.target_step = args.env.max_step * 2

args.eval_gap = 2 ** 8
args.eval_times1 = 2 ** 1
args.eval_times2 = 2 ** 4
args.break_step = int(8e7)
args.if_allow_break = False

args.worker_num = 4

train_and_evaluate(args)
