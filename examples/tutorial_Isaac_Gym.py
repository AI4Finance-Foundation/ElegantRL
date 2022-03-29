# When using Isaac Gym, isaacgym must be imported before torch.
import isaacgym
import os
import torch
from elegantrl.agents import AgentPPO
from elegantrl.train.config import Arguments
from elegantrl.envs.IsaacGym import IsaacVecEnv, IsaacOneEnv
from elegantrl.envs.utils.config_utils import get_isaac_env_args
from elegantrl.train.run import train_and_evaluate_mp

# Choose an environment by name. If you want to see what's available, just put a random
# string here and run the code. :)
env_name = "Ant"

# Establish CUDA_LAUNCH_BLOCKING so we can see proper CUDA tracebacks if an error
# occurs.
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Define the training function and training arguments.
env_func = IsaacVecEnv
env_args = get_isaac_env_args(env_name)

# Construct a new set of arguments with the desired agent. Note that all Isaac Gym
# environments use continuous action spaces, so your agent must account for this.
args = Arguments(agent=AgentPPO, env_func=env_func, env_args=env_args)

# Define the evaluator function and evaluator arguments. Note that the evaluator is
# just the training environment, but not vectorized (i.e. a single environment).
args.eval_env_func = IsaacOneEnv
args.eval_env_args = args.env_args.copy()
args.eval_env_args["env_num"] = 1

# Change any arguments you'd like here...
args.net_dim = 512
args.batch_size = 2048
args.target_step = args.max_step
args.repeat_times = 16

args.save_gap = 512
args.eval_gap = 256
args.eval_times1 = 1
args.eval_times2 = 4

args.worker_num = 1
args.learner_gpus = 0

# ...and train!
if __name__ == "__main__":
    train_and_evaluate_mp(args)
