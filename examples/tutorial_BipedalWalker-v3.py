import gym
from elegantrl.agent import AgentPPO
from elegantrl.config import get_gym_env_args, Arguments
from elegantrl.run import *

gym.logger.set_level(40)  # Block warning

get_gym_env_args(gym.make("BipedalWalker-v3"), if_print=True)

env_func = gym.make
env_args = {
    "env_num": 1,
    "env_name": "BipedalWalker-v3",
    "max_step": 1600,
    "state_dim": 24,
    "action_dim": 4,
    "if_discrete": False,
    "target_return": 300,
    "id": "BipedalWalker-v3",
}
args = Arguments(AgentPPO, env_func=env_func, env_args=env_args)

args.target_step = args.max_step * 4
args.gamma = 0.98
args.eval_times = 2**4

flag = "SingleProcess"

if flag == "SingleProcess":
    args.learner_gpus = 0
    train_and_evaluate(args)
elif flag == "MultiProcess":
    args.learner_gpus = 0
    train_and_evaluate_mp(args)
elif flag == "MultiGPU":
    args.learner_gpus = [0, 1, 2, 3]
    train_and_evaluate_mp(args)
elif flag == "Tournament-based":
    args.learner_gpus = [
        [
            i,
        ]
        for i in range(4)
    ]  # [[0,], [1, ], [2, ]] or [[0, 1], [2, 3]]
    python_path = "../bin/python3"
    train_and_evaluate_mp(args, python_path)  # multiple processing
else:
    raise ValueError(f"Unknown flag: {flag}")
