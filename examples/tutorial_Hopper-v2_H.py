import sys

from elegantrl.train.demo import *


def demo_continuous_action_on_policy():
    gpu_id = (
        int(sys.argv[1]) if len(sys.argv) > 1 else 0
    )  # >=0 means GPU ID, -1 means CPU
    drl_id = 1  # int(sys.argv[2])
    env_id = 4  # int(sys.argv[3])

    env_name = "Hopper-v2"
    agent = AgentPPO_H

    print("agent", agent.__name__)
    print("gpu_id", gpu_id)
    print("env_name", env_name)

    env_func = gym.make
    env_args = {
        "env_num": 1,
        "env_name": "Hopper-v2",
        "max_step": 1000,
        "state_dim": 11,
        "action_dim": 3,
        "if_discrete": False,
        "target_return": 3800.0,
    }
    args = Arguments(agent, env_func=env_func, env_args=env_args)
    args.eval_times = 2**1
    args.reward_scale = 2**-4

    args.target_step = args.max_step * 4  # 6
    args.worker_num = 2

    args.net_dim = 2**7
    args.layer_num = 3
    args.batch_size = int(args.net_dim * 2)
    args.repeat_times = 2**4
    args.ratio_clip = 0.25
    args.gamma = 0.993
    args.lambda_entropy = 0.02
    args.lambda_h_term = 2**-5

    args.if_allow_break = False
    args.break_step = int(8e6)

    args.learner_gpus = gpu_id
    args.random_seed += gpu_id

    if_check = 0
    if if_check:
        train_and_evaluate(args)
    else:
        train_and_evaluate_mp(args)


if __name__ == "__main__":
    demo_continuous_action_on_policy()
