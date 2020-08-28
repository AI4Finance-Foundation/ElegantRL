from AgentRun import *

"""GymMultiWalkerEnv
"""


def run_continuous_action(gpu_id=None):
    import AgentZoo as Zoo
    """offline policy"""  # plan to check args.max_total_step
    args = Arguments(rl_agent=Zoo.AgentInterSAC, gpu_id=gpu_id)

    args.env_name = 'MultiWalker'
    args.max_total_step = int(5e5 * 8)
    args.reward_scale = 2 ** -1  # beta terminal
    args.init_for_training()
    train_agent(**vars(args))
    build_for_mp(args)
    exit()


run_continuous_action()
