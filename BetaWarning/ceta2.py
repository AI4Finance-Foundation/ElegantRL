from AgentRun import *
from AgentNet import *
from AgentZoo import *


def run__discrete_action(gpu_id=None):
    import AgentZoo as Zoo

    """offline policy"""  # plan to check args.max_total_step
    args = Arguments(rl_agent=Zoo.AgentDoubleDQN, gpu_id=gpu_id)
    assert args.rl_agent in {
        Zoo.AgentDQN,  # 2014.
        Zoo.AgentDoubleDQN,  # 2016. stable
        Zoo.AgentDuelingDQN,  # 2016. fast
    }
    args.if_break_early = False

    # args.env_name = "CartPole-v0"
    # args.break_step = int(1e4 * 8)
    # args.reward_scale = 2 ** 0  # 0 ~ 198 (200)
    # args.net_dim = 2 ** 6
    # args.init_for_training()
    # train_agent_mp(args)  # train_agent(**vars(args))
    # exit()

    args.env_name = "LunarLander-v2"
    args.break_step = int(1e5 * 8)  # (-1000) -150 ~ 200 (250)
    args.net_dim = 2 ** 7
    args.init_for_training()
    train_agent_mp(args)  # train_agent(**vars(args))
    exit()


run__discrete_action()
