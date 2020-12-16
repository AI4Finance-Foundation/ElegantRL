from AgentRun import *
from AgentZoo import *
from AgentNet import *


def run__demo():
    import AgentZoo as Zoo

    env = decorate_env("LunarLander-v2")

    # args = Arguments(rl_agent=Zoo.AgentDoubleDQN, env_name="LunarLander-v2", gpu_id=0)
    args = Arguments(rl_agent=Zoo.AgentD3QN, env="LunarLander-v2", gpu_id=0)
    args.break_step = int(1e5 * 8)  # used time 600s
    args.net_dim = 2 ** 7
    args.init_for_training()
    # train_agent_mp(args)
    train_agent(args)
    exit()

    # args = Arguments(rl_agent=Zoo.AgentSAC, env_name="LunarLanderContinuous-v2", gpu_id=0)
    args = Arguments(rl_agent=Zoo.AgentModSAC, env="LunarLanderContinuous-v2", gpu_id=0)
    args.break_step = int(5e5 * 8)  # used time 1500s
    args.net_dim = 2 ** 7
    args.init_for_training()
    train_agent_mp(args)  # train_agent(args)
    exit()


run__demo()
