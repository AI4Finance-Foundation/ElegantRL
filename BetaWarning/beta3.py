from AgentRun import *
from AgentZoo import *
from AgentNet import *

"""
ModPPO use_dn=True
beta0 BW 256
beta1 BW 128
ceta0 Mini 256
ceta1 Mini 128

ModPPO StdPPO
beta2
"""


def run__discrete_action():
    import AgentZoo as Zoo
    args = Arguments(rl_agent=None, env_name=None, gpu_id=None)
    args.rl_agent = [
        Zoo.AgentDQN,  # 2014.
        Zoo.AgentDoubleDQN,  # 2016. stable
        Zoo.AgentDuelingDQN,  # 2016. stable and fast
        Zoo.AgentD3QN,  # 2016+ Dueling + Double DQN (Not a creative work)
    ][3]  # I suggest to use D3QN
    args.random_seed += 412

    # args.env_name = "CartPole-v0"
    # args.break_step = int(1e4 * 8)  # (3e5) 1e4, used time 20s
    # args.reward_scale = 2 ** 0  # 0 ~ 200
    # args.net_dim = 2 ** 6
    # args.init_for_training()
    # train_agent_mp(args)  # train_agent(args)
    # exit()

    args.env_name = "LunarLander-v2"
    args.break_step = int(1e5 * 8)  # (5e4) 1e5 (3e5), used time (355s) 1000s (2000s)
    args.reward_scale = 2 ** -1  # (-1000) -150 ~ 200 (285)
    args.net_dim = 2 ** 7
    args.init_for_training()
    train_agent_mp(args)  # train_agent(args)
    exit()


run__discrete_action()
