from AgentRun import *
from AgentZoo import *
from AgentNet import *


def train__car_racing(gpu_id=None, random_seed=0):
    import AgentZoo as Zoo

    '''DEMO 4: Fix gym Box2D env CarRacing-v0 (pixel-level 2D-state, continuous action) using PPO'''
    import gym  # gym of OpenAI is not necessary for ElegantRL (even RL)
    gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'
    env = gym.make('CarRacing-v0')
    env = fix_car_racing_env(env)

    args = Arguments(rl_agent=Zoo.AgentPPO, env=env, gpu_id=gpu_id)
    args.if_break_early = True
    args.random_seed += random_seed
    args.eval_times2 = 1
    args.eval_times2 = 3  # CarRacing Env is so slow. The GPU-util is low while training CarRacing.

    args.break_step = int(5e5 * 4)  # (2e5) 5e5, used time 25000s
    args.reward_scale = 2 ** -2  # (-1) 50 ~ 900 (1001)
    args.max_memo = 2 ** 11
    args.batch_size = 2 ** 7
    args.repeat_times = 2 ** 4
    args.net_dim = 2 ** 7
    args.max_step = 2 ** 10
    args.show_gap = 2 ** 8  # for Recorder
    args.init_for_training()
    train_agent_mp(args)  # train_agent(args)
    exit()


train__car_racing()
