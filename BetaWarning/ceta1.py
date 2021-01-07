from AgentRun import *


def train__demo():
    pass

    # '''DEMO 2: Standard gym env LunarLanderContinuous-v2 (continuous action) using ModSAC (Modify SAC, off-policy)'''
    # import gym  # gym of OpenAI is not necessary for ElegantRL (even RL)
    # gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'
    # env = gym.make("Pendulum-v0")
    # env = decorate_env(env, if_print=True)
    #
    # from AgentZoo import AgentModSAC
    # args = Arguments(rl_agent=AgentModSAC, env=env)
    # args.rollout_num = 2
    #
    # args.break_step = int(1e4 * 8)  # 1e4 means the average total training step of InterSAC to reach target_reward
    # args.reward_scale = 2 ** -2  # (-1800) -1000 ~ -200 (-50)
    # args.init_for_training()
    # # train_agent(args)  # Train agent using single process. Recommend run on PC.
    # train_agent_mp(args)  # Train using multi process. Recommend run on Server. Mix CPU(eval) GPU(train)
    # exit()

    '''DEMO 2: Standard gym env LunarLanderContinuous-v2 (continuous action) using ModSAC (Modify SAC, off-policy)'''
    import gym  # gym of OpenAI is not necessary for ElegantRL (even RL)
    gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'
    env = gym.make("BipedalWalker-v3")
    env = decorate_env(env, if_print=True)

    from AgentZoo import AgentModSAC
    args = Arguments(rl_agent=AgentModSAC, env=env)
    # args.rollout_num = 2  # todo ceta2
    args.rollout_num = 4  # todo ceta1

    args.break_step = int(2e5 * 8)  # (1e5) 2e5, used time 3500s
    args.reward_scale = 2 ** -1  # (-200) -140 ~ 300 (341)
    args.max_step = 2 ** 11  # todo beta3
    args.init_for_training()
    # train_agent(args)  # Train agent using single process. Recommend run on PC.
    train_agent_mp(args)  # Train using multi process. Recommend run on Server. Mix CPU(eval) GPU(train)
    exit()


if __name__ == '__main__':
    train__demo()
