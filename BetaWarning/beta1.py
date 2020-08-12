from AgentRun import *


def run__dqn(gpu_id=None, cwd='RL__DDQN'):  # 2020-07-07
    import AgentZoo as Zoo
    # rl_agent = Zoo.AgentDuelingDQN
    rl_agent = Zoo.AgentDoubleDQN  # todo I haven't test DQN, DDQN after 2020-07-07

    assert rl_agent in {Zoo.AgentDQN, Zoo.AgentDoubleDQN, Zoo.AgentDuelingDQN}
    args = ArgumentsBeta(rl_agent=rl_agent, env_name="CartPole-v0", gpu_id=gpu_id, cwd=cwd)
    args.init_for_training()
    train_agent_discrete(**vars(args))

    args = ArgumentsBeta(rl_agent=rl_agent, env_name="LunarLander-v2", gpu_id=gpu_id, cwd=cwd)
    args.init_for_training()
    train_agent_discrete(**vars(args))

run__dqn()
