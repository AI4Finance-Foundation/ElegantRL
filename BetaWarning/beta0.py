from AgentRun import *


def run_continuous_action(gpu_id=None):
    import AgentZoo as Zoo

    """online policy"""  # plan to check args.max_total_step
    args = Arguments(rl_agent=Zoo.AgentGAE, gpu_id=gpu_id)
    assert args.rl_agent in {Zoo.AgentPPO, Zoo.AgentGAE}
    """PPO and GAE is online policy.
    The memory in replay buffer will only be saved for one episode.

    TRPO's author use a surrogate object to simplify the KL penalty and get PPO.
    So I provide PPO instead of TRPO here.

    GAE is Generalization Advantage Estimate.
    RL algorithm that use advantage function (such as A2C, PPO, SAC) can use this technique.
    AgentGAE is a PPO using GAE and output log_std of action by an actor network.
    """

    args.max_memo = 2 ** 12
    args.repeat_times = 2 ** 4
    args.batch_size = 2 ** 9
    args.net_dim = 2 ** 8

    args.env_name = "LunarLanderContinuous-v2"
    args.max_total_step = int(4e5 * 4)
    args.init_for_training()
    train_agent(**vars(args))

    args.env_name = "Pendulum-v0"  # It is easy to reach target score -200.0 (-100 is harder)
    args.max_total_step = int(1e5 * 4)
    args.max_memo = 2 ** 11
    args.repeat_times = 2 ** 3
    args.batch_size = 2 ** 8
    args.net_dim = 2 ** 7
    args.reward_scale = 2 ** -1
    args.init_for_training()
    train_agent(**vars(args))
    exit()


if __name__ == '__main__':
    run_continuous_action()
