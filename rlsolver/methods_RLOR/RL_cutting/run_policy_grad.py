from tqdm import tqdm
import numpy as np

# import wandb
# wandb.login()
# run=wandb.init(project="finalproject", entity="ieor-4575", tags=["n=10,m=20,i=1"])


from config import gen_actor_params, gen_critic_params, env_configs, gen_rnd_params
from env.gymenv_v2 import make_multiple_env
from src.rollout_gen import RolloutGenerator
from src.helper import plot_arr
from src.logger import RewardLogger
from models.build_actor_critic import build_actor, build_critic
from models.build_rnd import build_rnd

def policy_grad(env_config,
                policy_params,
                critic_params,
                rnd_params,
                iterations=150,
                num_processes=8,
                num_trajs_per_process=1,
                gamma=0.99,
                intrinsic_gamma=0.99
                ):
    hyperparams = {"iterations": iterations,
                   "num_processes": num_processes,
                   "num_trajs_per_process": num_trajs_per_process,
                   "gamma": gamma,
                   "intrinsic_gamma" : intrinsic_gamma
                   }
    logger = RewardLogger(env_config, policy_params, critic_params, rnd_params, hyperparams)
    rrecord = []

    env = make_multiple_env(**env_config)
    actor = build_actor(policy_params)
    critic = build_critic(critic_params)

    rollout_gen = RolloutGenerator(num_processes, num_trajs_per_process, verbose=True)
    rnd = build_rnd(rnd_params)

    for _ in tqdm(range(iterations)):
        memory = rollout_gen.generate_trajs(env, actor, rnd, gamma, intrinsic_gamma)

        # TODO: make a actor-critic policy gradient
        critic.train(memory)
        baselines = critic.compute_values(memory.states)
        memory.advantages = np.array(memory.values) - baselines + memory.intrinsic_values
        memory.advantages = (memory.advantages - np.mean(memory.advantages)) \
                            / (np.std(memory.advantages) + 1e-8)

        rnd.train(memory)
        actor.train(memory)
        # log results
        rrecord.extend(memory.reward_sums)
        logger.record(memory.reward_sums) # writes record to a file

    plot_arr(rrecord, label="Moving Avg Reward " + policy_params['model'], window_size=101)
    plot_arr(rrecord, label="Reward " + policy_params['model'], window_size=1)


def main():
    env_config = env_configs.test_config
    policy_params = gen_actor_params.gen_rand_params()
    critic_params = gen_critic_params.gen_no_critic()
    rnd_params = gen_rnd_params.gen_no_rnd()
    #policy_params = gen_actor_params.gen_attention_params(n=25, h=32)
    #critic_params = gen_critic_params.gen_critic_dense(m=25, n=25, t=20, lr=0.001)
    # critic_params = gen_critic_params.gen_critic_dense(m=20, n=10, t=10, lr=0.001)

    # rnd_params = gen_rnd_params.gen_rnd_dense(m=25, n=25, t=20, lr=0.001)

    hyperparams = {"iterations": 1000,  # number of iterations to run policy gradient
                   "num_processes": 6,  # number of processes running in parallel
                   "num_trajs_per_process": 1,  # number of trajectories per process
                   "gamma": 0.99,  # discount factor
                   "intrinsic_gamma": 0.999  # intrinsic reward discount factor
                   }
    policy_grad(env_config,  # environment configuration
                policy_params,  # actor definition
                critic_params,
                rnd_params,
                **hyperparams
                )







if __name__ == '__main__':
    main()
