import matplotlib.pyplot as plt
plt.style.use('seaborn')

import seaborn as sns
import numpy as np

sns.set(style='darkgrid')
import time

from src.helper import plot_arr, get_reward_sums


def plot_rewards(filepaths, labels = None):
    plt.figure(figsize=(15, 10), dpi=80)
    for i, filepath in enumerate(filepaths):
        reward_sums = get_reward_sums(filepath)[:3500]
        if labels == None:
            label = filepath
        else:
            label = labels[i]
        plot_arr(reward_sums, label=label, window_size=100)
    plt.legend()
    curr_time = time.strftime("%Y%m%d-%H%M%S")

    plt.xlabel("trajectory")
    plt.ylabel("Reward Summation")

    plt.savefig(f"figures/{curr_time}.jpeg")
def main():
    filepaths = ["records/train_10_n60_m60/idx_0_9/ppo_actor_dense_critic_dense_rnd_dense/20210419-231641.txt",
                 "records/train_100_n60_m60/idx_0_98/ppo_actor_dense_critic_dense_rnd_dense/20210420-222013.txt"]
    labels = ["PPO w RND (Dense Actor, Dense Critic) [Easy Config]",
              "PPO w RND (Dense Actor, Dense Critic) [Hard Config]"]
    plot_rewards(filepaths, labels)


    # plot the mean rewards

    reward_random_sums = get_reward_sums("records/train_10_n60_m60/idx_0_9/random/.txt")[:300]

    random_mean = np.mean(reward_random_sums)
    print(random_mean)
    plt.plot([0, 3500], [random_mean, random_mean], label = "Random Policy Average Reward [Easy Config]")

    reward_random_sums = get_reward_sums("records/train_100_n60_m60/idx_0_98/actor_random_critic_None_rnd_None/20210423-112825.txt")[:300]
    random_mean = np.mean(reward_random_sums)
    print(random_mean)
    plt.plot([0, 3500], [random_mean, random_mean], label="Random Policy Average Reward (Hard Config)")

    plt.legend()
    curr_time = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(f"figures/{curr_time}.jpeg")
    plt.show()

    ##### TEST PLOT

    filepaths = ["records/test_100_n60_m60/idx_0_9/ppo_actor_dense_critic_dense_rnd_dense/20210423-140347.txt",
                 "records/test_100_n60_m60/idx_0_9/ppo_actor_dense_critic_dense_rnd_dense/20210423-155948.txt",
                 "records/test_100_n60_m60/idx_0_9/actor_random_critic_None_rnd_None/20210423-141840.txt",
                 ]
    labels = ["PPO w RND (Dense Actor, Dense Critic) [Easy Config]",
              "PPO w RND (Dense Actor, Dense Critic) [Hard Config]",
              "Random Policy"]
    plot_rewards(filepaths, labels)
    plt.show()







if __name__ == '__main__':
    main()



