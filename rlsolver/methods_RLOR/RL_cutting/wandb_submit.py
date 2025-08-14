from src.helper import get_reward_sums, moving_avg
import wandb

wandb.login()
run=wandb.init(project="finalproject", entity="ieor-4575", tags=["training-easy"])
filepath = "records/train_10_n60_m60/idx_0_9/ppo_actor_dense_critic_dense_rnd_dense/20210419-231641.txt"
reward_sums = get_reward_sums(filepath)
reward_avgs = moving_avg(reward_sums, window_size =100)
for reward_sum, reward_avg in zip(reward_sums,reward_avgs):
    wandb.log({"training reward": reward_sum})
    wandb.log({"training reward moving average": reward_avg})



