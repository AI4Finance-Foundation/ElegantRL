import gym
import tianshou as ts
from model import *
import warnings
warnings.filterwarnings('ignore')

# config
type = 'Easy'
numTestInstance = 1000
noise = 0.0
step_per_epoch = 30000
episode_per_collect = 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# environment
# to use your own environment, you need to follow specification and register your environment in gym
# train_envs = gym.make('Your environment', ...)
# test_envs = gym.make('Your environment', ...)
train_envs = gym.make('CuttingStockProblem-v0', instance_attrs=type)
test_envs = gym.make('CuttingStockProblem-v0', instance_attrs=type,
                     numTestInstance=numTestInstance, rootPath='eval')  # first unzip eval.zip

# model
graph_encoder = Graph_encoder()
instance_encoder = Instance_encoder()
actor_midlayer = Actor_midlayer()
actor_decoder = Actor_decoder()
critic_decoder = Critic_decoder()
actor = Actor_net(
    graph_encoder=graph_encoder,
    instance_encoder=instance_encoder,
    actor_decoder=actor_decoder,
    actor_midlayer=actor_midlayer,
    noise=noise
).to(device)
critic = Critic_net(
    graph_encoder=graph_encoder,
    instance_encoder=instance_encoder,
    critic_decoder=critic_decoder
).to(device)
params = list(graph_encoder.parameters()) + \
         list(instance_encoder.parameters()) + \
         list(actor_midlayer.parameters()) + \
         list(actor_decoder.parameters()) + \
         list(critic_decoder.parameters())
optimizer = torch.optim.Adam(params, lr=1e-3)

# ppo policy
policy = ts.policy.PPOPolicy(actor, critic, optimizer, dist_fn=torch.distributions.Categorical, discount_factor=0.9)

# ts collector
train_collector = ts.data.Collector(policy, train_envs, ts.data.ReplayBuffer(1024), exploration_noise=True)
test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)

# training
result = ts.trainer.onpolicy_trainer(
    policy, train_collector, test_collector,
    max_epoch=500, step_per_epoch=step_per_epoch, episode_per_collect=episode_per_collect,
    repeat_per_collect=1, episode_per_test=numTestInstance, batch_size=1024,
    save_best_fn=lambda policy: torch.save(policy.state_dict(), 'ckpt/ppo.pth'))
print(f'Finished training! Use {result["duration"]}')
