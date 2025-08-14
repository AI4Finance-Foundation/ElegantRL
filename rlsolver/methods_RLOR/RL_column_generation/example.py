"""
A toy example of the inference of network.
We use tianshou (https://tianshou.readthedocs.io/en/stable/) as the DRL framework,
and one can modify the format of input and output of the environment and network to match their framework.
"""

import pickle
import tianshou as ts
from model import *

data = [pickle.load(file=open('state/state_{:d}.pkl'.format(i), 'rb')) for i in range(8)]
# state data from CSP, batch_size = 8 as the example
# 'biGraph': dgl.heterograph(), 'fcGraph': dgl.graph, 'globalFeat': torch.tensor
data = ts.data.Batch(data)

# default network hyper-parameters
# modify the input dimensions to fit your features
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
    noise=0.0
)
critic = Critic_net(
    graph_encoder=graph_encoder,
    instance_encoder=instance_encoder,
    critic_decoder=critic_decoder
)

# Select 5 from 10, and the most negative reduced cost is always selected
# -> action_space = C(9, 4) = 126
action_distribution, _ = actor.forward(obs=data)  # (batch, action_space)
print(action_distribution)