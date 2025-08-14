import torch
import torch.nn as nn
import numpy as np
from functorch import vmap

class MIMOEnv():
    def __init__(self, N=4, episode_length=6, num_env=4096, device=torch.device("cuda:0")):
        self.N = N
        self.device = device
        self.num_env = num_env
        self.episode_length = episode_length
        self.epsilon = 1
        self.test_dim = torch.randint(1,10,(1,5), device=self.device).to(torch.float32)
        self.test_dim[0, 2:] = 2.
        self.test_dim[0, 0] = 2.
        self.test_dim[0, 1] = 10.



    def reset(self, test=False):
        self.dim = torch.randint(1,10, (self.num_env, 5), device=self.device).to(torch.float32)
        if test:
            self.dim = self.test_dim
        self.num_steps = 0
        self.done = False
        return self.dim

    def step(self, action ):
        reward = (self.dim[:, 0] * self.dim[:,1] * self.dim[:,2] + self.dim[:,0] * self.dim[:,2] * self.dim[:,3] * self.dim[:,4]) * action[:,0] + \
                (self.dim[:, 1] * self.dim[:,2] * self.dim[:,3] * self.dim[:,4] + self.dim[:,0] * self.dim[:,1] * self.dim[:,3] * self.dim[:,4]) * action[:,1]
        self.reward = reward
        self.num_steps += 1
        self.done = True if self.num_steps >= self.episode_length else False
        return self.dim, self.reward.mean(), self.done, reward.detach()

class Policy_Net_MIMO(nn.Module):
    def __init__(self, mid_dim=1024, N=4, ):
        super(Policy_Net_MIMO, self).__init__()
        self.N = N
        self.action_dim = 2
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mid_dim = mid_dim
        self.net = nn.Sequential(
        nn.Linear(5, mid_dim * 2),
        nn.Linear(mid_dim * 2, mid_dim * 2), nn.ReLU(),
        nn.Linear(mid_dim * 2, mid_dim * 2), nn.ReLU(),
        nn.Linear(mid_dim * 2, self.action_dim),
        )
        self.output_layer = nn.Sigmoid().to(self.device)

    def forward(self, state):
        action = self.output_layer(self.net(state))
        action = action / action.sum(dim=-1, keepdim=True)
        return action

def train_curriculum_learning(policy_net_mimo, optimizer, device, N=4,  num_epochs=100000000, num_env=512):
    env_mimo_relay = MIMOEnv(N=N, device=device, num_env=num_env, episode_length=1)
    for epoch in range(num_epochs):
        test = False
        if epoch % 2 == 0:
            test = True
        state = env_mimo_relay.reset(test)
        loss = 0
        while(1):
            action = policy_net_mimo(state)
            next_state, reward, done, _ = env_mimo_relay.step(action)
            loss += reward
            state = next_state
            if done and test == False:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                break
            if done and test == True:
                print(epoch, env_mimo_relay.reward[0], action[0])
                wandb.log({"flops":env_mimo_relay.reward[0]})
                break


if __name__  == "__main__":
    N = 4

    mid_dim = 256
    learning_rate = 5e-5

    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy_net_mimo = Policy_Net_MIMO(mid_dim= mid_dim, N=N).to(device)
    optimizer = torch.optim.Adam(policy_net_mimo.parameters(), lr=learning_rate)
    import wandb
    wandb.init(
        project='classical_simulation',
        entity="beamforming",
        sync_tensorboard=True,
    )
    train_curriculum_learning(policy_net_mimo, optimizer,N=N,  device=device, )