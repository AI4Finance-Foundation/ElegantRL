import argparse
import random
import copy
import os
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import Distribution

from AgentRun import get_env_info

"""
Reference: https://github.com/TianhongDai/reinforcement-learning-algorithms/tree/master/rl_algorithms/sac
Modify: Yonv1943 Zen4 Jia1Hao2
"""


def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--env-name', type=str, default='HalfCheetah-v2', help='the environment name')
    parse.add_argument('--cuda', default=True, action='store_true', help='use GPU do the training')
    parse.add_argument('--seed', type=int, default=123, help='the random seed to reproduce results')
    parse.add_argument('--hidden-size', type=int, default=256, help='the size of the hidden layer')
    parse.add_argument('--train-loop-per-epoch', type=int, default=1, help='the training loop per epoch')
    parse.add_argument('--q-lr', type=float, default=3e-4, help='the learning rate')
    parse.add_argument('--p-lr', type=float, default=3e-4, help='the learning rate of the actor')
    parse.add_argument('--n-epochs', type=int, default=int(3e3), help='the number of total epochs')
    parse.add_argument('--epoch-length', type=int, default=int(1e3), help='the lenght of each epoch')
    parse.add_argument('--n-updates', type=int, default=int(1e3), help='the number of training updates execute')
    parse.add_argument('--init-exploration-steps', type=int, default=int(1e3),
                       help='the steps of the initial exploration')
    parse.add_argument('--init-exploration-policy', type=str, default='gaussian', help='the inital exploration policy')
    parse.add_argument('--buffer-size', type=int, default=int(1e6), help='the size of the replay buffer')
    parse.add_argument('--batch-size', type=int, default=256, help='the batch size of samples for training')
    parse.add_argument('--reward-scale', type=float, default=1, help='the reward scale')
    parse.add_argument('--gamma', type=float, default=0.99, help='the discount factor')
    parse.add_argument('--log-std-max', type=float, default=2, help='the maximum log std value')
    parse.add_argument('--log-std-min', type=float, default=-20, help='the minimum log std value')
    parse.add_argument('--entropy-weights', type=float, default=0.2, help='the entropy weights')
    parse.add_argument('--tau', type=float, default=5e-3, help='the soft update coefficient')
    parse.add_argument('--target-update-interval', type=int, default=1, help='the interval to update target network')
    parse.add_argument('--update-cycles', type=int, default=int(1e3), help='how many updates apply in the update')
    parse.add_argument('--eval-episodes', type=int, default=10, help='the episodes that used for evaluation')
    parse.add_argument('--display-interval', type=int, default=1, help='the display interval')
    parse.add_argument('--save-dir', type=str, default='saved_models/', help='the place to save models')
    parse.add_argument('--reg', type=float, default=1e-3, help='the reg term')
    parse.add_argument('--auto-ent-tuning', action='store_true', help='tune the entorpy automatically')
    parse.add_argument('--log-dir', type=str, default='logs/', help='dir to save log information')
    parse.add_argument('--env-type', type=str, default=None, help='environment type')

    return parse.parse_args()


class TanhNormal(Distribution):
    def __init__(self, normal_mean, normal_std, epsilon=1e-6, cuda=False):
        super(object, self).__init__()
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.cuda = cuda
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.normal.sample_n(n)
        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def log_prob(self, value, pre_tanh_value=None):
        """
        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        if pre_tanh_value is None:
            pre_tanh_value = torch.log((1 + value) / (1 - value)) / 2
        return self.normal.log_prob(pre_tanh_value) - torch.log(1 - value * value + self.epsilon)

    def sample(self, return_pretanh_value=False):
        """
        Gradients will and should *not* pass through this operation.
        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.normal.sample().detach()
        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def rsample(self, return_pretanh_value=False):
        """
        Sampling in the reparameterization case.
        """
        sample_mean = torch.zeros(self.normal_mean.size(), dtype=torch.float32, device='cuda' if self.cuda else 'cpu')
        sample_std = torch.ones(self.normal_std.size(), dtype=torch.float32, device='cuda' if self.cuda else 'cpu')
        z = (self.normal_mean + self.normal_std * Normal(sample_mean, sample_std).sample())
        z.requires_grad_()
        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)


class ActionInfo:
    def __init__(self, pis, cuda=False):
        self.mean, self.std = pis
        self.dist = TanhNormal(normal_mean=self.mean, normal_std=self.std, cuda=cuda)

    # select actions
    def select_actions(self, exploration=True, reparameterize=True):
        if exploration:
            if reparameterize:
                actions, pretanh = self.dist.rsample(return_pretanh_value=True)
                return actions, pretanh
            else:
                actions = self.dist.sample()
        else:
            actions = torch.tanh(self.mean)
        return actions

    def get_log_prob(self, actions, pre_tanh_value):
        log_prob = self.dist.log_prob(actions, pre_tanh_value=pre_tanh_value)
        return log_prob.sum(dim=1, keepdim=True)



class ActorSAC(nn.Module):
    def __init__(self, input_dims, action_dims, hidden_size, log_std_min, log_std_max):
        super(ActorSAC, self).__init__()
        self.fc1 = nn.Linear(input_dims, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, action_dims)
        self.log_std = nn.Linear(hidden_size, action_dims)
        # the log_std_min and log_std_max
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        # clamp the log std
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        # the reparameterization trick
        # return mean and std
        return (mean, torch.exp(log_std))


class ReplayBuffer:
    def __init__(self, memory_size):
        self.storge = []
        self.memory_size = memory_size
        self.next_idx = 0

    # add the samples
    def add(self, obs, action, reward, obs_, done):
        data = (obs, action, reward, obs_, done)
        if self.next_idx >= len(self.storge):
            self.storge.append(data)
        else:
            self.storge[self.next_idx] = data
        # get the next idx
        self.next_idx = (self.next_idx + 1) % self.memory_size

    # encode samples
    def _encode_sample(self, idx):
        obses, actions, rewards, obses_, dones = [], [], [], [], []
        for i in idx:
            data = self.storge[i]
            obs, action, reward, obs_, done = data
            obses.append(np.array(obs, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_.append(np.array(obs_, copy=False))
            dones.append(done)
        return np.array(obses), np.array(actions), np.array(rewards), np.array(obses_), np.array(dones)

    # sample from the memory
    def sample(self, batch_size):
        idxes = [random.randint(0, len(self.storge) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class AgentSAC:
    def __init__(self, env, args):
        self.args = args
        self.env = env

        state_dim, action_dim, max_action, target_reward = get_env_info(env)
        net_dim = args.hidden_size
        use_densenet = False
        use_spectral_norm = False
        # create eval environment
        self.eval_env = gym.make(self.args.env_name)
        self.eval_env.seed(args.seed * 2)

        from AgentNetwork import Critic, CriticAdvantage
        # build up the network that will be used.
        self.qf1 = Critic(state_dim, action_dim, net_dim, use_densenet, use_spectral_norm)
        self.qf2 = Critic(state_dim, action_dim, net_dim, use_densenet, use_spectral_norm)
        # self.qf1 = CriticValue(self.env.observation_space.shape[0], self.args.hidden_size,
        #                        self.env.action_space.shape[0])
        # self.qf2 = CriticValue(self.env.observation_space.shape[0], self.args.hidden_size,
        #                        self.env.action_space.shape[0])

        # set the target q functions
        self.target_qf1 = copy.deepcopy(self.qf1)
        self.target_qf2 = copy.deepcopy(self.qf2)
        # build up the policy network
        self.actor_net = ActorSAC(self.env.observation_space.shape[0], self.env.action_space.shape[0],
                                  self.args.hidden_size, \
                                  self.args.log_std_min, self.args.log_std_max)
        # define the optimizer for them
        self.qf1_optim = torch.optim.Adam(self.qf1.parameters(), lr=self.args.q_lr)
        self.qf2_optim = torch.optim.Adam(self.qf2.parameters(), lr=self.args.q_lr)
        # the optimizer for the policy network
        self.actor_optim = torch.optim.Adam(self.actor_net.parameters(), lr=self.args.p_lr)
        # entorpy target
        self.target_entropy = -np.prod(self.env.action_space.shape).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device='cuda' if self.args.cuda else 'cpu')
        # define the optimizer
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.args.p_lr)
        # define the replay buffer
        self.buffer = ReplayBuffer(self.args.buffer_size)
        # get the action max
        self.action_max = self.env.action_space.high[0]
        # if use cuda, put tensor onto the gpu
        if self.args.cuda:
            self.actor_net.cuda()
            self.qf1.cuda()
            self.qf2.cuda()
            self.target_qf1.cuda()
            self.target_qf2.cuda()
        # automatically create the folders to save models
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        self.model_path = os.path.join(self.args.save_dir, self.args.env_name)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

    # train the agent
    def learn(self):
        global_timesteps = 0
        # before the official training, do the initial exploration to add episodes into the replay buffer
        self._initial_exploration(exploration_policy=self.args.init_exploration_policy)
        # reset the environment
        obs = self.env.reset()
        for epoch in range(self.args.n_epochs):
            for _ in range(self.args.train_loop_per_epoch):
                # for each epoch, it will reset the environment
                for t in range(self.args.epoch_length):
                    # start to collect samples
                    with torch.no_grad():
                        obs_tensor = self._get_tensor_inputs(obs)
                        pi = self.actor_net(obs_tensor)
                        action = ActionInfo(pi, cuda=self.args.cuda).select_actions(reparameterize=False)
                        action = action.cpu().numpy()[0]
                    # input the actions into the environment
                    obs_, reward, done, _ = self.env.step(self.action_max * action)
                    # store the samples
                    self.buffer.add(obs, action, reward, obs_, float(done))
                    # reassign the observations
                    obs = obs_
                    if done:
                        # reset the environment
                        obs = self.env.reset()
                # after collect the samples, start to update the network
                for _ in range(self.args.update_cycles):
                    qf1_loss, qf2_loss, actor_loss, alpha, alpha_loss = self._update_newtork()
                    # update the target network
                    if global_timesteps % self.args.target_update_interval == 0:
                        self._update_target_network(self.target_qf1, self.qf1)
                        self._update_target_network(self.target_qf2, self.qf2)
                    global_timesteps += 1
            # print the log information
            if epoch % self.args.display_interval == 0:
                # start to do the evaluation
                mean_rewards = self._evaluate_agent()
                print('E&R {:4} {:7.2f} |QF1&2 {:7.2f} {:7.2f} |AL {:7.3f} |Alpha&L {:7.3f} {:7.3f}'.format(
                    epoch, mean_rewards, qf1_loss, qf2_loss, actor_loss, alpha, alpha_loss))
                torch.save(self.actor_net.state_dict(), self.model_path + '/model.pt')

    # do the initial exploration by using the uniform policy
    def _initial_exploration(self, exploration_policy='gaussian'):
        # get the action information of the environment
        obs = self.env.reset()
        for _ in range(self.args.init_exploration_steps):
            if exploration_policy == 'uniform':
                raise NotImplementedError
            elif exploration_policy == 'gaussian':
                # the sac does not need normalize?
                with torch.no_grad():
                    obs_tensor = self._get_tensor_inputs(obs)
                    # generate the policy
                    pi = self.actor_net(obs_tensor)
                    action = ActionInfo(pi).select_actions(reparameterize=False)
                    action = action.cpu().numpy()[0]
                # input the action input the environment
                obs_, reward, done, _ = self.env.step(self.action_max * action)
                # store the episodes
                self.buffer.add(obs, action, reward, obs_, float(done))
                obs = obs_
                if done:
                    # if done, reset the environment
                    obs = self.env.reset()
        print("Initial exploration has been finished!")

    # get tensors
    def _get_tensor_inputs(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu').unsqueeze(0)
        return obs_tensor

    # update the network
    def _update_newtork(self):
        # smaple batch of samples from the replay buffer
        obses, actions, rewards, obses_, dones = self.buffer.sample(self.args.batch_size)
        # preprocessing the data into the tensors, will support GPU later
        obses = torch.tensor(obses, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        actions = torch.tensor(actions, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        rewards = torch.tensor(rewards, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu').unsqueeze(-1)
        obses_ = torch.tensor(obses_, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
        inverse_dones = torch.tensor(1 - dones, dtype=torch.float32,
                                     device='cuda' if self.args.cuda else 'cpu').unsqueeze(-1)
        # start to update the actor network
        pis = self.actor_net(obses)
        actions_info = ActionInfo(pis, cuda=self.args.cuda)
        actions_, pre_tanh_value = actions_info.select_actions(reparameterize=True)
        log_prob = actions_info.get_log_prob(actions_, pre_tanh_value)
        # use the automatically tuning
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        # get the param
        alpha = self.log_alpha.exp()
        # get the q_value for new actions
        q_actions_ = torch.min(self.qf1(obses, actions_), self.qf2(obses, actions_))
        actor_loss = (alpha * log_prob - q_actions_).mean()
        # q value function loss
        q1_value = self.qf1(obses, actions)
        q2_value = self.qf2(obses, actions)
        with torch.no_grad():
            pis_next = self.actor_net(obses_)
            actions_info_next = ActionInfo(pis_next, cuda=self.args.cuda)
            actions_next_, pre_tanh_value_next = actions_info_next.select_actions(reparameterize=True)
            log_prob_next = actions_info_next.get_log_prob(actions_next_, pre_tanh_value_next)
            target_q_value_next = torch.min(self.target_qf1(obses_, actions_next_),
                                            self.target_qf2(obses_, actions_next_)) - alpha * log_prob_next
            target_q_value = self.args.reward_scale * rewards + inverse_dones * self.args.gamma * target_q_value_next
        qf1_loss = (q1_value - target_q_value).pow(2).mean()
        qf2_loss = (q2_value - target_q_value).pow(2).mean()
        # qf1
        self.qf1_optim.zero_grad()
        qf1_loss.backward()
        self.qf1_optim.step()
        # qf2
        self.qf2_optim.zero_grad()
        qf2_loss.backward()
        self.qf2_optim.step()
        # policy loss
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        return qf1_loss.item(), qf2_loss.item(), actor_loss.item(), alpha.item(), alpha_loss.item()

    # update the target network
    def _update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

    # evaluate the agent
    def _evaluate_agent(self):
        total_reward = 0
        for _ in range(self.args.eval_episodes):
            obs = self.eval_env.reset()
            episode_reward = 0
            while True:
                with torch.no_grad():
                    obs_tensor = self._get_tensor_inputs(obs)
                    pi = self.actor_net(obs_tensor)
                    action = ActionInfo(pi, cuda=self.args.cuda).select_actions(exploration=False,
                                                                                reparameterize=False)
                    action = action.detach().cpu().numpy()[0]
                # input the action into the environment
                obs_, reward, done, _ = self.eval_env.step(self.action_max * action)
                episode_reward += reward
                if done:
                    break
                obs = obs_
            total_reward += episode_reward
        return total_reward / self.args.eval_episodes


if __name__ == '__main__':
    args = get_args()
    args.env_name = "LunarLanderContinuous-v2"
    env = gym.make(args.env_name)
    sac_trainer = AgentSAC(env, args)
    sac_trainer.learn()
