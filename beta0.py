import argparse
import random
import copy
import os
from abc import ABC

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions import Distribution

from AgentNetwork import LinearNet, layer_norm
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


class TanhNormal(Distribution, ABC):
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


class ActorSAC(nn.Module):  # no tanh
    def __init__(self, state_dim, action_dim, mid_dim, use_densenet):
        use_densenet += 1  # backup
        super(ActorSAC, self).__init__()
        self.log_std_max = 2.0
        self.log_std_min = -20.0

        self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                 LinearNet(mid_dim), )
        self.net__a_mean = nn.Linear(mid_dim, action_dim)
        self.net__log_std = nn.Linear(mid_dim, action_dim)

        # layer_norm(self.net[0], std=1.0)
        layer_norm(self.net__a_mean, std=0.01)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, s, noise_std=0.0):
        x = self.net(s)
        a_mean = self.net__a_mean(x)
        if noise_std:
            a_log_std = self.net__log_std(x).clamp(self.log_std_min, self.log_std_max)
            a_std = torch.exp(a_log_std)
            return a_mean, a_std
        else:  # noise_std == 0
            return a_mean


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
        learning_rate = 3e-4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_reward = target_reward
        self.env_name = self.args.env_name

        # create eval environment
        self.eval_env = gym.make(self.args.env_name)
        self.eval_env.seed(args.seed * 2)

        from AgentNetwork import Critic
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

        self.act = ActorSAC(state_dim, action_dim, net_dim, use_densenet)
        # self.actor_net = ActorSAC(self.env.observation_space.shape[0], self.env.action_space.shape[0],
        #                           self.args.hidden_size, \
        #                           self.args.log_std_min, self.args.log_std_max)

        # define the optimizer for them
        self.qf1_optim = torch.optim.Adam(self.qf1.parameters(), lr=learning_rate)
        self.qf2_optim = torch.optim.Adam(self.qf2.parameters(), lr=learning_rate)
        # the optimizer for the policy network
        self.actor_optim = torch.optim.Adam(self.act.parameters(), lr=learning_rate)
        # entorpy target
        self.target_entropy = -np.prod(self.env.action_space.shape).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device='cuda' if self.args.cuda else 'cpu')
        # define the optimizer
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=learning_rate)
        # define the replay buffer
        self.buffer = ReplayBuffer(self.args.buffer_size)
        # get the action max
        self.action_max = max_action
        # if use cuda, put tensor onto the gpu
        if self.args.cuda:
            self.act.cuda()
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

    def select_actions(self, states, explore_noise=0.0):  # CPU array to GPU tensor to CPU array
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = self.act(states)

        if explore_noise == 0.0:
            actions = actions.cpu().data.numpy()
            return actions
        else:
            a_noise, log_prob = self.act.get__a__log_prob(actions)
            a_noise = a_noise.cpu().data.numpy()

            log_prob = log_prob.cpu().data.numpy()

            q_value = self.act.critic(states)
            q_value = q_value.cpu().data.numpy()
            return a_noise, log_prob, q_value

    def save_or_load_model(self, mod_dir, is_save):
        act_save_path = '{}/actor.pth'.format(mod_dir)
        # cri_save_path = '{}/critic.pth'.format(mod_dir)

        if is_save:
            torch.save(self.act.state_dict(), act_save_path)
            # torch.save(self.cri.state_dict(), cri_save_path)
            # print("Saved act and cri:", mod_dir)
        elif os.path.exists(act_save_path):
            act_dict = torch.load(act_save_path, map_location=lambda storage, loc: storage)
            self.act.load_state_dict(act_dict)
            # self.act_target.load_state_dict(act_dict)
            # cri_dict = torch.load(cri_save_path, map_location=lambda storage, loc: storage)
            # self.cri.load_state_dict(cri_dict)
            # self.cri_target.load_state_dict(cri_dict)
        else:
            print("FileNotFound when load_model: {}".format(mod_dir))

    # train the agent
    def learn(self):
        from AgentRun import Recorder
        max_step = self.args.epoch_length
        max_action = self.action_max
        target_reward = self.target_reward
        env_name = self.env_name
        recorder = Recorder(self, max_step, max_action, target_reward, env_name)
        cwd = self.args.save_dir

        qf1_loss = qf2_loss = actor_loss = None
        # qf1_loss = qf2_loss = actor_loss = alpha = alpha_loss = None
        global_timesteps = 0
        # before the official training, do the initial exploration to add episodes into the replay buffer
        self._initial_exploration(exploration_policy=self.args.init_exploration_policy)
        # reset the environment
        state = self.env.reset()
        reward_sum = 0.0
        step = 0
        for epoch in range(self.args.n_epochs):

            rewards = list()
            steps = list()

            for _ in range(self.args.train_loop_per_epoch):
                # for each epoch, it will reset the environment
                for t in range(self.args.epoch_length):
                    # start to collect samples
                    with torch.no_grad():
                        obs_tensor = self._get_tensor_inputs(state)
                        pi = self.act(obs_tensor, True)
                        action = ActionInfo(pi, cuda=self.args.cuda).select_actions(reparameterize=False)
                        action = action.cpu().numpy()[0]
                    # input the actions into the environment
                    next_state, reward, done, _ = self.env.step(self.action_max * action)
                    step += 1
                    # store the samples
                    self.buffer.add(state, action, reward, next_state, float(done))
                    # reassign the observations
                    state = next_state
                    if done:
                        rewards.append(reward_sum)
                        steps.append(step)

                        # reset the environment
                        state = self.env.reset()
                        reward_sum = 0.0
                        step = 0

                # after collect the samples, start to update the network
                for _ in range(self.args.update_cycles):
                    qf1_loss, qf2_loss, actor_loss, alpha, alpha_loss = self._update_newtork()
                    # update the target network
                    if global_timesteps % self.args.target_update_interval == 0:
                        self._update_target_network(self.target_qf1, self.qf1)
                        self._update_target_network(self.target_qf2, self.qf2)
                    global_timesteps += 1
            # # print the log information
            # if epoch % self.args.display_interval == 0:
            #     # start to do the evaluation
            #     mean_rewards = self._evaluate_agent()
            #     print('E&R {:4} {:7.2f} |QF1&2 {:7.2f} {:7.2f} |AL {:7.3f} |Alpha&L {:7.3f} {:7.3f}'.format(
            #         epoch, mean_rewards, qf1_loss, qf2_loss, actor_loss, alpha, alpha_loss))
            #     torch.save(self.act.state_dict(), self.model_path + '/model.pt')
            with torch.no_grad():  # just the GPU memory
                # is_solved = recorder.show_and_check_reward(
                #     epoch, epoch_reward, iter_num, actor_loss, critic_loss, cwd)
                critic_loss = (qf1_loss + qf2_loss) * 0.5
                recorder.show_reward(epoch, rewards, steps, actor_loss, critic_loss)
                is_solved = recorder.check_reward(cwd, actor_loss, critic_loss)
                if is_solved:
                    break

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
                    pi = self.act(obs_tensor, True)
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
        # print("Initial exploration has been finished!")

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
        pis = self.act(obses, True)
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
            pis_next = self.act(obses_, True)
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
                    pi = self.act(obs_tensor, True)
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


def run__sac(gpu_id=0, cwd='AC_SAC'):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    args = get_args()
    args.env_name = "LunarLanderContinuous-v2"
    args.save_dir = './{}/LL_{}'.format(cwd, gpu_id)
    env = gym.make(args.env_name)
    sac_trainer = AgentSAC(env, args)
    sac_trainer.learn()
    env.close()


def run__sac_bw(gpu_id=0, cwd='AC_SAC'):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    args = get_args()
    args.env_name = "BipedalWalker-v3"
    args.save_dir = './{}/BW_{}'.format(cwd, gpu_id)
    env = gym.make(args.env_name)
    sac_trainer = AgentSAC(env, args)
    sac_trainer.learn()
    env.close()


if __name__ == '__main__':
    from AgentRun import run__multi_process
    from AgentRun import run__td3

    run__multi_process(run__td3, gpu_tuple=(0, 1,), cwd='AC_TD3')

    # run__sac(gpu_id=0, cwd='AC_SAC')
    run__multi_process(run__sac, gpu_tuple=(0, 1,), cwd='AC_SAC')
    run__multi_process(run__sac_bw, gpu_tuple=(0, 1,), cwd='AC_SAC')
