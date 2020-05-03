import os

import gym
import torch
import torch.nn as nn
import numpy as np
import numpy.random as rd

from AgentZoo import Recorder  # for train_agent_ppo()
from AgentRun import Arguments  # for run__ppo()
from AgentRun import get_env_info, draw_plot_with_npy  # for run__ppo()

'''AgentNetwork'''


class ActorCritic(nn.Module):
    def __init__(self, action_dim, critic_dim, mid_dim, layer_norm=True):
        super(ActorCritic, self).__init__()

        actor_fc1 = nn.Linear(action_dim, mid_dim)
        actor_fc2 = nn.Linear(mid_dim, mid_dim)
        actor_fc3 = nn.Linear(mid_dim, critic_dim)
        self.actor_fc = nn.Sequential(
            actor_fc1, HardSwish(),
            actor_fc2, HardSwish(),
            actor_fc3,
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, critic_dim), requires_grad=True)

        critic_fc1 = nn.Linear(action_dim, mid_dim)
        critic_fc2 = nn.Linear(mid_dim, mid_dim)
        critic_fc3 = nn.Linear(mid_dim, 1)
        self.critic_fc = nn.Sequential(
            critic_fc1, HardSwish(),
            critic_fc2, HardSwish(),
            critic_fc3,
        )

        if layer_norm:
            self.layer_norm(actor_fc1, std=1.0)
            self.layer_norm(actor_fc2, std=1.0)
            self.layer_norm(actor_fc3, std=0.01)

            self.layer_norm(critic_fc1, std=1.0)
            self.layer_norm(critic_fc2, std=1.0)
            self.layer_norm(critic_fc3, std=1.0)

    @staticmethod
    def layer_norm(layer, std=1.0, bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)

    def forward(self, s):
        a_mean = self.actor_fc(s)
        return a_mean

    def get__log_prob(self, s, a_inp):
        a_mean = self.actor_fc(s)
        a_log_std = self.actor_logstd.expand_as(a_mean)
        a_std = torch.exp(a_log_std)
        log_prob = -(a_log_std + (a_inp - a_mean).pow(2) / (2 * a_std.pow(2)) + np.log(2 * np.pi) * 0.5)
        log_prob = log_prob.sum(1)
        return log_prob

    def critic(self, s):
        q = self.critic_fc(s)
        return q

    def get__a__log_prob(self, a_mean):
        a_log_std = self.actor_logstd.expand_as(a_mean)
        a_std = torch.exp(a_log_std)
        a_noise = torch.normal(a_mean, a_std)

        log_prob = -(a_log_std + (a_noise - a_mean).pow(2) / (2 * a_std.pow(2)) + np.log(2 * np.pi) * 0.5)
        log_prob = log_prob.sum(1)
        return a_noise, log_prob


class HardSwish(nn.Module):
    def __init__(self):
        super(HardSwish, self).__init__()
        self.relu6 = nn.ReLU6()

    def forward(self, x):
        return self.relu6(x + 3.) / 6. * x


'''AgentZoo'''


class AgentPPO:
    def __init__(self, state_dim, action_dim, net_dim):
        # super(AgentSNAC, self).__init__()
        """
        Refer: https://github.com/zhangchuheng123/Reinforcement-Implementation/blob/master/code/ppo.py
        Refer: Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
        Refer: https://github.com/Jiankai-Sun/Proximal-Policy-Optimization-in-Pytorch/blob/master/ppo.py
        Refer: https://github.com/openai/baselines/tree/master/baselines/ppo2
        """
        self.act_lr = 4e-4  # learning rate of actor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        '''network'''
        act = ActorCritic(state_dim, action_dim, net_dim).to(self.device)
        act.train()
        self.act = act
        self.act_optimizer = torch.optim.Adam(act.parameters(), lr=self.act_lr, betas=(0.5, 0.99))

        self.criterion = nn.SmoothL1Loss()

        self.clip = 0.5  # constant
        self.update_counter = 0
        self.loss_c_sum = 0.0
        self.loss_coeff_value = 0.5
        self.loss_coeff_entropy = 0.02  # 0.01

    def inactive_in_env_ppo(self, env, max_step, max_memo, max_action, running_state):
        # step1: perform current policy to collect trajectories
        # this is an on-policy method!
        memory = MemoryList()
        rewards = []
        steps = []

        step_counter = 0
        while step_counter < max_memo:
            state = env.reset()
            reward_sum = 0
            t = 0

            state = running_state(state)  # if state_norm:
            for t in range(max_step):
                actions, log_probs, q_value = self.select_actions(state[np.newaxis], explore_noise=True)
                action = actions[0]
                log_prob = log_probs[0]
                q_value = q_value[0]

                next_state, reward, done, _ = env.step(action * max_action)
                reward_sum += reward

                next_state = running_state(next_state)  # if state_norm:
                mask = 0 if done else 1

                # memory.push(state, q_value, action, log_prob, mask, next_state, reward)
                memory.push(state, q_value, action, log_prob, mask, reward)

                if done:
                    break

                state = next_state
            rewards.append(reward_sum)

            t += 1
            steps.append(t)
            step_counter += t
        return rewards, steps, memory

    def update_parameter_ppo(self, memory, batch_size, gamma, ep_ratio):
        clip = 0.2
        lamda = 0.97
        num_epoch = 10

        all_batch = memory.sample()
        max_memo = len(memory)

        all_reward = torch.tensor(all_batch.reward, dtype=torch.float32, device=self.device)
        all_value = torch.tensor(all_batch.value, dtype=torch.float32, device=self.device)
        all_mask = torch.tensor(all_batch.mask, dtype=torch.float32, device=self.device)
        all_action = torch.tensor(all_batch.action, dtype=torch.float32, device=self.device)
        all_state = torch.tensor(all_batch.state, dtype=torch.float32, device=self.device)
        all_log_prob = torch.tensor(all_batch.log_prob, dtype=torch.float32, device=self.device)
        # next_state not use?

        '''calculate prev (return, value, advantage)'''
        all_deltas = torch.empty(max_memo, dtype=torch.float32, device=self.device)
        all_returns = torch.empty(max_memo, dtype=torch.float32, device=self.device)
        all_advantages = torch.empty(max_memo, dtype=torch.float32, device=self.device)

        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        for i in range(max_memo - 1, -1, -1):
            all_deltas[i] = all_reward[i] + gamma * prev_value * all_mask[i] - all_value[i]
            all_returns[i] = all_reward[i] + gamma * prev_return * all_mask[i]
            # ref: https://arxiv.org/pdf/1506.02438.pdf (generalization advantage estimate)
            all_advantages[i] = all_deltas[i] + gamma * lamda * prev_advantage * all_mask[i]

            prev_return = all_returns[i]
            prev_value = all_value[i]
            prev_advantage = all_advantages[i]

        all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-6)  # if advantage_norm:

        '''mini all_batch sample'''
        loss_total = loss_value = None

        for i_epoch in range(int(num_epoch * max_memo / batch_size)):
            # sample from current all_batch
            ind = rd.choice(max_memo, batch_size, replace=False)
            states = all_state[ind]
            actions = all_action[ind]
            old_log_probs = all_log_prob[ind]
            new_log_probs = self.act.get__log_prob(states, actions)
            advantages = all_advantages[ind]
            returns = all_returns[ind]

            new_values = self.act.critic(states).flatten()

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = ratio.clamp(1 - self.clip, 1 + self.clip) * advantages
            loss_surr = - torch.mean(torch.min(surr1, surr2))

            minibatch_return_6std = 6 * returns.std()  # if lossvalue_norm:
            loss_value = torch.mean((new_values - returns).pow(2)) / minibatch_return_6std

            loss_entropy = torch.mean(torch.exp(new_log_probs) * new_log_probs)

            loss_total = loss_surr + self.loss_coeff_value * loss_value + self.loss_coeff_entropy * loss_entropy
            self.act_optimizer.zero_grad()
            loss_total.backward()
            self.act_optimizer.step()

        '''schedule (clip, adam)'''
        # ep_ratio = 1 - (now_epoch / max_epoch)
        self.clip = clip * ep_ratio
        self.act_optimizer.param_groups[0]['lr'] = self.act_lr * ep_ratio

        # return loss_total.data, loss_surr.data, loss_value.data, loss_entropy.data
        return loss_total.item(), loss_value.item(),

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


class MemoryList:
    def __init__(self, ):
        self.memory = []
        from collections import namedtuple
        self.transition = namedtuple(
            'Transition',
            # ('state', 'value', 'action', 'log_prob', 'mask', 'next_state', 'reward')
            ('state', 'value', 'action', 'log_prob', 'mask', 'reward')
        )

    def push(self, *args):
        self.memory.append(self.transition(*args))

    def sample(self):
        return self.transition(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)


class RunningStat:
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        # assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            pre_memo = self._M.copy()
            self._M[...] = pre_memo + (x - pre_memo) / self._n
            self._S[...] = self._S + (x - pre_memo) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update:
            self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    @staticmethod
    def output_shape(input_space):
        return input_space.shape


"""AgentRun Demo"""


def train_agent_ppo(agent_class, env_name, cwd, net_dim, max_step, max_memo, max_epoch,  # env
                    batch_size, gamma,
                    **_kwargs):  # 2020-0430
    env = gym.make(env_name)
    state_dim, action_dim, max_action, target_reward = get_env_info(env)

    agent = agent_class(state_dim, action_dim, net_dim)
    agent.save_or_load_model(cwd, is_save=False)

    # memo_action_dim = action_dim if max_action else 1  # Discrete action space
    # memo = Memories(max_memo, memo_dim=1 + 1 + state_dim + memo_action_dim + state_dim)
    # memo.save_or_load_memo(cwd, is_save=False)

    running_stat = ZFilter((state_dim,), clip=5.0)
    recorder = Recorder(agent, max_step, max_action, target_reward, env_name,
                        running_stat=running_stat)
    # r_norm = RewardNorm(n_max=target_reward, n_min=recorder.reward_avg)
    try:
        for epoch in range(max_epoch):
            with torch.no_grad():  # just the GPU memory
                rewards, steps, memory = agent.inactive_in_env_ppo(
                    env, max_step, max_memo, max_action, running_stat)

            l_total, l_value = agent.update_parameter_ppo(
                memory, batch_size, gamma, ep_ratio=1 - epoch / max_epoch)

            if np.isnan(l_total) or np.isnan(l_value):
                print("ValueError: loss value should not be 'nan'. Please run again.")
                return False

            with torch.no_grad():  # just the GPU memory
                # is_solved = recorder.show_and_check_reward(
                #     epoch, epoch_reward, iter_num, actor_loss, critic_loss, cwd)
                recorder.show_reward(epoch, rewards, steps, l_value, l_total)
                is_solved = recorder.check_reward(cwd, l_value, l_total)
                if is_solved:
                    break

    except KeyboardInterrupt:
        print("raise KeyboardInterrupt while training.")
    except AssertionError:  # for BipedWalker BUG 2020-03-03
        print("AssertionError: OpenAI gym r.LengthSquared() > 0.0f ??? Please run again.")
        return False

    train_time = recorder.show_and_save(env_name, cwd)

    # agent.save_or_load_model(cwd, is_save=True)  # save max reward agent in Recorder
    # memo.save_or_load_memo(cwd, is_save=True)

    draw_plot_with_npy(cwd, train_time)
    return True


def run__ppo(gpu_id, cwd):
    # from AgentZoo import AgentPPO
    args = Arguments()

    args.agent_class = AgentPPO
    args.gpu_id = gpu_id
    args.max_memo = 2 ** 11
    args.batch_size = 2 ** 8
    args.net_dim = 96
    args.gamma = 0.995

    args.init_for_training()

    args.env_name = "LunarLanderContinuous-v2"
    args.cwd = './{}/LL_{}'.format(cwd, gpu_id)
    args.init_for_training()
    while not train_agent_ppo(**vars(args)):
        args.random_seed += 42

    args.env_name = "BipedalWalker-v3"
    args.cwd = './{}/BW_{}'.format(cwd, gpu_id)
    args.init_for_training()
    while not train_agent_ppo(**vars(args)):
        args.random_seed += 42


if __name__ == '__main__':
    from AgentRun import run__multi_process

    # run__ppo(gpu_id=0, cwd='AC_PPO')
    run__multi_process(run__ppo, gpu_tuple=(0, 1, 2, 3), cwd='AC_PPO')
