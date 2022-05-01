import os
import time
from copy import deepcopy
import pandas as pd
import numpy as np
import numpy.random as rd
import torch
import torch.nn as nn
import sys

"""environment"""

import gym


class PendulumEnv(gym.Wrapper):
    def __init__(self, gym_env_id="Pendulum-v1", target_return=-200):
        # Pendulum-v0 gym.__version__ == 0.17.0
        # Pendulum-v1 gym.__version__ == 0.21.0
        gym.logger.set_level(40)  # Block warning
        super().__init__(env=gym.make(gym_env_id))

        # from elegantrl.envs.Gym import get_gym_env_info
        # get_gym_env_info(env, if_print=True)  # use this function to print the env information
        self.env_num = 1  # the env number of VectorEnv is greater than 1
        self.env_name = gym_env_id  # the name of this env.
        self.max_step = 200  # the max step of each episode
        self.state_dim = 3  # feature number of state
        self.action_dim = 1  # feature number of action
        self.if_discrete = False  # discrete action or continuous action

        self.cumulative_returns = 0  # is between (-1600, 0)
        self.rewards = list()

    def reset(self):
        self.rewards = list()
        return self.env.reset().astype(np.float32)

    def step(self, action: np.ndarray):
        # PendulumEnv set its action space as (-2, +2). It is bad.  # https://github.com/openai/gym/wiki/Pendulum-v0
        # I suggest to set action space as (-1, +1) when you design your own env.
        state, reward, done, info_dict = self.env.step(
            action * 2
        )  # state, reward, done, info_dict
        self.rewards.append(reward)
        if done:
            self.cumulative_returns = sum(self.rewards)
        return state.astype(np.float32), reward, done, info_dict


class StockTradingEnv:
    def __init__(self, initial_amount=1e6, max_stock=1e2, buy_cost_pct=1e-3, sell_cost_pct=1e-3, gamma=0.99, ):
        self.close_ary, self.tech_ary = self.load_data_from_disk()

        self.max_stock = max_stock
        self.buy_cost_rate = 1 + buy_cost_pct
        self.sell_cost_rate = 1 - sell_cost_pct
        self.initial_amount = initial_amount
        self.gamma = gamma

        # reset()
        self.day = None
        self.rewards = None
        self.total_asset = None
        self.cumulative_returns = 0

        self.amount = None
        self.shares = None
        self.shares_num = self.close_ary.shape[1]
        amount_dim = 1

        # environment information
        self.env_name = 'StockTradingEnv-v2'
        self.state_dim = self.shares_num + self.close_ary.shape[1] + self.tech_ary.shape[1] + amount_dim
        self.action_dim = self.shares_num
        self.if_discrete = False
        self.max_step = len(self.close_ary)

    def reset(self):
        self.day = 0
        self.amount = self.initial_amount * rd.uniform(0.9, 1.1)
        if rd.rand() < 2 ** -2:
            self.shares = np.zeros(self.shares_num, dtype=np.float32)
        else:
            self.shares = (np.abs(rd.randn(self.shares_num).clip(-2, +2)) * 2 ** 6).astype(int)
        self.rewards = list()

        self.total_asset = (self.close_ary[self.day] * self.shares).sum() + self.amount
        return self.get_state()

    def get_state(self):
        # state = np.hstack((np.array(self.amount * 2 ** -4),
        #                    self.shares * 2 ** -9,
        #                    self.close_ary[self.day] * 2 ** -7,
        #                    self.tech_ary[self.day] * 2 ** -6,))
        state = torch.tensor((
            self.amount * 2 ** -16,
            *(self.shares * 2 ** -9),
            *(self.close_ary[self.day] * 2 ** -7),
            *(self.tech_ary[self.day] * 2 ** -6),
        ), dtype=torch.float32)
        return state

    def step(self, action):
        self.day += 1

        action = action.copy()
        action[(-0.1 < action) & (action < 0.1)] = 0
        action_int = (action * self.max_stock).astype(int)
        # actions initially is scaled between -1 and 1
        # convert into integer because we can't buy fraction of shares
        for index in range(self.action_dim):
            stock_action = action_int[index]
            adj_close_price = self.close_ary[self.day, index]  # `adjcp` denotes adjusted close price
            if stock_action > 0:  # buy_stock
                delta_stock = min(self.amount // adj_close_price, stock_action)
                self.amount -= adj_close_price * delta_stock * self.buy_cost_rate
                self.shares[index] += delta_stock
            elif self.shares[index] > 0:  # sell_stock
                delta_stock = min(-stock_action, self.shares[index])
                self.amount += adj_close_price * delta_stock * self.sell_cost_rate
                self.shares[index] -= delta_stock

        state = self.get_state()

        total_asset = (self.close_ary[self.day] * self.shares).sum() + self.amount
        reward = (total_asset - self.total_asset) * 2 ** -6
        self.rewards.append(reward)
        self.total_asset = total_asset

        done = self.day == self.max_step - 1
        if done:
            reward += 1 / (1 - self.gamma) * np.mean(self.rewards)
            self.cumulative_returns = total_asset / self.initial_amount
        return state, reward, done, {}

    def load_data_from_disk(self, tech_id_list=None):
        tech_id_list = [
            "macd",
            "boll_ub",
            "boll_lb",
            "rsi_30",
            "cci_30",
            "dx_30",
            "close_30_sma",
            "close_60_sma",
        ] if tech_id_list is None else tech_id_list

        npz_pwd = './China_A_shares.numpy.array.npz'
        if os.path.exists(npz_pwd):
            ary_dict = np.load(npz_pwd, allow_pickle=True)
            close_ary = ary_dict['close_ary']
            tech_ary = ary_dict['tech_ary']
            return close_ary, tech_ary

        df_pwd = 'train.pandas.dataframe'
        if os.path.exists(df_pwd):  # convert pandas.DataFrame to numpy.array
            df = pd.read_pickle(df_pwd)

            tech_ary = list()
            close_ary = list()
            df_len = len(df.index.unique())  # df_len = max_step
            for day in range(df_len):
                item = df.loc[day]

                tech_items = [item[tech].values.tolist() for tech in tech_id_list]
                tech_items_flatten = sum(tech_items, [])
                tech_ary.append(tech_items_flatten)

                close_ary.append(item.close)

            close_ary = np.array(close_ary)
            tech_ary = np.array(tech_ary)
            print(f"| get_numpy_arrays, close_ary.shape: {close_ary.shape}")
            print(f"| get_numpy_arrays, tech_ary.shape: {tech_ary.shape}")

            np.savez_compressed(
                npz_pwd,
                close_ary=close_ary,
                tech_ary=tech_ary,
            )
            return close_ary, tech_ary

        raise FileNotFoundError(f"| {self.__module__} need {df_pwd} or {npz_pwd}")


def check_env():
    env = StockTradingEnv()
    """
    cumulative_returns of random action   :      1.63
    cumulative_returns of buy all share   :      2.80
    cumulative_returns of buy half share  :      3.13
    """

    policy_name = 'random action'
    state = env.reset()
    dir(state)
    for _ in range(env.max_step):
        action = rd.uniform(-1, +1, env.action_dim)
        state, reward, done, _ = env.step(action)
        if done:
            break
    print(f'cumulative_returns of {policy_name:16}: {env.cumulative_returns:9.2f}')

    policy_name = 'buy all share'
    state = env.reset()
    dir(state)
    for _ in range(env.max_step):
        action = np.ones(env.action_dim, dtype=np.float32)
        state, reward, done, _ = env.step(action)
        if done:
            break
    print(f'cumulative_returns of {policy_name:16}: {env.cumulative_returns:9.2f}')

    policy_name = 'buy half share'
    state = env.reset()
    dir(state)
    for _ in range(env.max_step):
        action = np.ones(env.action_dim, dtype=np.float32)
        action[:env.action_dim // 2] = 0
        state, reward, done, _ = env.step(action)
        if done:
            break
    print(f'cumulative_returns of {policy_name:16}: {env.cumulative_returns:9.2f}')


'''reinforcement learning: net.py'''


class ActorPPO(nn.Module):
    def __init__(self, mid_dim, mid_layer_num, state_dim, action_dim):
        super().__init__()
        self.net = build_fcn(mid_dim, mid_layer_num, inp_dim=state_dim, out_dim=action_dim)

        # the logarithm (log) of standard deviation (std) of action, it is a trainable parameter
        self.a_std_log = nn.Parameter(torch.zeros((1, action_dim)) - 0.5, requires_grad=True)
        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))

    def forward(self, state):
        return self.net(state).tanh()  # action

    def get_action(self, state):
        a_avg = self.net(state)
        a_std = self.a_std_log.exp()

        noise = torch.randn_like(a_avg)
        action = a_avg + noise * a_std
        return action, noise

    def get_old_logprob(self, _action, noise):
        """
        Compute the log of probability with old network.

        :param _action: the action made.
        :param noise: the noised added. `noise = action - noisy_action`
        :return: the log of probability of old network.
        """
        delta = noise.pow(2) * 0.5
        return -(self.a_std_log + self.sqrt_2pi_log + delta).sum(1)  # old_logprob

    def get_logprob_entropy(self, state, action):
        """
        Compute the log of probability with current network.

        :param state: the input state.
        :param action: the action.
        :return: the log of probability and entropy.
        """
        a_avg = self.net(state)
        a_std = self.a_std_log.exp()

        delta = ((a_avg - action) / a_std).pow(2) * 0.5
        logprob = -(self.a_std_log + self.sqrt_2pi_log + delta).sum(1)  # new_logprob

        dist_entropy = (logprob.exp() * logprob).mean()  # policy entropy
        return logprob, dist_entropy

    @staticmethod
    def get_a_to_e(action):  # convert action of network to action of environment
        return action.tanh()


class CriticPPO(nn.Module):
    def __init__(self, mid_dim, mid_layer_num, state_dim, _action_dim):
        super().__init__()
        self.net = build_fcn(mid_dim, mid_layer_num, inp_dim=state_dim, out_dim=1)

    def forward(self, state):
        return self.net(state)  # advantage value


def build_fcn(mid_dim, mid_layer_num, inp_dim, out_dim):  # fcn (Fully Connected Network)
    net_list = [nn.Linear(inp_dim, mid_dim), nn.ReLU(), ]
    for _ in range(mid_layer_num):
        net_list += [nn.Linear(mid_dim, mid_dim), nn.ReLU(), ]
    net_list += [nn.Linear(mid_dim, out_dim), ]
    return nn.Sequential(*net_list)


'''reinforcement learning: agent.py'''


class AgentBase:
    def __init__(self, net_dim: int, state_dim: int, action_dim: int, gpu_id=0, args=None):
        self.gamma = getattr(args, 'gamma', 0.99)
        self.env_num = getattr(args, 'env_num', 1)
        self.batch_size = getattr(args, 'batch_size', 128)
        self.repeat_times = getattr(args, 'repeat_times', 1.)
        self.reward_scale = getattr(args, 'reward_scale', 1.)
        self.mid_layer_num = getattr(args, 'mid_layer_num', 1)
        self.learning_rate = getattr(args, 'learning_rate', 2 ** -12)
        self.soft_update_tau = getattr(args, 'soft_update_tau', 2 ** -8)

        self.if_off_policy = getattr(args, 'if_off_policy', True)
        self.if_act_target = getattr(args, 'if_act_target', False)
        self.if_cri_target = getattr(args, 'if_cri_target', False)

        self.states = None  # assert self.states == (self.env_num, state_dim)
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        self.traj_list = [[list() for _ in range(4 if self.if_off_policy else 5)]
                          for _ in range(self.env_num)]  # for `self.explore_vec_env()`

        act_class = getattr(self, "act_class", None)
        cri_class = getattr(self, "cri_class", None)
        self.act = act_class(net_dim, self.mid_layer_num, state_dim, action_dim).to(self.device)
        self.cri = cri_class(net_dim, self.mid_layer_num, state_dim, action_dim).to(self.device) \
            if cri_class else self.act
        self.act_target = deepcopy(self.act) if self.if_act_target else self.act
        self.cri_target = deepcopy(self.cri) if self.if_cri_target else self.cri

        self.act_optimizer = torch.optim.Adam(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), self.learning_rate) \
            if cri_class else self.act_optimizer

        """attribute"""
        self.criterion = torch.nn.SmoothL1Loss()

    def explore_env(self, env, target_step: int) -> list:
        """
        Collect trajectories through the actor-environment interaction for a **single** environment instance.

        :param env: the DRL environment instance.
        :param target_step: the total step for the interaction.
        :return: a list of trajectories [traj, ...] where each trajectory is a list of transitions [(state, ...), ...].
        """
        traj_list = []
        last_done = [0, ]
        state = self.states[0]

        step_i = 0
        done = False
        get_action = self.act.get_action
        while step_i < target_step or not done:
            ten_s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            ten_a = get_action(ten_s.to(self.device)).detach().cpu()
            next_s, reward, done, _ = env.step(ten_a[0].numpy())

            traj_list.append((ten_s, reward, done, ten_a))

            step_i += 1
            state = env.reset() if done else next_s

        self.states[0] = state
        last_done[0] = step_i
        return self.convert_trajectory(traj_list, last_done)  # traj_list

    def convert_trajectory(self, traj_list, last_done):  # [ElegantRL.2022.01.01]
        # assert len(buf_items) == step_i
        # assert len(buf_items[0]) in {4, 5}
        # assert len(buf_items[0][0]) == self.env_num
        traj_list = list(map(list, zip(*traj_list)))  # state, reward, done, action, noise
        # assert len(buf_items) == {4, 5}
        # assert len(buf_items[0]) == step
        # assert len(buf_items[0][0]) == self.env_num

        '''stack items'''
        traj_list[0] = torch.stack(traj_list[0])
        traj_list[3:] = [torch.stack(item) for item in traj_list[3:]]

        if len(traj_list[3].shape) == 2:
            traj_list[3] = traj_list[3].unsqueeze(2)

        if self.env_num > 1:
            traj_list[1] = (torch.stack(traj_list[1]) * self.reward_scale).unsqueeze(2)
            traj_list[2] = ((1 - torch.stack(traj_list[2])) * self.gamma).unsqueeze(2)
        else:
            traj_list[1] = (torch.tensor(traj_list[1], dtype=torch.float32) * self.reward_scale
                            ).unsqueeze(1).unsqueeze(2)
            traj_list[2] = ((1 - torch.tensor(traj_list[2], dtype=torch.float32)) * self.gamma
                            ).unsqueeze(1).unsqueeze(2)
        # assert all([buf_item.shape[:2] == (step, self.env_num) for buf_item in buf_items])

        '''splice items'''
        for j in range(len(traj_list)):
            cur_item = list()
            buf_item = traj_list[j]

            for env_i in range(self.env_num):
                last_step = last_done[env_i]

                pre_item = self.traj_list[env_i][j]
                if len(pre_item):
                    cur_item.append(pre_item)

                cur_item.append(buf_item[:last_step, env_i])

            traj_list[j] = torch.vstack(cur_item)

        # on-policy:  buf_item = [states, rewards, dones, actions, noises]
        # off-policy: buf_item = [states, rewards, dones, actions]
        # buf_items = [buf_item, ...]
        return traj_list

    def get_obj_critic(self, buffer, batch_size):
        """
        Calculate the loss of networks with **uniform sampling**.

        :param buffer: the `ReplayBuffer` instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and states.
        """
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_a = self.act_target(next_s)
            next_q = self.cri_target(next_s, next_a)
            q_label = reward + mask * next_q

        q = self.cri(state, action)
        obj_critic = self.criterion(q, q_label)
        return obj_critic, state

    def get_reward_sum(self, buf_len, buf_reward, buf_mask, buf_value):
        """
        Calculate the **reward-to-go** and **advantage estimation**.

        :param buf_len: the length of the `ReplayBuffer`.
        :param buf_reward: a list of rewards for the state-action pairs.
        :param buf_mask: a list of masks computed by the product of done signal and discount factor.
        :param buf_value: a list of state values estimated by the `Critic` network.
        :return: the reward-to-go and advantage estimation.
        """
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # reward sum

        pre_r_sum = 0
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
        buf_adv_v = buf_r_sum - buf_value[:, 0]
        return buf_r_sum, buf_adv_v

    def save_or_load_agent(self, cwd, if_save):
        def load_torch_file(model, _path):
            state_dict = torch.load(_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(state_dict)

        save_path = f"{cwd}/actor.pth"
        if if_save:
            torch.save(self.act.state_dict(), save_path)
        else:
            load_torch_file(self.act, save_path) if os.path.isfile(save_path) else None

    @staticmethod
    def optimizer_update(optimizer, objective):
        """
        Optimize networks through backpropagation.

        :param optimizer: the optimizer instance.
        :param objective: the loss.
        """
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    @staticmethod
    def soft_update(target_net, current_net, tau):
        """
        Soft update method for target networks.

        :param target_net: the target network.
        :param current_net: the current network.
        :param tau: the ratio for update.
        """
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))


class AgentPPO(AgentBase):
    def __init__(self, net_dim, state_dim, action_dim, gpu_id=0, args=None):
        self.if_off_policy = False
        self.act_class = getattr(self, "act_class", ActorPPO)
        self.cri_class = getattr(self, "cri_class", CriticPPO)
        self.if_act_target = getattr(args, 'if_act_target', False)
        self.if_cri_target = getattr(args, "if_cri_target", False)
        AgentBase.__init__(self, net_dim, state_dim, action_dim, gpu_id, args)

        self.ratio_clip = getattr(args, "ratio_clip", 0.25)  # `ratio.clamp(1 - clip, 1 + clip)`
        self.lambda_entropy = getattr(args, "lambda_entropy", 0.02)  # could be 0.00~0.10

    def explore_env(self, env, target_step) -> list:
        traj_list = []
        last_done = [0, ]
        state = self.states[0]

        step_i = 0
        done = False
        get_action = self.act.get_action
        get_a_to_e = self.act.get_a_to_e
        while step_i < target_step or not done:
            ten_s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            ten_a, ten_n = [ten.cpu() for ten in get_action(ten_s.to(self.device))]
            next_s, reward, done, _ = env.step(get_a_to_e(ten_a)[0].numpy())

            traj_list.append((ten_s, reward, done, ten_a, ten_n))

            step_i += 1
            state = env.reset() if done else next_s

        self.states[0] = state
        last_done[0] = step_i
        return self.convert_trajectory(traj_list, last_done)

    def update_net(self, buffer):
        with torch.no_grad():
            buf_state, buf_reward, buf_mask, buf_action, buf_noise = [ten.to(self.device) for ten in buffer]
            buf_len = buf_state.shape[0]

            '''get buf_r_sum, buf_logprob'''
            bs = 2 ** 10  # set a smaller 'BatchSize' when out of GPU memory.
            buf_value = [self.cri_target(buf_state[i:i + bs]) for i in range(0, buf_len, bs)]
            buf_value = torch.cat(buf_value, dim=0)
            buf_logprob = self.act.get_old_logprob(buf_action, buf_noise)

            buf_r_sum, buf_adv_v = self.get_reward_sum(buf_len, buf_reward, buf_mask, buf_value)  # detach()
            buf_adv_v = (buf_adv_v - buf_adv_v.mean()) / (buf_adv_v.std() + 1e-5)
            # buf_adv_v: buffer data of adv_v value
            del buf_noise

        '''update network'''
        obj_critic = obj_actor = None
        update_times = int(1 + buf_len * self.repeat_times / self.batch_size)
        for _ in range(update_times):
            indices = torch.randint(buf_len, size=(self.batch_size,), requires_grad=False, device=self.device)

            state = buf_state[indices]
            r_sum = buf_r_sum[indices]
            adv_v = buf_adv_v[indices]
            action = buf_action[indices]
            logprob = buf_logprob[indices]

            '''PPO: Surrogate objective of Trust Region'''
            new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)  # it is obj_actor
            ratio = (new_logprob - logprob.detach()).exp()
            surrogate1 = adv_v * ratio
            surrogate2 = adv_v * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy
            self.optimizer_update(self.act_optimizer, obj_actor)

            value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, r_sum)
            self.optimizer_update(self.cri_optimizer, obj_critic)
            if self.if_cri_target:
                self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

        a_std_log = getattr(self.act, 'a_std_log', torch.zeros(1)).mean()
        return obj_critic.item(), -obj_actor.item(), a_std_log.item()  # logging_tuple

    def get_reward_sum(self, buf_len, buf_reward, buf_mask, buf_value):
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # reward sum

        pre_r_sum = 0
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
        buf_adv_v = buf_r_sum - buf_value[:, 0]
        return buf_r_sum, buf_adv_v


class ReplayBufferList(list):  # for on-policy
    def __init__(self):
        list.__init__(self)

    def update_buffer(self, traj_list):
        cur_items = list(map(list, zip(*traj_list)))
        self[:] = [torch.cat(item, dim=0) for item in cur_items]

        steps = self[1].shape[0]
        r_exp = self[1].mean().item()
        return steps, r_exp


'''reinforcement learning: run.py'''


class Arguments:
    def __init__(self, agent, env_func=None, env_args=None):
        self.env_func = env_func  # env = env_func(*env_args)
        self.env_args = env_args  # env = env_func(*env_args)

        self.env_num = self.env_args['env_num']  # env_num = 1. In vector env, env_num > 1.
        self.max_step = self.env_args['max_step']  # the max step of an episode
        self.env_name = self.env_args['env_name']  # the env name. Be used to set 'cwd'.
        self.state_dim = self.env_args['state_dim']  # vector dimension (feature number) of state
        self.action_dim = self.env_args['action_dim']  # vector dimension (feature number) of action
        self.if_discrete = self.env_args['if_discrete']  # discrete or continuous action space

        self.agent = agent  # DRL algorithm
        self.net_dim = 2 ** 7  # the middle layer dimension of Fully Connected Network
        self.batch_size = 2 ** 7  # num of transitions sampled from replay buffer.
        self.mid_layer_num = 1  # the middle layer number of Fully Connected Network
        self.if_off_policy = self.get_if_off_policy()  # agent is on-policy or off-policy
        self.if_use_old_traj = False  # save old data to splice and get a complete trajectory (for vector env)
        if self.if_off_policy:  # off-policy
            self.max_memo = 2 ** 21  # capacity of replay buffer
            self.target_step = 2 ** 10  # repeatedly update network to keep critic's loss small
            self.repeat_times = 2 ** 0  # collect target_step, then update network
        else:  # on-policy
            self.max_memo = 2 ** 12  # capacity of replay buffer
            self.target_step = self.max_memo  # repeatedly update network to keep critic's loss small
            self.repeat_times = 2 ** 4  # collect target_step, then update network

        '''Arguments for training'''
        self.gamma = 0.99  # discount factor of future rewards
        self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256
        self.learning_rate = 2 ** -12  # 2 ** -15 ~= 3e-5
        self.soft_update_tau = 2 ** -8  # 2 ** -8 ~= 5e-3

        '''Arguments for device'''
        self.worker_num = 2  # rollout workers number pre GPU (adjust it to get high GPU usage)
        self.thread_num = 8  # cpu_num for pytorch, `torch.set_num_threads(self.num_threads)`
        self.random_seed = 0  # initialize random seed in self.init_before_training()
        self.learner_gpus = 0  # `int` means the ID of single GPU, -1 means CPU

        '''Arguments for evaluate'''
        self.cwd = None  # current working directory to save model. None means set automatically
        self.if_remove = True  # remove the cwd folder? (True, False, None:ask me)
        self.break_step = +np.inf  # break training if 'total_step > break_step'

        '''Arguments for evaluate'''
        self.eval_gap = 2 ** 7  # evaluate the agent per eval_gap seconds
        self.eval_times = 2 ** 4  # number of times that get episode return

    def init_before_training(self):
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.set_num_threads(self.thread_num)
        torch.set_default_dtype(torch.float32)

        '''auto set cwd (current working directory)'''
        if self.cwd is None:
            self.cwd = f'./{self.env_name}_{self.agent.__name__[5:]}_{self.learner_gpus}'

        '''remove history'''
        if self.if_remove is None:
            self.if_remove = bool(input(f"| Arguments PRESS 'y' to REMOVE: {self.cwd}? ") == 'y')
        elif self.if_remove:
            import shutil
            shutil.rmtree(self.cwd, ignore_errors=True)
            print(f"| Arguments Remove cwd: {self.cwd}")
        else:
            print(f"| Arguments Keep cwd: {self.cwd}")
        os.makedirs(self.cwd, exist_ok=True)

    def get_if_off_policy(self):
        name = self.agent.__name__
        return all((name.find('PPO') == -1, name.find('A2C') == -1))  # if_off_policy


def train_and_evaluate(args):
    torch.set_grad_enabled(False)
    args.init_before_training()
    gpu_id = args.learner_gpus

    '''init'''
    env = build_env(args.env_func, args.env_args)

    agent = args.agent(args.net_dim, args.state_dim, args.action_dim, gpu_id=gpu_id, args=args)
    agent.states = [env.reset(), ]

    buffer = ReplayBufferList()

    '''start training'''
    cwd = args.cwd
    break_step = args.break_step
    target_step = args.target_step
    del args

    start_time = time.time()
    total_step = 0
    save_gap = int(5e4)
    total_step_counter = -save_gap
    while True:
        trajectory = agent.explore_env(env, target_step)
        steps, r_exp = buffer.update_buffer((trajectory,))

        torch.set_grad_enabled(True)
        logging_tuple = agent.update_net(buffer)
        torch.set_grad_enabled(False)

        total_step += steps

        if total_step_counter + save_gap < total_step:
            total_step_counter = total_step
            print(
                f"Step:{total_step:8.2e}  "
                f"ExpR:{r_exp:8.2f}  "
                f"Returns:{env.cumulative_returns:8.2f}  "
                f"ObjC:{logging_tuple[0]:8.2f}  "
                f"ObjA:{logging_tuple[1]:8.2f}  "
            )
            save_path = f"{cwd}/actor_{total_step:014.0f}_{time.time() - start_time:08.0f}_{r_exp:08.2f}.pth"
            torch.save(agent.act.state_dict(), save_path)

        if (total_step > break_step) or os.path.exists(f"{cwd}/stop"):
            # stop training when reach `break_step` or `mkdir cwd/stop`
            break

    print(f'| UsedTime: {time.time() - start_time:.0f} | SavedDir: {cwd}')


def get_gym_env_args(env, if_print) -> dict:  # [ElegantRL.2021.12.12]
    """
    Get a dict ``env_args`` about a standard OpenAI gym env information.

    :param env: a standard OpenAI gym env
    :param if_print: [bool] print the dict about env information.
    :return: env_args [dict]

    env_args = {
        'env_num': 1,               # [int] the environment number, 'env_num>1' in vectorized env
        'env_name': env_name,       # [str] the environment name, such as XxxXxx-v0
        'max_step': max_step,       # [int] the steps in an episode. (from env.reset to done).
        'state_dim': state_dim,     # [int] the dimension of state
        'action_dim': action_dim,   # [int] the dimension of action or the number of discrete action
        'if_discrete': if_discrete, # [bool] action space is discrete or continuous
    }
    """
    import gym

    env_num = getattr(env, 'env_num') if hasattr(env, 'env_num') else 1

    if {'unwrapped', 'observation_space', 'action_space', 'spec'}.issubset(dir(env)):  # isinstance(env, gym.Env):
        env_name = getattr(env, 'env_name', None)
        env_name = env.unwrapped.spec.id if env_name is None else env_name

        state_shape = env.observation_space.shape
        state_dim = state_shape[0] if len(state_shape) == 1 else state_shape  # sometimes state_dim is a list

        max_step = getattr(env, 'max_step', None)
        max_step_default = getattr(env, '_max_episode_steps', None)
        if max_step is None:
            max_step = max_step_default
        if max_step is None:
            max_step = 2 ** 10

        if_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        if if_discrete:  # make sure it is discrete action space
            action_dim = env.action_space.n
        elif isinstance(env.action_space, gym.spaces.Box):  # make sure it is continuous action space
            action_dim = env.action_space.shape[0]
            if not any(env.action_space.high - 1):
                print('WARNING: env.action_space.high', env.action_space.high)
            if not any(env.action_space.low - 1):
                print('WARNING: env.action_space.low', env.action_space.low)
        else:
            raise RuntimeError('\n| Error in get_gym_env_info()'
                               '\n  Please set these value manually: if_discrete=bool, action_dim=int.'
                               '\n  And keep action_space in (-1, 1).')
    else:
        env_name = env.env_name
        max_step = env.max_step
        state_dim = env.state_dim
        action_dim = env.action_dim
        if_discrete = env.if_discrete

    env_args = {'env_num': env_num,
                'env_name': env_name,
                'max_step': max_step,
                'state_dim': state_dim,
                'action_dim': action_dim,
                'if_discrete': if_discrete, }
    if if_print:
        env_args_repr = repr(env_args)
        env_args_repr = env_args_repr.replace(',', f",\n   ")
        env_args_repr = env_args_repr.replace('{', "{\n    ")
        env_args_repr = env_args_repr.replace('}', ",\n}")
        print(f"env_args = {env_args_repr}")
    return env_args


def kwargs_filter(func, kwargs: dict):
    """
    Filter the variable in env func.

    :param func: the function for creating an env.
    :param kwargs: args for the env.
    :return: filtered args.
    """
    import inspect

    sign = inspect.signature(func).parameters.values()
    sign = {val.name for val in sign}

    common_args = sign.intersection(kwargs.keys())
    return {key: kwargs[key] for key in common_args}  # filtered kwargs


def build_env(env_func=None, env_args=None):
    env = env_func(**kwargs_filter(env_func.__init__, env_args.copy()))
    return env


def get_episode_return_and_step(env, act) -> (float, int):  # [ElegantRL.2022.01.01]
    """
    Evaluate the actor (policy) network on testing environment.

    :param env: environment object in ElegantRL.
    :param act: Actor (policy) network.
    :return: episodic reward and number of steps needed.
    """
    max_step = env.max_step
    if_discrete = env.if_discrete
    device = next(act.parameters()).device  # net.parameters() is a Python generator.

    state = env.reset()
    episode_step = None
    episode_return = 0.0  # sum of rewards in an episode
    for episode_step in range(max_step):
        s_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        a_tensor = act(s_tensor)
        if if_discrete:
            a_tensor = a_tensor.argmax(dim=1)
        action = a_tensor.detach().cpu().numpy()[0]  # not need detach(), because using torch.no_grad() outside
        state, reward, done, _ = env.step(action)
        episode_return += reward
        if done:
            break
    episode_return = getattr(env, 'cumulative_returns', episode_return)
    episode_step += 1
    return episode_return, episode_step


"""train and evaluate"""


def run():
    import sys
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    env = StockTradingEnv()
    env_func = StockTradingEnv
    env_args = get_gym_env_args(env=env, if_print=True)

    args = Arguments(AgentPPO, env_func=env_func, env_args=env_args)
    args.target_step = args.max_step * 4
    args.reward_scale = 2 ** -7
    args.learning_rate = 2 ** -14
    args.break_step = int(4e6)

    args.learner_gpus = gpu_id
    args.random_seed += gpu_id + 1943
    train_and_evaluate(args)


def run1():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    env = PendulumEnv("Pendulum-v1", target_return=-500)
    env_func = PendulumEnv
    env_args = get_gym_env_args(env=env, if_print=True)

    args = Arguments(AgentPPO, env_func=env_func, env_args=env_args)
    args.reward_scale = 2 ** -1  # RewardRange: -1800 < -200 < -50 < 0
    args.gamma = 0.97
    args.target_step = args.max_step * 8
    args.eval_times = 2 ** 3

    args.learner_gpus = gpu_id
    train_and_evaluate(args)


def evaluate_models_in_directory(dir_path=None):
    if dir_path is None:
        gpu_id = int(sys.argv[1])
        dir_path = f'StockTradingEnv-v2_PPO_{gpu_id}'
        print(f"| evaluate_models_in_directory: gpu_id {gpu_id}")
        print(f"| evaluate_models_in_directory: dir_path {dir_path}")
    else:
        gpu_id = 0
        print(f"| evaluate_models_in_directory: gpu_id {gpu_id}")
        print(f"| evaluate_models_in_directory: dir_path {dir_path}")

    model_names = [name for name in os.listdir(dir_path) if name[:6] == 'actor_']
    model_names.sort()

    env = StockTradingEnv()
    env_func = StockTradingEnv
    env_args = get_gym_env_args(env=env, if_print=True)

    args = Arguments(AgentPPO, env_func=env_func, env_args=env_args)
    device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
    actor = ActorPPO(mid_dim=args.net_dim,
                     mid_layer_num=args.mid_layer_num,
                     state_dim=args.state_dim,
                     action_dim=args.action_dim).to(device)

    def load_torch_file(model, _path):
        state_dict = torch.load(_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)

    for model_name in model_names:
        model_path = f"{dir_path}/{model_name}"
        load_torch_file(actor, model_path)

        cumulative_returns_list = [get_episode_return_and_step(env, actor)[0] for _ in range(4)]
        cumulative_returns = np.mean(cumulative_returns_list)
        print(f"cumulative_returns {cumulative_returns:9.3f}  {model_name}")

"""
python3 demo_FinRL_ElegantRL_ChinaAshares.py 3 GPU
cumulative_returns of random action   :      2.08
cumulative_returns of buy all share   :      2.95
cumulative_returns of buy half share  :      3.07
env_args = {
    'env_num': 1,
    'env_name': 'StockTradingEnv-v2',
    'max_step': 1113,
    'state_dim': 151,
    'action_dim': 15,
    'if_discrete': False,
}   
| Arguments Remove cwd: ./StockTradingEnv-v2_PPO_3
Step:5.56e+03  ExpR:    0.05  Returns:    1.62  ObjC:    5.33  ObjA:    0.03
Step:5.56e+04  ExpR:    0.06  Returns:    1.52  ObjC:    1.18  ObjA:   -0.07
Step:1.06e+05  ExpR:    0.08  Returns:    1.69  ObjC:    1.13  ObjA:    0.01
Step:1.56e+05  ExpR:    0.16  Returns:    2.17  ObjC:    1.25  ObjA:    0.01
Step:2.06e+05  ExpR:    0.22  Returns:    3.21  ObjC:    1.51  ObjA:    0.11
Step:2.56e+05  ExpR:    0.24  Returns:    3.51  ObjC:    1.79  ObjA:    0.02
Step:3.06e+05  ExpR:    0.27  Returns:    3.20  ObjC:    1.67  ObjA:   -0.10
Step:3.56e+05  ExpR:    0.25  Returns:    3.33  ObjC:    1.72  ObjA:    0.16
Step:4.06e+05  ExpR:    0.29  Returns:    3.11  ObjC:    2.08  ObjA:    0.02
Step:4.56e+05  ExpR:    0.36  Returns:    4.28  ObjC:    1.76  ObjA:    0.03
Step:5.06e+05  ExpR:    0.41  Returns:    4.50  ObjC:    1.85  ObjA:    0.14
Step:5.56e+05  ExpR:    0.38  Returns:    4.11  ObjC:    1.51  ObjA:    0.10
Step:6.06e+05  ExpR:    0.44  Returns:    3.79  ObjC:    2.16  ObjA:    0.12
Step:6.56e+05  ExpR:    0.41  Returns:    4.34  ObjC:    1.46  ObjA:    0.02
Step:7.06e+05  ExpR:    0.45  Returns:    4.18  ObjC:    2.07  ObjA:    0.04
Step:7.56e+05  ExpR:    0.52  Returns:    5.25  ObjC:    1.70  ObjA:    0.02
Step:8.06e+05  ExpR:    0.59  Returns:    5.70  ObjC:    1.56  ObjA:    0.05
Step:8.56e+05  ExpR:    0.67  Returns:    6.94  ObjC:    1.26  ObjA:    0.14
Step:9.06e+05  ExpR:    0.63  Returns:    6.32  ObjC:    1.69  ObjA:    0.24
Step:9.56e+05  ExpR:    0.68  Returns:    6.89  ObjC:    1.10  ObjA:    0.08
Step:1.01e+06  ExpR:    0.67  Returns:    6.76  ObjC:    1.14  ObjA:    0.08
Step:1.06e+06  ExpR:    0.60  Returns:    6.46  ObjC:    1.41  ObjA:    0.11
Step:1.11e+06  ExpR:    0.72  Returns:    6.79  ObjC:    0.98  ObjA:    0.05
Step:1.16e+06  ExpR:    0.74  Returns:    7.38  ObjC:    0.95  ObjA:    0.10
Step:1.21e+06  ExpR:    0.78  Returns:    7.37  ObjC:    0.85  ObjA:    0.22
Step:1.26e+06  ExpR:    0.75  Returns:    7.20  ObjC:    0.98  ObjA:    0.21
Step:1.31e+06  ExpR:    0.77  Returns:    7.39  ObjC:    1.04  ObjA:    0.10
Step:1.36e+06  ExpR:    0.80  Returns:    7.70  ObjC:    0.63  ObjA:   -0.03
Step:1.41e+06  ExpR:    0.77  Returns:    7.29  ObjC:    0.80  ObjA:   -0.01
Step:1.46e+06  ExpR:    0.82  Returns:    8.32  ObjC:    0.83  ObjA:    0.12
Step:1.51e+06  ExpR:    0.82  Returns:    7.28  ObjC:    0.43  ObjA:    0.16
Step:1.56e+06  ExpR:    0.86  Returns:    7.89  ObjC:    0.59  ObjA:    0.24
Step:1.61e+06  ExpR:    0.87  Returns:    8.03  ObjC:    0.61  ObjA:    0.12
Step:1.66e+06  ExpR:    0.86  Returns:    7.36  ObjC:    0.59  ObjA:    0.10
Step:1.71e+06  ExpR:    0.87  Returns:    8.27  ObjC:    0.55  ObjA:    0.08
Step:1.76e+06  ExpR:    0.87  Returns:    8.72  ObjC:    0.70  ObjA:    0.26
Step:1.81e+06  ExpR:    0.89  Returns:    8.59  ObjC:    0.51  ObjA:    0.07
Step:1.86e+06  ExpR:    0.91  Returns:    8.53  ObjC:    0.36  ObjA:    0.01
Step:1.91e+06  ExpR:    0.93  Returns:    9.21  ObjC:    0.77  ObjA:   -0.15
Step:1.96e+06  ExpR:    0.92  Returns:    8.21  ObjC:    0.62  ObjA:    0.19
Step:2.01e+06  ExpR:    0.92  Returns:    8.02  ObjC:    0.83  ObjA:    0.21
Step:2.06e+06  ExpR:    0.86  Returns:    8.83  ObjC:    1.27  ObjA:    0.20
Step:2.11e+06  ExpR:    0.88  Returns:    7.84  ObjC:    1.10  ObjA:    0.05
Step:2.16e+06  ExpR:    0.96  Returns:    9.46  ObjC:    0.46  ObjA:    0.14
Step:2.21e+06  ExpR:    0.97  Returns:    9.54  ObjC:    0.70  ObjA:    0.04
Step:2.26e+06  ExpR:    0.96  Returns:    9.49  ObjC:    0.49  ObjA:    0.05
Step:2.31e+06  ExpR:    0.82  Returns:    7.46  ObjC:    0.73  ObjA:    0.15
Step:2.36e+06  ExpR:    0.96  Returns:    9.76  ObjC:    0.70  ObjA:    0.11
Step:2.41e+06  ExpR:    0.92  Returns:    9.52  ObjC:    0.77  ObjA:    0.17
Step:2.46e+06  ExpR:    0.91  Returns:    7.37  ObjC:    0.74  ObjA:    0.23
Step:2.51e+06  ExpR:    0.87  Returns:    8.31  ObjC:    0.80  ObjA:    0.10
Step:2.56e+06  ExpR:    0.90  Returns:    9.98  ObjC:    0.62  ObjA:    0.11
Step:2.61e+06  ExpR:    1.03  Returns:    9.04  ObjC:    0.54  ObjA:    0.15
Step:2.66e+06  ExpR:    1.00  Returns:    9.71  ObjC:    0.41  ObjA:    0.11
Step:2.71e+06  ExpR:    1.01  Returns:    9.65  ObjC:    0.70  ObjA:   -0.03
Step:2.76e+06  ExpR:    1.01  Returns:    9.17  ObjC:    0.43  ObjA:    0.01
Step:2.81e+06  ExpR:    1.05  Returns:    9.05  ObjC:    0.33  ObjA:    0.25
Step:2.86e+06  ExpR:    1.02  Returns:   10.30  ObjC:    0.42  ObjA:    0.08
Step:2.91e+06  ExpR:    1.07  Returns:    9.56  ObjC:    0.30  ObjA:    0.11
Step:2.96e+06  ExpR:    1.06  Returns:   10.26  ObjC:    0.39  ObjA:    0.25
Step:3.01e+06  ExpR:    1.08  Returns:   10.26  ObjC:    0.25  ObjA:   -0.03
Step:3.06e+06  ExpR:    1.08  Returns:   10.74  ObjC:    0.20  ObjA:    0.07
Step:3.11e+06  ExpR:    1.12  Returns:   10.69  ObjC:    0.32  ObjA:    0.08
Step:3.16e+06  ExpR:    1.09  Returns:   10.79  ObjC:    0.26  ObjA:    0.13
Step:3.21e+06  ExpR:    1.10  Returns:   10.07  ObjC:    0.20  ObjA:    0.20
Step:3.26e+06  ExpR:    1.12  Returns:    9.94  ObjC:    0.55  ObjA:    0.03
Step:3.31e+06  ExpR:    1.08  Returns:   10.16  ObjC:    0.27  ObjA:    0.20
Step:3.36e+06  ExpR:    1.11  Returns:    9.68  ObjC:    0.23  ObjA:    0.02
Step:3.41e+06  ExpR:    1.12  Returns:   10.80  ObjC:    0.17  ObjA:    0.14
Step:3.46e+06  ExpR:    1.14  Returns:   11.00  ObjC:    0.15  ObjA:    0.21
Step:3.51e+06  ExpR:    1.10  Returns:   10.42  ObjC:    0.19  ObjA:   -0.00
Step:3.56e+06  ExpR:    1.11  Returns:    9.83  ObjC:    0.14  ObjA:    0.12
Step:3.61e+06  ExpR:    1.11  Returns:    9.98  ObjC:    0.21  ObjA:    0.12
Step:3.66e+06  ExpR:    1.15  Returns:   11.09  ObjC:    0.13  ObjA:    0.20
Step:3.71e+06  ExpR:    1.12  Returns:   10.28  ObjC:    0.35  ObjA:    0.23
Step:3.76e+06  ExpR:    1.14  Returns:   10.72  ObjC:    0.20  ObjA:    0.22
Step:3.81e+06  ExpR:    1.15  Returns:    9.84  ObjC:    0.17  ObjA:    0.14
Step:3.86e+06  ExpR:    1.13  Returns:   10.58  ObjC:    0.16  ObjA:    0.04
Step:3.91e+06  ExpR:    1.12  Returns:   10.65  ObjC:    0.36  ObjA:    0.10
Step:3.96e+06  ExpR:    1.13  Returns:   10.39  ObjC:    0.19  ObjA:    0.36
| UsedTime: 3034 | SavedDir: ./StockTradingEnv-v2_PPO_3
| evaluate_models_in_directory: gpu_id 3
| evaluate_models_in_directory: dir_path StockTradingEnv-v2_PPO_3
env_args = {
    'env_num': 1,
    'env_name': 'StockTradingEnv-v2',
    'max_step': 1113,
    'state_dim': 151,
    'action_dim': 15,
    'if_discrete': False,
}
cumulative_returns     1.491  actor_00000000005560_00000005_00000.05.pth
cumulative_returns     1.565  actor_00000000055600_00000044_00000.06.pth
cumulative_returns     1.870  actor_00000000105640_00000083_00000.08.pth
cumulative_returns     2.855  actor_00000000155680_00000121_00000.16.pth
cumulative_returns     3.240  actor_00000000205720_00000159_00000.22.pth
cumulative_returns     2.988  actor_00000000255760_00000196_00000.24.pth
cumulative_returns     3.643  actor_00000000305800_00000234_00000.27.pth
cumulative_returns     3.378  actor_00000000355840_00000272_00000.25.pth
cumulative_returns     4.731  actor_00000000405880_00000310_00000.29.pth
cumulative_returns     4.488  actor_00000000455920_00000348_00000.36.pth
cumulative_returns     4.890  actor_00000000505960_00000386_00000.41.pth
cumulative_returns     5.222  actor_00000000556000_00000424_00000.38.pth
cumulative_returns     4.721  actor_00000000606040_00000462_00000.44.pth
cumulative_returns     4.908  actor_00000000656080_00000500_00000.41.pth
cumulative_returns     5.209  actor_00000000706120_00000538_00000.45.pth
cumulative_returns     5.472  actor_00000000756160_00000576_00000.52.pth
cumulative_returns     6.287  actor_00000000806200_00000615_00000.59.pth
cumulative_returns     6.145  actor_00000000856240_00000653_00000.67.pth
cumulative_returns     6.074  actor_00000000906280_00000690_00000.63.pth
cumulative_returns     6.171  actor_00000000956320_00000728_00000.68.pth
cumulative_returns     6.151  actor_00000001006360_00000766_00000.67.pth
cumulative_returns     5.698  actor_00000001056400_00000804_00000.60.pth
cumulative_returns     6.448  actor_00000001106440_00000843_00000.72.pth
cumulative_returns     6.478  actor_00000001156480_00000880_00000.74.pth
cumulative_returns     6.799  actor_00000001206520_00000919_00000.78.pth
cumulative_returns     6.982  actor_00000001256560_00000957_00000.75.pth
cumulative_returns     6.883  actor_00000001306600_00000995_00000.77.pth
cumulative_returns     7.124  actor_00000001356640_00001033_00000.80.pth
cumulative_returns     7.439  actor_00000001406680_00001071_00000.77.pth
cumulative_returns     6.969  actor_00000001456720_00001109_00000.82.pth
cumulative_returns     6.966  actor_00000001506760_00001147_00000.82.pth
cumulative_returns     7.403  actor_00000001556800_00001184_00000.86.pth
cumulative_returns     7.216  actor_00000001606840_00001222_00000.87.pth
cumulative_returns     7.091  actor_00000001656880_00001260_00000.86.pth
cumulative_returns     7.079  actor_00000001706920_00001298_00000.87.pth
cumulative_returns     7.429  actor_00000001756960_00001335_00000.87.pth
cumulative_returns     7.468  actor_00000001807000_00001373_00000.89.pth
cumulative_returns     8.226  actor_00000001857040_00001411_00000.91.pth
cumulative_returns     8.261  actor_00000001907080_00001448_00000.93.pth
cumulative_returns     7.515  actor_00000001957120_00001486_00000.92.pth
cumulative_returns     7.369  actor_00000002007160_00001524_00000.92.pth
cumulative_returns     7.497  actor_00000002057200_00001562_00000.86.pth
cumulative_returns     7.437  actor_00000002107240_00001599_00000.88.pth
cumulative_returns     8.199  actor_00000002157280_00001637_00000.96.pth
cumulative_returns     7.469  actor_00000002207320_00001676_00000.97.pth
cumulative_returns     8.029  actor_00000002257360_00001714_00000.96.pth
cumulative_returns     8.250  actor_00000002307400_00001753_00000.82.pth
cumulative_returns     8.114  actor_00000002357440_00001791_00000.96.pth
cumulative_returns     7.784  actor_00000002407480_00001830_00000.92.pth
cumulative_returns     8.424  actor_00000002457520_00001868_00000.91.pth
cumulative_returns     7.943  actor_00000002507560_00001905_00000.87.pth
cumulative_returns     8.089  actor_00000002557600_00001943_00000.90.pth
cumulative_returns     9.417  actor_00000002607640_00001981_00001.03.pth
cumulative_returns     9.441  actor_00000002657680_00002018_00001.00.pth
cumulative_returns     9.605  actor_00000002707720_00002056_00001.01.pth
cumulative_returns     9.587  actor_00000002757760_00002094_00001.01.pth
cumulative_returns     8.381  actor_00000002807800_00002131_00001.05.pth
cumulative_returns     9.903  actor_00000002857840_00002169_00001.02.pth
cumulative_returns    10.101  actor_00000002907880_00002206_00001.07.pth
cumulative_returns    10.434  actor_00000002957920_00002244_00001.06.pth
cumulative_returns     9.839  actor_00000003007960_00002283_00001.08.pth
cumulative_returns     7.881  actor_00000003058000_00002320_00001.08.pth
cumulative_returns    10.067  actor_00000003108040_00002357_00001.12.pth
cumulative_returns    10.066  actor_00000003158080_00002395_00001.09.pth
cumulative_returns    10.382  actor_00000003208120_00002434_00001.10.pth
cumulative_returns    10.144  actor_00000003258160_00002472_00001.12.pth
cumulative_returns     8.542  actor_00000003308200_00002511_00001.08.pth
cumulative_returns    10.239  actor_00000003358240_00002549_00001.11.pth
cumulative_returns    10.132  actor_00000003408280_00002587_00001.12.pth
cumulative_returns    10.428  actor_00000003458320_00002625_00001.14.pth
cumulative_returns    10.220  actor_00000003508360_00002662_00001.10.pth
cumulative_returns    10.310  actor_00000003558400_00002700_00001.11.pth
cumulative_returns    10.692  actor_00000003608440_00002738_00001.11.pth
cumulative_returns    10.515  actor_00000003658480_00002775_00001.15.pth
cumulative_returns    10.586  actor_00000003708520_00002813_00001.12.pth
cumulative_returns    10.727  actor_00000003758560_00002851_00001.14.pth
cumulative_returns    10.552  actor_00000003808600_00002889_00001.15.pth
cumulative_returns    10.654  actor_00000003858640_00002926_00001.13.pth
cumulative_returns    10.687  actor_00000003908680_00002964_00001.12.pth
cumulative_returns    10.345  actor_00000003958720_00003001_00001.13.pth
"""

"""
cumulative_returns of random action   :      1.72
cumulative_returns of buy all share   :      2.96
cumulative_returns of buy half share  :      3.15
env_args = {
    'env_num': 1,
    'env_name': 'StockTradingEnv-v2',
    'max_step': 1113,
    'state_dim': 151,
    'action_dim': 15,
    'if_discrete': False,
}   
| Arguments Remove cwd: ./StockTradingEnv-v2_PPO_4
Step:5.56e+03  ExpR:    0.19  Returns:    2.42  ObjC:    8.28  ObjA:    0.05
Step:5.56e+04  ExpR:    0.22  Returns:    3.25  ObjC:    2.98  ObjA:   -0.03
Step:1.06e+05  ExpR:    0.30  Returns:    3.44  ObjC:    2.83  ObjA:    0.11
Step:1.56e+05  ExpR:    0.39  Returns:    4.29  ObjC:    2.38  ObjA:   -0.05
Step:2.06e+05  ExpR:    0.38  Returns:    4.03  ObjC:    1.80  ObjA:    0.06
Step:2.56e+05  ExpR:    0.38  Returns:    4.26  ObjC:    1.71  ObjA:    0.18
Step:3.06e+05  ExpR:    0.43  Returns:    4.46  ObjC:    1.89  ObjA:    0.09
Step:3.56e+05  ExpR:    0.45  Returns:    4.93  ObjC:    1.50  ObjA:    0.02
Step:4.06e+05  ExpR:    0.50  Returns:    5.26  ObjC:    1.38  ObjA:   -0.08
Step:4.56e+05  ExpR:    0.53  Returns:    5.11  ObjC:    1.22  ObjA:   -0.09
Step:5.06e+05  ExpR:    0.55  Returns:    5.50  ObjC:    1.10  ObjA:    0.10
Step:5.56e+05  ExpR:    0.56  Returns:    5.90  ObjC:    1.38  ObjA:    0.17
Step:6.06e+05  ExpR:    0.59  Returns:    6.14  ObjC:    1.05  ObjA:    0.05
Step:6.56e+05  ExpR:    0.63  Returns:    6.36  ObjC:    1.11  ObjA:    0.07
Step:7.06e+05  ExpR:    0.62  Returns:    6.24  ObjC:    1.36  ObjA:    0.24
Step:7.56e+05  ExpR:    0.62  Returns:    5.38  ObjC:    1.29  ObjA:   -0.01
Step:8.06e+05  ExpR:    0.68  Returns:    6.71  ObjC:    0.99  ObjA:    0.18
Step:8.56e+05  ExpR:    0.71  Returns:    6.99  ObjC:    0.86  ObjA:    0.20
Step:9.06e+05  ExpR:    0.73  Returns:    7.25  ObjC:    0.94  ObjA:    0.13
Step:9.56e+05  ExpR:    0.72  Returns:    7.27  ObjC:    0.68  ObjA:    0.12
Step:1.01e+06  ExpR:    0.73  Returns:    6.94  ObjC:    0.69  ObjA:   -0.05
Step:1.06e+06  ExpR:    0.75  Returns:    6.85  ObjC:    0.99  ObjA:    0.03
Step:1.11e+06  ExpR:    0.80  Returns:    7.97  ObjC:    0.60  ObjA:    0.11
Step:1.16e+06  ExpR:    0.81  Returns:    7.25  ObjC:    0.69  ObjA:    0.08
Step:1.21e+06  ExpR:    0.80  Returns:    7.88  ObjC:    0.72  ObjA:    0.06
Step:1.26e+06  ExpR:    0.82  Returns:    8.32  ObjC:    0.77  ObjA:   -0.05
Step:1.31e+06  ExpR:    0.82  Returns:    8.02  ObjC:    0.69  ObjA:    0.26
Step:1.36e+06  ExpR:    0.81  Returns:    7.26  ObjC:    0.69  ObjA:    0.21
Step:1.41e+06  ExpR:    0.79  Returns:    7.55  ObjC:    0.63  ObjA:    0.14
Step:1.46e+06  ExpR:    0.80  Returns:    7.47  ObjC:    0.55  ObjA:   -0.02
Step:1.51e+06  ExpR:    0.87  Returns:    8.39  ObjC:    0.50  ObjA:    0.12
Step:1.56e+06  ExpR:    0.86  Returns:    7.75  ObjC:    0.49  ObjA:    0.14
Step:1.61e+06  ExpR:    0.86  Returns:    8.33  ObjC:    0.57  ObjA:    0.07
Step:1.66e+06  ExpR:    0.85  Returns:    8.08  ObjC:    0.71  ObjA:    0.14
Step:1.71e+06  ExpR:    0.84  Returns:    7.76  ObjC:    0.58  ObjA:   -0.12
Step:1.76e+06  ExpR:    0.89  Returns:    8.25  ObjC:    0.63  ObjA:    0.25
Step:1.81e+06  ExpR:    0.83  Returns:    8.34  ObjC:    0.88  ObjA:   -0.04
Step:1.86e+06  ExpR:    0.70  Returns:    7.86  ObjC:    1.17  ObjA:    0.01
Step:1.91e+06  ExpR:    0.66  Returns:    6.55  ObjC:    1.73  ObjA:    0.18
Step:1.96e+06  ExpR:    0.76  Returns:    7.07  ObjC:    1.87  ObjA:    0.17
Step:2.01e+06  ExpR:    0.88  Returns:    8.85  ObjC:    0.87  ObjA:    0.09
Step:2.06e+06  ExpR:    0.86  Returns:    8.03  ObjC:    0.87  ObjA:    0.03
Step:2.11e+06  ExpR:    0.87  Returns:    8.40  ObjC:    0.98  ObjA:    0.05
Step:2.16e+06  ExpR:    0.87  Returns:    8.50  ObjC:    1.35  ObjA:    0.08
Step:2.21e+06  ExpR:    0.86  Returns:    7.85  ObjC:    1.03  ObjA:    0.09
Step:2.26e+06  ExpR:    0.83  Returns:    7.82  ObjC:    1.07  ObjA:   -0.04
Step:2.31e+06  ExpR:    0.90  Returns:    8.56  ObjC:    0.62  ObjA:    0.09
Step:2.36e+06  ExpR:    0.93  Returns:    8.67  ObjC:    0.54  ObjA:    0.06
Step:2.41e+06  ExpR:    0.92  Returns:    8.14  ObjC:    0.54  ObjA:    0.22
Step:2.46e+06  ExpR:    0.95  Returns:    8.96  ObjC:    0.56  ObjA:    0.02
Step:2.51e+06  ExpR:    0.96  Returns:    8.95  ObjC:    0.61  ObjA:    0.09
Step:2.56e+06  ExpR:    1.02  Returns:    9.76  ObjC:    0.58  ObjA:    0.12
Step:2.61e+06  ExpR:    0.99  Returns:    9.95  ObjC:    0.76  ObjA:    0.21
Step:2.66e+06  ExpR:    1.04  Returns:   10.19  ObjC:    0.37  ObjA:    0.16
Step:2.71e+06  ExpR:    1.06  Returns:   10.22  ObjC:    0.69  ObjA:   -0.04
Step:2.76e+06  ExpR:    0.97  Returns:    9.36  ObjC:    2.30  ObjA:    0.07
Step:2.81e+06  ExpR:    1.05  Returns:    9.82  ObjC:    0.80  ObjA:   -0.02
Step:2.86e+06  ExpR:    0.93  Returns:    6.90  ObjC:    2.05  ObjA:    0.02
Step:2.91e+06  ExpR:    1.03  Returns:    9.65  ObjC:    0.47  ObjA:    0.18
Step:2.96e+06  ExpR:    1.06  Returns:    9.94  ObjC:    0.61  ObjA:    0.40
Step:3.01e+06  ExpR:    1.00  Returns:    9.50  ObjC:    0.83  ObjA:    0.16
Step:3.06e+06  ExpR:    0.95  Returns:    8.79  ObjC:    0.81  ObjA:    0.09
Step:3.11e+06  ExpR:    1.02  Returns:    9.20  ObjC:    0.70  ObjA:    0.23
Step:3.16e+06  ExpR:    0.79  Returns:    7.86  ObjC:    0.70  ObjA:    0.03
Step:3.21e+06  ExpR:    0.86  Returns:    8.12  ObjC:    1.54  ObjA:   -0.05
Step:3.26e+06  ExpR:    0.83  Returns:    7.55  ObjC:    2.32  ObjA:    0.18
Step:3.31e+06  ExpR:    0.85  Returns:    8.31  ObjC:    1.78  ObjA:    0.09
Step:3.36e+06  ExpR:    0.81  Returns:    7.70  ObjC:    1.65  ObjA:    0.21
Step:3.41e+06  ExpR:    0.80  Returns:    7.34  ObjC:    0.76  ObjA:    0.04
Step:3.46e+06  ExpR:    0.83  Returns:    7.80  ObjC:    1.51  ObjA:    0.05
Step:3.51e+06  ExpR:    0.83  Returns:    8.19  ObjC:    0.65  ObjA:    0.06
Step:3.56e+06  ExpR:    0.84  Returns:    7.87  ObjC:    0.38  ObjA:    0.02
Step:3.61e+06  ExpR:    0.85  Returns:    7.88  ObjC:    0.26  ObjA:   -0.07
Step:3.66e+06  ExpR:    0.83  Returns:    8.17  ObjC:    0.23  ObjA:    0.08
Step:3.71e+06  ExpR:    0.86  Returns:    8.15  ObjC:    0.39  ObjA:    0.11
Step:3.76e+06  ExpR:    0.86  Returns:    8.17  ObjC:    0.46  ObjA:    0.17
Step:3.81e+06  ExpR:    0.89  Returns:    8.55  ObjC:    0.56  ObjA:    0.12
Step:3.86e+06  ExpR:    0.88  Returns:    8.38  ObjC:    0.32  ObjA:    0.15
Step:3.91e+06  ExpR:    0.94  Returns:    8.75  ObjC:    1.27  ObjA:   -0.00
Step:3.96e+06  ExpR:    0.91  Returns:    8.90  ObjC:    0.49  ObjA:   -0.00
| UsedTime: 3002 | SavedDir: ./StockTradingEnv-v2_PPO_4



| evaluate_models_in_directory: gpu_id 4
| evaluate_models_in_directory: dir_path StockTradingEnv-v2_PPO_4
env_args = {
    'env_num': 1,
    'env_name': 'StockTradingEnv-v2',
    'max_step': 1113,
    'state_dim': 151,
    'action_dim': 15,
    'if_discrete': False,
}
cumulative_returns     3.445  actor_00000000005560_00000005_00000.19.pth
cumulative_returns     3.877  actor_00000000055600_00000043_00000.22.pth
cumulative_returns     4.253  actor_00000000105640_00000080_00000.30.pth
cumulative_returns     4.544  actor_00000000155680_00000118_00000.39.pth
cumulative_returns     4.655  actor_00000000205720_00000156_00000.38.pth
cumulative_returns     4.806  actor_00000000255760_00000194_00000.38.pth
cumulative_returns     5.088  actor_00000000305800_00000231_00000.43.pth
cumulative_returns     4.872  actor_00000000355840_00000269_00000.45.pth
cumulative_returns     5.162  actor_00000000405880_00000307_00000.50.pth
cumulative_returns     5.486  actor_00000000455920_00000344_00000.53.pth
cumulative_returns     6.302  actor_00000000505960_00000381_00000.55.pth
cumulative_returns     6.222  actor_00000000556000_00000419_00000.56.pth
cumulative_returns     6.196  actor_00000000606040_00000457_00000.59.pth
cumulative_returns     6.363  actor_00000000656080_00000494_00000.63.pth
cumulative_returns     6.556  actor_00000000706120_00000533_00000.62.pth
cumulative_returns     6.791  actor_00000000756160_00000571_00000.62.pth
cumulative_returns     6.692  actor_00000000806200_00000608_00000.68.pth
cumulative_returns     7.250  actor_00000000856240_00000646_00000.71.pth
cumulative_returns     7.131  actor_00000000906280_00000684_00000.73.pth
cumulative_returns     7.694  actor_00000000956320_00000722_00000.72.pth
cumulative_returns     7.485  actor_00000001006360_00000759_00000.73.pth
cumulative_returns     7.490  actor_00000001056400_00000797_00000.75.pth
cumulative_returns     7.515  actor_00000001106440_00000834_00000.80.pth
cumulative_returns     7.734  actor_00000001156480_00000871_00000.81.pth
cumulative_returns     7.557  actor_00000001206520_00000908_00000.80.pth
cumulative_returns     7.957  actor_00000001256560_00000946_00000.82.pth
cumulative_returns     7.734  actor_00000001306600_00000983_00000.82.pth
cumulative_returns     7.861  actor_00000001356640_00001020_00000.81.pth
cumulative_returns     8.066  actor_00000001406680_00001058_00000.79.pth
cumulative_returns     7.875  actor_00000001456720_00001096_00000.80.pth
cumulative_returns     8.139  actor_00000001506760_00001134_00000.87.pth
cumulative_returns     8.102  actor_00000001556800_00001171_00000.86.pth
cumulative_returns     8.404  actor_00000001606840_00001208_00000.86.pth
cumulative_returns     7.719  actor_00000001656880_00001247_00000.85.pth
cumulative_returns     8.120  actor_00000001706920_00001284_00000.84.pth
cumulative_returns     8.381  actor_00000001756960_00001322_00000.89.pth
cumulative_returns     7.702  actor_00000001807000_00001360_00000.83.pth
cumulative_returns     7.180  actor_00000001857040_00001397_00000.70.pth
cumulative_returns     6.649  actor_00000001907080_00001435_00000.66.pth
cumulative_returns     8.175  actor_00000001957120_00001472_00000.76.pth
cumulative_returns     8.707  actor_00000002007160_00001509_00000.88.pth
cumulative_returns     8.096  actor_00000002057200_00001547_00000.86.pth
cumulative_returns     8.363  actor_00000002107240_00001584_00000.87.pth
cumulative_returns     8.246  actor_00000002157280_00001621_00000.87.pth
cumulative_returns     7.900  actor_00000002207320_00001658_00000.86.pth
cumulative_returns     7.751  actor_00000002257360_00001695_00000.83.pth
cumulative_returns     8.310  actor_00000002307400_00001732_00000.90.pth
cumulative_returns     8.666  actor_00000002357440_00001770_00000.93.pth
cumulative_returns     8.749  actor_00000002407480_00001807_00000.92.pth
cumulative_returns     8.836  actor_00000002457520_00001845_00000.95.pth
cumulative_returns     8.638  actor_00000002507560_00001883_00000.96.pth
cumulative_returns     9.200  actor_00000002557600_00001921_00001.02.pth
cumulative_returns     9.067  actor_00000002607640_00001958_00000.99.pth
cumulative_returns     9.569  actor_00000002657680_00001996_00001.04.pth
cumulative_returns     9.104  actor_00000002707720_00002034_00001.06.pth
cumulative_returns     8.855  actor_00000002757760_00002071_00000.97.pth
cumulative_returns     9.594  actor_00000002807800_00002109_00001.05.pth
cumulative_returns     9.352  actor_00000002857840_00002147_00000.93.pth
cumulative_returns     9.696  actor_00000002907880_00002184_00001.03.pth
cumulative_returns     9.824  actor_00000002957920_00002221_00001.06.pth
cumulative_returns     9.852  actor_00000003007960_00002257_00001.00.pth
cumulative_returns     8.801  actor_00000003058000_00002294_00000.95.pth
cumulative_returns     9.790  actor_00000003108040_00002332_00001.02.pth
cumulative_returns     8.641  actor_00000003158080_00002369_00000.79.pth
cumulative_returns     8.262  actor_00000003208120_00002407_00000.86.pth
cumulative_returns     8.323  actor_00000003258160_00002445_00000.83.pth
cumulative_returns     8.028  actor_00000003308200_00002482_00000.85.pth
cumulative_returns     7.846  actor_00000003358240_00002519_00000.81.pth
cumulative_returns     7.597  actor_00000003408280_00002556_00000.80.pth
cumulative_returns     8.021  actor_00000003458320_00002594_00000.83.pth
cumulative_returns     8.260  actor_00000003508360_00002631_00000.83.pth
cumulative_returns     7.796  actor_00000003558400_00002669_00000.84.pth
cumulative_returns     7.863  actor_00000003608440_00002706_00000.85.pth
cumulative_returns     8.199  actor_00000003658480_00002743_00000.83.pth
cumulative_returns     8.078  actor_00000003708520_00002780_00000.86.pth
cumulative_returns     8.368  actor_00000003758560_00002817_00000.86.pth
cumulative_returns     8.600  actor_00000003808600_00002855_00000.89.pth
cumulative_returns     8.370  actor_00000003858640_00002893_00000.88.pth
cumulative_returns     8.595  actor_00000003908680_00002931_00000.94.pth
cumulative_returns     8.812  actor_00000003958720_00002969_00000.91.pth
"""

if __name__ == '__main__':
    check_env()
    run()
    # run1()
    evaluate_models_in_directory()
