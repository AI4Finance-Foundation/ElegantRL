import os
import time
import sys
from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np
import numpy.random as rd
import pandas as pd

"""finance environment
Source: https://github.com/AI4Finance-Foundation/FinRL-Meta/blob/master/Demo_China_A_share_market.ipynb
Source: https://github.com/AI4Fiance-Foundation/ElegantRL
Modify: Github YonV1943
"""


class StockTradingEnv:
    def __init__(self, initial_amount=1e6, max_stock=1e2, buy_cost_pct=1e-3, sell_cost_pct=1e-3, gamma=0.99,
                 beg_idx=0, end_idx=1113):
        self.df_pwd = './China_A_shares.pandas.dataframe'
        self.npz_pwd = './China_A_shares.numpy.npz'

        self.close_ary, self.tech_ary = self.load_data_from_disk()
        self.close_ary = self.close_ary[beg_idx:end_idx]
        self.tech_ary = self.tech_ary[beg_idx:end_idx]
        print(f"| StockTradingEnv: close_ary.shape {self.close_ary.shape}")
        print(f"| StockTradingEnv: tech_ary.shape {self.tech_ary.shape}")

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
        self.if_random_reset = True

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
        self.target_return = +np.inf

    def reset(self):
        self.day = 0
        if self.if_random_reset:
            self.amount = self.initial_amount * rd.uniform(0.9, 1.1)
            self.shares = (np.abs(rd.randn(self.shares_num).clip(-2, +2)) * 2 ** 6).astype(int)
        else:
            self.amount = self.initial_amount
            self.shares = np.zeros(self.shares_num, dtype=np.float32)

        self.rewards = list()
        self.total_asset = (self.close_ary[self.day] * self.shares).sum() + self.amount
        return self.get_state()

    def get_state(self):
        state = np.hstack((np.array(self.amount * 2 ** -16),
                           self.shares * 2 ** -9,
                           self.close_ary[self.day] * 2 ** -7,
                           self.tech_ary[self.day] * 2 ** -6,))
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
            "macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30", "close_30_sma", "close_60_sma",
        ] if tech_id_list is None else tech_id_list

        if os.path.exists(self.npz_pwd):
            ary_dict = np.load(self.npz_pwd, allow_pickle=True)
            close_ary = ary_dict['close_ary']
            tech_ary = ary_dict['tech_ary']
        elif os.path.exists(self.df_pwd):  # convert pandas.DataFrame to numpy.array
            df = pd.read_pickle(self.df_pwd)

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

            np.savez_compressed(self.npz_pwd, close_ary=close_ary, tech_ary=tech_ary, )
        else:
            error_str = f"| StockTradingEnv need {self.df_pwd} or {self.npz_pwd}" \
                        f"\n  download the following files and save in `.`" \
                        f"\n  https://github.com/Yonv1943/Python/blob/master/scow/China_A_shares.numpy.npz" \
                        f"\n  https://github.com/Yonv1943/Python/blob/master/scow/China_A_shares.pandas.dataframe (2.1MB)"
            raise FileNotFoundError(error_str)
        return close_ary, tech_ary


def check_env():
    env = StockTradingEnv(beg_idx=834, end_idx=1113)
    env.if_random_reset = False
    evaluate_time = 4
    """
    env = StockTradingEnv(beg_idx=0, end_idx=1113)
    cumulative_returns of random action   :      1.63
    cumulative_returns of buy all share   :      2.80

    env = StockTradingEnv(beg_idx=0, end_idx=834)
    cumulative_returns of random action   :      1.94
    cumulative_returns of buy all share   :      2.51

    env = StockTradingEnv(beg_idx=834, end_idx=1113)
    cumulative_returns of random action   :      1.12
    cumulative_returns of buy all share   :      1.19
    """

    print()
    policy_name = 'random action'
    state = env.reset()
    for _ in range(env.max_step * evaluate_time):
        action = rd.uniform(-1, +1, env.action_dim)
        state, reward, done, _ = env.step(action)
        if done:
            print(f'cumulative_returns of {policy_name}: {env.cumulative_returns:9.2f}')
            state = env.reset()
    dir(state)

    print()
    policy_name = 'buy all share'
    state = env.reset()
    for _ in range(env.max_step * evaluate_time):
        action = np.ones(env.action_dim, dtype=np.float32)
        state, reward, done, _ = env.step(action)
        if done:
            print(f'cumulative_returns of {policy_name}: {env.cumulative_returns:9.2f}')
            state = env.reset()
    dir(state)
    print()


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


'''reinforcement learning
Source: https://github.com/AI4Finance-Foundation/ElegantRL/tree/master/elegantrl_helloworld
Modify: Github YonV1943
'''


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
        delta = noise.pow(2) * 0.5
        return -(self.a_std_log + self.sqrt_2pi_log + delta).sum(1)  # old_logprob

    def get_logprob_entropy(self, state, action):
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


class AgentPPO:
    def __init__(self, net_dim, state_dim, action_dim, gpu_id=0, args=None):
        self.if_off_policy = False
        self.act_class = getattr(self, "act_class", ActorPPO)
        self.cri_class = getattr(self, "cri_class", CriticPPO)
        self.if_act_target = getattr(args, 'if_act_target', False)
        self.if_cri_target = getattr(args, "if_cri_target", False)
        # AgentBase.__init__(self, net_dim, state_dim, action_dim, gpu_id, args)
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

    def convert_trajectory(self, traj_list, _last_done):  # [ElegantRL.2022.01.01]
        # assert len(buf_items) == step_i
        # assert len(buf_items[0]) in {4, 5}
        # assert len(buf_items[0][0]) == self.env_num
        traj_list = list(map(list, zip(*traj_list)))  # state, reward, done, action, noise
        # assert len(buf_items) == {4, 5}
        # assert len(buf_items[0]) == step
        # assert len(buf_items[0][0]) == self.env_num

        '''stack items'''
        traj_list[0] = torch.stack(traj_list[0]).squeeze(1)
        traj_list[1] = (torch.tensor(traj_list[1], dtype=torch.float32) * self.reward_scale).unsqueeze(1)
        traj_list[2] = ((1 - torch.tensor(traj_list[2], dtype=torch.float32)) * self.gamma).unsqueeze(1)
        traj_list[3:] = [torch.stack(item).squeeze(1) for item in traj_list[3:]]
        # assert all([buf_item.shape[:2] == (step, self.env_num) for buf_item in buf_items])
        return traj_list

    @staticmethod
    def optimizer_update(optimizer, objective):
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()


class ReplayBufferList(list):  # for on-policy
    def __init__(self):
        list.__init__(self)

    def update_buffer(self, traj_list):
        cur_items = list(map(list, zip(*traj_list)))
        self[:] = [torch.cat(item, dim=0) for item in cur_items]

        steps = self[1].shape[0]
        r_exp = self[1].mean().item()
        return steps, r_exp


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


def train_agent(args):
    torch.set_grad_enabled(False)
    args.init_before_training()
    gpu_id = args.learner_gpus

    '''init'''
    env = build_env(args.env_func, args.env_args)

    agent = args.agent_class(args.net_dim, args.state_dim, args.action_dim, gpu_id=gpu_id, args=args)
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


def load_torch_file(model, _path):
    state_dict = torch.load(_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)


"""train and evaluate"""


def run():
    import sys
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    env = StockTradingEnv()
    env_func = StockTradingEnv
    env_args = get_gym_env_args(env=env, if_print=False)
    env_args['beg_idx'] = 0  # training set
    env_args['end_idx'] = 834  # training set

    args = Arguments(AgentPPO, env_func=env_func, env_args=env_args)
    args.target_step = args.max_step * 4
    args.reward_scale = 2 ** -7
    args.learning_rate = 2 ** -14
    args.break_step = int(5e5)

    args.learner_gpus = gpu_id
    args.random_seed += gpu_id + 1943
    train_agent(args)


def evaluate_models_in_directory(dir_path=None):
    if dir_path is None:
        gpu_id = int(sys.argv[1])
        dir_path = f'StockTradingEnv-v2_PPO_{gpu_id}'
        print(f"| evaluate_models_in_directory: gpu_id {gpu_id}")
        print(f"| evaluate_models_in_directory: dir_path {dir_path}")
    else:
        gpu_id = -1
        print(f"| evaluate_models_in_directory: gpu_id {gpu_id}")
        print(f"| evaluate_models_in_directory: dir_path {dir_path}")

    model_names = [name for name in os.listdir(dir_path) if name[:6] == 'actor_']
    model_names.sort()

    env_func = StockTradingEnv
    env_args = {
        'env_num': 1,
        'env_name': 'StockTradingEnv-v2',
        'max_step': 1113 - 834,
        'state_dim': 151,
        'action_dim': 15,
        'if_discrete': False,

        'beg_idx': 834,  # testing set
        'end_idx': 1113,  # testing set
    }
    env = build_env(env_func=env_func, env_args=env_args)
    env.if_random_reset = False

    args = Arguments(AgentPPO, env_func=env_func, env_args=env_args)

    device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
    actor = ActorPPO(mid_dim=args.net_dim,
                     mid_layer_num=args.mid_layer_num,
                     state_dim=args.state_dim,
                     action_dim=args.action_dim).to(device)

    for model_name in model_names:
        model_path = f"{dir_path}/{model_name}"
        load_torch_file(actor, model_path)

        cumulative_returns_list = [get_episode_return_and_step(env, actor)[0] for _ in range(4)]
        cumulative_returns = np.mean(cumulative_returns_list)
        print(f"cumulative_returns {cumulative_returns:9.3f}  {model_name}")


if __name__ == '__main__':
    check_env()
    run()
    evaluate_models_in_directory()
