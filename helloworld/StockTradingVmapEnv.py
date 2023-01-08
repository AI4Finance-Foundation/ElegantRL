import os

import torch
import numpy as np
import numpy.random as rd
import pandas as pd

from functorch import vmap

"""finance environment
Source: 
https://github.com/AI4Finance-Foundation/FinRL-Meta/blob/master/Demo_China_A_share_market.ipynb
Modify: Github YonV1943
"""

'''vmap function'''


def _get_total_asset(close, shares, amount):
    return (close * shares).sum() + amount  # total_asset


def _get_state(amount, shares, close, tech):
    return torch.cat((amount, shares, close, tech))


def _inplace_amount_shares_when_buy(amount, shares, stock_action, close, buy_cost_rate):
    stock_delta = torch.min(stock_action, torch.div(amount, close, rounding_mode='floor'))
    amount -= close * stock_delta * buy_cost_rate
    shares += stock_delta
    return torch.zeros(1)


def _inplace_amount_shares_when_sell(amount, shares, stock_action, close, sell_cost_rate):
    stock_delta = torch.min(-stock_action, shares)
    amount += close * stock_delta * sell_cost_rate
    shares -= stock_delta
    return torch.zeros(1)


class StockTradingVmapEnv:
    def __init__(self, initial_amount=1e6, max_stock=100, buy_cost_pct=1e-3, sell_cost_pct=1e-3, gamma=0.99,
                 beg_idx=0, end_idx=1113, gpu_id: int = 0, num_envs: int = 4):
        self.df_pwd = './China_A_shares.pandas.dataframe'

        '''load data'''
        close_ary, tech_ary = self.load_data_from_disk()
        close_ary = close_ary[beg_idx:end_idx]
        tech_ary = tech_ary[beg_idx:end_idx]
        print(f"| StockTradingEnv: close_ary.shape {close_ary.shape}")
        print(f"| StockTradingEnv: tech_ary.shape {tech_ary.shape}")

        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        self.num_envs = num_envs

        self.close_price = torch.tensor(close_ary, dtype=torch.float32, device=self.device)
        self.tech_factor = torch.tensor(tech_ary, dtype=torch.float32, device=self.device)

        '''init'''
        self.gamma = gamma
        self.max_stock = max_stock
        self.initial_amount = initial_amount

        self.max_step = self.close_price.shape[0]
        self.buy_cost_rate = 1. + buy_cost_pct
        self.sell_cost_rate = 1. - sell_cost_pct

        '''init (set in reset)'''
        self.day = None
        self.rewards = None
        self.total_asset = None
        self.if_random_reset = True
        self.cumulative_returns = None

        self.amount = None
        self.shares = None
        self.shares_num = self.close_price.shape[1]
        amount_dim = 1

        '''environment information'''
        self.env_name = 'StockTradingEnvVMAP-v2'
        self.state_dim = self.shares_num + self.close_price.shape[1] + self.tech_factor.shape[1] + amount_dim
        self.action_dim = self.shares_num
        self.if_discrete = False

        '''vmap function'''
        self.vmap_get_total_asset = vmap(
            func=_get_total_asset, in_dims=(None, 0, 0), out_dims=0)

        self.vmap_get_state = vmap(
            func=_get_state, in_dims=(0, 0, None, None), out_dims=0)

        self.vmap_inplace_amount_shares_when_buy = vmap(
            func=_inplace_amount_shares_when_buy, in_dims=(0, 0, 0, None, None), out_dims=0)

        self.vmap_inplace_amount_shares_when_sell = vmap(
            func=_inplace_amount_shares_when_sell, in_dims=(0, 0, 0, None, None), out_dims=0)

    def reset(self):
        self.day = 0

        self.amount = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device) + self.initial_amount
        self.shares = torch.zeros((self.num_envs, self.shares_num), dtype=torch.float32, device=self.device)

        if self.if_random_reset:
            self.amount *= torch.rand((self.num_envs, 1), dtype=torch.float32, device=self.device) * 0.10 + 0.95
            self.shares += torch.randint(0, int(self.max_stock),
                                         size=(self.num_envs, self.shares_num), device=self.device)

        self.rewards = list()

        self.total_asset = self.vmap_get_total_asset(self.close_price[self.day],
                                                     self.shares,
                                                     self.amount)
        state = self.get_state()
        return state

    def get_state(self):
        return self.vmap_get_state(self.amount * 2 ** 16,
                                   self.shares * 2 ** -9,
                                   self.close_price[self.day] * 2 ** -7,
                                   self.tech_factor[self.day] * 2 ** -6)  # state

    def step(self, action):
        self.day += 1

        action = action.clone()
        action[(-0.1 < action) & (action < 0.1)] = 0
        stock_action = (action * self.max_stock).to(torch.int32)
        # actions initially is scaled between -1 and 1
        # convert `action` into integer as `stock_action`, because we can't buy fraction of shares

        for i in range(self.shares_num):
            buy_idx = torch.where(stock_action[:, i] > 0)[0]
            if buy_idx.shape[0] > 0:
                part_amount = self.amount[buy_idx]
                part_shares = self.shares[buy_idx, i]
                self.vmap_inplace_amount_shares_when_buy(part_amount,
                                                         part_shares,
                                                         stock_action[buy_idx, i],
                                                         self.close_price[self.day, i],
                                                         self.buy_cost_rate)
                self.amount[buy_idx] = part_amount
                self.shares[buy_idx, i] = part_shares

            sell_idx = torch.where((stock_action < 0) & (self.shares > 0))[0]
            if sell_idx.shape[0] > 0:
                part_amount = self.amount[sell_idx]
                part_shares = self.shares[sell_idx, i]
                self.vmap_inplace_amount_shares_when_sell(part_amount,
                                                          part_shares,
                                                          stock_action[sell_idx, i],
                                                          self.close_price[self.day, i],
                                                          self.sell_cost_rate)
                self.amount[sell_idx] = part_amount
                self.shares[sell_idx, i] = part_shares

        state = self.get_state()

        total_asset = self.vmap_get_total_asset(self.close_price[self.day],
                                                self.shares,
                                                self.amount)

        reward = (total_asset - self.total_asset) * 2 ** -6
        self.rewards.append(reward)
        self.total_asset = total_asset

        done = self.day == self.max_step - 1
        if done:
            reward += 1. / (1. - self.gamma) * torch.stack(self.rewards).mean(dim=0)
            self.cumulative_returns = total_asset / self.initial_amount
            self.cumulative_returns = self.cumulative_returns.mean().item()
        done = torch.tensor(done, dtype=torch.bool, device=self.device).expand(self.num_envs)
        return state, reward, done, {}

    def load_data_from_disk(self, tech_id_list=None):
        tech_id_list = [
            "macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30", "close_30_sma", "close_60_sma",
        ] if tech_id_list is None else tech_id_list

        if os.path.exists(self.df_pwd):  # convert pandas.DataFrame to numpy.array
            df = pd.read_pickle(self.df_pwd)

            tech_ary = []
            close_ary = []
            df_len = len(df.index.unique())  # df_len = max_step
            for day in range(df_len):
                item = df.loc[day]

                tech_items = [item[tech].values.tolist() for tech in tech_id_list]
                tech_items_flatten = sum(tech_items, [])
                tech_ary.append(tech_items_flatten)

                close_ary.append(item.close)

            close_ary = np.array(close_ary)
            tech_ary = np.array(tech_ary)
        else:
            error_str = f"| StockTradingEnv need {self.df_pwd}" \
                        f"\n  download the following files and save in `.`" \
                        f"\n  https://github.com/Yonv1943/Python/blob/master/scow/China_A_shares.pandas.dataframe (2MB)"
            raise FileNotFoundError(error_str)
        return close_ary, tech_ary


def check_env():
    gpu_id = 0
    env_num = 32

    env = StockTradingVmapEnv(beg_idx=834, end_idx=1113, gpu_id=gpu_id, num_envs=env_num)
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
        action = torch.rand((env.num_envs, env.action_dim), dtype=torch.float32, device=env.device) * 2. - 1.
        state, reward, done, _ = env.step(action)
        if torch.all(done):
            print(f'cumulative_returns of {policy_name}: {env.cumulative_returns:9.2f}')
            state = env.reset()
    dir(state)

    print()
    policy_name = 'buy all share (if_random_reset = False)'
    env.if_random_reset = False
    state = env.reset()
    for _ in range(env.max_step * evaluate_time):
        action = torch.ones((env.num_envs, env.action_dim), dtype=torch.float32, device=env.device) * 2. - 1.
        state, reward, done, _ = env.step(action)
        if torch.all(done):
            print(f'cumulative_returns of {policy_name}: {env.cumulative_returns:9.2f}')
            state = env.reset()
    dir(state)
    print()

    print()
    policy_name = 'buy all share (if_random_reset = True)'
    env.if_random_reset = True
    state = env.reset()
    for _ in range(env.max_step * evaluate_time):
        action = torch.ones((env.num_envs, env.action_dim), dtype=torch.float32, device=env.device) * 2. - 1.
        state, reward, done, _ = env.step(action)
        if torch.all(done):
            print(f'cumulative_returns of {policy_name}: {env.cumulative_returns:9.2f}')
            state = env.reset()
    dir(state)
    print()


if __name__ == '__main__':
    check_env()
