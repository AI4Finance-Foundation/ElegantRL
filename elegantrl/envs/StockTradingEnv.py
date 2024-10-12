import os
import torch as th
import numpy as np
import numpy.random as rd
from typing import Tuple

ARY = np.ndarray


class StockTradingEnv:
    def __init__(self, initial_amount=1e6, max_stock=1e2, cost_pct=1e-3, gamma=0.99,
                 beg_idx=0, end_idx=1113):
        self.df_pwd = './China_A_shares.pandas.dataframe'
        self.npz_pwd = './China_A_shares.numpy.npz'

        self.close_ary, self.tech_ary = self.load_data_from_disk()
        self.close_ary = self.close_ary[beg_idx:end_idx]
        self.tech_ary = self.tech_ary[beg_idx:end_idx]
        # print(f"| StockTradingEnv: close_ary.shape {self.close_ary.shape}")
        # print(f"| StockTradingEnv: tech_ary.shape {self.tech_ary.shape}")

        self.max_stock = max_stock
        self.cost_pct = cost_pct
        self.reward_scale = 2 ** -12
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
        self.max_step = self.close_ary.shape[0] - 1
        self.target_return = +np.inf

    def reset(self) -> Tuple[ARY, dict]:
        self.day = 0
        if self.if_random_reset:
            self.amount = self.initial_amount * rd.uniform(0.9, 1.1)
            self.shares = (np.abs(rd.randn(self.shares_num).clip(-2, +2)) * 2 ** 6).astype(int)
        else:
            self.amount = self.initial_amount
            self.shares = np.zeros(self.shares_num, dtype=np.float32)

        self.rewards = []
        self.total_asset = (self.close_ary[self.day] * self.shares).sum() + self.amount
        return self.get_state(), {}

    def get_state(self) -> ARY:
        state = np.hstack((np.tanh(np.array(self.amount * 2 ** -16)),
                           self.shares * 2 ** -9,
                           self.close_ary[self.day] * 2 ** -7,
                           self.tech_ary[self.day] * 2 ** -6,))
        return state

    def step(self, action) -> Tuple[ARY, float, bool, bool, dict]:
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
                self.amount -= adj_close_price * delta_stock * (1 + self.cost_pct)
                self.shares[index] += delta_stock
            elif self.shares[index] > 0:  # sell_stock
                delta_stock = min(-stock_action, self.shares[index])
                self.amount += adj_close_price * delta_stock * (1 - self.cost_pct)
                self.shares[index] -= delta_stock

        total_asset = (self.close_ary[self.day] * self.shares).sum() + self.amount
        reward = (total_asset - self.total_asset) * self.reward_scale
        self.rewards.append(reward)
        self.total_asset = total_asset

        terminal = self.day == self.max_step
        if terminal:
            reward += 1 / (1 - self.gamma) * np.mean(self.rewards)
            self.cumulative_returns = total_asset / self.initial_amount * 100

        state = self.get_state()
        truncated = False
        return state, reward, terminal, truncated, {}

    def load_data_from_disk(self, tech_id_list=None) -> Tuple[ARY, ARY]:
        tech_id_list = [
            "macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30", "close_30_sma", "close_60_sma",
        ] if tech_id_list is None else tech_id_list

        if os.path.exists(self.npz_pwd):
            ary_dict = np.load(self.npz_pwd, allow_pickle=True)
            close_ary = ary_dict['close_ary']
            tech_ary = ary_dict['tech_ary']
        elif os.path.exists(self.df_pwd):  # convert pandas.DataFrame to numpy.array
            import pandas as pd

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

            np.savez_compressed(self.npz_pwd, close_ary=close_ary, tech_ary=tech_ary, )
        else:
            error_str = f"| StockTradingEnv need {self.df_pwd} or {self.npz_pwd}" \
                        f"\n  download the following files and save in `.`" \
                        f"\n  https://github.com/Yonv1943/Python/blob/master/scow/China_A_shares.numpy.npz" \
                        f"\n  https://github.com/Yonv1943/Python/blob/master/scow/China_A_shares.pandas.dataframe"
            raise FileNotFoundError(error_str)
        return close_ary, tech_ary


'''function for vmap'''


def _inplace_amount_shares_when_buy(amount, shares, stock_action, close, cost_pct):
    stock_delta = th.min(stock_action, th.div(amount, close, rounding_mode='floor'))
    amount -= close * stock_delta * (1 + cost_pct)
    shares += stock_delta
    return th.zeros(1)


def _inplace_amount_shares_when_sell(amount, shares, stock_action, close, cost_rate):
    stock_delta = th.min(-stock_action, shares)
    amount += close * stock_delta * (1 - cost_rate)
    shares -= stock_delta
    return th.zeros(1)


class StockTradingVecEnv:
    def __init__(self, initial_amount=1e6, max_stock=1e2, cost_pct=1e-3, gamma=0.99,
                 beg_idx=0, end_idx=1113, num_envs=4, gpu_id=0):
        self.df_pwd = './elegantrl/envs/China_A_shares.pandas.dataframe'
        self.npz_pwd = './elegantrl/envs/China_A_shares.numpy.npz'
        self.device = th.device(f"cuda:{gpu_id}" if (th.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        '''load data'''
        close_ary, tech_ary = self.load_data_from_disk()
        close_ary = close_ary[beg_idx:end_idx]
        tech_ary = tech_ary[beg_idx:end_idx]
        self.close_price = th.tensor(close_ary, dtype=th.float32, device=self.device)
        self.tech_factor = th.tensor(tech_ary, dtype=th.float32, device=self.device)
        # print(f"| StockTradingEnv: close_ary.shape {close_ary.shape}")
        # print(f"| StockTradingEnv: tech_ary.shape {tech_ary.shape}")

        '''init'''
        self.gamma = gamma
        self.cost_pct = cost_pct
        self.max_stock = max_stock
        self.reward_scale = 2 ** -12
        self.initial_amount = initial_amount
        self.if_random_reset = True

        '''init (reset)'''
        self.day = None
        self.rewards = None
        self.total_asset = None
        self.cumulative_returns = None

        self.amount = None
        self.shares = None
        self.clears = None
        self.num_shares = self.close_price.shape[1]
        amount_dim = 1

        '''environment information'''
        self.env_name = 'StockTradingEnv-v2'
        self.num_envs = num_envs
        self.max_step = self.close_price.shape[0] - 1
        self.state_dim = self.num_shares + self.close_price.shape[1] + self.tech_factor.shape[1] + amount_dim
        self.action_dim = self.num_shares
        self.if_discrete = False

        '''vmap function'''
        self.vmap_get_state = th.vmap(
            func=lambda amount, shares, close, techs: th.cat((amount, shares, close, techs)),
            in_dims=(0, 0, None, None), out_dims=0)

        self.vmap_get_total_asset = th.vmap(
            func=lambda close, shares, amount: (close * shares).sum() + amount,
            in_dims=(None, 0, 0), out_dims=0)

        self.vmap_inplace_amount_shares_when_buy = th.vmap(
            func=_inplace_amount_shares_when_buy, in_dims=(0, 0, 0, None, None), out_dims=0)

        self.vmap_inplace_amount_shares_when_sell = th.vmap(
            func=_inplace_amount_shares_when_sell, in_dims=(0, 0, 0, None, None), out_dims=0)

    def reset(self):
        self.day = 0

        self.amount = th.zeros((self.num_envs, 1), dtype=th.float32, device=self.device) + self.initial_amount
        self.shares = th.zeros((self.num_envs, self.num_shares), dtype=th.float32, device=self.device)

        if self.if_random_reset:
            rand_amount = th.rand((self.num_envs, 1), dtype=th.float32, device=self.device) * 0.5 + 0.75
            self.amount = self.amount * rand_amount

            rand_shares = th.randn((self.num_envs, self.num_shares), dtype=th.float32, device=self.device)
            rand_shares = rand_shares.clip(-2, +2) * 2 ** 7
            self.shares = self.shares + th.abs(rand_shares).type(th.int32)

        self.rewards = list()
        self.total_asset = self.vmap_get_total_asset(self.close_price[self.day], self.shares, self.amount)
        return self.get_state()

    def get_state(self):
        return self.vmap_get_state((self.amount * 2 ** -18).tanh(),
                                   (self.shares * 2 ** -10).tanh(),
                                   self.close_price[self.day] * 2 ** -7,
                                   self.tech_factor[self.day] * 2 ** -6)  # state

    def step(self, action):
        self.day += 1
        if self.day == 1:
            self.cumulative_returns = 0.

        # action = action.clone()
        action = th.ones_like(action)
        action[(-0.1 < action) & (action < 0.1)] = 0
        action_int = (action * self.max_stock).to(th.int32)
        # actions initially is scaled between -1 and 1
        # convert `action` into integer as `stock_action`, because we can't buy fraction of shares

        for i in range(self.num_shares):
            buy_idx = th.where(action_int[:, i] > 0)[0]
            if buy_idx.shape[0] > 0:
                part_amount = self.amount[buy_idx]
                part_shares = self.shares[buy_idx, i]
                self.vmap_inplace_amount_shares_when_buy(part_amount, part_shares, action_int[buy_idx, i],
                                                         self.close_price[self.day, i], self.cost_pct)
                self.amount[buy_idx] = part_amount
                self.shares[buy_idx, i] = part_shares

            sell_idx = th.where((action_int < 0) & (self.shares > 0))[0]
            if sell_idx.shape[0] > 0:
                part_amount = self.amount[sell_idx]
                part_shares = self.shares[sell_idx, i]
                self.vmap_inplace_amount_shares_when_sell(part_amount, part_shares, action_int[sell_idx, i],
                                                          self.close_price[self.day, i], self.cost_pct)
                self.amount[sell_idx] = part_amount
                self.shares[sell_idx, i] = part_shares
        # for index in range(self.action_dim):
        #     stock_actions = action_int[:, index]
        #     close_price = self.close_price[self.day, index]
        #
        #     # delta_stock.shape == ()
        #     for i in range(self.num_envs):
        #         if stock_actions[i] > 0:  # buy_stock
        #             delta_stock = th.div(self.amount[i], close_price, rounding_mode='floor')
        #             delta_stock = th.min(delta_stock, stock_actions[0])
        #             self.amount[i] -= close_price * delta_stock * (1 + self.cost_pct)
        #             self.shares[i, index] = self.shares[i, index] + delta_stock
        #         elif self.shares[i, index] > 0:  # sell_stock
        #             delta_stock = th.min(-stock_actions[i], self.shares[i, index])
        #             self.amount[i] += close_price * delta_stock * (1 - self.cost_pct)
        #             self.shares[i, index] = self.shares[i, index] + delta_stock

        '''random clear the inventory'''
        # reset_rate = 1e-2 * self.num_shares / self.max_step
        # if self.if_random_reset and (rd.rand() < reset_rate):
        #     env_i = rd.randint(self.num_envs)
        #     shares_i = rd.randint(self.num_shares)
        #
        #     self.amount[env_i] = (self.amount[env_i] +
        #                           self.shares[env_i, shares_i] * self.close_price[self.day, shares_i])  # not cost_pct
        #     self.shares[env_i, shares_i] = 0

        '''get reward'''
        total_asset = self.vmap_get_total_asset(self.close_price[self.day], self.shares, self.amount)
        reward = (total_asset - self.total_asset).squeeze(1) * self.reward_scale  # shape == (num_envs, )
        self.rewards.append(reward)
        self.total_asset = total_asset

        '''get done and state'''
        done = self.day == self.max_step
        if done:
            reward += th.stack(self.rewards).mean(dim=0) * (1. / (1. - self.gamma))
            self.cumulative_returns = (total_asset / self.initial_amount) * 100
            self.cumulative_returns = self.cumulative_returns.squeeze(1).cpu().data.tolist()

        state = self.reset() if done else self.get_state()  # automatically reset in vectorized env
        done = th.tensor(done, dtype=th.bool, device=self.device).expand(self.num_envs)
        return state, reward, done, ()

    def load_data_from_disk(self, tech_id_list=None):
        tech_id_list = [
            "macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30", "close_30_sma", "close_60_sma",
        ] if tech_id_list is None else tech_id_list

        if os.path.exists(self.npz_pwd):
            ary_dict = np.load(self.npz_pwd, allow_pickle=True)
            close_ary = ary_dict['close_ary']
            tech_ary = ary_dict['tech_ary']
        elif os.path.exists(self.df_pwd):  # convert pandas.DataFrame to numpy.array
            import pandas as pd

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

            np.savez_compressed(self.npz_pwd, close_ary=close_ary, tech_ary=tech_ary, )
        else:
            error_str = f"| StockTradingEnv need {self.df_pwd} or {self.npz_pwd}" \
                        f"\n  download the following files and save in `.`" \
                        f"\n  https://github.com/Yonv1943/Python/blob/master/scow/China_A_shares.numpy.npz" \
                        f"\n  https://github.com/Yonv1943/Python/blob/master/scow/China_A_shares.pandas.dataframe"
            raise FileNotFoundError(error_str)
        return close_ary, tech_ary


def check_stock_trading_env():
    env = StockTradingEnv(beg_idx=834, end_idx=1113)
    env.if_random_reset = False
    evaluate_time = 4

    print()
    policy_name = 'random action (if_random_reset = False)'
    state, info_dict = env.reset()
    for _ in range(env.max_step * evaluate_time):
        action = rd.uniform(-1, +1, env.action_dim)
        state, reward, terminal, truncated, info_dict = env.step(action)
        done = terminal or truncated
        if done:
            print(f'cumulative_returns of {policy_name}: {env.cumulative_returns:9.2f}')
            state, info_dict = env.reset()
    print(state.shape)

    print()
    policy_name = 'buy all share (if_random_reset = True)'
    env.if_random_reset = True
    state, info_dict = env.reset()
    for _ in range(env.max_step * evaluate_time):
        action = np.ones(env.action_dim, dtype=np.float32)
        state, reward, terminal, truncated, info_dict = env.step(action)
        done = terminal or truncated
        if done:
            print(f'cumulative_returns of {policy_name}: {env.cumulative_returns:9.2f}')
            state, info_dict = env.reset()
    print(state.shape)
    print()


if __name__ == '__main__':
    check_stock_trading_env()
