import os
import numpy as np
import numpy.random as rd

"""
1441 args.max_step = 1699
beta0 max_memo = 1699 * 16, batch_size = 2 ** 10
beta2 max_memo = 1699 * 8,  batch_size = 2 ** 9

ceta1 max_memo = 1699 * 32, batch_size = 2 ** 11
ceta4 max_memo = 1699 * 16, batch_size = 2 ** 11
ceta2 max_memo = 1699 * 8, batch_size = 2 ** 10

1647 args.max_step = 1699
ceta2 gamma_r args.max_memo = 1699 * 4
ceta3 gamma_r args.max_memo = 1699 * 8
"""


class FinanceMultiStock2018:  # todo 2020-12-21 16:00
    """FinRL
    Paper: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance
           https://arxiv.org/abs/2011.09607 NeurIPS 2020: Deep RL Workshop.
    Source: Github https://github.com/AI4Finance-LLC/FinRL-Library
    Modify: Github Yonv1943 ElegantRL
    """

    def __init__(self, initial_account=1e6, transaction_fee_percent=1e-3, max_stock=100):
        self.stock_dim = 30
        self.initial_account = initial_account
        self.transaction_fee_percent = transaction_fee_percent
        self.max_stock = max_stock

        self.ary = self.load_csv_for_multi_stock()
        assert self.ary.shape == (1699, 5 * 30)  # ary: (date, item*stock_dim), item: (adjcp, macd, rsi, cci, adx)
        self.max_day = self.ary.shape[0] - 1

        # reset
        self.day = 0
        self.account = self.initial_account
        self.day_npy = self.ary[self.day]
        self.stocks = np.zeros(self.stock_dim, dtype=np.float32)  # multi-stack
        self.begin_total_asset = self.account + (self.day_npy[:self.stock_dim] * self.stocks).sum()
        # total_asset = account + (adjcp * stocks).sum()

        '''env information'''
        self.env_name = 'FinanceStock-v1'
        self.state_dim = 1 + (5 + 1) * self.stock_dim
        self.action_dim = self.stock_dim
        self.if_discrete = False
        self.target_reward = 800  # todo need to update

    def reset(self):
        self.day = 0
        self.account = self.initial_account * rd.uniform(0.99, 1.01)  # todo
        self.day_npy = self.ary[self.day]
        self.stocks = np.zeros(self.stock_dim, dtype=np.float32)
        self.begin_total_asset = self.account + (self.day_npy[:self.stock_dim] * self.stocks).sum()
        # total_asset = account + (adjcp * stocks).sum()

        state = np.hstack((
            self.account * 2 ** -16,
            self.day_npy * 2 ** -8,
            self.stocks * 2 ** -12,
        ), ).astype(np.float32)

        return state

    def step(self, actions):
        actions = actions * self.max_stock

        """bug or sell stock"""
        for index in range(self.stock_dim):
            action = actions[index]
            adj = self.day_npy[index]
            if action > 0:  # buy_stock
                available_amount = self.account // adj
                delta_stock = min(available_amount, action)
                self.account -= adj * delta_stock * (1 + self.transaction_fee_percent)
                self.stocks[index] += delta_stock
            elif self.stocks[index] > 0:  # sell_stock
                delta_stock = min(-action, self.stocks[index])
                self.account += adj * delta_stock * (1 - self.transaction_fee_percent)
                self.stocks[index] -= delta_stock

        """update day"""
        self.day += 1
        self.day_npy = self.ary[self.day]

        state = np.hstack((
            self.account * 2 ** -16,
            self.day_npy * 2 ** -8,
            self.stocks * 2 ** -12,
        ), ).astype(np.float32)

        end_total_asset = self.account + (self.day_npy[:self.stock_dim] * self.stocks).sum()
        reward = (end_total_asset - self.begin_total_asset) * 2 ** -16  # notice scaling! todo -14
        self.begin_total_asset = end_total_asset

        done = self.day == self.max_day

        # self.gamma_r = self.gamma_r * 0.99 + reward  # todo gamma_r seems good?
        # if done:
        #     reward += self.gamma_r
        #     self.gamma_r = 0.0

        return state, reward, done, None

    @staticmethod
    def load_csv_for_multi_stock(if_load=True):  # todo need independent


        from preprocessing.preprocessors import pd, data_split, preprocess_data, add_turbulence

        # the following is same as part of run_model()
        preprocessed_path = "done_data.csv"
        if if_load and os.path.exists(preprocessed_path):
            data = pd.read_csv(preprocessed_path, index_col=0)
        else:
            data = preprocess_data()
            data = add_turbulence(data)
            data.to_csv(preprocessed_path)

        df = data
        rebalance_window = 63
        validation_window = 63
        i = rebalance_window + validation_window

        unique_trade_date = data[(data.datadate > 20151001) & (data.datadate <= 20200707)].datadate.unique()
        train__df = data_split(df, start=20090000, end=unique_trade_date[i - rebalance_window - validation_window])
        # print(train__df) # df: DataFrame of Pandas

        train_ary = train__df.to_numpy().reshape((-1, 30, 12))
        '''state_dim = 1 + 6 * stock_dim, stock_dim=30
        n   item    index
        1   ACCOUNT -
        30  adjcp   2  
        30  stock   -
        30  macd    7
        30  rsi     8
        30  cci     9
        30  adx     10
        '''
        data_ary = np.empty((train_ary.shape[0], 5, 30), dtype=np.float32)
        data_ary[:, 0] = train_ary[:, :, 2]  # adjcp
        data_ary[:, 1] = train_ary[:, :, 7]  # macd
        data_ary[:, 2] = train_ary[:, :, 8]  # rsi
        data_ary[:, 3] = train_ary[:, :, 9]  # cci
        data_ary[:, 4] = train_ary[:, :, 10]  # adx

        data_ary = data_ary.reshape((-1, 5 * 30))
        return data_ary


def run__fin_rl_1441():
    env = FinanceMultiStock2018()  # todo 2020-12-21 16:00

    from AgentRun import Arguments, train_agent_mp
    from AgentZoo import AgentPPO

    args = Arguments(rl_agent=AgentPPO, env=env)
    args.eval_times1 = 1
    args.eval_times2 = 2

    args.reward_scale = 2 ** 0  # 10 ~ 200+
    args.break_step = int(4e6 * 4)  # UsedTime: 3800
    args.net_dim = 2 ** 8
    args.max_step = 1699
    args.max_memo = 1699 * 16  # todo  larger is better?
    args.batch_size = 2 ** 10  # todo
    args.repeat_times = 2 ** 4  # larger is better?
    args.init_for_training()
    train_agent_mp(args)  # train_agent(args)


if __name__ == '__main__':
    run__fin_rl_1441()
