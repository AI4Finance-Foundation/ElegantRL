import numpy as np
import numpy.random as rd


class SingleStockFinEnv:  # adjust state, inner df_pandas, beta3 pass
    """FinRL
    Paper: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance
           https://arxiv.org/abs/2011.09607 NeurIPS 2020: Deep RL Workshop.
    Source: Github https://github.com/AI4Finance-LLC/FinRL-Library
    Modify: Github Yonv1943 ElegantRL
    """

    """ Update Log 2020-12-12 by Github Yonv1943
    change download_preprocess_data: If the data had been downloaded, then don't download again

    # env 
    move reward_memory out of Env
    move plt.savefig('account_value.png') out of Env
    cancel SingleStockEnv(gym.Env): There is not need to use OpenAI's gym
    change pandas to numpy
    fix bug in comment: ('open', 'high', 'low', 'close', 'adjcp', 'volume', 'macd'), lack 'macd' before
    change slow 'state'
    change repeat compute 'begin_total_asset', 'end_total_asset'
    cancel self.asset_memory
    cancel self.cost
    cancel self.trade
    merge '_sell_stock' and '_bug_stock' to _sell_or_but_stock
    adjust order of state 
    reserved expansion interface on self.stock self.stocks

    # compatibility
    move global variable into Env.__init__()
    cancel matplotlib.use('Agg'): It will cause compatibility issues for ssh connection
    """

    def __init__(self, initial_account=100000, transaction_fee_percent=0.001, max_stock=200):
        # state_dim, action_dim = 4, 1

        self.stock_dim = 1
        self.initial_account = initial_account
        self.transaction_fee_percent = transaction_fee_percent
        self.max_stock = max_stock
        self.state_div_std = np.array((2 ** -14, 2 ** -4, 2 ** 0, 2 ** -11))

        self.ary = self.download_data_as_csv__load_as_array()
        assert self.ary.shape == (2517, 9)  # ary: (date, item)
        self.ary = self.ary[1:, 2:].astype(np.float32)
        assert self.ary.shape == (2516, 7)  # ary: (date, item), item: (open, high, low, close, adjcp, volume, macd)
        self.ary = np.concatenate((
            self.ary[:, 4:5],  # adjcp? What is this? unit price?
            self.ary[:, 6:7],  # macd? What is this?
        ), axis=1)
        self.max_day = self.ary.shape[0] - 1

        # reset
        self.day = 0
        self.account = self.initial_account
        self.day_npy = self.ary[self.day]
        # self.stocks = np.zeros(self.stock_dim, dtype=np.float32) # multi-stack
        self.stock = 0
        # self.begin_total_asset = self.account + (self.day_npy[:self.stock_dim] * self.stocks).sum()
        self.begin_total_asset = self.account + self.day_npy[0] * self.stock

    def reset(self):
        self.day = 0
        self.account = self.initial_account
        self.day_npy = self.ary[self.day]
        # self.stocks = np.zeros(self.stock_dim, dtype=np.float32)
        self.stock = 0
        # self.begin_total_asset = self.account + (self.day_npy[:self.stock_dim] * self.stocks).sum()
        self.begin_total_asset = self.account + self.day_npy[0] * self.stock
        # state = np.hstack((self.account, self.day_npy, self.stocks)
        #                   ).astype(np.float32) * self.state_div_std
        state = np.hstack((self.account, self.day_npy, self.stock)
                          ).astype(np.float32) * self.state_div_std
        return state

    def step(self, actions):
        actions = actions * self.max_stock

        """bug or sell stock"""
        index = 0
        action = actions[index]
        adj = self.day_npy[index]
        if action > 0:  # buy_stock
            available_amount = self.account // adj
            delta_stock = min(available_amount, action)
            self.account -= adj * delta_stock * (1 + self.transaction_fee_percent)
            # self.stocks[index] += delta_stock
            self.stock += delta_stock
        # elif self.stocks[index] > 0:  # sell_stock
        #     delta_stock = min(-action, self.stocks[index])
        #     self.account += adj * delta_stock * (1 - self.transaction_fee_percent)
        #     self.stocks[index] -= delta_stock
        elif self.stock > 0:  # sell_stock
            delta_stock = min(-action, self.stock)
            self.account += adj * delta_stock * (1 - self.transaction_fee_percent)
            self.stock -= delta_stock

        """update day"""
        self.day += 1
        # self.data = self.df.loc[self.day, :]
        self.day_npy = self.ary[self.day]

        # state = np.hstack((self.account, self.day_npy, self.stocks)
        #                   ).astype(np.float32) * self.state_div_std
        state = np.hstack((self.account, self.day_npy, self.stock)
                          ).astype(np.float32) * self.state_div_std

        # end_total_asset = self.account + (self.day_npy[:self.stock_dim] * self.stocks).sum()
        end_total_asset = self.account + self.day_npy[0] * self.stock
        reward = end_total_asset - self.begin_total_asset
        self.begin_total_asset = end_total_asset

        done = self.day == self.max_day  # 2516 is over
        return state, reward * 2 ** -10, done, None

    @staticmethod
    def download_data_as_csv__load_as_array(if_load=True):
        save_path = './AAPL_2009_2020.csv'

        import os
        if if_load and os.path.isfile(save_path):
            ary = np.genfromtxt(save_path, delimiter=',')
            assert isinstance(ary, np.ndarray)
            return ary

        import yfinance as yf
        from stockstats import StockDataFrame as Sdf
        """ pip install
        !pip install yfinance
        !pip install pandas
        !pip install matplotlib
        !pip install stockstats
        """

        """# Part 1: Download Data
        Yahoo Finance is a website that provides stock data, financial news, financial reports, etc. 
        All the data provided by Yahoo Finance is free.
        """
        print('| download_preprocess_data_as_csv: Download Data')

        data_pd = yf.download("AAPL", start="2009-01-01", end="2020-10-23")
        assert data_pd.shape == (2974, 6)

        data_pd = data_pd.reset_index()

        data_pd.columns = ['datadate', 'open', 'high', 'low', 'close', 'adjcp', 'volume']

        """# Part 2: Preprocess Data
        Data preprocessing is a crucial step for training a high quality machine learning model. 
        We need to check for missing data and do feature engineering 
        in order to convert the data into a model-ready state.
        """
        print('| download_preprocess_data_as_csv: Preprocess Data')

        # check missing data
        data_pd.isnull().values.any()

        # calculate technical indicators like MACD
        stock = Sdf.retype(data_pd.copy())
        # we need to use adjusted close price instead of close price
        stock['close'] = stock['adjcp']
        data_pd['macd'] = stock['macd']

        # check missing data again
        data_pd.isnull().values.any()
        data_pd.head()
        # data_pd=data_pd.fillna(method='bfill')

        # Note that I always use a copy of the original data to try it track step by step.
        data_clean = data_pd.copy()
        data_clean.head()
        data_clean.tail()

        data = data_clean[(data_clean.datadate >= '2009-01-01') & (data_clean.datadate < '2019-01-01')]
        data = data.reset_index(drop=True)  # the index needs to start from 0

        data.to_csv(save_path)  # save *.csv
        # assert isinstance(data_pd, pd.DataFrame)

        df_pandas = data[(data.datadate >= '2009-01-01') & (data.datadate < '2019-01-01')]
        df_pandas = df_pandas.reset_index(drop=True)  # the index needs to start from 0
        ary = df_pandas.to_numpy()
        return ary


def original_download_pandas_data():
    import yfinance as yf
    from stockstats import StockDataFrame as Sdf

    # Download and save the data in a pandas DataFrame:
    data_df = yf.download("AAPL", start="2009-01-01", end="2020-10-23")

    data_df.shape

    # reset the index, we want to use numbers instead of dates
    data_df = data_df.reset_index()

    data_df.head()

    data_df.columns

    # convert the column names to standardized names
    data_df.columns = ['datadate', 'open', 'high', 'low', 'close', 'adjcp', 'volume']

    # save the data to a csv file in your current folder
    # data_df.to_csv('AAPL_2009_2020.csv')

    """# Part 2: Preprocess Data
    Data preprocessing is a crucial step for training a high quality machine learning model. We need to check for missing data and do feature engineering in order to convert the data into a model-ready state.
    """

    # check missing data
    data_df.isnull().values.any()

    # calculate technical indicators like MACD
    stock = Sdf.retype(data_df.copy())
    # we need to use adjusted close price instead of close price
    stock['close'] = stock['adjcp']
    data_df['macd'] = stock['macd']

    # check missing data again
    data_df.isnull().values.any()

    data_df.head()

    # data_df=data_df.fillna(method='bfill')

    # Note that I always use a copy of the original data to try it track step by step.
    data_clean = data_df.copy()

    data_clean.head()

    data_clean.tail()
    return data_clean


import gym


class SingleStockFinEnvForStableBaseLines(gym.Env):  # adjust state, inner df_pandas, beta3 pass
    """FinRL
    Paper: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance
           https://arxiv.org/abs/2011.09607 NeurIPS 2020: Deep RL Workshop.
    Source: Github https://github.com/AI4Finance-LLC/FinRL-Library
    Modify: Github Yonv1943 ElegantRL
    """

    """ Update Log 2020-12-12 by Github Yonv1943
    change download_preprocess_data: If the data had been downloaded, then don't download again

    # env 
    move reward_memory out of Env
    move plt.savefig('account_value.png') out of Env
    cancel SingleStockEnv(gym.Env): There is not need to use OpenAI's gym
    change pandas to numpy
    fix bug in comment: ('open', 'high', 'low', 'close', 'adjcp', 'volume', 'macd'), lack 'macd' before
    change slow 'state'
    change repeat compute 'begin_total_asset', 'end_total_asset'
    cancel self.asset_memory
    cancel self.cost
    cancel self.trade
    merge '_sell_stock' and '_bug_stock' to _sell_or_but_stock
    adjust order of state 
    reserved expansion interface on self.stock self.stocks

    # compatibility
    move global variable into Env.__init__()
    cancel matplotlib.use('Agg'): It will cause compatibility issues for ssh connection
    """
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, initial_account=100000, transaction_fee_rate=0.001, max_stock=200):
        self.stock_dim = 1

        # not necessary
        self.observation_space = gym.spaces.Box(low=0, high=2 ** 24, shape=(4,))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.stock_dim,))

        self.initial_account = initial_account
        self.transaction_fee_rate = transaction_fee_rate
        self.max_stock = max_stock
        self.state_div_std = np.array((2 ** -14, 2 ** -4, 2 ** 0, 2 ** -11))

        self.ary = self.download_data_as_csv__load_as_array()
        assert self.ary.shape == (2517, 9)  # ary: (date, item)
        self.ary = self.ary[1:, 2:].astype(np.float32)
        assert self.ary.shape == (2516, 7)  # ary: (date, item), item: (open, high, low, close, adjcp, volume, macd)
        self.ary = np.concatenate((
            self.ary[:, 4:5],  # adjcp? What is this? unit price?
            self.ary[:, 6:7],  # macd? What is this?
        ), axis=1)
        self.max_day = self.ary.shape[0] - 1

        # reset
        self.day = 0
        self.account = self.initial_account
        self.day_npy = self.ary[self.day]
        # self.stocks = np.zeros(self.stock_dim, dtype=np.float32) # multi-stack
        self.stock = 0
        # self.begin_total_asset = self.account + (self.day_npy[:self.stock_dim] * self.stocks).sum()
        self.begin_total_asset = self.account + self.day_npy[0] * self.stock

        self.step_sum = 0  # todo
        self.reward_sum = 0.0  # todo

    def reset(self):
        self.reward_sum = 0.0  # todo

        self.day = 0
        self.account = self.initial_account
        self.day_npy = self.ary[self.day]
        # self.stocks = np.zeros(self.stock_dim, dtype=np.float32)
        self.stock = 0
        # self.begin_total_asset = self.account + (self.day_npy[:self.stock_dim] * self.stocks).sum()
        self.begin_total_asset = self.account + self.day_npy[0] * self.stock
        # state = np.hstack((self.account, self.day_npy, self.stocks)
        #                   ).astype(np.float32) * self.state_div_std
        state = np.hstack((self.account, self.day_npy, self.stock)
                          ).astype(np.float32) * self.state_div_std
        return state

    def step(self, actions):
        actions = actions * self.max_stock

        """bug or sell stock"""
        index = 0
        action = actions[index]
        adj = self.day_npy[index]
        if action > 0:  # buy_stock
            available_amount = self.account // adj
            delta_stock = min(available_amount, action)
            self.account -= adj * delta_stock * (1 + self.transaction_fee_rate)
            # self.stocks[index] += delta_stock
            self.stock += delta_stock
        # elif self.stocks[index] > 0:  # sell_stock
        #     delta_stock = min(-action, self.stocks[index])
        #     self.account += adj * delta_stock * (1 - self.transaction_fee_percent)
        #     self.stocks[index] -= delta_stock
        elif self.stock > 0:  # sell_stock
            delta_stock = min(-action, self.stock)
            self.account += adj * delta_stock * (1 - self.transaction_fee_rate)
            self.stock -= delta_stock

        """update day"""
        self.day += 1
        # self.data = self.df.loc[self.day, :]
        self.day_npy = self.ary[self.day]

        # state = np.hstack((self.account, self.day_npy, self.stocks)
        #                   ).astype(np.float32) * self.state_div_std
        state = np.hstack((self.account, self.day_npy, self.stock)
                          ).astype(np.float32) * self.state_div_std

        # end_total_asset = self.account + (self.day_npy[:self.stock_dim] * self.stocks).sum()
        end_total_asset = self.account + self.day_npy[0] * self.stock
        reward = end_total_asset - self.begin_total_asset
        self.begin_total_asset = end_total_asset

        done = self.day == self.max_day  # 2516 is over
        reward_ = reward * 2 ** -10

        self.reward_sum += reward * 2 ** -10
        self.step_sum += 1
        if done:  # todo
            print(f'{self.step_sum:8}   {self.reward_sum:8.1f}')
        return state, reward_, done, {}

    def render(self, mode='human'):
        pass

    @staticmethod
    def download_data_as_csv__load_as_array(if_load=True):
        save_path = './AAPL_2009_2020.csv'

        import os
        if if_load and os.path.isfile(save_path):
            ary = np.genfromtxt(save_path, delimiter=',')
            assert isinstance(ary, np.ndarray)
            return ary
        import yfinance as yf
        from stockstats import StockDataFrame as Sdf
        """ pip install
        !pip install yfinance
        !pip install pandas
        !pip install matplotlib
        !pip install stockstats
        """

        """# Part 1: Download Data
        Yahoo Finance is a website that provides stock data, financial news, financial reports, etc. 
        All the data provided by Yahoo Finance is free.
        """
        print('| download_preprocess_data_as_csv: Download Data')

        data_pd = yf.download("AAPL", start="2009-01-01", end="2020-10-23")
        assert data_pd.shape == (2974, 6)

        data_pd = data_pd.reset_index()

        data_pd.columns = ['datadate', 'open', 'high', 'low', 'close', 'adjcp', 'volume']

        """# Part 2: Preprocess Data
        Data preprocessing is a crucial step for training a high quality machine learning model. 
        We need to check for missing data and do feature engineering 
        in order to convert the data into a model-ready state.
        """
        print('| download_preprocess_data_as_csv: Preprocess Data')

        # check missing data
        data_pd.isnull().values.any()

        # calculate technical indicators like MACD
        stock = Sdf.retype(data_pd.copy())
        # we need to use adjusted close price instead of close price
        stock['close'] = stock['adjcp']
        data_pd['macd'] = stock['macd']

        # check missing data again
        data_pd.isnull().values.any()
        data_pd.head()
        # data_pd=data_pd.fillna(method='bfill')

        # Note that I always use a copy of the original data to try it track step by step.
        data_clean = data_pd.copy()
        data_clean.head()
        data_clean.tail()

        data = data_clean[(data_clean.datadate >= '2009-01-01') & (data_clean.datadate < '2019-01-01')]
        data = data.reset_index(drop=True)  # the index needs to start from 0

        data.to_csv(save_path)  # save *.csv
        # assert isinstance(data_pd, pd.DataFrame)

        df_pandas = data[(data.datadate >= '2009-01-01') & (data.datadate < '2019-01-01')]
        df_pandas = df_pandas.reset_index(drop=True)  # the index needs to start from 0
        ary = df_pandas.to_numpy()
        return ary


def test():
    env = SingleStockFinEnv()
    ary = env.download_data_as_csv__load_as_array(if_load=True)  # data_frame_pandas
    print(ary.shape)
    ary = env.download_data_as_csv__load_as_array(if_load=True)  # data_frame_pandas
    print(ary.shape)

    env = SingleStockFinEnv(ary)
    state_dim, action_dim = 4, 1

    # state = env.reset()
    # done = False
    reward_sum = 0
    for i in range(2514):
        state, reward, done, info = env.step(rd.uniform(-1, 1, size=action_dim))
        reward_sum += reward
        # print(f'{i:5} {done:5} {reward:8.1f}', state)
    print(';', reward_sum)

    # state = env.reset()
    # done = False
    for _ in range(4):
        state, reward, done, info = env.step(rd.uniform(-1, 1, size=action_dim))
        print(f'{done:5} {reward:8.1f}', state)


# def train__baselines_rl():
#     from stable_baselines import PPO2, DDPG, A2C, ACKTR, TD3
#     from stable_baselines import DDPG
#     from stable_baselines import A2C
#     from stable_baselines import SAC
#     from stable_baselines.common.vec_env import DummyVecEnv
#
#     data_clean = download_preprocess_data()
#
#     train = data_clean[(data_clean.datadate >= '2009-01-01') & (data_clean.datadate < '2019-01-01')]
#     train = train.reset_index(drop=True)  # the index needs to start from 0
#     train.head()
#
#     env_train = DummyVecEnv([lambda: SingleStockEnv(train)])
#     model_ppo = PPO2('MlpPolicy', env_train, tensorboard_log="./single_stock_trading_2_tensorboard/")
#     model_ppo.learn(total_timesteps=100000, tb_log_name="run_aapl_ppo")

def train():
    from AgentRun import Arguments, train_agent_mp
    from AgentZoo import AgentPPO, AgentModSAC
    args = Arguments(rl_agent=None, env_name=None, gpu_id=None)
    args.rl_agent = AgentModSAC
    """
    | GPU: 0 | CWD: ./AgentModSAC/FinRL_0
    ID      Step      MaxR |    avgR      stdR |    ExpR     LossA     LossC
    0.0
    0   1.01e+04      1.30 |
    0   1.76e+04    485.86 |  158.33      0.00 |  430.58      0.67      0.07
    0   2.01e+04    916.53 |
    ID      Step   TargetR |    avgR      stdR |    ExpR  UsedTime  ########
    0   2.01e+04    800.00 |  916.53      0.00 |  141.46       322  ########
    0   3.27e+04    976.58 |  589.64      0.00 |  379.84      0.26      0.19 
    0   5.78e+04    976.58 |  628.69      0.00 |  517.65     -0.02      0.41
    
    | GPU: 1 | CWD: ./AgentModSAC/FinRL_1
    ID      Step      MaxR |    avgR      stdR |    ExpR     LossA     LossC
    0.0
    1   5.03e+03      0.40 |
    1   1.76e+04      6.63 |    6.63      0.00 |   24.55      0.69      0.01
    1   2.26e+04    652.01 |
    1   2.77e+04    836.11 |
    ID      Step   TargetR |    avgR      stdR |    ExpR  UsedTime  ########
    1   2.77e+04    800.00 |  836.11      0.00 |  650.18       430  ########
    1   3.02e+04    873.22 |
    1   3.52e+04    913.21 |
    1   3.52e+04    913.21 |  913.21      0.00 |  842.45     -0.12      0.35
    """
    args.if_break_early = False
    args.break_step = 2 ** 20

    args.max_memo = 2 ** 16
    args.gamma = 0.99  # important hyper-parameter, related to episode steps
    args.reward_scale = 2 ** -2
    args.max_step = 2515
    args.eval_times1 = 1
    args.eval_times2 = 1

    args.env_name = 'FinRL'
    args.init_for_training()
    train_agent_mp(args)

    args = Arguments(rl_agent=None, env_name=None, gpu_id=None)
    args.rl_agent = AgentPPO
    """
    | GPU: 3 | CWD: ./AgentPPO/FinRL_3
    ID      Step      MaxR |    avgR      stdR |    ExpR     LossA     LossC
    3   2.51e+04     48.87 |
    3   5.53e+04    441.54 |
    3   1.06e+05    707.73 |
    ID      Step   TargetR |    avgR      stdR |    ExpR  UsedTime  ########
    3   2.01e+05    800.00 |  810.60      0.00 |  748.02       197  ########
    3   2.66e+05    862.69 |  855.54      0.00 |  797.89     -0.03      0.53
    3   5.33e+05    915.21 |  888.58      0.00 |  874.78      0.06      0.48
    3   1.07e+06    942.40 |  938.52      0.00 |  932.97     -0.10      0.44
    3   1.34e+06    948.64 |  944.51      0.00 |  945.21     -0.05      0.41
    """
    args.if_break_early = False
    args.break_step = 2 ** 21  # 2**20==1e6, 2**21

    args.net_dim = 2 ** 8
    args.max_memo = 2 ** 12
    args.batch_size = 2 ** 9
    args.repeat_times = 2 ** 4
    # args.reward_scale = 2 ** -10  # unimportant hyper-parameter in PPO which do normalization on Q value
    args.gamma = 0.95  # important hyper-parameter, related to episode steps
    args.max_step = 2515 * 2
    args.eval_times1 = 1
    args.eval_times2 = 1

    args.env_name = 'FinRL'
    args.init_for_training()
    train_agent_mp(args)
    exit()


if __name__ == '__main__':
    # run()
    test()
    # train()
    # train__baselines_rl()
