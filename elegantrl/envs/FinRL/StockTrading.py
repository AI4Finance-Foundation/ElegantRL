import os
import pandas as pd
import numpy as np
import numpy.random as rd
import torch

from elegantrl.replay import ReplayBuffer

class StockTradingEnv:
    def __init__(self, cwd='./', gamma=0.99, max_stock=1.0,
                 initial_capital=1e6, buy_cost_pct=1e-3, sell_cost_pct=1e-3,
                 start_date='2008-03-19', end_date='2016-01-01', env_eval_date='2021-01-01',
                 stock_name='AAPL', tech_indicator_list=None, initial_stocks=0, if_eval=False, if_save=False):

        self.price_ary, self.tech_ary, self.turbulence_ary = self.load_data(cwd, if_eval, if_save, stock_name, tech_indicator_list,
                                                                            start_date, end_date, env_eval_date)

        self.gamma = gamma
        self.max_stock = max_stock
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.initial_capital = initial_capital
        self.initial_stocks = initial_stocks
        self.if_eval = if_eval

        # reset()
        self.day = None
        self.amount = None
        self.stocks = None
        self.total_asset = None
        self.initial_total_asset = None
        self.gamma_reward = 0.0

        # environment information
        self.env_name = 'StockTradingEnv-v1'
        self.state_dim = 3 + self.tech_ary.shape[1]
        self.action_dim = 3
        self.max_step = len(self.price_ary) - 1
        self.if_discrete = True
        self.target_return = 3.5
        self.episode_return = 0.0

    def reset(self):
        self.day = 0
        price = self.price_ary[self.day]

        if self.if_eval:
            self.stocks = self.initial_stocks
            self.amount = self.initial_capital
        else:
            self.stocks = self.initial_stocks + rd.randint(0, 64)
            self.amount = self.initial_capital * rd.uniform(0.95, 1.05) - self.stocks * price

        self.total_asset = self.amount + self.stocks * price
        self.initial_total_asset = self.total_asset
        self.gamma_reward = 0.0

        state = np.hstack((self.amount * 2 ** -13,
                           price,
                           self.stocks,
                           self.tech_ary[self.day],)).astype(np.float32) * 2 ** -5
        return state

    def get_episode_return(self):
        price = self.price_ary[self.day]
        total_asset = self.amount + self.stocks * price
        return total_asset / self.initial_total_asset

    def step(self, action):

        self.day += 1
        price = self.price_ary[self.day]

        if action == 0:
            self.stocks -= 1
            self.amount += price * (1 - self.sell_cost_pct)
        elif action == 2:
            self.stocks += 1
            self.amount -= price * (1 - self.buy_cost_pct)

        state = np.hstack((self.amount * 2 ** -13,
                           price,
                           self.stocks,
                           self.tech_ary[self.day],)).astype(np.float32) * 2 ** -5

        total_asset = self.amount + self.stocks * price
        reward = (total_asset - self.total_asset) * 2 ** -14  # reward scaling
        self.total_asset = total_asset

        self.gamma_reward = self.gamma_reward * self.gamma + reward #!!!!!
        done = self.day == self.max_step
        if done:
            reward = self.gamma_reward
            self.episode_return = total_asset / self.initial_total_asset

        return state, reward, done, dict()

    def load_data(self, cwd='./', if_eval=None, if_save=False,
                  stock_name='AAPL', tech_indicator_list=None,
                  start_date='2008-03-19', end_date='2016-01-01', env_eval_date='2021-01-01'):
        raw_data_path = f'{cwd}/StockTradingEnv_raw_data.df'
        processed_data_path = f'{cwd}/StockTradingEnv_processed_data.df'
        data_path_array = f'{cwd}/StockTradingEnv_arrays_float16.npz'

        tech_indicator_list = [
            'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30', 'close_30_sma', 'close_60_sma'
        ] if tech_indicator_list is None else tech_indicator_list

        '''get: train_price_ary, train_tech_ary, eval_price_ary, eval_tech_ary'''
        if os.path.exists(data_path_array):
            load_dict = np.load(data_path_array)

            train_price_ary = load_dict['train_price_ary'].astype(np.float32)
            train_tech_ary = load_dict['train_tech_ary'].astype(np.float32)
            train_turbulence_ary = load_dict['train_turbulence_ary'].astype(np.float32)
            eval_price_ary = load_dict['eval_price_ary'].astype(np.float32)
            eval_tech_ary = load_dict['eval_tech_ary'].astype(np.float32)
            eval_turbulence_ary = load_dict['eval_turbulence_ary'].astype(np.float32)
        else:
            processed_df = self.processed_raw_data(raw_data_path, processed_data_path,
                                                   stock_name, tech_indicator_list, if_save)

            def data_split(df, start, end):
                data = df[(df.date >= start) & (df.date < end)]
                data = data.sort_values(["date", "tic"], ignore_index=True)
                data.index = data.date.factorize()[0]
                return data

            train_df = data_split(processed_df, start_date, end_date)
            eval_df = data_split(processed_df, end_date, env_eval_date)

            train_price_ary, train_tech_ary, train_turbulence_ary = self.convert_df_to_ary(train_df, tech_indicator_list)
            eval_price_ary, eval_tech_ary, eval_turbulence_ary = self.convert_df_to_ary(eval_df, tech_indicator_list)

            if if_save:
                np.savez_compressed(data_path_array,
                                    train_price_ary=train_price_ary.astype(np.float16),
                                    train_tech_ary=train_tech_ary.astype(np.float16),
                                    train_turbulence_ary=train_turbulence_ary.astype(np.float16),
                                    eval_price_ary=eval_price_ary.astype(np.float16),
                                    eval_tech_ary=eval_tech_ary.astype(np.float16), 
                                    eval_turbulence_ary=eval_turbulence_ary.astype(np.float16), )

        if if_eval is None:
            price_ary = np.concatenate((train_price_ary, eval_price_ary), axis=0)
            tech_ary = np.concatenate((train_tech_ary, eval_tech_ary), axis=0)
            turbulence_ary = np.concatenate((train_turbulence_ary, eval_turbulence_ary), axis=0)
        elif if_eval:
            price_ary = eval_price_ary
            tech_ary = eval_tech_ary
            turbulence_ary = eval_turbulence_ary
        else:
            price_ary = train_price_ary
            tech_ary = train_tech_ary
            turbulence_ary = train_turbulence_ary
        
        return price_ary, tech_ary, turbulence_ary

    def processed_raw_data(self, raw_data_path, processed_data_path,
                           stock_name, tech_indicator_list, if_save):
        if os.path.exists(processed_data_path):
            processed_df = pd.read_pickle(processed_data_path)  # DataFrame of Pandas
            # print('| processed_df.columns.values:', processed_df.columns.values)
            print(f"| load data: {processed_data_path}")
        else:
            print("| FeatureEngineer: start processing data (2 minutes)")
            fe = FeatureEngineer(use_turbulence=True,
                                 user_defined_feature=False,
                                 use_technical_indicator=True,
                                 tech_indicator_list=tech_indicator_list, )
            raw_df = self.get_raw_data(raw_data_path, stock_name, if_save)

            processed_df = fe.preprocess_data(raw_df)
            if if_save:
                processed_df.to_pickle(processed_data_path)
            print("| FeatureEngineer: finish processing data")

        return processed_df

    @staticmethod
    def get_raw_data(raw_data_path, stock_name, if_save=False):
        if os.path.exists(raw_data_path):
            raw_df = pd.read_pickle(raw_data_path)  # DataFrame of Pandas
            # print('| raw_df.columns.values:', raw_df.columns.values)
            print(f"| load data: {raw_data_path}")
        else:
            print("| YahooDownloader: start downloading data (1 minute)")
            raw_df = YahooDownloader(start_date="2000-01-01",
                                     end_date="2021-01-01",
                                     stock_name=stock_name, ).fetch_data()
            if if_save:
                raw_df.to_pickle(raw_data_path)
            print("| YahooDownloader: finish downloading data")
        return raw_df

    def convert_df_to_ary(self, df, tech_indicator_list):
        tech_ary = list()
        price_ary = list()
        turbulence_ary = list()
        for day in range(len(df.index.unique())):
            item = df.loc[day]

            tech_items = [[item[tech]] for tech in tech_indicator_list]
            tech_items_flatten = sum(tech_items, [])
            tech_ary.append(tech_items_flatten)
            price_ary.append(item.close)  # adjusted close price (adjcp)
            turbulence_ary.append(item.turbulence)

        price_ary = np.array(price_ary)
        tech_ary = np.array(tech_ary)
        turbulence_ary = np.array(turbulence_ary)
        print(f'| price_ary.shape: {price_ary.shape}, tech_ary.shape: {tech_ary.shape}, turbulence_ary.shape: {turbulence_ary.shape}')
        return price_ary, tech_ary, turbulence_ary

    def draw_cumulative_return(self, args, _torch) -> list:
        state_dim = self.state_dim
        action_dim = self.action_dim

        agent = args.agent
        net_dim = args.net_dim
        cwd = args.cwd

        agent.init(net_dim, state_dim, action_dim)
        agent.save_load_model(cwd=cwd, if_save=False)
        act = agent.act
        device = agent.device

        state = self.reset()
        episode_returns = list()  # the cumulative_return / initial_account
        action_choice = list()
        print('The initial captial is {}'.format(self.initial_capital))
        print('The initial number of stocks is {}'.format(self.initial_stocks))
        with _torch.no_grad():
            for i in range(self.max_step):
                s_tensor = _torch.as_tensor((state,), device=device)
                action = agent.get_best_act(s_tensor)
                state, reward, done, _ = self.step(action)

                total_asset = self.amount + self.price_ary[self.day] * self.stocks
                episode_return = total_asset / self.initial_total_asset
                episode_returns.append(episode_return)
                action_choice.append(action)
                if done:
                    break

        import matplotlib.pyplot as plt
        plt.plot(episode_returns)
        plt.grid()
        plt.title('cumulative return over time')
        plt.xlabel('day')
        plt.ylabel('fraction of initial asset')
        plt.savefig(f'{cwd}/cumulative_return.jpg')

        plt.figure()
        plt.plot(self.price_ary)
        plt.grid()
        plt.title('stock price over time')
        plt.xlabel('day')
        plt.ylabel('price')
        plt.savefig(f'{cwd}/price_over_time.jpg')

        plt.figure()
        plt.plot(action_choice)
        plt.grid()
        plt.title('action choice over time')
        plt.xlabel('day')
        plt.ylabel('action')
        plt.savefig(f'{cwd}/action_over_time.jpg')

        return episode_returns
    
    def draw_cumulative_return_while_learning(self, args, _torch) -> list:
        state_dim = self.state_dim
        action_dim = self.action_dim

        agent = args.agent
        net_dim = args.net_dim
        cwd = args.cwd

        agent.init(net_dim, state_dim, action_dim)
        agent.save_load_model(cwd=cwd, if_save=False)
        act = agent.act
        device = agent.device

        state = self.reset()
        episode_returns = list()  # the cumulative_return / initial_account

        buffer = ReplayBuffer(max_len=1000 + self.max_step, state_dim=state_dim, action_dim=1,
                          if_on_policy=False, if_per=False, if_gpu=True)
                          
        for i in range(self.max_step):
            action = agent.select_action(state)
            new_state, reward, done, _ = self.step(action)

            other = (reward * 1, 0.0 if done else self.gamma, action)
            buffer.append_buffer(state, other)
            state = new_state

            if i%50 == 49: 
                print('updating network: {}'.format(i))
                agent.update_net(buffer, 50, 32, 1)

            total_asset = self.amount + self.price_ary[self.day] * self.stocks
            episode_return = total_asset / self.initial_total_asset
            episode_returns.append(episode_return)
            if done:
                break

        import matplotlib.pyplot as plt
        plt.plot(episode_returns)
        plt.grid()
        plt.title('cumulative return')
        plt.xlabel('day')
        plt.xlabel('multiple of initial_account')
        plt.savefig(f'{cwd}/cumulative_return.jpg')
        return episode_returns


def check_stock_trading_env():
    if_eval = True  # False

    env = StockTradingEnv(if_eval=if_eval)
    action_dim = env.action_dim

    state = env.reset()
    print('state_dim', len(state))

    from time import time
    timer = time()

    step = 1
    done = False
    reward = None
    while not done:
        action = rd.rand(action_dim) * 2 - 1
        next_state, reward, done, _ = env.step(action)
        # print(';', len(next_state), env.day, reward)
        step += 1

    print(f"| Random action: step {step}, UsedTime {time() - timer:.3f}")
    print(f"| Random action: terminal reward {reward:.3f}")
    print(f"| Random action: episode return {env.episode_return:.3f}")

    '''draw_cumulative_return'''
    from elegantrl.agent import AgentPPO
    from elegantrl.run import Arguments
    args = Arguments(if_on_policy=True)
    args.agent = AgentPPO()
    args.env = StockTradingEnv(if_eval=True)
    args.if_remove = False
    args.cwd = './AgentPPO/StockTradingEnv-v1_0'
    args.init_before_training()

    env.draw_cumulative_return(args, torch)


"""Copy from FinRL"""


class YahooDownloader:
    """Provides methods for retrieving daily stock data from
    Yahoo Finance API
    from finrl.marketdata.yahoodownloader import YahooDownloader

    Attributes
    ----------
        start_date : str
            start date of the data (modified from config.py)
        end_date : str
            end date of the data (modified from config.py)
        stock_name : string
            stock to be downloaded

    Methods
    -------
    fetch_data()
        Fetches data from yahoo API

    """

    def __init__(self, start_date: str, end_date: str, stock_name: list):

        self.start_date = start_date
        self.end_date = end_date
        self.stock_name = stock_name

    def fetch_data(self) -> pd.DataFrame:
        import yfinance as yf  # Yahoo Finance
        """Fetches data from Yahoo API
        Parameters
        ----------

        Returns
        -------
        `pd.DataFrame`
            7 columns: A date, open, high, low, close, volume and tick symbol
            for the specified stock ticker
        """
        # Download and save the data in a pandas DataFrame:
        data_df = pd.DataFrame()
        temp_df = yf.download(self.stock_name, start=self.start_date, end=self.end_date)
        temp_df["tic"] = self.stock_name
        data_df = data_df.append(temp_df)
        # reset the index, we want to use numbers as index instead of dates
        data_df = data_df.reset_index()
        try:
            # convert the column names to standardized names
            data_df.columns = [
                "date",
                "open",
                "high",
                "low",
                "close",
                "adjcp",
                "volume",
                "tic",
            ]
            # use adjusted close price instead of close price
            data_df["close"] = data_df["adjcp"]
            # drop the adjusted close price column
            data_df = data_df.drop("adjcp", 1)
        except NotImplementedError:
            print("the features are not supported currently")
        # create day of the week column (monday = 0)
        data_df["day"] = data_df["date"].dt.dayofweek
        # convert date to standard string format, easy to filter
        data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
        # drop missing data
        data_df = data_df.dropna()
        data_df = data_df.reset_index(drop=True)
        print("Shape of DataFrame: ", data_df.shape)
        # print("Display DataFrame: ", data_df.head())

        data_df = data_df.sort_values(by=['date', 'tic']).reset_index(drop=True)

        return data_df


class FeatureEngineer:
    """Provides methods for preprocessing the stock price data
    from finrl.preprocessing.preprocessors import FeatureEngineer

    Attributes
    ----------
        use_technical_indicator : boolean
            we technical indicator or not
        tech_indicator_list : list
            a list of technical indicator names (modified from config.py)
        use_turbulence : boolean
            use turbulence index or not
        user_defined_feature:boolean
            user user defined features or not

    Methods
    -------
    preprocess_data()
        main method to do the feature engineering

    """

    def __init__(
            self,
            use_technical_indicator=True,
            tech_indicator_list=None,  # config.TECHNICAL_INDICATORS_LIST,
            use_turbulence=False,
            user_defined_feature=False,
    ):
        self.use_technical_indicator = use_technical_indicator
        self.tech_indicator_list = tech_indicator_list
        self.use_turbulence = use_turbulence
        self.user_defined_feature = user_defined_feature

    def preprocess_data(self, df):
        """main method to do the feature engineering
        @:param config: source dataframe
        @:return: a DataMatrices object
        """

        if self.use_technical_indicator:
            # add technical indicators using stockstats
            df = self.add_technical_indicator(df)
            print("Successfully added technical indicators")

        # add turbulence index for multiple stock
        if self.use_turbulence:
            df = self.add_turbulence(df)
            print("Successfully added turbulence index")

        # add user defined feature
        if self.user_defined_feature:
            df = self.add_user_defined_feature(df)
            print("Successfully added user defined features")

        # fill the missing values at the beginning and the end
        df = df.fillna(method="bfill").fillna(method="ffill")
        return df

    def add_technical_indicator(self, data):
        """
        calculate technical indicators
        use stockstats package to add technical inidactors
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        from stockstats import StockDataFrame as Sdf  # for Sdf.retype

        df = data.copy()
        df = df.sort_values(by=['tic', 'date'])
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tic.unique()

        for indicator in self.tech_indicator_list:
            indicator_df = pd.DataFrame()
            for i in range(len(unique_ticker)):
                try:
                    temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
                    temp_indicator = pd.DataFrame(temp_indicator)
                    temp_indicator['tic'] = unique_ticker[i]
                    temp_indicator['date'] = df[df.tic == unique_ticker[i]]['date'].to_list()
                    indicator_df = indicator_df.append(
                        temp_indicator, ignore_index=True
                    )
                except Exception as e:
                    print(e)
            df = df.merge(indicator_df[['tic', 'date', indicator]], on=['tic', 'date'], how='left')
        df = df.sort_values(by=['date', 'tic'])
        return df

    def add_turbulence(self, data):
        """
        add turbulence index from a precalcualted dataframe
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        turbulence_index = self.calculate_turbulence(df)
        df = df.merge(turbulence_index, on="date")
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        return df

    @staticmethod
    def add_user_defined_feature(data):
        """
         add user defined features
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df["daily_return"] = df.close.pct_change(1)
        # df['return_lag_1']=df.close.pct_change(2)
        # df['return_lag_2']=df.close.pct_change(3)
        # df['return_lag_3']=df.close.pct_change(4)
        # df['return_lag_4']=df.close.pct_change(5)
        return df

    @staticmethod
    def calculate_turbulence(data):
        """calculate turbulence index based on dow 30"""
        # can add other market assets
        df = data.copy()
        df_price_pivot = df.pivot(index="date", columns="tic", values="close")
        # use returns to calculate turbulence
        df_price_pivot = df_price_pivot.pct_change()

        unique_date = df.date.unique()
        # start after a year
        start = 252
        turbulence_index = [0] * start
        # turbulence_index = [0]
        count = 0
        for i in range(start, len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
            # use one year rolling window to calcualte covariance
            hist_price = df_price_pivot[
                (df_price_pivot.index < unique_date[i])
                & (df_price_pivot.index >= unique_date[i - 252])
                ]
            # Drop tickers which has number missing values more than the "oldest" ticker
            filtered_hist_price = hist_price.iloc[hist_price.isna().sum().min():].dropna(axis=1)

            cov_temp = filtered_hist_price.cov()
            current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(filtered_hist_price, axis=0)
            temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(
                current_temp.values.T
            )
            if temp > 0:
                count += 1
                if count > 2:
                    turbulence_temp = temp[0][0]
                else:
                    # avoid large outlier because of the calculation just begins
                    turbulence_temp = 0
            else:
                turbulence_temp = 0
            turbulence_index.append(turbulence_temp)

        turbulence_index = pd.DataFrame(
            {"date": df_price_pivot.index, "turbulence": turbulence_index}
        )
        return turbulence_index


if __name__ == '__main__':
    check_stock_trading_env()
