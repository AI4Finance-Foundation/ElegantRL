"""
https://github.com/AI4Finance-Foundation/FinRL-Meta/blob/master/examples/FinRL_PaperTrading_Demo.ipynb
"""

"""Part I"""

API_KEY = "PKAVSDVA8AIK4YBOOL3S"
API_SECRET = "U6TKEjt9C77Dw21ca8zVGUhsZxTUohaLYdmOrO3L"
API_BASE_URL = 'https://paper-api.alpaca.markets'
data_url = 'wss://data.alpaca.markets'

from finrl.config_tickers import DOW_30_TICKER
from finrl.config import INDICATORS
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from finrl.meta.env_stock_trading.env_stock_papertrading import AlpacaPaperTrading
from finrl.meta.data_processor import DataProcessor
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline

import numpy as np
import pandas as pd


def train(
        start_date,
        end_date,
        ticker_list,
        data_source,
        time_interval,
        technical_indicator_list,
        drl_lib,
        env,
        model_name,
        if_vix=True,
        **kwargs,
):
    # download data
    dp = DataProcessor(data_source, **kwargs)
    data = dp.download_data(ticker_list, start_date, end_date, time_interval)
    data = dp.clean_data(data)
    data = dp.add_technical_indicator(data, technical_indicator_list)
    if if_vix:
        data = dp.add_vix(data)
    else:
        data = dp.add_turbulence(data)
    price_array, tech_array, turbulence_array = dp.df_to_array(data, if_vix)

    np.save('price_array.npy', price_array)
    np.save('tech_array.npy', tech_array)
    np.save('turbulence_array.npy', turbulence_array)
    print("| save in '.'")


def run():
    ticker_list = DOW_30_TICKER
    env = StockTradingEnv
    erl_params = {"learning_rate": 3e-6, "batch_size": 2048, "gamma": 0.985,
                  "seed": 312, "net_dimension": [128, 64], "target_step": 5000, "eval_gap": 30,
                  "eval_times": 1}

    train(start_date='2022-08-25',
          end_date='2022-08-31',
          ticker_list=ticker_list,
          data_source='alpaca',
          time_interval='1Min',
          technical_indicator_list=INDICATORS,
          drl_lib='elegantrl',
          env=env,
          model_name='ppo',
          if_vix=True,
          API_KEY=API_KEY,
          API_SECRET=API_SECRET,
          API_BASE_URL=API_BASE_URL,
          erl_params=erl_params,
          cwd='./papertrading_erl',  # current_working_dir
          break_step=1e5)


if __name__ == '__main__':
    run()


"""
(base) develop@rlsmartagent-dev:~/workspace/ElegantRL0101/examples$ pip install git+https://github.com/AI4Finance-Foundation/FinRL.git
Defaulting to user installation because normal site-packages is not writeable
Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple
Collecting git+https://github.com/AI4Finance-Foundation/FinRL.git
  Cloning https://github.com/AI4Finance-Foundation/FinRL.git to /tmp/pip-req-build-er0zbi_n
  Running command git clone -q https://github.com/AI4Finance-Foundation/FinRL.git /tmp/pip-req-build-er0zbi_n



  fatal: unable to access 'https://github.com/AI4Finance-Foundation/FinRL.git/': GnuTLS recv error (-110): The TLS connection was non-properly terminated.
WARNING: Discarding git+https://github.com/AI4Finance-Foundation/FinRL.git. Command errored out with exit status 128: git clone -q https://github.com/AI4Finance-Foundation/FinRL.git /tmp/pip-req-build-er0zbi_n Check the logs for full command output.
ERROR: Command errored out with exit status 128: git clone -q https://github.com/AI4Finance-Foundation/FinRL.git /tmp/pip-req-build-er0zbi_n Check the logs for full command output.
"""