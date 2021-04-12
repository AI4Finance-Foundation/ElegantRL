import numpy as np

from elegantrl.demo import *

"""
FinRL PPO
beta1 beta2 gamma reward
beta2 "reward_scaling": 1e-6
beta3 beta2 repeat_times = 2 ** 3

Ant ModSAC Alpha reliable_lambda
ceta2 if if_update_a
ceta3 ceta2 repeat_times = 2 ** 0
"""


def demo4_bullet_mujoco_off_policy():
    args = Arguments(if_on_policy=False)
    args.random_seed = 100867

    from elegantrl.agent import AgentModSAC  # AgentSAC, AgentTD3, AgentDDPG
    args.agent = AgentModSAC()  # AgentSAC(), AgentTD3(), AgentDDPG()
    args.agent.if_use_dn = True
    args.net_dim = 2 ** 7

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)

    "TotalStep:  3e5, TargetReward: 1500, UsedTime:  8ks, AntBulletEnv-v0"
    "TotalStep:  6e5, TargetReward: 2500, UsedTime: 18ks, AntBulletEnv-v0"
    "TotalStep: 10e5, TargetReward: 2800, UsedTime:   ks, AntBulletEnv-v0"
    args.env = PreprocessEnv(env=gym.make('AntBulletEnv-v0'))
    args.break_step = int(6e5 * 8)  # (5e5) 1e6, UsedTime: (15,000s) 30,000s
    args.if_allow_break = False
    args.reward_scale = 2 ** -2  # todo # RewardRange: -50 < 0 < 2500 < 3340
    args.max_memo = 2 ** 18
    args.batch_size = 2 ** 9
    args.show_gap = 2 ** 8  # for Recorder
    args.eva_size1 = 2 ** 1  # for Recorder
    args.eva_size2 = 2 ** 3  # for Recorder

    # train_and_evaluate(args)
    args.rollout_num = 4
    train_and_evaluate_mp(args)


def check_finrl():
    from FinRL import StockTradingEnv
    from numpy import random as rd
    from finrl.config import config
    from finrl.preprocessing.data import data_split
    import pandas as pd

    # df = pd.read_pickle('finrl_data.df')  # DataFrame of Pandas
    #
    # from finrl.preprocessing.preprocessors import FeatureEngineer
    # fe = FeatureEngineer(
    #     use_technical_indicator=True,
    #     tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
    #     use_turbulence=True,
    #     user_defined_feature=False,
    # )
    #
    # processed_df = fe.preprocess_data(df)
    # processed_df.to_pickle('finrl_processed_data.df')  # DataFrame of Pandas
    processed_data_path = 'StockTradingEnv_processed_data.df'
    processed_df = pd.read_pickle(processed_data_path)  # DataFrame of Pandas
    print(processed_df.columns.values)

    split_df = data_split(processed_df, start='2008-03-19', end='2021-01-01')
    # `start`

    env = StockTradingEnv(df=split_df, tech_indicator_list=config.TECHNICAL_INDICATORS_LIST)
    action_dim = env.action_dim

    state = env.reset()
    print('state_dim', len(state))

    done = False
    step = 1
    from time import time
    timer = time()
    while not done:
        action = rd.rand(action_dim) * 2 - 1
        next_state, reward, done, _ = env.step(action)
        print(';', step, len(next_state), env.day, reward)
        step += 1

    print(';;', step, int(time() - timer))  # 44 seconds


if __name__ == '__main__':
    # demo2_continuous_action_space_off_policy()
    check_finrl()
