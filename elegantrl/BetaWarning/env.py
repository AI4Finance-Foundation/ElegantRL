import os
import numpy as np
import numpy.random as rd


def prep_env(env, data_type=np.float32, if_print=True):  # preprocess environment
    """preprocess environment
    In OpenAI gym, it is class Wrapper(). In ElegantRL, it is a decorator

    What preprocess environment do?
    1.1 use get_gym_env_info() to get state_dim, action_dim ... from gym-env
    1.2 assignment env.state_dim, env.action_dim ...,
    1.3 then DRL agent can build network according to these information
    2.1 let state = state.astype(float32), because sometimes state is float64
    2.2 do normalization on state before training (no necessary). (Other people do running state while training)
    """
    if not all([hasattr(env, attr) for attr in (
            'env_name', 'state_dim', 'action_dim', 'if_discrete', 'target_reward', 'max_step',)]):
        (env_name, state_dim, action_dim, action_max, if_discrete, target_reward, max_step
         ) = _get_gym_env_info(env, if_print)
        setattr(env, 'env_name', env_name)
        setattr(env, 'state_dim', state_dim)
        setattr(env, 'action_dim', action_dim)
        setattr(env, 'if_discrete', if_discrete)
        setattr(env, 'target_reward', target_reward)
        setattr(env, 'max_step', max_step)
    else:
        action_max = 1

    '''state_norm is useful but non-necessary'''
    state_avg, state_std = _get_avg_std__for_state_norm(env.env_name)
    if state_avg is None:
        neg_state_avg = 0
        div_state_std = 1
    else:
        state_avg = state_avg.astype(np.float32)
        state_std = state_std.astype(np.float32)
        neg_state_avg = -state_avg
        div_state_std = 1 / (state_std + 1e-4)
    setattr(env, 'neg_state_avg', neg_state_avg)  # for def print_norm() agent.py
    setattr(env, 'div_state_std', div_state_std)  # for def print_norm() agent.py

    def decorate_step(env_step):
        if neg_state_avg is not None and action_max != 1:
            def new_env_step(action):
                state, reward, done, info = env_step(action * action_max)
                return (state.astype(data_type) + neg_state_avg) * div_state_std, reward, done, info
        elif neg_state_avg is not None:
            def new_env_step(action):
                state, reward, done, info = env_step(action)
                return (state.astype(data_type) + neg_state_avg) * div_state_std, reward, done, info
        else:
            def new_env_step(action):
                state, reward, done, info = env_step(action * action_max)
                return state.astype(data_type), reward, done, info
        return new_env_step

    env.step = decorate_step(env.step)

    def decorate_reset(env_reset):
        if neg_state_avg is not None:
            def new_env_reset():
                state = env_reset()
                return state.astype(data_type) + neg_state_avg
        else:
            def new_env_reset():
                state = env_reset()
                return state.astype(data_type)
        return new_env_reset

    env.reset = decorate_reset(env.reset)
    return env


def _get_avg_std__for_state_norm(env_name):
    avg = None
    std = None
    if env_name == 'LunarLanderContinuous-v2':
        avg = np.array([1.65470898e-02, -1.29684399e-01, 4.26883133e-03, -3.42124557e-02,
                        -7.39076972e-03, -7.67103031e-04, 1.12640885e+00, 1.12409466e+00])
        std = np.array([0.15094465, 0.29366297, 0.23490797, 0.25931464, 0.21603736,
                        0.25886878, 0.277233, 0.27771219])
    elif env_name == "BipedalWalker-v3":
        avg = np.array([1.42211734e-01, -2.74547996e-03, 1.65104509e-01, -1.33418152e-02,
                        -2.43243194e-01, -1.73886203e-02, 4.24114229e-02, -6.57800099e-02,
                        4.53460692e-01, 6.08022244e-01, -8.64884810e-04, -2.08789053e-01,
                        -2.92092949e-02, 5.04791247e-01, 3.33571745e-01, 3.37325723e-01,
                        3.49106580e-01, 3.70363115e-01, 4.04074671e-01, 4.55838055e-01,
                        5.36685407e-01, 6.70771701e-01, 8.80356865e-01, 9.97987386e-01])
        std = np.array([0.84419678, 0.06317835, 0.16532085, 0.09356959, 0.486594,
                        0.55477525, 0.44076614, 0.85030824, 0.29159821, 0.48093035,
                        0.50323634, 0.48110776, 0.69684234, 0.29161077, 0.06962932,
                        0.0705558, 0.07322677, 0.07793258, 0.08624322, 0.09846895,
                        0.11752805, 0.14116005, 0.13839757, 0.07760469])
    elif env_name == 'ReacherBulletEnv-v0':
        avg = np.array([0.03149641, 0.0485873, -0.04949671, -0.06938662, -0.14157104,
                        0.02433294, -0.09097818, 0.4405931, 0.10299437], dtype=np.float32)
        std = np.array([0.12277275, 0.1347579, 0.14567468, 0.14747661, 0.51311225,
                        0.5199606, 0.2710207, 0.48395795, 0.40876198], dtype=np.float32)
    elif env_name == 'AntBulletEnv-v0':
        avg = np.array([-2.2785307e-01, -4.1971792e-02, 9.2752278e-01, 8.3731368e-02,
                        1.2131270e-03, -5.7878396e-03, 1.8127944e-02, -1.1823924e-02,
                        1.5717462e-01, 1.2224792e-03, -1.9672018e-01, 6.4919023e-03,
                        -2.0346987e-01, 5.1609759e-04, 1.6572942e-01, -6.0344036e-03,
                        -1.6024958e-02, -1.3426526e-03, 3.8138664e-01, -5.6816568e-03,
                        -1.8004493e-01, -3.2685725e-03, -1.5989083e-01, 7.0396746e-03,
                        7.2912598e-01, 8.3666992e-01, 8.2824707e-01, 7.6196289e-01],
                       dtype=np.float32)
        std = np.array([0.09652393, 0.33918667, 0.23290202, 0.13423778, 0.10426794,
                        0.11678293, 0.39058578, 0.28871638, 0.5447721, 0.36814892,
                        0.73530555, 0.29377502, 0.5031936, 0.36130348, 0.71889997,
                        0.2496559, 0.5484764, 0.39613277, 0.7103549, 0.25976712,
                        0.56372136, 0.36917716, 0.7030704, 0.26312646, 0.30555955,
                        0.2681793, 0.27192947, 0.29626447], dtype=np.float32)
    #     avg = np.array([
    #         0.4838, -0.047, 0.3500, 1.3028, -0.249, 0.0000, -0.281, 0.0573,
    #         -0.261, 0.0000, 0.0424, 0.0000, 0.2278, 0.0000, -0.072, 0.0000,
    #         0.0000, 0.0000, -0.175, 0.0000, -0.319, 0.0000, 0.1387, 0.0000,
    #         0.1949, 0.0000, -0.136, -0.060])
    #     std = np.array([
    #         0.0601, 0.2267, 0.0838, 0.2680, 0.1161, 0.0757, 0.1495, 0.1235,
    #         0.6733, 0.4326, 0.6723, 0.3422, 0.7444, 0.5129, 0.6561, 0.2732,
    #         0.6805, 0.4793, 0.5637, 0.2586, 0.5928, 0.3876, 0.6005, 0.2369,
    #         0.4858, 0.4227, 0.4428, 0.4831])
    # elif env_name == 'MinitaurBulletEnv-v0': # need check
    #     # avg = np.array([0.90172989, 1.54730119, 1.24560906, 1.97365306, 1.9413892,
    #     #                 1.03866835, 1.69646277, 1.18655352, -0.45842347, 0.17845232,
    #     #                 0.38784456, 0.58572877, 0.91414561, -0.45410697, 0.7591031,
    #     #                 -0.07008998, 3.43842258, 0.61032482, 0.86689961, -0.33910894,
    #     #                 0.47030415, 4.5623528, -2.39108079, 3.03559422, -0.36328256,
    #     #                 -0.20753499, -0.47758384, 0.86756409])
    #     # std = np.array([0.34192648, 0.51169916, 0.39370621, 0.55568461, 0.46910769,
    #     #                 0.28387504, 0.51807949, 0.37723445, 13.16686185, 17.51240024,
    #     #                 14.80264211, 16.60461412, 15.72930229, 11.38926597, 15.40598346,
    #     #                 13.03124941, 2.47718145, 2.55088804, 2.35964651, 2.51025567,
    #     #                 2.66379017, 2.37224904, 2.55892521, 2.41716885, 0.07529733,
    #     #                 0.05903034, 0.1314812, 0.0221248])
    # elif env_name == "BipedalWalkerHardcore-v3": # need check
    #     avg = np.array([-3.6378160e-02, -2.5788052e-03, 3.4413573e-01, -8.4189959e-03,
    #                     -9.1864385e-02, 3.2804706e-04, -6.4693891e-02, -9.8939031e-02,
    #                     3.5180664e-01, 6.8103075e-01, 2.2930240e-03, -4.5893672e-01,
    #                     -7.6047562e-02, 4.6414185e-01, 3.9363885e-01, 3.9603019e-01,
    #                     4.0758255e-01, 4.3053803e-01, 4.6186063e-01, 5.0293463e-01,
    #                     5.7822973e-01, 6.9820738e-01, 8.9829963e-01, 9.8080903e-01])
    #     std = np.array([0.5771428, 0.05302362, 0.18906464, 0.10137994, 0.41284004,
    #                     0.68852615, 0.43710527, 0.87153363, 0.3210142, 0.36864948,
    #                     0.6926624, 0.38297284, 0.76805115, 0.33138904, 0.09618598,
    #                     0.09843876, 0.10035378, 0.11045089, 0.11910835, 0.13400233,
    #                     0.15718603, 0.17106676, 0.14363566, 0.10100251])
    return avg, std


def _get_gym_env_info(env, if_print):
    import gym  # gym of OpenAI is not necessary for ElegantRL (even RL)
    gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'
    assert isinstance(env, gym.Env)

    env_name = env.unwrapped.spec.id

    state_shape = env.observation_space.shape
    state_dim = state_shape[0] if len(state_shape) == 1 else state_shape  # sometimes state_dim is a list

    if env.spec.reward_threshold is None:
        target_reward = env.target_reward if hasattr(env, 'target_reward') else 2 ** 16
        print(f"| env.spec.reward_threshold is None, so I set target_reward={target_reward}")
    else:
        target_reward = env.spec.reward_threshold

    max_step = getattr(env, '_max_episode_steps', 2 ** 10)
    if max_step is None or max_step == 2 ** 10:
        print(f"| env._max_episode_steps is None, so I set max_step={max_step}")

    if_discrete = isinstance(env.action_space, gym.spaces.Discrete)
    if if_discrete:  # make sure it is discrete action space
        action_dim = env.action_space.n
        action_max = int(1)
    elif isinstance(env.action_space, gym.spaces.Box):  # make sure it is continuous action space
        action_dim = env.action_space.shape[0]
        action_max = float(env.action_space.high[0])
    else:
        raise RuntimeError('| Please set these value manually: if_discrete=bool, action_dim=int, action_max=1.0')

    if if_print:
        print("| env_name: {}, action space: {}".format(env_name, 'if_discrete' if if_discrete else 'Continuous'))
        print("| state_dim: {}, action_dim: {}, action_max: {}, target_reward: {}".format(
            state_dim, action_dim, action_max, target_reward))
    return env_name, state_dim, action_dim, action_max, if_discrete, target_reward, max_step


"""Custom environment: Finance RL, Github AI4Finance-LLC"""


class FinanceMultiStockEnv:  # 2021-02-02
    """FinRL
    Paper: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance
           https://arxiv.org/abs/2011.09607 NeurIPS 2020: Deep RL Workshop.
    Source: Github https://github.com/AI4Finance-LLC/FinRL-Library
    Modify: Github Yonv1943 ElegantRL
    """

    def __init__(self, initial_account=1e6, max_stock=1e2, transaction_fee_percent=1e-3, if_train=True,
                 train_beg=0, train_len=1024):
        self.stock_dim = 30
        self.initial_account = initial_account
        self.transaction_fee_percent = transaction_fee_percent
        self.max_stock = max_stock

        ary = self.load_training_data_for_multi_stock()
        assert ary.shape == (1699, 5 * 30)  # ary: (date, item*stock_dim), item: (adjcp, macd, rsi, cci, adx)
        assert train_beg < train_len
        assert train_len < ary.shape[0]  # ary.shape[0] == 1699
        self.ary_train = ary[:train_len]
        self.ary_valid = ary[train_len:]
        self.ary = self.ary_train if if_train else self.ary_valid

        # reset
        self.day = 0
        self.initial_account__reset = self.initial_account
        self.account = self.initial_account__reset
        self.day_npy = self.ary[self.day]
        self.stocks = np.zeros(self.stock_dim, dtype=np.float32)  # multi-stack

        self.total_asset = self.account + (self.day_npy[:self.stock_dim] * self.stocks).sum()
        self.episode_return = 0.0  # Compatibility for ElegantRL 2020-12-21
        self.gamma_return = 0.0

        '''env information'''
        self.env_name = 'FinanceStock-v2'
        self.state_dim = 1 + (5 + 1) * self.stock_dim
        self.action_dim = self.stock_dim
        self.if_discrete = False
        self.target_reward = 1.25  # convergence 1.5
        self.max_step = self.ary.shape[0]

    def reset(self):
        self.initial_account__reset = self.initial_account * rd.uniform(0.9, 1.1)  # reset()
        self.account = self.initial_account__reset
        self.stocks = np.zeros(self.stock_dim, dtype=np.float32)
        self.total_asset = self.account + (self.day_npy[:self.stock_dim] * self.stocks).sum()
        # total_asset = account + (adjcp * stocks).sum()

        self.day = 0
        self.day_npy = self.ary[self.day]
        self.day += 1

        state = np.hstack((self.account * 2 ** -16,
                           self.day_npy * 2 ** -8,
                           self.stocks * 2 ** -12,), ).astype(np.float32)
        return state

    def step(self, action):
        action = action * self.max_stock

        """bug or sell stock"""
        for index in range(self.stock_dim):
            stock_action = action[index]
            adj = self.day_npy[index]
            if stock_action > 0:  # buy_stock
                available_amount = self.account // adj
                delta_stock = min(available_amount, stock_action)
                self.account -= adj * delta_stock * (1 + self.transaction_fee_percent)
                self.stocks[index] += delta_stock
            elif self.stocks[index] > 0:  # sell_stock
                delta_stock = min(-stock_action, self.stocks[index])
                self.account += adj * delta_stock * (1 - self.transaction_fee_percent)
                self.stocks[index] -= delta_stock

        """update day"""
        self.day_npy = self.ary[self.day]
        self.day += 1
        done = self.day == self.max_step  # 2020-12-21

        state = np.hstack((self.account * 2 ** -16,
                           self.day_npy * 2 ** -8,
                           self.stocks * 2 ** -12,), ).astype(np.float32)

        next_total_asset = self.account + (self.day_npy[:self.stock_dim] * self.stocks).sum()
        reward = (next_total_asset - self.total_asset) * 2 ** -16  # notice scaling!
        self.total_asset = next_total_asset

        self.gamma_return = self.gamma_return * 0.99 + reward  # notice: gamma_r seems good? Yes
        if done:
            reward += self.gamma_return
            self.gamma_return = 0.0  # env.reset()

            # cumulative_return_rate
            self.episode_return = next_total_asset / self.initial_account

        return state, reward, done, None

    @staticmethod
    def load_training_data_for_multi_stock(if_load=True):  # need more independent
        npy_path = './FinanceMultiStock.npy'
        if if_load and os.path.exists(npy_path):
            data_ary = np.load(npy_path).astype(np.float32)
            assert data_ary.shape[1] == 5 * 30
            return data_ary
        else:
            raise RuntimeError(
                f'| Download and put it into: {npy_path}\n for FinanceMultiStockEnv()'
                f'| https://github.com/Yonv1943/ElegantRL/blob/master/FinanceMultiStock.npy'
                f'| Or you can use the following code to generate it from a csv file.')

        # from preprocessing.preprocessors import pd, data_split, preprocess_data, add_turbulence
        #
        # # the following is same as part of run_model()
        # preprocessed_path = "done_data.csv"
        # if if_load and os.path.exists(preprocessed_path):
        #     data = pd.read_csv(preprocessed_path, index_col=0)
        # else:
        #     data = preprocess_data()
        #     data = add_turbulence(data)
        #     data.to_csv(preprocessed_path)
        #
        # df = data
        # rebalance_window = 63
        # validation_window = 63
        # i = rebalance_window + validation_window
        #
        # unique_trade_date = data[(data.datadate > 20151001) & (data.datadate <= 20200707)].datadate.unique()
        # train__df = data_split(df, start=20090000, end=unique_trade_date[i - rebalance_window - validation_window])
        # # print(train__df) # df: DataFrame of Pandas
        #
        # train_ary = train__df.to_numpy().reshape((-1, 30, 12))
        # '''state_dim = 1 + 6 * stock_dim, stock_dim=30
        # n   item    index
        # 1   ACCOUNT -
        # 30  adjcp   2
        # 30  stock   -
        # 30  macd    7
        # 30  rsi     8
        # 30  cci     9
        # 30  adx     10
        # '''
        # data_ary = np.empty((train_ary.shape[0], 5, 30), dtype=np.float32)
        # data_ary[:, 0] = train_ary[:, :, 2]  # adjcp
        # data_ary[:, 1] = train_ary[:, :, 7]  # macd
        # data_ary[:, 2] = train_ary[:, :, 8]  # rsi
        # data_ary[:, 3] = train_ary[:, :, 9]  # cci
        # data_ary[:, 4] = train_ary[:, :, 10]  # adx
        #
        # data_ary = data_ary.reshape((-1, 5 * 30))
        #
        # os.makedirs(npy_path[:npy_path.rfind('/')])
        # np.save(npy_path, data_ary.astype(np.float16))  # save as float16 (0.5 MB), float32 (1.0 MB)
        # print('| FinanceMultiStockEnv(): save in:', npy_path)
        # return data_ary

    def draw_cumulative_return(self, args, torch):
        state_dim = self.state_dim
        action_dim = self.action_dim

        agent_rl = args.agent_rl
        net_dim = args.net_dim
        cwd = args.cwd

        agent = agent_rl(net_dim, state_dim, action_dim)  # build AgentRL
        act = agent.act
        device = agent.device

        state = self.reset()
        episode_returns = list()  # the cumulative_return / initial_account
        with torch.no_grad():
            for i in range(self.max_step):
                s_tensor = torch.as_tensor((state,), device=device)
                a_tensor = act(s_tensor)
                action = a_tensor.cpu().numpy()[0]  # not need detach(), because with torch.no_grad() outside
                state, reward, done, _ = self.step(action)

                episode_return = (self.account + (self.day_npy[:self.stock_dim] * self.stocks).sum()
                                  ) / self.initial_account__reset
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


"""Custom environment: Fix Env CarRacing-v0 - Box2D"""


def fix_car_racing_env(env, frame_num=3, action_num=3):  # 2020-12-12
    setattr(env, 'old_step', env.step)  # env.old_step = env.step
    setattr(env, 'env_name', 'CarRacing-Fix')
    setattr(env, 'state_dim', (frame_num, 96, 96))
    setattr(env, 'action_dim', 3)
    setattr(env, 'if_discrete', False)
    setattr(env, 'target_reward', 700)  # 900 in default

    setattr(env, 'state_stack', None)  # env.state_stack = None
    setattr(env, 'avg_reward', 0)  # env.avg_reward = 0
    """ cancel the print() in environment
    comment 'car_racing.py' line 233-234: print('Track generation ...
    comment 'car_racing.py' line 308-309: print("retry to generate track ...
    """

    def rgb2gray(rgb):
        # # rgb image -> gray [0, 1]
        # gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114]).astype(np.float32)
        # if norm:
        #     # normalize
        #     gray = gray / 128. - 1.
        # return gray

        state = rgb[:, :, 1]  # show green
        state[86:, 24:36] = rgb[86:, 24:36, 2]  # show red
        state[86:, 72:] = rgb[86:, 72:, 0]  # show blue
        state = (state - 128).astype(np.float32) / 128.
        return state

    def decorator_step(env_step):
        def new_env_step(action):
            action = action.copy()
            action[1:] = (action[1:] + 1) / 2  # fix action_space.low

            reward_sum = 0
            done = state = None
            try:
                for _ in range(action_num):
                    state, reward, done, info = env_step(action)
                    state = rgb2gray(state)

                    if done:
                        reward += 100  # don't penalize "die state"
                    if state.mean() > 192:  # 185.0:  # penalize when outside of road
                        reward -= 0.05

                    env.avg_reward = env.avg_reward * 0.95 + reward * 0.05
                    if env.avg_reward <= -0.1:  # done if car don't move
                        done = True

                    reward_sum += reward

                    if done:
                        break
            except Exception as error:
                print(f"| CarRacing-v0 Error 'stack underflow'? {error}")
                reward_sum = 0
                done = True
            env.state_stack.pop(0)
            env.state_stack.append(state)

            return np.array(env.state_stack).flatten(), reward_sum, done, {}

        return new_env_step

    env.step = decorator_step(env.step)

    def decorator_reset(env_reset):
        def new_env_reset():
            state = rgb2gray(env_reset())
            env.state_stack = [state, ] * frame_num
            return np.array(env.state_stack).flatten()

        return new_env_reset

    env.reset = decorator_reset(env.reset)
    return env


def render__car_racing():
    import gym  # gym of OpenAI is not necessary for ElegantRL (even RL)
    gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'
    env = gym.make('CarRacing-v0')
    env = fix_car_racing_env(env)

    state_dim = env.state_dim

    _state = env.reset()
    import cv2
    action = np.array((0, 1.0, -1.0))
    for i in range(321):
        # action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        # env.render
        show = state.reshape(state_dim)
        show = ((show[0] + 1.0) * 128).astype(np.uint8)
        cv2.imshow('', show)
        cv2.waitKey(1)
        if done:
            break
        # env.render()
