import gym
import numpy as np
import numpy.random as rd


class PreprocessEnv(gym.Wrapper):  # environment wrapper
    def __init__(self, env, if_print=True):
        self.env = gym.make(env) if isinstance(env, str) else env
        super(PreprocessEnv, self).__init__(self.env)
        self.step = self.step_type

        (self.env_name, self.state_dim, self.action_dim, self.action_max, self.max_step,
         self.if_discrete, self.target_reward) = get_gym_env_info(self.env, if_print)

    def reset(self) -> np.ndarray:
        state = self.env.reset()
        return state.astype(np.float32)

    def step_type(self, action) -> (np.ndarray, float, bool, dict):
        state, reward, done, info = self.env.step(action * self.action_max)
        return state.astype(np.float32), reward, done, info


def get_gym_env_info(env, if_print) -> (str, int, int, int, int, bool, float):
    import gym  # gym of OpenAI is not necessary for ElegantRL (even RL)
    gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'
    assert isinstance(env, gym.Env)

    env_name = env.unwrapped.spec.id

    state_shape = env.observation_space.shape
    state_dim = state_shape[0] if len(state_shape) == 1 else state_shape  # sometimes state_dim is a list

    target_reward = getattr(env, 'target_reward', None)
    target_reward_default = getattr(env.spec, 'reward_threshold', None)
    if target_reward is None:
        target_reward = target_reward_default
    if target_reward is None:
        target_reward = 2 ** 16

    max_step = getattr(env, 'max_step', None)
    max_step_default = getattr(env, '_max_episode_steps', None)
    if max_step is None:
        max_step = max_step_default
    if max_step is None:
        max_step = 2 ** 10

    if_discrete = isinstance(env.action_space, gym.spaces.Discrete)
    if if_discrete:  # make sure it is discrete action space
        action_dim = env.action_space.n
        action_max = int(1)
    elif isinstance(env.action_space, gym.spaces.Box):  # make sure it is continuous action space
        action_dim = env.action_space.shape[0]
        action_max = float(env.action_space.high[0])
    else:
        raise RuntimeError('| Please set these value manually: if_discrete=bool, action_dim=int, action_max=1.0')

    print(f"\n| env_name: {env_name}, action space if_discrete: {if_discrete}"
          f"\n| state_dim: {state_dim}, action_dim: {action_dim}, action_max: {action_max}"
          f"\n| max_step: {max_step} target_reward: {target_reward}") if if_print else None
    return env_name, state_dim, action_dim, action_max, max_step, if_discrete, target_reward


class FinanceStockEnv:  # custom env
    def __init__(self, initial_account=1e6, max_stock=1e2, transaction_fee_percent=1e-3, if_train=True, ):
        self.stock_dim = 30
        self.initial_account = initial_account
        self.transaction_fee_percent = transaction_fee_percent
        self.max_stock = max_stock

        data_ary = np.load('./FinanceStock.npy').astype(np.float32)
        assert data_ary.shape == (1699, 5 * 30)  # ary: (date, item*stock_dim), item: (adjcp, macd, rsi, cci, adx)
        self.ary_train = data_ary[:1024]
        self.ary_valid = data_ary[1024:]
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
        self.target_reward = 1.3  # denotes 1.3 times the initial_account. convergence to 1.5+
        self.max_step = self.ary.shape[0]

    def reset(self) -> np.ndarray:
        self.initial_account__reset = self.initial_account * rd.uniform(0.9, 1.1)  # reset()
        self.account = self.initial_account__reset
        self.stocks = np.zeros(self.stock_dim, dtype=np.float32)
        self.total_asset = self.account + (self.day_npy[:self.stock_dim] * self.stocks).sum()
        # total_asset = account + (adjcp * stocks).sum()

        self.day = 0
        self.day_npy = self.ary[self.day]
        self.day += 1

        state = np.hstack((self.account * 2 ** -16, self.day_npy * 2 ** -8, self.stocks * 2 ** -12,)).astype(np.float32)
        return state

    def step(self, action) -> (np.ndarray, float, bool, None):
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
        state = np.hstack((self.account * 2 ** -16, self.day_npy * 2 ** -8, self.stocks * 2 ** -12,)).astype(np.float32)

        next_total_asset = self.account + (self.day_npy[:self.stock_dim] * self.stocks).sum()
        reward = (next_total_asset - self.total_asset) * 2 ** -16  # notice scaling!
        self.total_asset = next_total_asset

        self.gamma_return = self.gamma_return * 0.99 + reward  # for infinite horizon case
        if done:
            reward += self.gamma_return
            self.gamma_return = 0.0  # env.reset()
            self.episode_return = next_total_asset / self.initial_account  # cumulative_return_rate

        return state, reward, done, None
