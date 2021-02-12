import os
import sys
import time

import numpy as np
import numpy.random as rd
import torch

from AgentZoo import AgentModSAC, AgentGaePPO


class Arguments:
    def __init__(self, rl_agent=None, env=None, gpu_id=None):
        self.rl_agent = rl_agent  # Deep Reinforcement Learning algorithm
        self.gpu_id = gpu_id  # choose the GPU for running. gpu_id is None means set it automatically
        self.cwd = None  # current work directory. cwd is None means set it automatically
        self.env = env  # the environment for training

        '''Arguments for training'''
        self.net_dim = 2 ** 8  # the network width
        self.max_memo = 2 ** 17  # memories capacity (memories: replay buffer)
        self.max_step = 2 ** 10  # max steps in one training episode
        self.batch_size = 2 ** 7  # num of transitions sampled from replay buffer.
        self.repeat_times = 2 ** 0  # repeatedly update network to keep critic's loss small
        self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256
        self.gamma = 0.99  # discount factor of future rewards
        self.rollout_num = 2  # the number of rollout workers (larger is not always faster)
        self.num_threads = 4  # cpu_num for evaluate model, torch.set_num_threads(self.num_threads)

        '''Arguments for evaluate'''
        self.break_step = 2 ** 17  # break training after 'total_step > break_step'
        self.if_break_early = True  # break training after 'eval_reward > target reward'
        self.if_remove = True  # remove the cwd folder? (True, False, None:ask me)
        self.show_gap = 2 ** 8  # show the Reward and Loss of actor and critic per show_gap seconds
        self.eval_times = 2 ** 3  # evaluation times if 'eval_reward > target_reward'
        self.random_seed = 1943  # Github: YonV1943

    def init_before_training(self):
        self.gpu_id = sys.argv[-1][-4] if self.gpu_id is None else str(self.gpu_id)
        self.cwd = f'./{self.rl_agent.__name__}/{self.env.env_name}_{self.gpu_id}' if self.cwd is None else self.cwd
        print(f'| GPU id: {self.gpu_id}, cwd: {self.cwd}')
        whether_remove_history(self.cwd, self.if_remove)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        torch.set_num_threads(self.num_threads)
        torch.set_default_dtype(torch.float32)
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)


def whether_remove_history(cwd, if_remove=None):
    import shutil
    if_remove = bool(input("PRESS 'y' to REMOVE: {}? ".format(cwd)) == 'y') if if_remove is None else if_remove
    shutil.rmtree(cwd, ignore_errors=True) if if_remove else print("| Keep history")
    os.makedirs(cwd, exist_ok=True)
    del shutil


def train_agent(args):
    args.init_before_training()

    '''basic arguments'''
    rl_agent = args.rl_agent
    gpu_id = args.gpu_id
    env = args.env
    cwd = args.cwd

    '''training arguments'''
    gamma = args.gamma
    net_dim = args.net_dim
    max_memo = args.max_memo
    max_step = args.max_step
    batch_size = args.batch_size
    repeat_times = args.repeat_times
    reward_scale = args.reward_scale

    '''evaluate arguments'''
    break_step = args.break_step
    if_break_early = args.if_break_early
    show_gap = args.show_gap
    eval_times = args.eval_times
    del args  # In order to show these hyper-parameters clearly, I put them above.

    if_on_policy = rl_agent.__name__ in {'AgentPPO', 'AgentGaePPO'}

    '''init: env'''
    state_dim = env.state_dim
    action_dim = env.action_dim
    if_discrete = env.if_discrete
    target_reward = env.target_reward
    from copy import deepcopy  # built-in library of Python
    env = deepcopy(env)
    env_eval = deepcopy(env)  # 2020-12-12

    '''build rl_agent'''
    agent = rl_agent(state_dim, action_dim, net_dim)  # training agent
    agent.state = env.reset()

    '''build ReplayBuffer'''
    buffer = ReplayBuffer(max_memo, state_dim, action_dim=1 if if_discrete else action_dim,
                          if_on_policy=if_on_policy)
    total_step = 0
    if if_on_policy:
        steps = 0
    else:
        with torch.no_grad():  # update replay buffer
            steps = explore_before_train(env, buffer, max_step, if_discrete, reward_scale, gamma, action_dim)
        '''pre training and hard update before training loop'''
        buffer.update__now_len__before_sample()
        agent.update_policy(buffer, max_step, batch_size, repeat_times)
        agent.act_target.load_state_dict(agent.act.state_dict()) if 'act_target' in dir(agent) else None
    total_step += steps

    '''build Recorder'''
    recorder = Recorder(eval_times)
    with torch.no_grad():
        recorder.update_recorder(env_eval, agent.act, agent.device, steps, agent.obj_a, agent.obj_c)

    '''loop'''
    if_solve = False
    while not ((if_break_early and if_solve) or total_step > break_step or os.path.exists(f'{cwd}/stop')):
        with torch.no_grad():  # speed up running
            steps = agent.update_buffer(env, buffer, max_step, reward_scale, gamma)
        total_step += steps

        buffer.update__now_len__before_sample()
        agent.update_policy(buffer, max_step, batch_size, repeat_times)

        with torch.no_grad():  # for saving the GPU buffer
            if_save = recorder.update_recorder(env_eval, agent.act, agent.device, steps, agent.obj_a, agent.obj_c)
            recorder.save_act(cwd, agent.act, gpu_id) if if_save else None
            if_solve = recorder.check__if_solved(target_reward, gpu_id, show_gap, cwd)


class ReplayBuffer:
    def __init__(self, max_len, state_dim, action_dim, if_on_policy=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.next_idx = 0
        self.is_full = False
        self.max_len = max_len
        self.now_len = self.max_len if self.is_full else self.next_idx

        self.state_idx = 1 + 1 + state_dim  # reward_dim=1, done_dim=1
        self.action_idx = self.state_idx + action_dim

        last_dim = action_dim if if_on_policy else state_dim
        self.memo_dim = 1 + 1 + state_dim + action_dim + last_dim
        self.memories = np.empty((max_len, self.memo_dim), dtype=np.float32)

    def append_memo(self, memo_tuple):
        self.memories[self.next_idx] = memo_tuple
        self.next_idx += 1
        if self.next_idx >= self.max_len:
            self.is_full = True
            self.next_idx = 0

    def random_sample(self, batch_size):
        indices = rd.randint(self.now_len, size=batch_size)
        memory = torch.as_tensor(self.memories[indices], device=self.device)
        return (memory[:, 0:1],  # rewards
                memory[:, 1:2],  # masks, mark == (1-float(done)) * gamma
                memory[:, 2:self.state_idx],  # states
                memory[:, self.state_idx:self.action_idx],  # actions
                memory[:, self.action_idx:],)  # next_states

    def all_sample(self):
        tensors = (self.memories[:self.now_len, 0:1],  # rewards
                   self.memories[:self.now_len, 1:2],  # masks, mark == (1-float(done)) * gamma
                   self.memories[:self.now_len, 2:self.state_idx],  # states
                   self.memories[:self.now_len, self.state_idx:self.action_idx],  # actions
                   self.memories[:self.now_len, self.action_idx:],)  # next_states or log_prob_sum
        return [torch.tensor(ary, device=self.device) for ary in tensors]

    def update__now_len__before_sample(self):
        self.now_len = self.max_len if self.is_full else self.next_idx

    def empty_memories__before_explore(self):
        self.next_idx = 0
        self.now_len = 0
        self.is_full = False


def explore_before_train(env, buffer, max_step, if_discrete, reward_scale, gamma, action_dim):
    state = env.reset()
    steps = 0

    while steps < max_step:
        action = rd.randint(action_dim) if if_discrete else rd.uniform(-1, 1, size=action_dim)
        next_state, reward, done, _ = env.step(action)
        steps += 1

        scaled_reward = reward * reward_scale
        mask = 0.0 if done else gamma
        memo_tuple = (scaled_reward, mask, *state, action, *next_state) if if_discrete else \
            (scaled_reward, mask, *state, *action, *next_state)  # not elegant but ok
        buffer.append_memo(memo_tuple)

        state = env.reset() if done else next_state
    return steps


class Recorder:
    def __init__(self, eval_size):
        self.recorder = [(0., -np.inf, 0., 0., 0.), ]  # total_step, r_avg, r_std, obj_a, obj_c
        self.r_max = -np.inf
        self.is_solved = False
        self.total_step = 0
        self.eva_size = eval_size  # constant

        self.used_time = None
        self.start_time = time.time()
        self.print_time = time.time()
        print(f"{'ID':>2}  {'Step':>8}  {'MaxR':>8} |{'avgR':>8}  {'stdR':>8}   {'objA':>8}  {'objC':>8}")

    def update_recorder(self, env, act, device, step_sum, obj_a, obj_c):
        is_saved = False
        reward_list = [get_episode_return(env, act, device) for _ in range(self.eva_size)]

        r_avg = np.average(reward_list)
        if r_avg > self.r_max:  # check final
            self.r_max = r_avg
            is_saved = True

        r_std = float(np.std(reward_list))
        self.total_step += step_sum
        self.recorder.append((self.total_step, r_avg, r_std, obj_a, obj_c))
        return is_saved

    def save_act(self, cwd, act, agent_id):
        act_save_path = f'{cwd}/actor.pth'
        torch.save(act.state_dict(), act_save_path)
        print(f"{agent_id:<2}  {self.total_step:8.2e}  {self.r_max:8.2f} |")

    def check__if_solved(self, target_reward, agent_id, show_gap, _cwd):
        total_step, r_avg, r_std, obj_a, obj_c = self.recorder[-1]

        self.is_solved = bool(self.r_max > target_reward)
        if self.is_solved and self.used_time is None:
            self.used_time = int(time.time() - self.start_time)
            print(f"{'ID':>2}  {'Step':>8}  {'TargetR':>8} |"
                  f"{'avgR':>8}  {'stdR':>8}   {'UsedTime':>8}  ########\n"
                  f"{agent_id:<2}  {total_step:8.2e}  {target_reward:8.2f} |"
                  f"{r_avg:8.2f}  {r_std:8.2f}   {self.used_time:>8}  ########")

        if time.time() - self.print_time > show_gap:
            self.print_time = time.time()
            print(f"{agent_id:<2}  {total_step:8.2e}  {self.r_max:8.2f} |"
                  f"{r_avg:8.2f}  {r_std:8.2f}   {obj_a:8.2f}  {obj_c:8.2f}")
        return self.is_solved


def get_episode_return(env, act, device) -> float:
    episode_return = 0.0  # sum of rewards in an episode
    max_step = env.max_step if hasattr(env, 'max_step') else 2 ** 10
    if_discrete = env.if_discrete

    state = env.reset()
    for _ in range(max_step):
        s_tensor = torch.as_tensor((state,), device=device)
        a_tensor = act(s_tensor)
        if if_discrete:
            a_tensor = a_tensor.argmax(dim=1)
        action = a_tensor.cpu().numpy()[0]  # not need detach(), because with torch.no_grad() outside

        state, reward, done, _ = env.step(action)
        episode_return += reward
        if done:
            break
    return env.episode_return if hasattr(env, 'episode_return') else episode_return


"""Environment"""


def decorate_env(env):
    if not all([hasattr(env, attr) for attr in (
            'env_name', 'state_dim', 'action_dim', 'target_reward', 'if_discrete')]):
        (env_name, state_dim, action_dim, action_max, if_discrete, target_reward) = get_gym_env_information(env)
        setattr(env, 'env_name', env_name)
        setattr(env, 'state_dim', state_dim)
        setattr(env, 'action_dim', action_dim)
        setattr(env, 'if_discrete', if_discrete)
        setattr(env, 'target_reward', target_reward)
    return env


def get_gym_env_information(env) -> (str, int, int, float, bool, float):
    import gym  # gym of OpenAI is not necessary for ElegantRL (even RL)
    gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'
    assert isinstance(env, gym.Env)

    env_name = env.unwrapped.spec.id

    state_shape = env.observation_space.shape
    state_dim = state_shape[0] if len(state_shape) == 1 else state_shape  # sometimes state_dim is a list

    if_discrete = isinstance(env.action_space, gym.spaces.Discrete)
    target_reward = env.spec.reward_threshold
    assert target_reward is not None

    if if_discrete:  # make sure it is discrete action space
        action_dim = env.action_space.n
        action_max = int(1)
    elif isinstance(env.action_space, gym.spaces.Box):  # make sure it is continuous action space
        action_dim = env.action_space.shape[0]
        action_max = float(env.action_space.high[0])
    else:
        raise RuntimeError('| Please set these value manually: if_discrete=bool, action_dim=int, action_max=1.0')
    return env_name, state_dim, action_dim, action_max, if_discrete, target_reward


class FinanceMultiStockEnv:  # 2021-02-02
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

        self.ary = self.load_training_data_for_multi_stock()
        assert self.ary.shape == (1699, 5 * 30)  # ary: (date, item*stock_dim), item: (adjcp, macd, rsi, cci, adx)

        # reset
        self.day = 0
        self.account = self.initial_account
        self.day_npy = self.ary[self.day]
        self.stocks = np.zeros(self.stock_dim, dtype=np.float32)  # multi-stack
        self.total_asset = self.account + (self.day_npy[:self.stock_dim] * self.stocks).sum()
        self.episode_return = 0.0  # Compatibility for ElegantRL 2020-12-21

        '''env information'''
        self.env_name = 'FinanceStock-v1'
        self.state_dim = 1 + (5 + 1) * self.stock_dim
        self.action_dim = self.stock_dim
        self.if_discrete = False
        self.target_reward = 15
        self.max_step = self.ary.shape[0]

        self.gamma_r = 0.0

    def reset(self):  # 2021-02-03
        self.account = self.initial_account * rd.uniform(0.9, 1.0)  # notice reset()
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
        self.day_npy = self.ary[self.day]
        self.day += 1
        done = self.day == self.max_step  # 2020-12-21

        state = np.hstack((
            self.account * 2 ** -16,
            self.day_npy * 2 ** -8,
            self.stocks * 2 ** -12,
        ), ).astype(np.float32)

        next_total_asset = self.account + (self.day_npy[:self.stock_dim] * self.stocks).sum()
        reward = (next_total_asset - self.total_asset) * 2 ** -16  # notice scaling!
        self.total_asset = next_total_asset

        self.gamma_r = self.gamma_r * 0.99 + reward  # notice: gamma_r seems good? Yes
        if done:
            reward += self.gamma_r
            self.gamma_r = 0.0  # env.reset()

            # cumulative_return_rate
            self.episode_return = next_total_asset / self.initial_account

        return state, reward, done, None

    @staticmethod
    def load_training_data_for_multi_stock(if_load=True):  # need more independent
        npy_path = './Result/FinanceMultiStock.npy'
        if if_load and os.path.exists(npy_path):
            data_ary = np.load(npy_path).astype(np.float32)
            assert data_ary.shape[1] == 5 * 30
            return data_ary
        else:
            raise RuntimeError(
                f'| FinanceMultiStockEnv(): Can you download and put it into: {npy_path}\n'
                f'| https://github.com/Yonv1943/ElegantRL/blob/master/Result/FinanceMultiStock.npy'
                f'| Or you can use the following code to generate it from a csv file.'
            )

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


def train__demo():
    args = Arguments(rl_agent=None, env=None, gpu_id=0)

    '''DEMO 2: Continuous action env: LunarLanderContinuous-v2 of gym.box2D'''
    import gym
    gym.logger.set_level(40)
    args.env = decorate_env(env=gym.make('LunarLanderContinuous-v2'))
    args.rl_agent = AgentModSAC  # Modified SAC (off-policy)

    args.break_step = int(6e4 * 8)  # UsedTime 900s (reach target_reward 200)
    args.net_dim = 2 ** 7
    train_agent(args)
    exit()

    '''DEMO 3: Custom Continuous action env: FinanceStock-v1'''
    args.env = FinanceMultiStockEnv()  # a standard env for ElegantRL, not need decorate_env()
    args.rl_agent = AgentGaePPO  # PPO+GAE (on-policy)

    args.break_step = int(5e6 * 4)  # 5e6 (15e6) UsedTime 3,000s (9,000s)
    args.net_dim = 2 ** 8
    args.max_step = 1699
    args.max_memo = (args.max_step - 1) * 16
    args.batch_size = 2 ** 11
    args.repeat_times = 2 ** 4
    args.init_before_training()
    train_agent(args)
    exit()


if __name__ == '__main__':
    train__demo()
