import os
import time
import numpy as np
import torch as th
import gymnasium as gym
import torch.nn as nn
import torch.distributions.normal
from typing import List, Tuple, Optional

"""finance environment
Source: https://github.com/AI4Finance-Foundation/ElegantRL/blob/master/examples/demo_FinRL_ElegantRL_China_A_shares.py
Modify: Github YonV1943
"""

ARY = np.ndarray
TEN = th.Tensor


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
            self.amount = self.initial_amount * np.random.uniform(0.9, 1.1)
            self.shares = (np.abs(np.random.randn(self.shares_num).clip(-2, +2)) * 2 ** 6).astype(int)
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
            self.cumulative_returns = total_asset / self.initial_amount * 100  # todo

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


def check_stock_trading_env():
    env = StockTradingEnv(beg_idx=834, end_idx=1113)
    env.if_random_reset = False
    evaluate_time = 4

    print()
    policy_name = 'random action (if_random_reset = False)'
    state, info_dict = env.reset()
    for _ in range(env.max_step * evaluate_time):
        action = np.random.uniform(-1, +1, env.action_dim)
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


'''config'''


class Config:
    def __init__(self, agent_class=None, env_class=None, env_args=None):
        self.agent_class = agent_class  # agent = agent_class(...)
        self.if_off_policy = self.get_if_off_policy()  # whether off-policy or on-policy of DRL algorithm

        self.env_class = env_class  # env = env_class(**env_args)
        self.env_args = env_args  # env = env_class(**env_args)
        if env_args is None:  # dummy env_args
            env_args = {'env_name': None, 'state_dim': None, 'action_dim': None, 'if_discrete': None}
        self.env_name = env_args['env_name']  # the name of environment. Be used to set 'cwd'.
        self.state_dim = env_args['state_dim']  # vector dimension (feature number) of state
        self.action_dim = env_args['action_dim']  # vector dimension (feature number) of action
        self.if_discrete = env_args['if_discrete']  # discrete or continuous action space

        '''Arguments for reward shaping'''
        self.gamma = 0.99  # discount factor of future rewards
        self.reward_scale = 1.0  # an approximate target reward usually be closed to 256

        '''Arguments for training'''
        self.net_dims = [64, 32]  # the middle layer dimension of MLP (MultiLayer Perceptron)
        self.learning_rate = 6e-5  # 2 ** -14 ~= 6e-5
        self.soft_update_tau = 5e-3  # 2 ** -8 ~= 5e-3
        if self.if_off_policy:  # off-policy
            self.batch_size = int(64)  # num of transitions sampled from replay buf.
            self.horizon_len = int(512)  # collect horizon_len step while exploring, then update network
            self.buffer_size = int(1e6)  # ReplayBuffer size. First in first out for off-policy.
            self.repeat_times = 1.0  # repeatedly update network using ReplayBuffer to keep critic's loss small
        else:  # on-policy
            self.batch_size = int(128)  # num of transitions sampled from replay buf.
            self.horizon_len = int(2048)  # collect horizon_len step while exploring, then update network
            self.buffer_size = None  # ReplayBuffer size. Empty the ReplayBuffer for on-policy.
            self.repeat_times = 8.0  # repeatedly update network using ReplayBuffer to keep critic's loss small

        '''Arguments for device'''
        self.gpu_id = int(0)  # `int` means the ID of single GPU, -1 means CPU
        self.thread_num = int(8)  # cpu_num for pytorch, `th.set_num_threads(self.num_threads)`
        self.random_seed = int(0)  # initialize random seed in self.init_before_training()

        '''Arguments for evaluate'''
        self.cwd = None  # current working directory to save model. None means set automatically
        self.if_remove = True  # remove the cwd folder? (True, False, None:ask me)
        self.break_step = +np.inf  # break training if 'total_step > break_step'

        self.eval_times = int(32)  # number of times that get episodic cumulative return
        self.eval_per_step = int(1e4)  # evaluate the agent per training steps

    def init_before_training(self):
        np.random.seed(self.random_seed)
        th.manual_seed(self.random_seed)
        th.set_num_threads(self.thread_num)
        th.set_default_dtype(th.float32)

        if self.cwd is None:  # set cwd (current working directory) for saving model
            self.cwd = f'./{self.env_name}_{self.agent_class.__name__[5:]}_{self.random_seed}'

        if self.if_remove is None:  # remove or keep the history files
            self.if_remove = bool(input(f"| Arguments PRESS 'y' to REMOVE: {self.cwd}? ") == 'y')
        if self.if_remove:
            import shutil
            shutil.rmtree(self.cwd, ignore_errors=True)
            print(f"| Arguments Remove cwd: {self.cwd}")
        else:
            print(f"| Arguments Keep cwd: {self.cwd}")
        os.makedirs(self.cwd, exist_ok=True)

    def get_if_off_policy(self) -> bool:
        agent_name = self.agent_class.__name__ if self.agent_class else ''
        on_policy_names = ('SARSA', 'VPG', 'A2C', 'A3C', 'TRPO', 'PPO', 'MPO')
        return all([agent_name.find(s) == -1 for s in on_policy_names])


def get_gym_env_args(env, if_print: bool) -> dict:
    """Get a dict ``env_args`` about a standard OpenAI gym env information.

    param env: a standard OpenAI gym env
    param if_print: [bool] print the dict about env information.
    return: env_args [dict]

    env_args = {
        'env_name': env_name,       # [str] the environment name, such as XxxXxx-v0
        'state_dim': state_dim,     # [int] the dimension of state
        'action_dim': action_dim,   # [int] the dimension of action or the number of discrete action
        'if_discrete': if_discrete, # [bool] action space is discrete or continuous
    }
    """
    if {'unwrapped', 'observation_space', 'action_space', 'spec'}.issubset(dir(env)):  # isinstance(env, gym.Env):
        env_name = env.unwrapped.spec.id

        state_shape = env.observation_space.shape
        state_dim = state_shape[0] if len(state_shape) == 1 else state_shape  # sometimes state_dim is a list

        if_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        if if_discrete:  # make sure it is discrete action space
            action_dim = env.action_space.n
        elif isinstance(env.action_space, gym.spaces.Box):  # make sure it is continuous action space
            action_dim = env.action_space.shape[0]
            if any(env.action_space.high - 1):
                print('WARNING: env.action_space.high', env.action_space.high)
            if any(env.action_space.low + 1):
                print('WARNING: env.action_space.low', env.action_space.low)
        else:
            raise RuntimeError('\n| Error in get_gym_env_info(). Please set these value manually:'
                               '\n  `state_dim=int; action_dim=int; if_discrete=bool;`'
                               '\n  And keep action_space in range (-1, 1).')
    else:
        env_name = env.env_name
        state_dim = env.state_dim
        action_dim = env.action_dim
        if_discrete = env.if_discrete

    env_args = {'env_name': env_name,
                'state_dim': state_dim,
                'action_dim': action_dim,
                'if_discrete': if_discrete, }
    if if_print:
        env_args_str = repr(env_args).replace(',', f",\n{'':11}")
        print(f"env_args = {env_args_str}")
    return env_args


def kwargs_filter(function, kwargs: dict) -> dict:
    import inspect
    sign = inspect.signature(function).parameters.values()
    sign = {val.name for val in sign}
    common_args = sign.intersection(kwargs.keys())
    return {key: kwargs[key] for key in common_args}  # filtered kwargs


def build_env(env_class=None, env_args=None):
    if env_class.__module__ == 'gymnasium.envs.registration':  # special rule
        env = env_class(id=env_args['env_name'])
    else:
        env = env_class(**kwargs_filter(env_class.__init__, env_args.copy()))
    for attr_str in ('env_name', 'state_dim', 'action_dim', 'if_discrete'):
        setattr(env, attr_str, env_args[attr_str])
    return env


'''utils of network'''


def build_mlp(dims: List[int]) -> nn.Sequential:  # MLP (MultiLayer Perceptron)
    net_list = []
    for i in range(len(dims) - 1):
        net_list.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU()])
    del net_list[-1]  # remove the activation of output layer
    return nn.Sequential(*net_list)


def layer_init_with_orthogonal(layer, std=1.0, bias_const=1e-6):
    th.nn.init.orthogonal_(layer.weight, std)
    th.nn.init.constant_(layer.bias, bias_const)


'''AgentPPO'''


class ActorPPO(nn.Module):
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int):
        super().__init__()
        self.net = build_mlp(dims=[state_dim, *net_dims, action_dim])
        self.action_std_log = nn.Parameter(th.zeros((1, action_dim)), requires_grad=True)  # trainable parameter
        self.ActionDist = torch.distributions.normal.Normal

    def forward(self, state: TEN) -> TEN:
        action = self.net(state)
        return self.convert_action_for_env(action)

    def get_action(self, state: TEN) -> Tuple[TEN, TEN]:  # for exploration
        action_avg = self.net(state)
        action_std = self.action_std_log.exp()

        dist = self.ActionDist(action_avg, action_std)
        action = dist.sample()
        logprob = dist.log_prob(action).sum(1)
        return action, logprob

    def get_logprob_entropy(self, state: TEN, action: TEN) -> Tuple[TEN, TEN]:
        action_avg = self.net(state)
        action_std = self.action_std_log.exp()

        dist = self.ActionDist(action_avg, action_std)
        logprob = dist.log_prob(action).sum(1)
        entropy = dist.entropy().sum(1)
        return logprob, entropy

    @staticmethod
    def convert_action_for_env(action: TEN) -> TEN:
        return action.tanh()


class CriticPPO(nn.Module):
    def __init__(self, net_dims: List[int], state_dim: int):
        super().__init__()
        self.net = build_mlp(dims=[state_dim, *net_dims, 1])

    def forward(self, state: TEN) -> TEN:
        return self.net(state)  # advantage value


class AgentPPO:
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.if_discrete: bool = False
        self.if_off_policy: bool = False

        self.net_dims: List[int] = net_dims
        self.state_dim: int = state_dim
        self.action_dim: int = action_dim

        self.gamma: float = args.gamma
        self.batch_size: int = args.batch_size
        self.repeat_times: float = args.repeat_times
        self.reward_scale: float = args.reward_scale
        self.learning_rate: float = args.learning_rate
        self.soft_update_tau: float = args.soft_update_tau

        self.explore_noise_std = getattr(args, 'explore_noise_std', 0.05)  # standard deviation of exploration noise

        self.last_state: Optional[ARY] = None  # state of the trajectory for training. `shape == (state_dim)`
        self.device = th.device(f"cuda:{gpu_id}" if (th.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        self.act = ActorPPO(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.cri = CriticPPO(net_dims=net_dims, state_dim=state_dim).to(self.device)
        self.act_optimizer = th.optim.Adam(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = th.optim.Adam(self.cri.parameters(), self.learning_rate)

        self.ratio_clip = getattr(args, "ratio_clip", 0.25)  # `ratio.clamp(1 - clip, 1 + clip)`
        self.lambda_gae_adv = getattr(args, "lambda_gae_adv", 0.95)  # could be 0.80~0.99
        self.lambda_entropy = getattr(args, "lambda_entropy", 0.01)  # could be 0.00~0.10
        self.lambda_entropy = th.tensor(self.lambda_entropy, dtype=th.float32, device=self.device)

        self.criterion = th.nn.SmoothL1Loss(reduction='none')

    def explore_env(self, env, horizon_len: int) -> Tuple[TEN, TEN, TEN, TEN, TEN, TEN]:
        states = th.zeros((horizon_len, self.state_dim), dtype=th.float32).to(self.device)
        actions = th.zeros((horizon_len, self.action_dim), dtype=th.float32).to(self.device)
        logprobs = th.zeros(horizon_len, dtype=th.float32).to(self.device)
        rewards = th.zeros(horizon_len, dtype=th.float32).to(self.device)
        terminals = th.zeros(horizon_len, dtype=th.bool).to(self.device)
        truncates = th.zeros(horizon_len, dtype=th.bool).to(self.device)

        ary_state = self.last_state
        convert = self.act.convert_action_for_env
        for i in range(horizon_len):
            state = th.as_tensor(ary_state, dtype=th.float32, device=self.device)
            action, logprob = self.explore_action(state)

            ary_action = convert(action).detach().cpu().numpy()
            ary_state, reward, terminal, truncate, _ = env.step(ary_action)
            if terminal or truncate:
                ary_state, info_dict = env.reset()

            states[i] = state
            actions[i] = action
            rewards[i] = reward
            terminals[i] = terminal
            truncates[i] = truncate

        self.last_state = ary_state
        rewards = rewards.unsqueeze(1)
        undones = th.logical_not(terminals).unsqueeze(1)
        unmasks = th.logical_not(truncates).unsqueeze(1)
        return states, actions, logprobs, rewards, undones, unmasks

    def explore_action(self, state: TEN) -> Tuple[TEN, TEN]:
        actions, logprobs = self.act.get_action(state.unsqueeze(0))
        return actions[0], logprobs[0]

    def update_net(self, buffer) -> Tuple[float, float, float]:
        states, actions, logprobs, rewards, undones, unmasks = buffer
        buffer_size = states.shape[0]

        '''get advantages reward_sums'''
        bs = 2 ** 10  # set a smaller 'seq_num' when out of GPU memory.
        values = [self.cri(states[i:i + bs]) for i in range(0, buffer_size, bs)]
        values = th.cat(values, dim=0).squeeze(1)  # values.shape == (buffer_size, )

        advantages = self.get_advantages(states, rewards, undones, unmasks, values)  # shape == (buffer_size, )
        reward_sums = advantages + values  # reward_sums.shape == (buffer_size, )
        del rewards, undones, values

        advantages = (advantages - advantages.mean()) / (advantages.std(dim=0) + 1e-5)
        assert logprobs.shape == advantages.shape == reward_sums.shape == (buffer_size,)

        buffer = states, actions, unmasks, logprobs, advantages, reward_sums

        '''update network'''
        obj_critics = []
        obj_actors = []

        th.set_grad_enabled(True)
        update_times = int(buffer_size * self.repeat_times / self.batch_size)
        assert update_times >= 1
        for update_t in range(update_times):
            obj_critic, obj_actor = self.update_objectives(buffer, update_t)
            obj_critics.append(obj_critic)
            obj_actors.append(obj_actor)
        th.set_grad_enabled(False)

        obj_critic_avg = np.array(obj_critics).mean() if len(obj_critics) else 0.0
        obj_actor_avg = np.array(obj_actors).mean() if len(obj_actors) else 0.0
        a_std_log = getattr(self.act, 'a_std_log', th.zeros(1)).mean()
        return obj_critic_avg, obj_actor_avg, a_std_log.item()

    def update_objectives(self, buffer: Tuple[TEN, ...], _update_t: int) -> Tuple[float, float]:
        states, actions, unmasks, logprobs, advantages, reward_sums = buffer

        buffer_size = states.shape[0]
        indices = th.randint(buffer_size, size=(self.batch_size,), requires_grad=False)
        state = states[indices]
        action = actions[indices]
        unmask = unmasks[indices]
        logprob = logprobs[indices]
        advantage = advantages[indices]
        reward_sum = reward_sums[indices]

        value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
        obj_critic = (self.criterion(value, reward_sum) * unmask).mean()
        self.optimizer_backward(self.cri_optimizer, obj_critic)

        new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)
        ratio = (new_logprob - logprob.detach()).exp()
        surrogate1 = advantage * ratio
        surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
        obj_surrogate = th.min(surrogate1, surrogate2).mean()
        obj_actor = obj_surrogate + obj_entropy.mean() * self.lambda_entropy
        self.optimizer_backward(self.act_optimizer, -obj_actor)
        return obj_critic.item(), obj_actor.item()

    def get_advantages(self, states: TEN, rewards: TEN, undones: TEN, unmasks: TEN, values: TEN) -> TEN:
        advantages = th.empty_like(values)  # advantage value

        # update undones rewards when truncated
        truncated = th.logical_not(unmasks)
        if th.any(truncated):
            rewards[truncated] += self.cri(states[truncated.squeeze(1)]).detach().squeeze(1)
            undones[truncated] = False

        masks = undones * self.gamma
        horizon_len = rewards.shape[0]

        next_state = th.tensor(self.last_state, dtype=th.float32).to(self.device)
        next_value = self.cri(next_state.unsqueeze(0)).detach().squeeze(1).squeeze(0)

        advantage = 0  # last_gae_lambda
        for t in range(horizon_len - 1, -1, -1):
            delta = rewards[t] + masks[t] * next_value - values[t]
            advantages[t] = advantage = delta + masks[t] * self.lambda_gae_adv * advantage
            next_value = values[t]
        return advantages

    @staticmethod
    def optimizer_backward(optimizer, objective: TEN):
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()


'''evaluate'''


class Evaluator:
    def __init__(self, eval_env, eval_per_step: int = 1e4, eval_times: int = 8, cwd: str = '.'):
        self.cwd = cwd
        self.env_eval = eval_env
        self.eval_step = 0
        self.total_step = 0
        self.start_time = time.time()
        self.eval_times = eval_times  # number of times that get episodic cumulative return
        self.eval_per_step = eval_per_step  # evaluate the agent per training steps

        self.recorder = []
        print("| Evaluator:"
              "\n| `step`: Number of samples, or total training steps, or running times of `env.step()`."
              "\n| `time`: Time spent from the start of training to this moment."
              "\n| `avgR`: Average value of cumulative rewards, which is the sum of rewards in an episode."
              "\n| `stdR`: Standard dev of cumulative rewards, which is the sum of rewards in an episode."
              "\n| `avgS`: Average of steps in an episode."
              "\n| `objC`: Objective of Critic network. Or call it loss function of critic network."
              "\n| `objA`: Objective of Actor network. It is the average Q value of the critic network."
              f"\n| {'step':>8}  {'time':>8}  | {'avgR':>8}  {'stdR':>6}  {'avgS':>6}  | {'objC':>8}  {'objA':>8}")

    def evaluate_and_save(self, actor, horizon_len: int, logging_tuple: tuple):
        self.total_step += horizon_len
        if self.eval_step + self.eval_per_step > self.total_step:
            return
        self.eval_step = self.total_step

        rewards_steps_ary = [get_rewards_and_steps(self.env_eval, actor) for _ in range(self.eval_times)]
        rewards_steps_ary = np.array(rewards_steps_ary, dtype=np.float32)
        avg_r = rewards_steps_ary[:, 0].mean()  # average of cumulative rewards
        std_r = rewards_steps_ary[:, 0].std()  # std of cumulative rewards
        avg_s = rewards_steps_ary[:, 1].mean()  # average of steps in an episode

        used_time = time.time() - self.start_time
        self.recorder.append((self.total_step, used_time, avg_r))

        save_path = f"{self.cwd}/actor_{self.total_step:012.0f}_{used_time:08.0f}_{avg_r:08.2f}.pth"
        th.save(actor.state_dict(), save_path)
        print(f"| {self.total_step:8.2e}  {used_time:8.0f}  "
              f"| {avg_r:8.2f}  {std_r:6.2f}  {avg_s:6.0f}  "
              f"| {logging_tuple[0]:8.2f}  {logging_tuple[1]:8.2f}")

    def close(self):
        np.save(f"{self.cwd}/recorder.npy", np.array(self.recorder))
        draw_learning_curve_using_recorder(self.cwd)


def get_rewards_and_steps(env, actor, if_render: bool = False) -> (float, int):
    device = next(actor.parameters()).device  # net.parameters() is a Python generator.

    state, info_dict = env.reset()
    episode_steps = 0
    cumulative_returns = 0.0  # sum of rewards in an episode
    for episode_steps in range(12345):
        tensor_state = th.as_tensor(state, dtype=th.float32, device=device).unsqueeze(0)
        tensor_action = actor(tensor_state)
        action = tensor_action.detach().cpu().numpy()[0]  # not need detach(), because using th.no_grad() outside
        state, reward, terminated, truncated, _ = env.step(action)
        cumulative_returns += reward

        if if_render:
            env.render()
        if terminated or truncated:
            break
    env_unwrapped = getattr(env, 'unwrapped', env)
    cumulative_returns = getattr(env_unwrapped, 'cumulative_returns', cumulative_returns)
    return cumulative_returns, episode_steps + 1


def draw_learning_curve_using_recorder(cwd: str):
    recorder = np.load(f"{cwd}/recorder.npy")

    import matplotlib as mpl
    mpl.use('Agg')  # write  before `import matplotlib.pyplot as plt`. `plt.savefig()` without a running X server
    import matplotlib.pyplot as plt
    x_axis = recorder[:, 0]
    y_axis = recorder[:, 2]
    plt.plot(x_axis, y_axis)
    plt.xlabel('#samples (Steps)')
    plt.ylabel('#Rewards (Score)')
    plt.grid()

    file_path = f"{cwd}/LearningCurve.jpg"
    # plt.show()  # if use `mpl.use('Agg')` to draw figures without GUI, then plt can't plt.show()
    plt.savefig(file_path)
    print(f"| Save learning curve in {file_path}")


'''run'''


def run(gpu_id: int = 0):
    agent_class = AgentPPO  # DRL algorithm name
    beg_idx, end_idx = (0, 1113)  # (0, 834)
    env_class = StockTradingEnv  # run a custom env: StockTradingEnv
    env_args = get_gym_env_args(env=StockTradingEnv(beg_idx=beg_idx, end_idx=end_idx), if_print=True)

    args = Config(agent_class, env_class, env_args)  # see `erl_config.py Arguments()` for hyperparameter explanation

    max_step = end_idx - beg_idx
    args.target_step = max_step * 4
    args.reward_scale = 2 ** -7
    args.learning_rate = 2 ** -14
    args.break_step = int(5e5)

    args.learner_gpus = gpu_id
    args.random_seed += gpu_id + 1943
    args.gpu_id = gpu_id

    train_agent(args)
    if input("| Press 'y' to load actor.pth and render:") == 'y':
        actor_name = sorted([s for s in os.listdir(args.cwd) if s[-4:] == '.pth'])[-1]
        actor_path = f"{args.cwd}/{actor_name}"
        env_args['beg_idx'], env_args['end_idx'] = (0, 1113)  # (834, 1113)
        env_args['end_idx'] = 1113
        valid_agent(env_class, env_args, args.net_dims, agent_class, actor_path)


def train_agent(args: Config):
    args.init_before_training()
    th.set_grad_enabled(False)

    evaluator = Evaluator(
        eval_env=build_env(args.env_class, args.env_args),
        eval_per_step=args.eval_per_step,
        eval_times=args.eval_times,
        cwd=args.cwd,
    )

    env = build_env(args.env_class, args.env_args)
    agent = args.agent_class(args.net_dims, args.state_dim, args.action_dim, gpu_id=args.gpu_id, args=args)
    agent.temp_observ, info_dict = env.reset()

    buffer = []

    '''start training'''
    while True:
        buffer_items = agent.explore_env(env, args.horizon_len)
        buffer[:] = buffer_items

        logging_tuple = agent.update_net(buffer)

        evaluator.evaluate_and_save(agent.act, args.horizon_len, logging_tuple)
        if (evaluator.total_step > args.break_step) or os.path.exists(f"{args.cwd}/stop"):
            break  # stop training when reach `break_step` or `mkdir cwd/stop`
    evaluator.close()


def valid_agent(env_class, env_args: dict, net_dims: List[int], agent_class, actor_path: str, render_times: int = 8):
    env = build_env(env_class, env_args)

    state_dim = env_args['state_dim']
    action_dim = env_args['action_dim']
    agent = agent_class(net_dims, state_dim, action_dim, gpu_id=-1)
    actor = agent.act

    print(f"| render and load actor from: {actor_path}")
    actor.load_state_dict(th.load(actor_path, map_location=lambda storage, loc: storage))
    for i in range(render_times):
        cumulative_reward, episode_step = get_rewards_and_steps(env, actor, if_render=True)
        print(f"|{i:4}  cumulative_reward {cumulative_reward:9.3f}  episode_step {episode_step:5.0f}")


if __name__ == '__main__':
    run(gpu_id=0)
