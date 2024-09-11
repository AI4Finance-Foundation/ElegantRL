import os
import sys
import time
import numpy as np
import torch as th
import torch.nn as nn
import gymnasium as gym
from copy import deepcopy
from typing import List, Optional, Tuple

ARY = np.ndarray
TEN = th.Tensor


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
            self.batch_size = int(64)  # num of transitions sampled from replay buffer.
            self.horizon_len = int(512)  # collect horizon_len step while exploring, then update network
            self.buffer_size = int(1e6)  # ReplayBuffer size. First in first out for off-policy.
            self.repeat_times = 1.0  # repeatedly update network using ReplayBuffer to keep critic's loss small
        else:  # on-policy
            self.batch_size = int(128)  # num of transitions sampled from replay buffer.
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


'''ReplayBuffer'''


class ReplayBuffer:  # for off-policy
    def __init__(self, max_size: int, state_dim: int, action_dim: int, gpu_id: int = 0):
        self.p = 0  # pointer
        self.if_full = False
        self.cur_size = 0
        self.max_size = max_size
        self.device = th.device(f"cuda:{gpu_id}" if (th.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        self.states = th.empty((max_size, state_dim), dtype=th.float32, device=self.device)
        self.actions = th.empty((max_size, action_dim), dtype=th.float32, device=self.device)
        self.rewards = th.empty((max_size, 1), dtype=th.float32, device=self.device)
        self.undones = th.empty((max_size, 1), dtype=th.float32, device=self.device)
        self.unmasks = th.empty((max_size, 1), dtype=th.float32, device=self.device)

    def update(self, items: Tuple[TEN, TEN, TEN, TEN, TEN]):
        states, actions, rewards, undones, unmasks = items
        p = self.p + rewards.shape[0]  # pointer
        if p > self.max_size:
            self.if_full = True
            p0 = self.p
            p1 = self.max_size
            p2 = self.max_size - self.p
            p = p - self.max_size

            self.states[p0:p1], self.states[0:p] = states[:p2], states[-p:]
            self.actions[p0:p1], self.actions[0:p] = actions[:p2], actions[-p:]
            self.rewards[p0:p1], self.rewards[0:p] = rewards[:p2], rewards[-p:]
            self.undones[p0:p1], self.undones[0:p] = undones[:p2], undones[-p:]
            self.unmasks[p0:p1], self.unmasks[0:p] = unmasks[:p2], unmasks[-p:]
        else:
            self.states[self.p:p] = states
            self.actions[self.p:p] = actions
            self.rewards[self.p:p] = rewards
            self.undones[self.p:p] = undones
            self.unmasks[self.p:p] = unmasks
        self.p = p
        self.cur_size = self.max_size if self.if_full else self.p

    def sample(self, batch_size: int) -> Tuple[TEN, TEN, TEN, TEN, TEN, TEN]:
        ids = th.randint(self.cur_size - 1, size=(batch_size,), requires_grad=False)
        return (
            self.states[ids],
            self.actions[ids],
            self.rewards[ids],
            self.undones[ids],
            self.unmasks[ids],
            self.states[ids + 1],
        )


'''Agent of DRL algorithms'''


class AgentBase:
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.if_discrete: bool = args.if_discrete
        self.if_off_policy: bool = args.if_off_policy

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

        self.act = None
        self.cri = None
        self.act_target = self.act
        self.cri_target = self.cri

        self.act_optimizer: Optional[th.optim] = None
        self.cri_optimizer: Optional[th.optim] = None
        self.criterion = th.nn.SmoothL1Loss(reduction='none')

    def explore_env(self, env, horizon_len: int) -> Tuple[TEN, TEN, TEN, TEN, TEN]:
        states = th.zeros((horizon_len, self.state_dim), dtype=th.float32).to(self.device)
        actions = th.zeros((horizon_len, self.action_dim), dtype=th.float32).to(self.device) \
            if not self.if_discrete else th.zeros((horizon_len, 1), dtype=th.int32).to(self.device)
        rewards = th.zeros(horizon_len, dtype=th.float32).to(self.device)
        terminals = th.zeros(horizon_len, dtype=th.bool).to(self.device)
        truncates = th.zeros(horizon_len, dtype=th.bool).to(self.device)

        ary_state = self.last_state
        for i in range(horizon_len):
            state = th.as_tensor(ary_state, dtype=th.float32, device=self.device)
            action = self.explore_action(state)

            states[i] = state
            actions[i] = action

            ary_action = action.detach().cpu().numpy()
            ary_state, reward, terminal, truncate, _ = env.step(ary_action)
            if terminal or truncate:
                ary_state, info_dict = env.reset()

            rewards[i] = reward
            terminals[i] = terminal
            truncates[i] = truncate

        self.last_state = ary_state
        rewards = rewards.unsqueeze(1)
        undones = th.logical_not(terminals).unsqueeze(1)
        unmasks = th.logical_not(truncates).unsqueeze(1)
        return states, actions, rewards, undones, unmasks

    def explore_action(self, state: TEN) -> TEN:
        return self.act.explore_action(state.unsqueeze(0), action_std=self.explore_noise_std)[0]

    def update_net(self, buffer) -> Tuple[float, ...]:
        objs_critic = []
        objs_actor = []

        th.set_grad_enabled(True)
        update_times = int(buffer.cur_size * self.repeat_times / self.batch_size)
        for update_t in range(update_times):
            obj_critic, obj_actor = self.update_objectives(buffer, self.batch_size, update_t)
            objs_critic.append(obj_critic)
            objs_actor.append(obj_actor) if isinstance(obj_actor, float) else None
        th.set_grad_enabled(False)

        obj_avg_critic = np.nanmean(objs_critic) if len(objs_critic) else 0.0
        obj_avg_actor = np.nanmean(objs_actor) if len(objs_actor) else 0.0
        return obj_avg_critic, obj_avg_actor

    def update_objectives(self, buffer: ReplayBuffer, batch_size: int, _update_t: int) -> Tuple[float, float]:
        with th.no_grad():
            state, action, reward, undone, unmask, next_state = buffer.sample(batch_size)

            next_action = self.act(next_state)  # deterministic policy
            next_q = self.cri_target(next_state, next_action)

            q_label = reward + undone * self.gamma * next_q

        q_value = self.cri(state, action) * unmask
        obj_critic = (self.criterion(q_value, q_label) * unmask).mean()
        self.optimizer_backward(self.cri_optimizer, obj_critic)
        self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

        action_pg = self.act(state)  # action to policy gradient
        obj_actor = self.cri(state, action_pg).mean()
        self.optimizer_backward(self.act_optimizer, -obj_actor)
        self.soft_update(self.act_target, self.act, self.soft_update_tau)
        return obj_critic.item(), obj_actor.item()

    @staticmethod
    def optimizer_backward(optimizer, objective: TEN):
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    @staticmethod
    def soft_update(target_net: th.nn.Module, current_net: th.nn.Module, tau: float):
        if target_net is current_net:
            return
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))


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


'''AgentDQN'''


class QNetwork(nn.Module):  # `nn.Module` is a PyTorch module for neural network
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int):
        super().__init__()
        self.net = build_mlp(dims=[state_dim, *net_dims, action_dim])
        self.action_dim = action_dim

    def forward(self, state: TEN) -> TEN:
        return self.net(state).argmax(dim=1)  # Q values for multiple actions

    def get_action(self, state: TEN, explore_rate: float) -> TEN:  # return the index List[int] of discrete action
        if explore_rate < th.rand(1):
            action = self.net(state).argmax(dim=1, keepdim=True)
        else:
            action = th.randint(self.action_dim, size=(state.shape[0], 1))
        return action

    def get_q_values(self, state: TEN) -> TEN:
        return self.net(state)


class AgentDQN:
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.if_discrete: bool = True
        self.if_off_policy: bool = True

        self.net_dims: List[int] = net_dims
        self.state_dim: int = state_dim
        self.action_dim: int = action_dim

        self.gamma: float = args.gamma
        self.batch_size: int = args.batch_size
        self.repeat_times: float = args.repeat_times
        self.reward_scale: float = args.reward_scale
        self.learning_rate: float = args.learning_rate
        self.soft_update_tau: float = args.soft_update_tau

        self.last_state: Optional[ARY] = None  # state of the trajectory for training. `shape == (state_dim)`
        self.device = th.device(f"cuda:{gpu_id}" if (th.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        self.act = self.cri = QNetwork(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.act_target = self.cri_target = deepcopy(self.act)
        self.act_optimizer = self.cri_optimizer = th.optim.Adam(self.act.parameters(), self.learning_rate)

        self.criterion = th.nn.SmoothL1Loss(reduction='none')

        self.explore_rate = getattr(args, "explore_rate", 0.25)  # set for `self.act.get_action()`
        # the probability of choosing action randomly in epsilon-greedy

    def explore_env(self, env, horizon_len: int) -> Tuple[TEN, TEN, TEN, TEN, TEN]:
        states = th.zeros((horizon_len, self.state_dim), dtype=th.float32).to(self.device)
        actions = th.zeros((horizon_len, self.action_dim), dtype=th.float32).to(self.device) \
            if not self.if_discrete else th.zeros((horizon_len, 1), dtype=th.int32).to(self.device)
        rewards = th.zeros(horizon_len, dtype=th.float32).to(self.device)
        terminals = th.zeros(horizon_len, dtype=th.bool).to(self.device)
        truncates = th.zeros(horizon_len, dtype=th.bool).to(self.device)

        ary_state = self.last_state
        for i in range(horizon_len):
            state = th.as_tensor(ary_state, dtype=th.float32, device=self.device)
            action = self.explore_action(state)

            states[i] = state
            actions[i] = action

            ary_action = action.detach().cpu().numpy()
            ary_state, reward, terminal, truncate, _ = env.step(ary_action)
            if terminal or truncate:
                ary_state, info_dict = env.reset()

            rewards[i] = reward
            terminals[i] = terminal
            truncates[i] = truncate

        self.last_state = ary_state
        rewards = rewards.unsqueeze(1)
        undones = th.logical_not(terminals).unsqueeze(1)
        unmasks = th.logical_not(truncates).unsqueeze(1)
        return states, actions, rewards, undones, unmasks

    def explore_action(self, state: TEN) -> TEN:
        return self.act.get_action(state.unsqueeze(0), explore_rate=self.explore_rate)[0, 0]

    def update_net(self, buffer) -> Tuple[float, ...]:
        objs_critic = []
        objs_actor = []

        th.set_grad_enabled(True)
        update_times = int(buffer.cur_size * self.repeat_times / self.batch_size)
        for update_t in range(update_times):
            obj_critic, obj_actor = self.update_objectives(buffer, self.batch_size, update_t)
            objs_critic.append(obj_critic)
            objs_actor.append(obj_actor) if isinstance(obj_actor, float) else None
        th.set_grad_enabled(False)

        obj_avg_critic = np.nanmean(objs_critic) if len(objs_critic) else 0.0
        obj_avg_actor = np.nanmean(objs_actor) if len(objs_actor) else 0.0
        return obj_avg_critic, obj_avg_actor

    def update_objectives(self, buffer: ReplayBuffer, batch_size: int, _update_t: int) -> Tuple[float, float]:
        with th.no_grad():
            state, action, reward, undone, unmask, next_state = buffer.sample(batch_size)

            next_q = self.cri_target.get_q_values(next_state).max(dim=1, keepdim=True)[0]
            q_label = reward + undone * self.gamma * next_q

        q_value = self.cri.get_q_values(state).gather(1, action.long())
        obj_critic = (self.criterion(q_value, q_label) * unmask).mean()
        self.optimizer_backward(self.cri_optimizer, obj_critic)
        self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

        obj_actor = q_value.detach().mean()
        return obj_critic.item(), obj_actor.item()

    @staticmethod
    def optimizer_backward(optimizer, objective: TEN):
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    @staticmethod
    def soft_update(target_net: th.nn.Module, current_net: th.nn.Module, tau: float):
        if target_net is current_net:
            return
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))


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
    agent: AgentBase = args.agent_class(args.net_dims, args.state_dim, args.action_dim, gpu_id=args.gpu_id, args=args)
    agent.last_state, info_dict = env.reset()

    if args.if_off_policy:
        buffer = ReplayBuffer(
            gpu_id=args.gpu_id,
            max_size=args.buffer_size,
            state_dim=args.state_dim,
            action_dim=1 if args.if_discrete else args.action_dim,
        )
        buffer_items = agent.explore_env(env, args.horizon_len * args.eval_times)
        buffer.update(buffer_items)  # warm up for ReplayBuffer
    else:
        buffer = []

    '''start training'''
    while True:
        buffer_items = agent.explore_env(env, args.horizon_len)
        if args.if_off_policy:
            buffer.update(buffer_items)
        else:
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


def train_dqn_for_cartpole(gpu_id: int = 0):
    env_class = gym.make
    env_args = {
        'env_name': 'CartPole-v1',  # A pole is attached by an un-actuated joint to a cart.
        'state_dim': 4,  # (CartPosition, CartVelocity, PoleAngle, PoleAngleVelocity)
        'action_dim': 2,  # (Push cart to the left, Push cart to the right)
        'if_discrete': True,  # discrete action space
    }  # env_args = get_gym_env_args(env=gym.make('CartPole-v0'), if_print=True)
    agent_class = AgentDQN

    args = Config(agent_class=agent_class, env_class=env_class, env_args=env_args)  # see `Config` for explanation
    args.break_step = int(1e5)  # break training if 'total_step > break_step'
    args.net_dims = [64, 32]  # the middle layer dimension of MultiLayer Perceptron
    args.gamma = 0.95  # discount factor of future rewards

    args.gpu_id = gpu_id  # the ID of single GPU, -1 means CPU
    train_agent(args)
    if input("| Press 'y' to load actor.pth and render:") == 'y':
        actor_name = sorted([s for s in os.listdir(args.cwd) if s[-4:] == '.pth'])[-1]
        actor_path = f"{args.cwd}/{actor_name}"
        valid_agent(env_class, env_args, args.net_dims, agent_class, actor_path)


if __name__ == '__main__':
    GPU_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    train_dqn_for_cartpole(gpu_id=GPU_ID)
"""
| Arguments Remove cwd: ./CartPole-v1_DQN_0
| Evaluator:
| `step`: Number of samples, or total training steps, or running times of `env.step()`.
| `time`: Time spent from the start of training to this moment.
| `avgR`: Average value of cumulative rewards, which is the sum of rewards in an episode.
| `stdR`: Standard dev of cumulative rewards, which is the sum of rewards in an episode.
| `avgS`: Average of steps in an episode.
| `objC`: Objective of Critic network. Or call it loss function of critic network.
| `objA`: Objective of Actor network. It is the average Q value of the critic network.
|     step      time  |     avgR    stdR    avgS  |     objC      objA
| 1.02e+04        19  |    17.31    2.11      17  |     0.92     19.77
| 2.05e+04        39  |     9.47    0.71       9  |     0.93     23.96
| 3.07e+04        66  |   191.25   18.10     191  |     1.38     31.52
| 4.10e+04        98  |   212.41   16.34     212  |     0.65     21.52
| 5.12e+04       141  |   183.41   10.96     183  |     0.47     21.10
| 6.14e+04       184  |   171.94    8.44     172  |     0.34     20.48
| 7.17e+04       233  |   173.00    8.85     173  |     0.27     19.88
| 8.19e+04       290  |   115.84    3.61     116  |     0.24     19.95
| 9.22e+04       349  |   128.44    5.99     128  |     0.19     19.80
"""
