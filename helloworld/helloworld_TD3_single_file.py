import os
import sys
import time
from copy import deepcopy

import gym
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class Config:  # for off-policy
    def __init__(self, agent_class=None, env_class=None, env_args=None):
        self.agent_class = agent_class  # agent = agent_class(...)
        self.if_off_policy = True  # whether off-policy or on-policy of DRL algorithm

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
        self.net_dims = (64, 32)  # the middle layer dimension of MLP (MultiLayer Perceptron)
        self.learning_rate = 6e-5  # 2 ** -14 ~= 6e-5
        self.soft_update_tau = 5e-3  # 2 ** -8 ~= 5e-3
        self.state_value_tau = 0.1  # 0.05 ~ 0.50
        self.batch_size = int(64)  # num of transitions sampled from replay buffer.
        self.horizon_len = int(256)  # collect horizon_len step while exploring, then update network
        self.buffer_size = int(1e6)  # ReplayBuffer size. First in first out for off-policy.
        self.repeat_times = 1.0  # repeatedly update network using ReplayBuffer to keep critic's loss small

        '''Arguments for device'''
        self.gpu_id = int(0)  # `int` means the ID of single GPU, -1 means CPU
        self.thread_num = int(8)  # cpu_num for pytorch, `torch.set_num_threads(self.num_threads)`
        self.random_seed = int(0)  # initialize random seed in self.init_before_training()

        '''Arguments for evaluate'''
        self.cwd = None  # current working directory to save model. None means set automatically
        self.if_remove = True  # remove the cwd folder? (True, False, None:ask me)
        self.break_step = +np.inf  # break training if 'total_step > break_step'

        self.eval_times = int(16)  # number of times that get episodic cumulative return
        self.eval_per_step = int(1e4)  # evaluate the agent per training steps

    def init_before_training(self):
        if self.cwd is None:  # set cwd (current working directory) for saving model
            self.cwd = f'./{self.env_name}_{self.agent_class.__name__[5:]}'
        os.makedirs(self.cwd, exist_ok=True)


class ActorBase(nn.Module):  # todo state_norm
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = None  # build_mlp(dims=[state_dim, *dims, action_dim])
        self.explore_noise_std = None  # standard deviation of exploration action noise
        self.ActionDist = torch.distributions.normal.Normal

        self.state_avg = nn.Parameter(torch.zeros((state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(torch.ones((state_dim,)), requires_grad=False)

    def state_norm(self, state: Tensor) -> Tensor:
        return (state - self.state_avg) / self.state_std  # todo state_norm


class Actor(ActorBase):
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.net = build_mlp(dims=[state_dim, *dims, action_dim])
        layer_init_with_orthogonal(self.net[-1], std=0.5)

    def forward(self, state: Tensor) -> Tensor:
        state = self.state_norm(state)
        action = self.net(state)
        return action.tanh()

    def get_action(self, state: Tensor) -> Tensor:  # for exploration
        state = self.state_norm(state)
        action_avg = self.net(state).tanh()
        dist = self.ActionDist(action_avg, self.explore_noise_std)
        action = dist.sample()
        return action.clip(-1.0, 1.0)


class CriticBase(nn.Module):  # todo state_norm, value_norm
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = None  # build_mlp(dims=[state_dim + action_dim, *dims, 1])

        self.state_avg = nn.Parameter(torch.zeros((state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(torch.ones((state_dim,)), requires_grad=False)
        self.value_avg = nn.Parameter(torch.zeros((1,)), requires_grad=False)
        self.value_std = nn.Parameter(torch.ones((1,)), requires_grad=False)

    def state_norm(self, state: Tensor) -> Tensor:
        return (state - self.state_avg) / self.state_std  # todo state_norm

    def value_re_norm(self, value: Tensor) -> Tensor:
        return value * self.value_std + self.value_avg  # todo value_norm


class CriticTwin(CriticBase):
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.enc_sa = build_mlp(dims=[state_dim + action_dim, *dims])  # encoder of state and action
        self.dec_q1 = build_mlp(dims=[dims[-1], action_dim])  # decoder of Q value 1
        self.dec_q2 = build_mlp(dims=[dims[-1], action_dim])  # decoder of Q value 2
        layer_init_with_orthogonal(self.dec_q1[-1], std=0.5)
        layer_init_with_orthogonal(self.dec_q2[-1], std=0.5)

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        state = self.state_norm(state)
        sa_tmp = self.enc_sa(torch.cat((state, action), dim=1))
        value = self.dec_q1(sa_tmp)
        value = self.value_re_norm(value)
        return value  # Q value

    def get_q1_q2(self, state, action):
        state = self.state_norm(state)
        sa_tmp = self.enc_sa(torch.cat((state, action), dim=1))
        value1 = self.value_re_norm(self.dec_q1(sa_tmp))
        value2 = self.value_re_norm(self.dec_q2(sa_tmp))
        return value1, value2  # two Q values


def layer_init_with_orthogonal(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)


def build_mlp(dims: [int]) -> nn.Sequential:  # MLP (MultiLayer Perceptron)
    net_list = []
    for i in range(len(dims) - 1):
        net_list.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU()])
    del net_list[-1]  # remove the activation of output layer
    return nn.Sequential(*net_list)


def get_gym_env_args(env, if_print: bool) -> dict:
    if {'unwrapped', 'observation_space', 'action_space', 'spec'}.issubset(dir(env)):  # isinstance(env, gym.Env):
        env_name = env.unwrapped.spec.id
        state_shape = env.observation_space.shape
        state_dim = state_shape[0] if len(state_shape) == 1 else state_shape  # sometimes state_dim is a list
        if_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        action_dim = env.action_space.n if if_discrete else env.action_space.shape[0]
    else:
        env_name = env.env_name
        state_dim = env.state_dim
        action_dim = env.action_dim
        if_discrete = env.if_discrete
    env_args = {'env_name': env_name, 'state_dim': state_dim, 'action_dim': action_dim, 'if_discrete': if_discrete}
    print(f"env_args = {repr(env_args)}") if if_print else None
    return env_args


def kwargs_filter(function, kwargs: dict) -> dict:
    import inspect
    sign = inspect.signature(function).parameters.values()
    sign = {val.name for val in sign}
    common_args = sign.intersection(kwargs.keys())
    return {key: kwargs[key] for key in common_args}  # filtered kwargs


def build_env(env_class=None, env_args=None):
    if env_class.__module__ == 'gym.envs.registration':  # special rule
        assert '0.18.0' <= gym.__version__ <= '0.25.2'  # pip3 install gym==0.24.0
        env = env_class(id=env_args['env_name'])
    else:
        env = env_class(**kwargs_filter(env_class.__init__, env_args.copy()))
    for attr_str in ('env_name', 'state_dim', 'action_dim', 'if_discrete'):
        setattr(env, attr_str, env_args[attr_str])
    return env


class AgentBase:
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.repeat_times = args.repeat_times
        self.reward_scale = args.reward_scale
        self.learning_rate = args.learning_rate
        self.if_off_policy = args.if_off_policy
        self.soft_update_tau = args.soft_update_tau
        self.state_value_tau = args.state_value_tau

        self.last_state = None  # save the last state of the trajectory for training. `last_state.shape == (state_dim)`
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        act_class = getattr(self, "act_class", None)
        cri_class = getattr(self, "cri_class", None)
        self.act = self.act_target = act_class(net_dims, state_dim, action_dim).to(self.device)
        self.cri = self.cri_target = cri_class(net_dims, state_dim, action_dim).to(self.device) \
            if cri_class else self.act

        self.act_optimizer = torch.optim.Adam(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), self.learning_rate) \
            if cri_class else self.act_optimizer

        self.criterion = torch.nn.SmoothL1Loss()

    @staticmethod
    def optimizer_update(optimizer, objective: Tensor):
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    @staticmethod
    def soft_update(target_net: torch.nn.Module, current_net: torch.nn.Module, tau: float):
        # assert target_net is not current_net
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

    def update_avg_std_for_state_value_norm(self, states: Tensor, returns: Tensor):
        tau = self.state_value_tau
        if tau == 0:
            return

        state_avg = states.mean(dim=0, keepdim=True)
        state_std = states.std(dim=0, keepdim=True)
        self.act.state_avg[:] = self.act.state_avg * (1 - tau) + state_avg * tau
        self.act.state_std[:] = self.cri.state_std * (1 - tau) + state_std * tau + 1e-4
        self.cri.state_avg[:] = self.act.state_avg
        self.cri.state_std[:] = self.cri.state_std

        returns_avg = returns.mean(dim=0)
        returns_std = returns.std(dim=0)
        self.cri.value_avg[:] = self.cri.value_avg * (1 - tau) + returns_avg * tau
        self.cri.value_std[:] = self.cri.value_std * (1 - tau) + returns_std * tau + 1e-4


class AgentTD3(AgentBase):
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.act_class = getattr(self, 'act_class', Actor)  # get the attribute of object `self`
        self.cri_class = getattr(self, 'cri_class', CriticTwin)  # get the attribute of object `self`
        super().__init__(net_dims, state_dim, action_dim, gpu_id, args)
        self.cri_target = deepcopy(self.cri)
        self.act_target = deepcopy(self.act)

        self.explore_noise_std = getattr(args, 'explore_noise_std', 0.06)  # standard deviation of exploration noise
        self.policy_noise_std = getattr(args, 'policy_noise_std', 0.12)  # standard deviation of exploration noise
        self.update_freq = getattr(args, 'update_freq', 2)  # standard deviation of exploration noise
        self.horizon_len = 0

    def explore_env(self, env, horizon_len: int, if_random: bool = False) -> [Tensor]:
        self.act.explore_noise_std = self.act_target.explore_noise_std = self.explore_noise_std
        self.horizon_len = 0

        states = torch.zeros((horizon_len, self.state_dim), dtype=torch.float32).to(self.device)
        actions = torch.zeros((horizon_len, self.action_dim), dtype=torch.float32).to(self.device)
        rewards = torch.zeros(horizon_len, dtype=torch.float32).to(self.device)
        dones = torch.zeros(horizon_len, dtype=torch.bool).to(self.device)

        ary_state = self.last_state
        get_action = self.act.get_action
        for i in range(horizon_len):
            state = torch.as_tensor(ary_state, dtype=torch.float32, device=self.device)
            action = torch.rand(self.action_dim) * 2 - 1.0 if if_random else get_action(state.unsqueeze(0))[0]

            states[i] = state
            actions[i] = action

            ary_action = action.detach().cpu().numpy()
            ary_state, reward, done, _ = env.step(ary_action)
            if done:
                ary_state = env.reset()

            rewards[i] = reward
            dones[i] = done

        self.last_state = ary_state
        rewards = rewards.unsqueeze(1)
        undones = (1.0 - dones.type(torch.float32)).unsqueeze(1)
        return states, actions, rewards, undones

    def update_net(self, buffer) -> [float]:
        self.act.explore_noise_std = self.act_target.explore_noise_std = self.policy_noise_std

        states = buffer.states[-self.horizon_len:]
        reward_sums = buffer.rewards[-self.horizon_len:] * (1 / (1 - self.gamma))
        self.update_avg_std_for_state_value_norm(
            states=states.reshape((-1, self.state_dim)),
            returns=reward_sums.reshape((-1,))
        )

        obj_critics = obj_actors = 0.0
        update_times = int(buffer.cur_size * self.repeat_times / self.batch_size)
        for t in range(update_times):
            obj_critic, state = self.get_obj_critic(buffer, self.batch_size)
            self.optimizer_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
            obj_critics += obj_critic.item()

            if t % self.update_freq == 0:
                action = self.act(state)  # policy gradient
                obj_actor = (self.cri(state, action)).mean()
                self.optimizer_update(self.act_optimizer, -obj_actor)
                self.soft_update(self.act_target, self.act, self.soft_update_tau)
                obj_actors += obj_actor.item()
        return obj_critics / update_times, obj_actors / (update_times / self.update_freq)

    def get_obj_critic(self, buffer, batch_size: int) -> (Tensor, Tensor):
        with torch.no_grad():
            state, action, reward, undone, next_state = buffer.sample(batch_size)

            next_action = self.act.get_action(next_state)  # stochastic policy
            next_q = torch.min(*self.cri_target.get_q1_q2(next_state, next_action))  # twin critics
            q_label = reward + undone * self.gamma * next_q
        q1, q2 = self.cri.get_q1_q2(state, action)
        obj_critic = (self.criterion(q1, q_label) + self.criterion(q2, q_label)) / 2.
        return obj_critic, state


class ReplayBuffer:  # for off-policy
    def __init__(self, max_size: int, state_dim: int, action_dim: int, gpu_id: int = 0):
        self.p = 0  # pointer
        self.if_full = False
        self.cur_size = 0
        self.max_size = max_size
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        self.states = torch.empty((max_size, state_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.empty((max_size, action_dim), dtype=torch.float32, device=self.device)
        self.rewards = torch.empty((max_size, 1), dtype=torch.float32, device=self.device)
        self.undones = torch.empty((max_size, 1), dtype=torch.float32, device=self.device)

    def update(self, items: [Tensor]):
        states, actions, rewards, undones = items
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
        else:
            self.states[self.p:p] = states
            self.actions[self.p:p] = actions
            self.rewards[self.p:p] = rewards
            self.undones[self.p:p] = undones
        self.p = p
        self.cur_size = self.max_size if self.if_full else self.p

    def sample(self, batch_size: int) -> [Tensor]:
        ids = torch.randint(self.cur_size - 1, size=(batch_size,), requires_grad=False)
        return self.states[ids], self.actions[ids], self.rewards[ids], self.undones[ids], self.states[ids + 1]


class PendulumEnv(gym.Wrapper):  # a demo of custom gym env
    def __init__(self, gym_env_name=None):
        gym.logger.set_level(40)  # Block warning
        assert '0.18.0' <= gym.__version__ <= '0.25.2'  # pip3 install gym==0.24.0
        if gym_env_name is None:
            gym_env_name = "Pendulum-v0" if gym.__version__ < '0.18.0' else "Pendulum-v1"
        super().__init__(env=gym.make(gym_env_name))

        '''the necessary env information when you design a custom env'''
        self.env_name = gym_env_name  # the name of this env.
        self.state_dim = self.observation_space.shape[0]  # feature number of state
        self.action_dim = self.action_space.shape[0]  # feature number of action
        self.if_discrete = False  # discrete action or continuous action

    def reset(self) -> np.ndarray:  # reset the agent in env
        return self.env.reset()

    def step(self, action: np.ndarray) -> (np.ndarray, float, bool, dict):  # agent interacts in env
        # OpenAI Pendulum env set its action space as (-2, +2). It is bad.
        # We suggest that adjust action space to (-1, +1) when designing a custom env.
        state, reward, done, info_dict = self.env.step(action * 2)
        state = state.reshape(self.state_dim)
        return state, float(reward * 0.5), done, info_dict


def train_agent(args: Config):
    args.init_before_training()
    gpu_id = args.gpu_id

    env = build_env(args.env_class, args.env_args)
    agent = args.agent_class(args.net_dims, args.state_dim, args.action_dim, gpu_id=gpu_id, args=args)
    agent.last_state = env.reset()
    buffer = ReplayBuffer(gpu_id=gpu_id, max_size=args.buffer_size,
                          state_dim=args.state_dim, action_dim=1 if args.if_discrete else args.action_dim, )
    buffer_items = agent.explore_env(env, args.horizon_len * args.eval_times, if_random=True)
    buffer.update(buffer_items)  # warm up for ReplayBuffer

    evaluator = Evaluator(eval_env=build_env(args.env_class, args.env_args),
                          eval_per_step=args.eval_per_step, eval_times=args.eval_times, cwd=args.cwd)
    torch.set_grad_enabled(False)
    while True:  # start training
        buffer_items = agent.explore_env(env, args.horizon_len)
        buffer.update(buffer_items)

        torch.set_grad_enabled(True)
        logging_tuple = agent.update_net(buffer)
        torch.set_grad_enabled(False)

        evaluator.evaluate_and_save(agent.act, args.horizon_len, logging_tuple)
        if (evaluator.total_step > args.break_step) or os.path.exists(f"{args.cwd}/stop"):
            break  # stop training when reach `break_step` or `mkdir cwd/stop`


class Evaluator:
    def __init__(self, eval_env, eval_per_step: int = 1e4, eval_times: int = 8, cwd: str = '.'):
        self.cwd = cwd
        self.env_eval = eval_env
        self.eval_step = 0
        self.total_step = 0
        self.start_time = time.time()
        self.eval_times = eval_times  # number of times that get episodic cumulative return
        self.eval_per_step = eval_per_step  # evaluate the agent per training steps

        self.recorder = list()
        print("\n| `step`: Number of samples, or total training steps, or running times of `env.step()`."
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

        print(f"| {self.total_step:8.2e}  {used_time:8.0f}  "
              f"| {avg_r:8.2f}  {std_r:6.2f}  {avg_s:6.0f}  "
              f"| {logging_tuple[0]:8.2f}  {logging_tuple[1]:8.2f}")


def get_rewards_and_steps(env, actor, if_render: bool = False) -> (float, int):  # cumulative_rewards and episode_steps
    device = next(actor.parameters()).device  # net.parameters() is a Python generator.

    state = env.reset()
    episode_steps = 0
    cumulative_returns = 0.0  # sum of rewards in an episode
    for episode_steps in range(12345):
        tensor_state = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        tensor_action = actor(tensor_state)
        action = tensor_action.detach().cpu().numpy()[0]  # not need detach(), because using torch.no_grad() outside
        state, reward, done, _ = env.step(action)
        cumulative_returns += reward

        if if_render:
            env.render()
        if done:
            break
    cumulative_returns = getattr(env, 'cumulative_returns', cumulative_returns)  # todo
    return cumulative_returns, episode_steps + 1


def train_sac_for_pendulum(gpu_id=0):
    env_args = {
        'env_name': 'Pendulum',  # Apply torque on the free end to swing a pendulum into an upright position
        'state_dim': 3,  # the x-y coordinates of the pendulum's free end and its angular velocity.
        'action_dim': 1,  # the torque applied to free end of the pendulum
        'if_discrete': False  # continuous action space, symbols → direction, value → force
    }  # env_args = get_gym_env_args(env=gym.make('CartPole-v0'), if_print=True)

    args = Config(agent_class=AgentTD3, env_class=PendulumEnv, env_args=env_args)  # see `Config` for explanation
    args.break_step = int(1e5)  # break training if 'total_step > break_step'
    args.net_dims = (64, 32)  # the middle layer dimension of MultiLayer Perceptron
    args.gpu_id = gpu_id  # the ID of single GPU, -1 means CPU
    args.gamma = 0.97  # discount factor of future rewards

    train_agent(args)


train_sac_for_pendulum(gpu_id=int(sys.argv[1]) if len(sys.argv) > 1 else -1)
"""
|     step      time  |     avgR    stdR    avgS  |     objC      objA
| 1.02e+04       108  |  -745.94   26.63     200  |     0.76    -54.69
| 2.05e+04       302  |  -409.87   22.87     200  |     1.14    -72.69
| 3.07e+04       501  |  -309.10   31.90     200  |     0.74    -58.23
| 4.10e+04       800  |   -83.88   46.36     200  |     0.68    -43.74
| 5.12e+04      1103  |   -79.66   53.86     200  |     0.48    -32.24
"""
