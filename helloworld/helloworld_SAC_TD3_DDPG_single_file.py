import os
import sys
import time
import numpy as np
import torch as th
import torch.nn as nn
import gymnasium as gym
import torch.distributions.normal
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
        return self.act.get_action(state.unsqueeze(0), action_std=self.explore_noise_std)[0]

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


'''AgentDDPG'''


class Actor(nn.Module):
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int):
        super().__init__()
        self.net = build_mlp(dims=[state_dim, *net_dims, action_dim])
        self.explore_noise_std = None  # standard deviation of exploration action noise
        self.ActionDist = torch.distributions.normal.Normal

    def forward(self, state: TEN) -> TEN:
        action = self.net(state)
        return action.tanh()

    def get_action(self, state: TEN, action_std: float) -> TEN:  # for exploration
        action_avg = self.net(state).tanh()
        dist = self.ActionDist(action_avg, action_std)
        action = dist.sample()
        return action.clip(-1.0, 1.0)


class Critic(nn.Module):
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int):
        super().__init__()
        self.net = build_mlp(dims=[state_dim + action_dim, *net_dims, 1])

    def forward(self, state: TEN, action: TEN) -> TEN:
        return self.net(th.cat((state, action), dim=1))  # Q value


class AgentDDPG(AgentBase):
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        super().__init__(net_dims, state_dim, action_dim, gpu_id, args)
        self.explore_noise_std = getattr(args, 'explore_noise', 0.05)  # set for `self.get_policy_action()`

        self.act = Actor(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.cri = Critic(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.act_target = deepcopy(self.act)
        self.cri_target = deepcopy(self.cri)
        self.act_optimizer = th.optim.Adam(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = th.optim.Adam(self.cri.parameters(), self.learning_rate)

    def explore_action(self, state: TEN) -> TEN:
        return self.act.get_action(state.unsqueeze(0), action_std=self.explore_noise_std)[0]


'''AgentTD3'''


class CriticTwin(nn.Module):
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int, num_ensembles: int = 8):
        super().__init__()
        self.net = build_mlp(dims=[state_dim + action_dim, *net_dims, num_ensembles])
        layer_init_with_orthogonal(self.net[-1], std=0.5)

    def forward(self, state: TEN, action: TEN) -> TEN:
        values = self.get_q_values(state=state, action=action)
        value = values.mean(dim=-1, keepdim=True)
        return value  # Q value

    def get_q_values(self, state: TEN, action: TEN) -> TEN:
        values = self.net(th.cat((state, action), dim=1))
        return values  # Q values


class AgentTD3(AgentBase):
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        super().__init__(net_dims, state_dim, action_dim, gpu_id, args)
        self.update_freq = getattr(args, 'update_freq', 2)  # standard deviation of exploration noise
        self.num_ensembles = getattr(args, 'num_ensembles', 8)  # the number of critic networks
        self.policy_noise_std = getattr(args, 'policy_noise_std', 0.10)  # standard deviation of exploration noise
        self.explore_noise_std = getattr(args, 'explore_noise_std', 0.05)  # standard deviation of exploration noise

        self.act = Actor(net_dims, state_dim, action_dim).to(self.device)
        self.cri = CriticTwin(net_dims, state_dim, action_dim, num_ensembles=self.num_ensembles).to(self.device)
        self.act_target = deepcopy(self.act)
        self.cri_target = deepcopy(self.cri)
        self.act_optimizer = th.optim.Adam(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = th.optim.Adam(self.cri.parameters(), self.learning_rate)

    def update_objectives(self, buffer: ReplayBuffer, batch_size: int, _update_t: int) -> Tuple[float, float]:
        with th.no_grad():
            state, action, reward, undone, unmask, next_state = buffer.sample(batch_size)

            next_action = self.act.get_action(next_state, action_std=self.policy_noise_std)  # deterministic policy
            next_q = self.cri_target.get_q_values(next_state, next_action).min(dim=1, keepdim=True)[0]

            q_label = reward + undone * self.gamma * next_q

        q_values = self.cri.get_q_values(state, action)
        q_labels = q_label.repeat(1, q_values.shape[1])
        obj_critic = (self.criterion(q_values, q_labels) * unmask).mean()
        self.optimizer_backward(self.cri_optimizer, obj_critic)
        self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

        if _update_t % self.update_freq == 0:
            action_pg = self.act(state)  # action to policy gradient
            obj_actor = self.cri(state, action_pg).mean()
            self.optimizer_backward(self.act_optimizer, -obj_actor)
            self.soft_update(self.act_target, self.act, self.soft_update_tau)
        else:
            obj_actor = torch.tensor(float('nan'))
        return obj_critic.item(), obj_actor.item()


'''AgentSAC'''


class ActorSAC(nn.Module):
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int):
        super().__init__()
        self.encoder_s = build_mlp(dims=[state_dim, *net_dims])  # encoder of state
        self.decoder_a_avg = build_mlp(dims=[net_dims[-1], action_dim])  # decoder of action mean
        self.decoder_a_std = build_mlp(dims=[net_dims[-1], action_dim])  # decoder of action log_std
        self.soft_plus = nn.Softplus()

    def forward(self, state: TEN) -> TEN:
        state_tmp = self.encoder_s(state)  # temporary tensor of state
        return self.decoder_a_avg(state_tmp).tanh()  # action

    def get_action(self, state: TEN, **_kwargs) -> TEN:  # for exploration
        state_tmp = self.encoder_s(state)  # temporary tensor of state
        action_avg = self.decoder_a_avg(state_tmp)
        action_std = self.decoder_a_std(state_tmp).clamp(-20, 2).exp()

        noise = th.randn_like(action_avg, requires_grad=True)
        action = action_avg + action_std * noise
        return action.tanh()  # action (re-parameterize)

    def get_action_logprob(self, state: TEN) -> Tuple[TEN, TEN]:
        state_tmp = self.encoder_s(state)  # temporary tensor of state
        action_log_std = self.decoder_a_std(state_tmp).clamp(-20, 2)
        action_std = action_log_std.exp()
        action_avg = self.decoder_a_avg(state_tmp)

        noise = th.randn_like(action_avg, requires_grad=True)
        action = action_avg + action_std * noise
        logprob = -action_log_std - noise.pow(2) * 0.5 - np.log(np.sqrt(2 * np.pi))
        # dist = self.Normal(action_avg, action_std)
        # action = dist.sample()
        # logprob = dist.log_prob(action)

        '''fix logprob by adding the derivative of y=tanh(x)'''
        logprob -= (np.log(2.) - action - self.soft_plus(-2. * action)) * 2.  # better than below
        # logprob -= (1.000001 - action.tanh().pow(2)).log()
        return action.tanh(), logprob.sum(1, keepdim=True)


class CriticEnsemble(nn.Module):
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int, num_ensembles: int = 4):
        super().__init__()
        self.encoder_sa = build_mlp(dims=[state_dim + action_dim, net_dims[0]])  # encoder of state and action
        self.decoder_qs = []
        for net_i in range(num_ensembles):
            decoder_q = build_mlp(dims=[*net_dims, 1])
            layer_init_with_orthogonal(decoder_q[-1], std=0.5)

            self.decoder_qs.append(decoder_q)
            setattr(self, f"decoder_q{net_i:02}", decoder_q)

    def forward(self, state: TEN, action: TEN) -> TEN:
        values = self.get_q_values(state=state, action=action)
        value = values.mean(dim=-1, keepdim=True)
        return value  # Q value

    def get_q_values(self, state: TEN, action: TEN) -> TEN:
        tensor_sa = self.encoder_sa(th.cat((state, action), dim=1))
        values = th.concat([decoder_q(tensor_sa) for decoder_q in self.decoder_qs], dim=-1)
        return values  # Q values   tr4


class AgentSAC(AgentBase):
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        super().__init__(net_dims, state_dim, action_dim, gpu_id, args)
        self.num_ensembles = getattr(args, 'num_ensembles', 8)  # the number of critic networks

        self.act = ActorSAC(net_dims, state_dim, action_dim).to(self.device)
        self.cri = CriticEnsemble(net_dims, state_dim, action_dim, num_ensembles=self.num_ensembles).to(self.device)
        self.cri_target = deepcopy(self.cri)
        self.act_optimizer = th.optim.Adam(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = th.optim.Adam(self.cri.parameters(), self.learning_rate)

        self.alpha_log = th.tensor((-1,), dtype=th.float32, requires_grad=True, device=self.device)  # trainable var
        self.alpha_optim = th.optim.Adam((self.alpha_log,), lr=args.learning_rate)
        self.target_entropy = -np.log(action_dim)

    def explore_action(self, state: TEN) -> TEN:
        return self.act.get_action(state.unsqueeze(0))[0]  # stochastic policy for exploration

    def update_objectives(self, buffer: ReplayBuffer, batch_size: int, _update_t: int) -> Tuple[float, float]:
        with th.no_grad():
            state, action, reward, undone, unmask, next_state = buffer.sample(batch_size)

            next_action, next_logprob = self.act.get_action_logprob(next_state)  # stochastic policy
            next_q = th.min(self.cri_target.get_q_values(next_state, next_action), dim=1, keepdim=True)[0]
            alpha = self.alpha_log.exp()
            q_label = reward + undone * self.gamma * (next_q - next_logprob * alpha)

        '''objective of critic (loss function of critic)'''
        q_values = self.cri.get_q_values(state, action)
        q_labels = q_label.repeat(1, q_values.shape[1])
        obj_critic = (self.criterion(q_values, q_labels) * unmask).mean()
        self.optimizer_backward(self.cri_optimizer, obj_critic)
        self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

        '''objective of alpha (temperature parameter automatic adjustment)'''
        action_pg, logprob = self.act.get_action_logprob(state)  # policy gradient
        obj_alpha = (self.alpha_log * (self.target_entropy - logprob).detach()).mean()
        self.optimizer_backward(self.alpha_optim, obj_alpha)

        '''objective of actor'''
        alpha = self.alpha_log.exp().detach()
        with torch.no_grad():
            self.alpha_log[:] = self.alpha_log.clamp(-16, 2)

        q_value_pg = self.cri_target(state, action_pg).mean()
        obj_actor = (q_value_pg - logprob * alpha).mean()
        self.optimizer_backward(self.act_optimizer, -obj_actor)
        # self.soft_update(self.act_target, self.act, self.soft_update_tau)
        return obj_critic.item(), obj_actor.item()


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


class PendulumEnv(gym.Wrapper):  # a demo of custom env
    def __init__(self):
        gym_env_name = 'Pendulum-v1'
        super().__init__(env=gym.make(gym_env_name))

        '''the necessary env information when you design a custom env'''
        self.env_name = gym_env_name  # the name of this env.
        self.state_dim = self.observation_space.shape[0]  # feature number of state
        self.action_dim = self.action_space.shape[0]  # feature number of action
        self.if_discrete = False  # discrete action or continuous action

    def reset(self, **kwargs) -> Tuple[ARY, dict]:  # reset the agent in env
        state, info_dict = self.env.reset()
        return state, info_dict

    def step(self, action: ARY) -> Tuple[ARY, float, bool, bool, dict]:  # agent interacts in env
        # OpenAI Pendulum env set its action space as (-2, +2). It is bad.
        # We suggest that adjust action space to (-1, +1) when designing a custom env.
        state, reward, terminated, truncated, info_dict = self.env.step(action * 2)
        state = state.reshape(self.state_dim)
        return state, float(reward) * 0.5, terminated, truncated, info_dict


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


def train_sac_td3_ddpg_for_pendulum(gpu_id: int = 0, drl_id: int = 0):
    agent_class = [AgentSAC, AgentTD3, AgentDDPG][drl_id]  # DRL algorithm name
    print(f"agent_class {agent_class.__name__}")

    env_class = PendulumEnv  # run a custom env: PendulumEnv, which based on OpenAI pendulum
    env_args = {
        'env_name': 'Pendulum-v1',  # Apply torque on the free end to swing a pendulum into an upright position
        'state_dim': 3,  # the x-y coordinates of the pendulum's free end and its angular velocity.
        'action_dim': 1,  # the torque applied to free end of the pendulum
        'if_discrete': False  # continuous action space, symbols → direction, value → force
    }
    get_gym_env_args(env=PendulumEnv(), if_print=True)  # return env_args

    args = Config(agent_class=agent_class, env_class=env_class, env_args=env_args)  # see `Config` for explanation
    args.break_step = int(1e5)  # break training if 'total_step > break_step'
    args.net_dims = [64, 32]  # the middle layer dimension of MultiLayer Perceptron
    args.gamma = 0.97  # discount factor of future rewards

    args.gpu_id = gpu_id  # the ID of single GPU, -1 means CPU
    train_agent(args)
    if input("| Press 'y' to load actor.pth and render:"):
        actor_name = sorted([s for s in os.listdir(args.cwd) if s[-4:] == '.pth'])[-1]
        actor_path = f"{args.cwd}/{actor_name}"
        valid_agent(env_class, env_args, args.net_dims, agent_class, actor_path)


def train_sac_td3_ddpg_for_lunar_lander(gpu_id: int = 0, drl_id: int = 0):
    agent_class = [AgentSAC, AgentTD3, AgentDDPG][drl_id]  # DRL algorithm name
    print(f"agent_class {agent_class.__name__}")

    env_class = gym.make
    env_args = {
        'env_name': 'LunarLanderContinuous-v2',  # A lander learns to land on a landing pad
        'state_dim': 8,  # coordinates xy, linear velocities xy, angle, angular velocity, two booleans
        'action_dim': 2,  # fire main engine or side engine.
        'if_discrete': False  # continuous action space, symbols → direction, value → force
    }
    get_gym_env_args(env=gym.make('LunarLanderContinuous-v2'), if_print=True)  # return env_args

    args = Config(agent_class=agent_class, env_class=env_class, env_args=env_args)  # see `Config` for explanation
    args.break_step = int(2e5)  # break training if 'total_step > break_step'
    args.net_dims = [128, 128]  # the middle layer dimension of MultiLayer Perceptron
    args.horizon_len = 256  # collect horizon_len step while exploring, then update network
    args.repeat_times = 1.0  # repeatedly update network using ReplayBuffer to keep critic's loss small
    args.state_update_tau = 1e-2  # do rolling normalization on state using soft update tau
    args.batch_size = 256  # do rolling normalization on state using soft update tau
    args.gamma = 0.98

    args.eval_times = 32
    args.eval_per_step = int(2e4)

    args.gpu_id = gpu_id  # the ID of single GPU, -1 means CPU
    train_agent(args)
    if input("| Press 'y' to load actor.pth and render:") == 'y':
        actor_name = sorted([s for s in os.listdir(args.cwd) if s[-4:] == '.pth'])[-1]
        actor_path = f"{args.cwd}/{actor_name}"
        valid_agent(env_class, env_args, args.net_dims, agent_class, actor_path)


if __name__ == '__main__':
    GPU_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    DRL_ID = int(sys.argv[2]) if len(sys.argv) > 1 else 0
    # agent_class = [AgentSAC, AgentTD3, AgentDDPG][drl_id]

    train_sac_td3_ddpg_for_pendulum(gpu_id=GPU_ID, drl_id=DRL_ID)
    train_sac_td3_ddpg_for_lunar_lander(gpu_id=GPU_ID, drl_id=DRL_ID)
"""   
cumulative returns range: -1000 < -700 < -100 < -50
AgentSAC  env_name Pendulum-v1
|     step      time  |     avgR    stdR    avgS  |     objC      objA
| 1.02e+04        37  |  -623.17  114.43     200  |     0.65    -62.60
| 2.05e+04       103  |  -482.81   27.75     200  |     0.94    -91.89
| 3.07e+04       190  |  -213.40   97.32     200  |     0.78    -77.40
| 4.10e+04       296  |   -77.75   41.87     200  |     0.62    -46.33
| 5.12e+04       427  |   -70.72   22.34     200  |     0.47    -32.85

AgentTD3  env_name Pendulum-v1
|     step      time  |     avgR    stdR    avgS  |     objC      objA
| 1.02e+04        80  |  -775.71   38.21     200  |     2.08    -58.31
| 2.05e+04       190  |  -682.85   23.14     200  |     1.22    -98.91
| 3.07e+04       330  |  -451.86   34.99     200  |     0.82   -115.65
| 4.10e+04       506  |  -100.34   84.90     200  |     0.75   -108.58
| 5.12e+04       715  |  -103.01   60.56     200  |     1.17    -85.16


cumulative returns range: -1500 < -140 < 200 < 280
AgentSAC  env_name LunarLanderContinuous-v2
|     step      time  |     avgR    stdR    avgS  |     objC      objA
| 2.02e+04       190  |   -24.88   99.83     854  |     1.93      2.25
| 4.04e+04       557  |   -27.52   43.49     995  |     2.19     14.83
| 6.07e+04      1104  |    10.36   45.61     997  |     1.70     14.53
| 8.09e+04      1834  |   104.48   65.05     916  |     1.54     11.82
| 1.01e+05      2743  |   160.15   67.38     783  |     1.56     10.33
| 1.21e+05      3832  |   146.69   57.42     824  |     1.30     10.22
| 1.42e+05      5105  |   162.57   53.90     799  |     1.45      9.00
| 1.62e+05      6552  |   139.77   79.76     787  |     1.42      8.45
| 1.82e+05      8191  |   121.63   79.18     869  |     1.46      6.80

AgentTD3  env_name LunarLanderContinuous-v2
|     step      time  |     avgR    stdR    avgS  |     objC      objA
| 2.02e+04       116  |   -11.51   75.17     272  |     1.51     35.14
| 4.04e+04       335  |   -73.55   75.77     556  |     1.20     37.81
| 6.07e+04       658  |   -83.49   80.27     964  |     1.12     17.95
| 8.09e+04      1084  |   184.33   80.48     543  |     1.13     36.24
| 1.01e+05      1616  |   142.96   96.74     744  |     0.90     27.96
| 1.21e+05      2254  |    62.56  102.60     795  |     1.05     27.80
| 1.42e+05      2998  |   -24.59   60.55     987  |     0.95     25.17
| 1.62e+05      3827  |   137.89  160.10     483  |     1.09     20.72
| 1.82e+05      4776  |    48.25  110.73     722  |     0.96     31.02
"""
