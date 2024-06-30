import os
import sys
import time
from copy import deepcopy
from typing import Tuple, List

import gymnasium
import numpy as np
import torch as th
import torch.nn as nn

ARY = np.ndarray
TEN = th.Tensor


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
        self.learning_rate = 1e-4  # 2 ** -14 ~= 6e-5
        self.soft_update_tau = 5e-3  # 2 ** -8 ~= 5e-3
        self.state_value_tau = 0.1  # 0.05 ~ 0.50
        self.batch_size = int(64)  # num of transitions sampled from replay buffer.
        self.horizon_len = int(256)  # collect horizon_len step while exploring, then update network
        self.buffer_size = int(1e6)  # ReplayBuffer size. First in first out for off-policy.
        self.repeat_times = 1.0  # repeatedly update network using ReplayBuffer to keep critic's loss small

        '''Arguments for device'''
        self.gpu_id = int(0)  # `int` means the ID of single GPU, -1 means CPU
        self.thread_num = int(8)  # cpu_num for pytorch, `th.set_num_threads(self.num_threads)`
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


class ActorBase(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = None  # build_mlp(dims=[state_dim, *dims, action_dim])
        self.ActionDist = th.distributions.normal.Normal
        self.action_std = None

        self.state_avg = nn.Parameter(th.zeros((state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(th.ones((state_dim,)), requires_grad=False)

    def state_norm(self, state: TEN) -> TEN:
        return (state - self.state_avg) / self.state_std


class Actor(ActorBase):
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.net = build_mlp(dims=[state_dim, *dims, action_dim])

    def forward(self, state: TEN) -> TEN:
        state = self.state_norm(state)
        action = self.net(state)
        return action.tanh()

    def get_action(self, state: TEN) -> TEN:  # for exploration
        state = self.state_norm(state)
        action_avg = self.net(state).tanh()
        dist = self.ActionDist(action_avg, self.action_std)
        action = dist.sample()
        return action.clip(-1.0, 1.0)


class ActorSAC(ActorBase):
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.enc_s = build_mlp(dims=[state_dim, *dims])  # encoder of state
        self.dec_a_avg = build_mlp(dims=[dims[-1], action_dim])  # decoder of action mean
        self.dec_a_std = build_mlp(dims=[dims[-1], action_dim])  # decoder of action log_std
        self.soft_plus = nn.Softplus()

    def forward(self, state: TEN) -> TEN:
        state = self.state_norm(state)
        state_tmp = self.enc_s(state)  # temporary tensor of state
        return self.dec_a_avg(state_tmp).tanh()  # action

    def get_action(self, state: TEN) -> TEN:  # for exploration
        state = self.state_norm(state)
        state_tmp = self.enc_s(state)  # temporary tensor of state
        action_avg = self.dec_a_avg(state_tmp)
        action_std = self.dec_a_std(state_tmp).clamp(-20, 2).exp()

        noise = th.randn_like(action_avg, requires_grad=True)
        action = action_avg + action_std * noise
        return action.tanh()  # action (re-parameterize)

    def get_action_logprob(self, state: TEN) -> Tuple[TEN, TEN]:
        state = self.state_norm(state)
        state_tmp = self.enc_s(state)  # temporary tensor of state
        action_log_std = self.dec_a_std(state_tmp).clamp(-20, 2)
        action_std = action_log_std.exp()
        action_avg = self.dec_a_avg(state_tmp)

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


class CriticBase(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = None  # build_mlp(dims=[state_dim + action_dim, *dims, 1])

        self.state_avg = nn.Parameter(th.zeros((state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(th.ones((state_dim,)), requires_grad=False)
        self.value_avg = nn.Parameter(th.zeros((1,)), requires_grad=False)
        self.value_std = nn.Parameter(th.ones((1,)), requires_grad=False)

    def state_norm(self, state: TEN) -> TEN:
        return (state - self.state_avg) / self.state_std

    def value_re_norm(self, value: TEN) -> TEN:
        return value * self.value_std + self.value_avg


class CriticTwin(CriticBase):
    def __init__(self, dims: [int], state_dim: int, action_dim: int, num_q_values: int = 8):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.net = build_mlp(dims=[state_dim + action_dim, *dims, num_q_values])  # encoder of state and action

    def forward(self, state: TEN, action: TEN) -> TEN:
        values = self.get_q_values(state=state, action=action)
        value = values.mean(dim=1, keepdim=True)
        return value  # Q value

    def get_q_values(self, state, action):
        state = self.state_norm(state)
        values = self.net(th.cat((state, action), dim=1))
        values = self.value_re_norm(values)
        return values  # Q values


class CriticEnsemble(CriticBase):
    def __init__(self, dims: [int], state_dim: int, action_dim: int, num_ensembles: int = 8):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.encoder_sa = build_mlp(dims=[state_dim + action_dim, dims[0]])  # encoder of state and action
        self.decoder_qs = [build_mlp(dims=[*dims, 1]) for _ in range(num_ensembles)]  # decoder of Q values

    def forward(self, state: TEN, action: TEN) -> TEN:
        values = self.get_q_values(state=state, action=action)
        value = values.mean(dim=1, keepdim=True)
        return value  # Q value

    def get_q_values(self, state, action):
        state = self.state_norm(state)
        sa_tmp = self.encoder_sa(th.cat((state, action), dim=1))
        values = th.concat([dec_q(sa_tmp) for dec_q in self.decoder_qs], dim=1)
        values = self.value_re_norm(values)
        return values  # Q values


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
        if_discrete = isinstance(env.action_space, gymnasium.spaces.Discrete)
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
    if env_class.__module__ == 'gymnasium.envs.registration':  # special rule
        env = env_class(id=env_args['env_name'])
    else:
        env = env_class(**kwargs_filter(env_class.__init__, env_args.copy()))
    for attr_str in ('env_name', 'state_dim', 'action_dim', 'if_discrete'):
        setattr(env, attr_str, env_args[attr_str])
    return env


class ReplayBuffer:  # for off-policy
    def __init__(self, max_size: int, state_dim: int, action_dim: int, gpu_id: int = 0):
        self.p = 0  # pointer
        self.if_full = False
        self.cur_size = 0
        self.add_size = 0
        self.max_size = max_size
        self.device = th.device(f"cuda:{gpu_id}" if (th.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        self.states = th.empty((max_size, state_dim), dtype=th.float32, device=self.device)
        self.actions = th.empty((max_size, action_dim), dtype=th.float32, device=self.device)
        self.rewards = th.empty((max_size, 1), dtype=th.float32, device=self.device)
        self.undones = th.empty((max_size, 1), dtype=th.float32, device=self.device)

    def update(self, items: List[TEN]):
        states, actions, rewards, undones = items
        add_size = rewards.shape[0]
        p = self.p + add_size  # pointer
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
        self.add_size = add_size
        self.cur_size = self.max_size if self.if_full else self.p

    def sample(self, batch_size: int) -> Tuple[TEN, TEN, TEN, TEN, TEN]:
        ids = th.randint(self.cur_size - 1, size=(batch_size,), requires_grad=False)
        return self.states[ids], self.actions[ids], self.rewards[ids], self.undones[ids], self.states[ids + 1]

    def slice(self, data: TEN, slice_size: int) -> TEN:
        slice_data = data[self.p - slice_size:self.p] if slice_size >= self.p \
            else th.vstack((data[slice_size - self.p:], data[:self.p]))
        return slice_data


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
        self.device = th.device(f"cuda:{gpu_id}" if (th.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        act_class = getattr(self, "act_class", None)
        cri_class = getattr(self, "cri_class", None)
        self.act = self.act_target = act_class(net_dims, state_dim, action_dim).to(self.device)
        self.cri = self.cri_target = cri_class(net_dims, state_dim, action_dim).to(self.device) \
            if cri_class else self.act

        self.act_optimizer = th.optim.Adam(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = th.optim.Adam(self.cri.parameters(), self.learning_rate) \
            if cri_class else self.act_optimizer

        self.criterion = th.nn.SmoothL1Loss()

    def explore_env(self, env, horizon_len: int, if_random: bool = False) -> Tuple[TEN, TEN, TEN, TEN]:
        states = th.zeros((horizon_len, self.state_dim), dtype=th.float32).to(self.device)
        actions = th.zeros((horizon_len, self.action_dim), dtype=th.float32).to(self.device)
        rewards = th.zeros(horizon_len, dtype=th.float32).to(self.device)
        dones = th.zeros(horizon_len, dtype=th.bool).to(self.device)

        state = self.last_state

        get_action = self.act.get_action
        for i in range(horizon_len):
            action = th.rand(self.action_dim) * 2 - 1.0 if if_random else get_action(state.unsqueeze(0))[0]
            states[i] = state

            ary_action = action.detach().cpu().numpy()
            old_state, reward, terminated, truncated, _ = env.step(ary_action)
            done = terminated or truncated

            new_state, info_dict = env.reset()
            state = th.as_tensor(new_state if done else old_state,
                                 dtype=th.float32, device=self.device)
            actions[i] = action
            rewards[i] = reward
            dones[i] = done

        self.last_state = state
        rewards = rewards.unsqueeze(1)
        undones = (1.0 - dones.type(th.float32)).unsqueeze(1)
        return states, actions, rewards, undones

    @staticmethod
    def optimizer_update(optimizer, objective: TEN):
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    @staticmethod
    def soft_update(target_net: th.nn.Module, current_net: th.nn.Module, tau: float):
        # assert target_net is not current_net
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

    def update_avg_std_for_state_value_norm(self, states: TEN, returns: TEN):
        tau = self.state_value_tau
        if tau == 0:
            return

        state_avg = states.mean(dim=0, keepdim=True)
        state_std = states.std(dim=0, keepdim=True)
        self.act.state_avg[:] = self.act.state_avg * (1 - tau) + state_avg * tau
        self.act.state_std[:] = self.cri.state_std * (1 - tau) + state_std * tau + 1e-4
        self.cri.state_avg[:] = self.act.state_avg
        self.cri.state_std[:] = self.act.state_std

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
        self.act.action_std = self.explore_noise_std
        self.update_freq = getattr(args, 'update_freq', 2)  # standard deviation of exploration noise
        self.horizon_len = 0

    def update_net(self, buffer: ReplayBuffer, if_update_act: bool = True) -> Tuple[float, float]:
        self.act.action_std = self.act_target.action_std = self.policy_noise_std
        with th.no_grad():
            add_states = buffer.slice(buffer.states, buffer.add_size)
            add_actions = buffer.slice(buffer.actions, buffer.add_size)
            add_returns = self.cri_target(add_states, add_actions)
            self.update_avg_std_for_state_value_norm(states=add_states, returns=add_returns)
            del add_states, add_actions, add_returns

        obj_critics = obj_actors = 0.0
        update_times = int(buffer.cur_size * self.repeat_times / self.batch_size)
        for t in range(update_times):
            obj_critic, state = self.get_obj_critic(buffer, self.batch_size)
            self.optimizer_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
            obj_critics += obj_critic.item()

            if if_update_act and (t % self.update_freq == 0):
                action = self.act(state)  # policy gradient
                obj_actor = self.cri(state, action).mean()
                self.optimizer_update(self.act_optimizer, -obj_actor)
                self.soft_update(self.act_target, self.act, self.soft_update_tau)
                obj_actors += obj_actor.item()

        self.act.action_std = self.act_target.action_std = self.explore_noise_std
        return obj_critics / update_times, obj_actors / (update_times / self.update_freq)

    def get_obj_critic(self, buffer, batch_size: int) -> Tuple[TEN, TEN]:
        with th.no_grad():
            state, action, reward, undone, next_state = buffer.sample(batch_size)
            next_action = self.act_target.get_action(next_state)  # stochastic policy
            next_q, _ = th.min(self.cri_target.get_q_values(next_state, next_action), dim=1, keepdim=True)
            q_label = reward + undone * self.gamma * next_q

        q_values = self.cri.get_q_values(state, action)
        q_labels = q_label.repeat(1, q_values.shape[1])
        obj_critic = self.criterion(q_values, q_labels)
        return obj_critic, state


class AgentSAC(AgentBase):
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.act_class = getattr(self, 'act_class', ActorSAC)  # get the attribute of object `self`
        self.cri_class = getattr(self, 'cri_class', CriticTwin)  # get the attribute of object `self`
        super().__init__(net_dims, state_dim, action_dim, gpu_id, args)
        self.cri_target = deepcopy(self.cri)

        self.alpha_log = th.tensor(-1, dtype=th.float32, requires_grad=True, device=self.device)  # trainable var
        self.alpha_optim = th.optim.Adam((self.alpha_log,), lr=args.learning_rate)
        self.target_entropy = -np.log(action_dim)

    def update_net(self, buffer: ReplayBuffer, if_update_act: bool = True) -> [float]:
        with th.no_grad():
            add_states = buffer.slice(buffer.states, buffer.add_size)
            add_actions = buffer.slice(buffer.actions, buffer.add_size)
            add_returns = self.cri_target(add_states, add_actions)
            self.update_avg_std_for_state_value_norm(states=add_states, returns=add_returns)
            del add_states, add_actions, add_returns

        obj_critics = obj_actors = 0.0
        update_times = int(buffer.cur_size * self.repeat_times / self.batch_size)
        for i in range(update_times):
            obj_critic, state = self.get_obj_critic(buffer, self.batch_size)
            self.optimizer_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
            obj_critics += obj_critic.item()

            if if_update_act:
                action, logprob = self.act.get_action_logprob(state)  # policy gradient
                obj_alpha = (self.alpha_log * (-logprob + self.target_entropy).detach()).mean()
                self.optimizer_update(self.alpha_optim, obj_alpha)

                alpha = self.alpha_log.exp().detach()
                obj_actor = (self.cri(state, action) - logprob * alpha).mean()
                self.optimizer_update(self.act_optimizer, -obj_actor)
                obj_actors += obj_actor.item()
        return obj_critics / update_times, obj_actors / update_times

    def get_obj_critic(self, buffer, batch_size: int) -> Tuple[TEN, TEN]:
        with th.no_grad():
            state, action, reward, undone, next_state = buffer.sample(batch_size)

            next_action, next_logprob = self.act.get_action_logprob(next_state)  # stochastic policy
            next_q, _ = th.min(self.cri_target.get_q_values(next_state, next_action), dim=1, keepdim=True)
            alpha = self.alpha_log.exp()
            q_label = reward + undone * self.gamma * (next_q - next_logprob * alpha)

        q_values = self.cri.get_q_values(state, action)
        q_labels = q_label.repeat(1, q_values.shape[1])
        obj_critic = self.criterion(q_values, q_labels)
        return obj_critic, state


class AgentReDQ(AgentSAC):  # Randomized Ensemble Double Q-Learning
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.act_class = getattr(self, 'act_class', ActorSAC)  # get the attribute of object `self`
        self.cri_class = getattr(self, 'cri_class', CriticEnsemble)  # get the attribute of object `self`
        super().__init__(net_dims, state_dim, action_dim, gpu_id, args)
        self.cri_target = deepcopy(self.cri)


class PendulumEnv(gymnasium.Wrapper):  # a demo of custom gym env
    def __init__(self):
        # gymnasium.logger.set_level(40)  # Block warning
        gym_env_name = 'Pendulum-v1'
        super().__init__(env=gymnasium.make(gym_env_name))

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
    gpu_id = args.gpu_id

    env = build_env(args.env_class, args.env_args)
    state, info_dict = env.reset()
    agent = args.agent_class(args.net_dims, args.state_dim, args.action_dim, gpu_id=gpu_id, args=args)
    agent.last_state = th.tensor(state, dtype=th.float32, device=agent.device)

    buffer = ReplayBuffer(gpu_id=gpu_id, max_size=args.buffer_size,
                          state_dim=args.state_dim, action_dim=1 if args.if_discrete else args.action_dim, )
    buffer_items = agent.explore_env(env, args.horizon_len * args.eval_times, if_random=True)
    buffer.update(buffer_items)  # warm up for ReplayBuffer

    evaluator = Evaluator(eval_env=build_env(args.env_class, args.env_args),
                          eval_per_step=args.eval_per_step, eval_times=args.eval_times, cwd=args.cwd)

    th.set_grad_enabled(True)
    agent.update_net(buffer=buffer, if_update_act=False)
    th.set_grad_enabled(False)
    while True:  # start training
        buffer_items = agent.explore_env(env, args.horizon_len)
        buffer.update(buffer_items)

        th.set_grad_enabled(True)
        logging_tuple = agent.update_net(buffer)
        th.set_grad_enabled(False)

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

    state, info_dict = env.reset()
    episode_steps = 0
    cumulative_returns = 0.0  # sum of rewards in an episode
    for episode_steps in range(12345):
        tensor_state = th.as_tensor(state, dtype=th.float32, device=device).unsqueeze(0)
        tensor_action = actor(tensor_state)
        action = tensor_action.detach().cpu().numpy()[0]  # not need detach(), because using th.no_grad() outside
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        cumulative_returns += reward

        if if_render:
            env.render()
        if done:
            break
    return cumulative_returns, episode_steps + 1


def train_sac_td3_for_pendulum():
    agent_class = [AgentSAC, AgentTD3][0]  # DRL algorithm name
    print(f"agent_class  {agent_class.__name__}")
    env_class = PendulumEnv  # run a custom env: PendulumEnv, which based on OpenAI pendulum
    env_args = {
        'env_name': 'Pendulum-v1',  # Apply torque on the free end to swing a pendulum into an upright position
        'state_dim': 3,  # the x-y coordinates of the pendulum's free end and its angular velocity.
        'action_dim': 1,  # the torque applied to free end of the pendulum
        'if_discrete': False  # continuous action space, symbols → direction, value → force
    }
    get_gym_env_args(env=PendulumEnv(), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `config.py Arguments()` for hyperparameter explanation
    args.gpu_id = GPU_ID
    args.break_step = int(2e5)  # break training if 'total_step > break_step'
    args.net_dims = (128, 128)  # the middle layer dimension of MultiLayer Perceptron
    args.gamma = 0.98  # discount factor of future rewards
    args.horizon_len = 128  # collect horizon_len step while exploring, then update network
    args.repeat_times = 1.0  # repeatedly update network using ReplayBuffer to keep critic's loss small
    args.state_value_tau = 0.05
    args.explore_noise_std = 0.05
    args.policy_noise_std = 0.10

    train_agent(args)
    """
    cumulative returns range: -2000 < -1000 < -200 < -80

    SAC
    |     step      time  |     avgR    stdR    avgS  |     objC      objA
    | 1.00e+04       135  |  -211.21   55.50     200  |     0.88    -69.34
    | 2.01e+04       479  |   -74.14   56.91     200  |     0.62    -22.68
    | 3.01e+04      1029  |   -69.16   36.39     200  |     0.36    -16.79

    TD3
    |     step      time  |     avgR    stdR    avgS  |     objC      objA
    | 1.00e+04       103  |  -771.30   38.15     200  |     1.03    -98.23
    | 2.01e+04       380  |   -89.88   62.76     200  |     0.73    -50.82
    | 3.01e+04       813  |   -91.69   42.66     200  |     0.45    -30.01
    """


def train_sac_td3_for_lunar_lander():
    agent_class = [AgentSAC, AgentTD3][0]  # DRL algorithm name
    env_class = gymnasium.make
    env_args = {
        'env_name': 'LunarLanderContinuous-v2',  # A lander learns to land on a landing pad
        'state_dim': 8,  # coordinates xy, linear velocities xy, angle, angular velocity, two booleans
        'action_dim': 2,  # fire main engine or side engine.
        'if_discrete': False  # continuous action space, symbols → direction, value → force
    }
    get_gym_env_args(env=gymnasium.make('LunarLanderContinuous-v2'), if_print=True)  # return env_args
    # pip install box2d box2d-kengz --user
    # https://stackoverflow.com/questions/50037674/attributeerror-module-box2d-has-no-attribute-rand-limit-swigconstant

    args = Config(agent_class, env_class, env_args)  # see `config.py Arguments()` for hyperparameter explanation
    args.break_step = int(8e4)  # break training if 'total_step > break_step'
    args.net_dims = (128, 128)  # the middle layer dimension of MultiLayer Perceptron
    args.horizon_len = 128  # collect horizon_len step while exploring, then update network
    args.repeat_times = 1.0  # repeatedly update network using ReplayBuffer to keep critic's loss small
    args.state_value_tau = 0.1  # do rolling normalization on state using soft update tau

    args.gpu_id = GPU_ID
    args.random_seed = GPU_ID
    train_agent(args)
    """   
    cumulative returns range: -1500 < -140 < 200 < 280

    SAC on CPU
    |     step      time  |     avgR    stdR    avgS  |     objC      objA
    | 1.01e+04        88  |    19.53  148.64     362  |     1.93     23.59
    | 2.02e+04       294  |   -60.15  120.83     805  |     2.59     60.84
    | 3.03e+04       617  |   -50.82   46.35     965  |     3.53    104.68
    | 4.04e+04      1051  |   -55.18   22.74     972  |     2.58     90.86
    | 5.06e+04      1560  |   172.70   84.48     664  |     2.06     66.80
    | 6.07e+04      2175  |   211.03   90.33     511  |     2.07     55.08
    """


if __name__ == '__main__':
    GPU_ID = int(sys.argv[1]) if len(sys.argv) > 1 else -1
    train_sac_td3_for_pendulum()
    # train_sac_td3_for_lunar_lander()
