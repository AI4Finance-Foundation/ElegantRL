import numpy as np
import torch as th
import torch.nn as nn
import torch.distributions.normal
from copy import deepcopy
from typing import List, Tuple, Optional


from erl_config import Config

ARY = np.ndarray
TEN = th.Tensor

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


class AgentDQN(AgentBase):
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        super().__init__(net_dims, state_dim, action_dim, gpu_id, args)

        self.act = self.cri = QNetwork(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.act_target = self.cri_target = deepcopy(self.act)
        self.act_optimizer = self.cri_optimizer = th.optim.Adam(self.act.parameters(), self.learning_rate)

        self.explore_rate = getattr(args, "explore_rate", 0.25)  # set for `self.act.get_action()`
        # the probability of choosing action randomly in epsilon-greedy

    def explore_action(self, state: TEN) -> TEN:
        return self.act.get_action(state.unsqueeze(0), explore_rate=self.explore_rate)[0, 0]

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


class AgentPPO(AgentBase):
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        super().__init__(net_dims, state_dim, action_dim, gpu_id, args)
        self.if_off_policy = False

        self.act = ActorPPO(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.cri = CriticPPO(net_dims=net_dims, state_dim=state_dim).to(self.device)
        self.act_optimizer = th.optim.Adam(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = th.optim.Adam(self.cri.parameters(), self.learning_rate)

        self.ratio_clip = getattr(args, "ratio_clip", 0.25)  # `ratio.clamp(1 - clip, 1 + clip)`
        self.lambda_gae_adv = getattr(args, "lambda_gae_adv", 0.95)  # could be 0.80~0.99
        self.lambda_entropy = getattr(args, "lambda_entropy", 0.01)  # could be 0.00~0.10
        self.lambda_entropy = th.tensor(self.lambda_entropy, dtype=th.float32, device=self.device)

    def explore_env(self, env, horizon_len: int, **kwargs) -> Tuple[TEN, TEN, TEN, TEN, TEN, TEN]:
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
            obj_critic, obj_actor = self.update_objectives(buffer, self.batch_size, update_t)
            obj_critics.append(obj_critic)
            obj_actors.append(obj_actor)
        th.set_grad_enabled(False)

        obj_critic_avg = np.array(obj_critics).mean() if len(obj_critics) else 0.0
        obj_actor_avg = np.array(obj_actors).mean() if len(obj_actors) else 0.0
        a_std_log = getattr(self.act, 'a_std_log', th.zeros(1)).mean()
        return obj_critic_avg, obj_actor_avg, a_std_log.item()

    def update_objectives(self, buffer: Tuple[TEN, ...], batch_size: int, _update_t: int) -> Tuple[float, float]:
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
        # self.act_target = deepcopy(self.act)  # TODO
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
