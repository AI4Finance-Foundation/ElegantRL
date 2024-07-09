from copy import deepcopy
from typing import List, Optional, Tuple

import numpy as np
import torch as th

from erl_config import Config
from erl_net import QNetwork  # DQN
from erl_net import Actor, Critic  # DDPG
from erl_net import ActorPPO, CriticPPO  # PPO

ARY = np.ndarray
TEN = th.Tensor


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

    def update(self, items: [TEN]):
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

    def sample(self, batch_size: int) -> [TEN]:
        ids = th.randint(self.cur_size - 1, size=(batch_size,), requires_grad=False)
        return self.states[ids], self.actions[ids], self.rewards[ids], self.undones[ids], self.states[ids + 1]


class AgentBase:
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.if_off_policy: bool = False
        self.net_dims: List[int] = net_dims
        self.state_dim: int = state_dim
        self.action_dim: int = action_dim

        self.gamma: float = args.gamma
        self.batch_size: int = args.batch_size
        self.horizon_len: int = args.horizon_len
        self.repeat_times: float = args.repeat_times
        self.reward_scale: float = args.reward_scale
        self.learning_rate: float = args.learning_rate
        self.soft_update_tau: float = args.soft_update_tau
        self.state_update_tau: float = args.state_update_tau

        self.explore_noise_std = getattr(args, 'explore_noise_std', 0.05)  # standard deviation of exploration noise

        self.last_state: Optional[ARY] = None  # state of the trajectory for training. `shape == (state_dim)`
        self.device = th.device(f"cuda:{gpu_id}" if (th.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        self.act = None
        self.cri = None
        self.act_target = self.act
        self.cri_target = self.cri

        self.act_optimizer: Optional[th.optim] = None
        self.cri_optimizer: Optional[th.optim] = None
        self.criterion = th.nn.SmoothL1Loss()

    def get_random_action(self) -> TEN:
        return th.rand(self.action_dim) * 2 - 1.0

    def get_policy_action(self, state: TEN) -> TEN:
        return self.act.get_action(state.unsqueeze(0), action_std=self.explore_noise_std)[0]

    def explore_env(self, env, horizon_len: int, if_random: bool = False) -> Tuple[TEN, TEN, TEN, TEN, TEN]:
        self.horizon_len = horizon_len  # update horizon_len for update_net()

        states = th.zeros((horizon_len, self.state_dim), dtype=th.float32).to(self.device)
        actions = th.zeros((horizon_len, self.action_dim), dtype=th.float32).to(self.device)
        rewards = th.zeros(horizon_len, dtype=th.float32).to(self.device)
        terminals = th.zeros(horizon_len, dtype=th.bool).to(self.device)
        truncates = th.zeros(horizon_len, dtype=th.bool).to(self.device)

        ary_state = self.last_state
        for i in range(horizon_len):
            state = th.as_tensor(ary_state, dtype=th.float32, device=self.device)
            action = self.get_random_action() if if_random \
                else self.get_policy_action(state)

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

    def update_critic_net(self, buffer: ReplayBuffer, batch_size: int) -> Tuple[TEN, TEN]:
        with th.no_grad():
            state, action, reward, undone, unmask, next_state = buffer.sample(batch_size)

            next_action = self.act(next_state)  # deterministic policy
            next_q = self.cri_target(next_state, next_action)

            q_label = reward + undone * self.gamma * next_q

        q_value = self.cri(state, action) * unmask
        obj_critic = self.criterion(q_value, q_label)
        return obj_critic, state

    def update_actor_net(self, state: TEN, update_t: int, if_skip: bool = False) -> Optional[TEN]:
        if if_skip:
            return None

        action_pg = self.act(state)  # action to policy gradient
        obj_actor = self.cri(state, action_pg).mean()
        return obj_actor

    def update_net(self, buffer, if_skip_actor: bool = False) -> Tuple[float, float]:
        states = buffer.states[-self.horizon_len:]
        state_update_tau = 1.0 if if_skip_actor else self.state_update_tau
        self.update_avg_std_for_state_norm(states=states, tau=state_update_tau)

        obj_critics = []
        obj_actors = []

        th.set_grad_enabled(True)
        update_times = int(buffer.cur_size * self.repeat_times / self.batch_size)
        for update_t in range(update_times):
            obj_critic, state = self.update_critic_net(buffer, self.batch_size)
            self.optimizer_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
            obj_critics.append(obj_critic.item())

            obj_actor = self.update_actor_net(state, update_t, if_skip_actor)
            if isinstance(obj_actor, TEN):
                self.optimizer_update(self.act_optimizer, -obj_actor)
                self.soft_update(self.act_target, self.act, self.soft_update_tau)
                obj_actors.append(obj_actor.item())
        th.set_grad_enabled(False)

        obj_critic_avg = np.array(obj_critics).mean() if len(obj_critics) else 0.0
        obj_actor_avg = np.array(obj_actors).mean() if len(obj_actors) else 0.0
        return obj_critic_avg, obj_actor_avg

    @staticmethod
    def optimizer_update(optimizer, objective: TEN):
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    @staticmethod
    def soft_update(target_net: th.nn.Module, current_net: th.nn.Module, tau: float):
        if target_net is current_net:
            return
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

    def update_avg_std_for_state_norm(self, states: TEN, tau: float):
        if tau == 0 or self.state_update_tau == 0:
            return
        state_avg = states.mean(dim=0, keepdim=True)
        state_std = states.std(dim=0, keepdim=True)
        self.act.state_avg[:] = self.act.state_avg * (1 - tau) + state_avg * tau
        self.act.state_std[:] = (self.cri.state_std * (1 - tau) + state_std * tau).clamp_min(1e-4)
        self.cri.state_avg[:] = self.act.state_avg
        self.cri.state_std[:] = self.cri.state_std


class AgentDQN(AgentBase):
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        super().__init__(net_dims, state_dim, action_dim, gpu_id, args)

        self.act = self.cri = QNetwork(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.act_target = self.cri_target = deepcopy(self.act)
        self.act_optimizer = self.cri_optimizer = th.optim.Adam(self.act.parameters(), self.learning_rate)

        self.explore_rate = getattr(args, "explore_rate", 0.25)  # set for `self.act.get_action()`
        # the probability of choosing action randomly in epsilon-greedy

    def get_random_action(self) -> TEN:
        return th.randint(self.action_dim, size=(1,))[0]

    def get_policy_action(self, state: TEN) -> TEN:
        return self.act.get_action(state.unsqueeze(0), explore_rate=self.explore_rate)[0]

    def update_critic_net(self, buffer: ReplayBuffer, batch_size: int) -> Tuple[TEN, TEN]:
        with th.no_grad():
            state, action, reward, undone, unmask, next_state = buffer.sample(batch_size)

            next_q = self.cri_target(next_state).max(dim=1, keepdim=True)[0]
            q_label = reward + undone * self.gamma * next_q

        q_value = self.cri(state).gather(1, action.long())
        obj_critic = self.criterion(q_value, q_label)
        return obj_critic, state

    def update_actor_net(self, state: TEN, update_t: int, if_skip: bool = False) -> Optional[TEN]:
        if if_skip:
            return None


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

    def get_policy_action(self, state: TEN) -> TEN:
        return self.act.get_action(state.unsqueeze(0), action_std=self.explore_noise_std)[0]


class AgentPPO(AgentBase):
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        super().__init__(net_dims, state_dim, action_dim, gpu_id, args)
        self.if_off_policy = False

        self.act = ActorPPO(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.cri = CriticPPO(net_dims=net_dims, state_dim=state_dim).to(self.device)
        self.act_optimizer = th.optim.Adam(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = th.optim.Adam(self.cri.parameters(), self.learning_rate)

        self.act_class = getattr(self, "act_class", ActorPPO)
        self.cri_class = getattr(self, "cri_class", CriticPPO)

        self.ratio_clip = getattr(args, "ratio_clip", 0.25)  # `ratio.clamp(1 - clip, 1 + clip)`
        self.lambda_gae_adv = getattr(args, "lambda_gae_adv", 0.95)  # could be 0.80~0.99
        self.lambda_entropy = getattr(args, "lambda_entropy", 0.01)  # could be 0.00~0.10
        self.lambda_entropy = th.tensor(self.lambda_entropy, dtype=th.float32, device=self.device)

    def get_policy_action(self, state: TEN) -> Tuple[TEN, TEN]:
        actions, logprobs = self.act.get_action(state.unsqueeze(0))
        return actions[0], logprobs[0]

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
            action, logprob = self.get_policy_action(state)

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

    def update_net(self, buffer: List[TEN, TEN, TEN, TEN, TEN, TEN], **kwargs) -> Tuple[float, float, float]:
        states, actions, logprobs, rewards, undones, unmasks = buffer
        buffer_size = states.shape[0]

        '''get advantages reward_sums'''
        bs = 2 ** 10  # set a smaller 'batch_size' when out of GPU memory.
        values = [self.cri(states[i:i + bs]) for i in range(0, buffer_size, bs)]
        values = th.cat(values, dim=0).squeeze(1)  # values.shape == (buffer_size, )

        advantages = self.get_advantages(states, rewards, undones, unmasks, values)  # shape == (buffer_size, )
        reward_sums = advantages + values  # reward_sums.shape == (buffer_size, )
        del rewards, undones, values

        advantages = (advantages - advantages.mean()) / (advantages.std(dim=0) + 1e-5)
        assert logprobs.shape == advantages.shape == reward_sums.shape == (buffer_size,)

        '''update network'''
        obj_critics = []
        obj_actors = []

        th.set_grad_enabled(True)
        update_times = int(buffer_size * self.repeat_times / self.batch_size)
        assert update_times >= 1
        for update_t in range(update_times):
            indices = th.randint(buffer_size, size=(self.batch_size,), requires_grad=False)
            state = states[indices]
            action = actions[indices]
            logprob = logprobs[indices]
            advantage = advantages[indices]
            reward_sum = reward_sums[indices]

            obj_critic = self.update_critic_net_with_advantage(state, reward_sum)
            self.optimizer_update(self.cri_optimizer, obj_critic)
            obj_critics.append(obj_critic.item())

            obj_actor = self.update_actor_net_with_advantage(state, action, logprob, advantage)
            self.optimizer_update(self.act_optimizer, -obj_actor)
            obj_actors.append(obj_actor.item())
        th.set_grad_enabled(False)

        obj_critic_avg = np.array(obj_critics).mean() if len(obj_critics) else 0.0
        obj_actor_avg = np.array(obj_actors).mean() if len(obj_actors) else 0.0
        a_std_log = getattr(self.act, 'a_std_log', th.zeros(1)).mean()
        return obj_critic_avg, obj_actor_avg, a_std_log.item()

    def update_critic_net_with_advantage(self, state: TEN, reward_sum: TEN) -> TEN:
        value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
        obj_critic = self.criterion(value, reward_sum)
        return obj_critic

    def update_actor_net_with_advantage(self, state: TEN, action: TEN, logprob: TEN, advantage: TEN) -> TEN:
        new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)
        ratio = (new_logprob - logprob.detach()).exp()
        surrogate1 = advantage * ratio
        surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
        obj_surrogate = th.min(surrogate1, surrogate2).mean()

        obj_actor = obj_surrogate + obj_entropy.mean() * self.lambda_entropy
        return obj_actor

    def get_advantages(self, states: TEN, rewards: TEN, undones: TEN, unmasks: TEN, values: TEN) -> TEN:
        advantages = th.empty_like(values)  # advantage value

        # update undones rewards when truncated
        truncated = th.logical_not(unmasks)
        if th.any(truncated):
            rewards[truncated] += self.cri(states[truncated]).detach().squeeze(1)
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
