import torch as th
from torch import nn
from copy import deepcopy
from typing import Tuple, List

from .AgentBase import AgentBase
from .AgentBase import build_mlp, layer_init_with_orthogonal
from ..train import Config
from ..train import ReplayBuffer

TEN = th.Tensor


class AgentDQN(AgentBase):
    """Deep Q-Network algorithm.
    “Human-Level Control Through Deep Reinforcement Learning”. 2015.
    """

    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        super().__init__(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, gpu_id=gpu_id, args=args)

        self.act = QNetwork(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.act_target = deepcopy(self.act)
        self.act_optimizer = th.optim.Adam(self.act.parameters(), self.learning_rate)

        self.cri = self.act
        self.cri_target = self.act_target
        self.cri_optimizer = self.act_optimizer

        self.explore_rate = getattr(args, "explore_rate", 0.25)  # set for `self.act.get_action()`
        # the probability of choosing action randomly in epsilon-greedy

    def explore_action(self, state: TEN) -> TEN:
        return self.act.get_action(state, explore_rate=self.explore_rate)[:, 0]

    def update_objectives(self, buffer: ReplayBuffer, update_t: int) -> Tuple[float, float]:
        assert isinstance(update_t, int)
        with th.no_grad():
            if self.if_use_per:
                (state, action, reward, undone, unmask, next_state,
                 is_weight, is_index) = buffer.sample_for_per(self.batch_size)
            else:
                state, action, reward, undone, unmask, next_state = buffer.sample(self.batch_size)
                is_weight, is_index = None, None

            next_q = self.cri_target.get_q_value(next_state).max(dim=1)[0]  # next q_values
            q_label = reward + undone * self.gamma * next_q

        q_value = self.cri.get_q_value(state).squeeze(-1).gather(dim=1, index=action.long())
        td_error = self.criterion(q_value, q_label) * unmask
        if self.if_use_per:
            obj_critic = (td_error * is_weight).mean()
            buffer.td_error_update_for_per(is_index.detach(), td_error.detach())
        else:
            obj_critic = td_error.mean()
        if self.lambda_fit_cum_r != 0:
            cum_reward_mean = buffer.cum_rewards[buffer.ids0, buffer.ids1].detach_().mean()
            obj_critic += self.criterion(cum_reward_mean, q_value.mean()) * self.lambda_fit_cum_r
        self.optimizer_backward(self.cri_optimizer, obj_critic)
        self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

        obj_actor = q_value.detach().mean()
        return obj_critic.item(), obj_actor.item()

    def get_cumulative_rewards(self, rewards: TEN, undones: TEN) -> TEN:
        returns = th.empty_like(rewards)

        masks = undones * self.gamma
        horizon_len = rewards.shape[0]

        last_state = self.last_state
        next_value = self.act_target(last_state).argmax(dim=1).detach()  # actor is Q Network in DQN style
        for t in range(horizon_len - 1, -1, -1):
            returns[t] = next_value = rewards[t] + masks[t] * next_value
        return returns


class AgentDoubleDQN(AgentDQN):
    """
    Double Deep Q-Network algorithm. “Deep Reinforcement Learning with Double Q-learning”. 2015.
    """

    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        AgentBase.__init__(self, net_dims, state_dim, action_dim, gpu_id=gpu_id, args=args)

        self.act = QNetTwin(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.act_target = deepcopy(self.act)
        self.act_optimizer = th.optim.Adam(self.act.parameters(), self.learning_rate)

        self.cri = self.act
        self.cri_target = self.act_target
        self.cri_optimizer = self.act_optimizer

        self.explore_rate = getattr(args, "explore_rate", 0.25)  # set for `self.act.get_action()`
        # the probability of choosing action randomly in epsilon-greedy

    def update_objectives(self, buffer: ReplayBuffer, update_t: int) -> Tuple[float, float]:
        assert isinstance(update_t, int)
        with th.no_grad():
            if self.if_use_per:
                (state, action, reward, undone, unmask, next_state,
                 is_weight, is_index) = buffer.sample_for_per(self.batch_size)
            else:
                state, action, reward, undone, unmask, next_state = buffer.sample(self.batch_size)
                is_weight, is_index = None, None

            next_q = th.min(*self.cri_target.get_q1_q2(next_state)).max(dim=1)[0]
            q_label = reward + undone * self.gamma * next_q

        q_value1, q_value2 = [qs.squeeze(1).gather(dim=1, index=action.long()) for qs in self.cri.get_q1_q2(state)]
        td_error = (self.criterion(q_value1, q_label) + self.criterion(q_value2, q_label)) * unmask
        if self.if_use_per:
            obj_critic = (td_error * is_weight).mean()
            buffer.td_error_update_for_per(is_index.detach(), td_error.detach())
        else:
            obj_critic = td_error.mean()
        if self.lambda_fit_cum_r != 0:
            cum_reward_mean = buffer.cum_rewards[buffer.ids0, buffer.ids1].detach_().mean()
            obj_critic += (self.criterion(cum_reward_mean, q_value1.mean()) +
                           self.criterion(cum_reward_mean, q_value2.mean()))
        self.optimizer_backward(self.cri_optimizer, obj_critic)
        self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

        obj_actor = q_value1.detach().mean()
        return obj_critic.item(), obj_actor.item()


'''add dueling q network'''


class AgentDuelingDQN(AgentDQN):
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        AgentBase.__init__(self, net_dims, state_dim, action_dim, gpu_id=gpu_id, args=args)

        self.act = QNetDuel(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.act_target = deepcopy(self.act)
        self.act_optimizer = th.optim.Adam(self.act.parameters(), self.learning_rate)

        self.cri = self.act
        self.cri_target = self.act_target
        self.cri_optimizer = self.act_optimizer

        self.explore_rate = getattr(args, "explore_rate", 0.25)  # set for `self.act.get_action()`
        # the probability of choosing action randomly in epsilon-greedy


class AgentD3QN(AgentDoubleDQN):  # Dueling Double Deep Q Network. (D3QN)
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        AgentBase.__init__(self, net_dims, state_dim, action_dim, gpu_id=gpu_id, args=args)

        self.act = QNetTwinDuel(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.act_target = deepcopy(self.act)
        self.act_optimizer = th.optim.Adam(self.act.parameters(), self.learning_rate)

        self.cri = self.act
        self.cri_target = self.act_target
        self.cri_optimizer = self.act_optimizer

        self.explore_rate = getattr(args, "explore_rate", 0.25)  # set for `self.act.get_action()`
        # the probability of choosing action randomly in epsilon-greedy


'''network'''


class QNetBase(nn.Module):  # nn.Module is a standard PyTorch Network
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = None  # build_mlp(net_dims=[state_dim + action_dim, *net_dims, 1])
        self.state_dim = state_dim
        self.action_dim = action_dim

    def forward(self, state):
        q_value = self.get_q_value(state)
        return q_value.argmax(dim=1)  # index of max Q values

    def get_q_value(self, state: TEN) -> TEN:
        q_value = self.net(state)
        return q_value

    def get_action(self, state: TEN, explore_rate: float):  # return the index List[int] of discrete action
        if explore_rate < th.rand(1):
            action = self.get_q_value(state).argmax(dim=1, keepdim=True)
        else:
            action = th.randint(self.action_dim, size=(state.shape[0], 1))
        return action


class QNetwork(QNetBase):
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.net = build_mlp(dims=[state_dim, *net_dims, action_dim])
        layer_init_with_orthogonal(self.net[-1], std=0.1)


class QNetDuel(QNetBase):  # Dueling DQN
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.net_state = build_mlp(dims=[state_dim, *net_dims])
        self.net_adv = build_mlp(dims=[net_dims[-1], 1])  # advantage value
        self.net_val = build_mlp(dims=[net_dims[-1], action_dim])  # Q value

        layer_init_with_orthogonal(self.net_adv[-1], std=0.1)
        layer_init_with_orthogonal(self.net_val[-1], std=0.1)

    def forward(self, state):
        s_enc = self.net_state(state)  # encoded state
        q_val = self.net_val(s_enc)  # q value
        q_adv = self.net_adv(s_enc)  # advantage value
        value = q_val - q_val.mean(dim=1, keepdim=True) + q_adv  # dueling Q value
        return value.argmax(dim=1)  # index of max Q values

    def get_q_value(self, state: TEN) -> TEN:
        s_enc = self.net_state(state)  # encoded state
        q_value = self.net_val(s_enc)
        return q_value


class QNetTwin(QNetBase):  # Double DQN
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.net_state = build_mlp(dims=[state_dim, *net_dims])
        self.net_val1 = build_mlp(dims=[net_dims[-1], action_dim])  # Q value 1
        self.net_val2 = build_mlp(dims=[net_dims[-1], action_dim])  # Q value 2
        self.soft_max = nn.Softmax(dim=-1)

        layer_init_with_orthogonal(self.net_val1[-1], std=0.1)
        layer_init_with_orthogonal(self.net_val2[-1], std=0.1)

    def get_q_value(self, state: TEN) -> TEN:
        s_enc = self.net_state(state)  # encoded state
        q_value = self.net_val1(s_enc)  # q value
        return q_value

    def get_q1_q2(self, state):
        s_enc = self.net_state(state)  # encoded state
        q_val1 = self.net_val1(s_enc)  # q value 1
        q_val2 = self.net_val2(s_enc)  # q value 2
        return q_val1, q_val2  # two groups of Q values


class QNetTwinDuel(QNetTwin):  # D3QN: Dueling Double DQN
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int):
        QNetBase.__init__(self, state_dim=state_dim, action_dim=action_dim)
        self.net_state = build_mlp(dims=[state_dim, *net_dims])
        self.net_adv1 = build_mlp(dims=[net_dims[-1], 1])  # advantage value 1
        self.net_val1 = build_mlp(dims=[net_dims[-1], action_dim])  # Q value 1
        self.net_adv2 = build_mlp(dims=[net_dims[-1], 1])  # advantage value 2
        self.net_val2 = build_mlp(dims=[net_dims[-1], action_dim])  # Q value 2
        self.soft_max = nn.Softmax(dim=1)

        layer_init_with_orthogonal(self.net_adv1[-1], std=0.1)
        layer_init_with_orthogonal(self.net_val1[-1], std=0.1)
        layer_init_with_orthogonal(self.net_adv2[-1], std=0.1)
        layer_init_with_orthogonal(self.net_val2[-1], std=0.1)

    def get_q_value(self, state):
        s_enc = self.net_state(state)  # encoded state
        q_val = self.net_val1(s_enc)  # q value
        q_adv = self.net_adv1(s_enc)  # advantage value
        q_value = q_val - q_val.mean(dim=1, keepdim=True) + q_adv  # one dueling Q value
        return q_value

    def get_q1_q2(self, state):
        s_enc = self.net_state(state)  # encoded state

        q_val1 = self.net_val1(s_enc)  # q value 1
        q_adv1 = self.net_adv1(s_enc)  # advantage value 1
        q_duel1 = q_val1 - q_val1.mean(dim=1, keepdim=True) + q_adv1

        q_val2 = self.net_val2(s_enc)  # q value 2
        q_adv2 = self.net_adv2(s_enc)  # advantage value 2
        q_duel2 = q_val2 - q_val2.mean(dim=1, keepdim=True) + q_adv2
        return q_duel1, q_duel2  # two dueling Q values
