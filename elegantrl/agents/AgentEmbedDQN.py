import torch as th
from torch import nn
from copy import deepcopy
from typing import Tuple, List

from .AgentBase import AgentBase
from .AgentBase import build_mlp, layer_init_with_orthogonal
from ..train import Config
from ..train import ReplayBuffer

TEN = th.Tensor


class AgentEmbedDQN(AgentBase):
    """Deep Q-Network algorithm. 
    “Human-Level Control Through Deep Reinforcement Learning”. 2015.
    
    DQN1 original:
    q_values = q_network(state)
    
    DQN2 modify by ElegantRL:
    q_values = q_critic(state, action)
    """

    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        super().__init__(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim, gpu_id=gpu_id, args=args)

        self.act = QEmbedTwin(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim).to(self.device)
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

        q_values = self.cri.get_q_values(state, action_int=action.long())
        q_labels = q_label.view((-1, 1)).repeat(1, q_values.shape[1])
        td_error = self.criterion(q_values, q_labels).mean(dim=1) * unmask
        if self.if_use_per:
            obj_critic = (td_error * is_weight).mean()
            buffer.td_error_update_for_per(is_index.detach(), td_error.detach())
        else:
            obj_critic = td_error.mean()
        if self.lambda_fit_cum_r != 0:
            cum_reward_mean = buffer.cum_rewards[buffer.ids0, buffer.ids1].detach_().mean().repeat(q_values.shape[1])
            obj_critic += self.criterion(cum_reward_mean, q_values.mean(dim=0)).mean() * self.lambda_fit_cum_r
        self.optimizer_backward(self.cri_optimizer, obj_critic)
        self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

        obj_actor = q_values.detach().mean()
        return obj_critic.item(), obj_actor.item()

    def get_cumulative_rewards(self, rewards: TEN, undones: TEN) -> TEN:
        returns = th.empty_like(rewards)

        masks = undones * self.gamma
        horizon_len = rewards.shape[0]

        last_state = self.last_state
        next_value = self.act_target.get_q_value(last_state).max(dim=1)[0].detach()  # next q_values
        for t in range(horizon_len - 1, -1, -1):
            returns[t] = next_value = rewards[t] + masks[t] * next_value
        return returns


class AgentEnsembleDQN(AgentEmbedDQN):
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        AgentBase.__init__(self, net_dims, state_dim, action_dim, gpu_id=gpu_id, args=args)

        self.act = QEmbedEnsemble(net_dims=net_dims, state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.act_target = deepcopy(self.act)
        self.act_optimizer = th.optim.Adam(self.act.parameters(), self.learning_rate)

        self.cri = self.act
        self.cri_target = self.act_target
        self.cri_optimizer = self.act_optimizer

        self.explore_rate = getattr(args, "explore_rate", 0.25)  # set for `self.act.get_action()`
        # the probability of choosing action randomly in epsilon-greedy


'''network'''


class QEmbedBase(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = None  # build_mlp(net_dims=[state_dim + action_dim, *net_dims, 1])

        self.embedding_dim = max(8, int(action_dim ** 0.5))
        self.action_emb = nn.Embedding(num_embeddings=action_dim, embedding_dim=self.embedding_dim)
        th.nn.init.orthogonal_(self.action_emb.weight, gain=0.5)

    def forward(self, state: TEN) -> TEN:
        all_q_values = self.get_all_q_values(state=state)  # (batch, action_dim, num_ensembles)
        all_q_value = all_q_values.mean(dim=2)  # (batch, action_dim)
        return all_q_value.argmax(dim=1)  # index of max Q values

    def get_q_value(self, state: TEN) -> TEN:
        all_q_values = self.get_all_q_values(state=state)  # (batch, action_dim, num_ensembles)
        all_q_value = all_q_values.mean(dim=2)  # (batch, action_dim)
        return all_q_value  # Q values

    def get_q_values(self, state: TEN, action_int: TEN) -> TEN:
        action = self.action_emb(action_int)  # Long: (batch, ) -> Float: (batch, embedding_dim)
        state_action = th.concat((state, action), dim=1)  # (batch, action_dim, state_dim+embedding)
        q_values = self.net(state_action)  # (batch, num_ensembles)
        return q_values

    def get_action(self, state: TEN, explore_rate: float):  # return the index List[int] of discrete action
        if explore_rate < th.rand(1):
            action = self.get_q_value(state).argmax(dim=1, keepdim=True)
        else:
            action = th.randint(self.action_dim, size=(state.shape[0], 1))
        return action

    def get_all_q_values(self, state: TEN) -> TEN:
        batch_size = state.shape[0]
        device = state.device

        action_int = th.arange(self.action_dim, device=device)  # (action_dim, )
        all_action_int = action_int.unsqueeze(0).repeat((batch_size, 1))  # (batch_size, action_dim)
        all_action = self.action_emb(all_action_int)  # (batch_size, action_dim, embedding_dim)

        all_state = state.unsqueeze(1).repeat((1, self.action_dim, 1))  # (batch, action_dim, state_dim)
        all_state_action = th.concat((all_state, all_action), dim=2)  # (batch, action_dim, state_dim+embedding)
        all_q_values = self.net(all_state_action)  # (batch, action_dim, num_ensembles)
        return all_q_values


class QEmbedTwin(QEmbedBase):  # shared parameter
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int, num_ensembles: int = 8):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.net = build_mlp(dims=[state_dim + self.embedding_dim, *net_dims, num_ensembles])
        layer_init_with_orthogonal(self.net[-1], std=0.5)


class QEmbedEnsemble(QEmbedBase):  # ensemble networks
    def __init__(self, net_dims: List[int], state_dim: int, action_dim: int, num_ensembles: int = 4):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.encoder_sa = build_mlp(dims=[state_dim + self.embedding_dim, net_dims[0]])  # encoder of state and action
        self.decoder_qs = []
        for net_i in range(num_ensembles):
            decoder_q = build_mlp(dims=[*net_dims, 1])
            layer_init_with_orthogonal(decoder_q[-1], std=0.5)

            self.decoder_qs.append(decoder_q)
            setattr(self, f"decoder_q{net_i:02}", decoder_q)

    def get_q_values(self, state: TEN, action_int: TEN) -> TEN:
        action = self.action_emb(action_int)  # Long: (batch, ) -> Float: (batch, embedding_dim)
        state_action = th.concat((state, action), dim=1)  # (batch, action_dim, state_dim+embedding)

        tensor_sa = self.encoder_sa(state_action)
        q_values = th.concat([decoder_q(tensor_sa) for decoder_q in self.decoder_qs], dim=-1)
        return q_values  # (batch, num_ensembles)

    def get_all_q_values(self, state: TEN) -> TEN:
        batch_size = state.shape[0]
        device = state.device

        action_int = th.arange(self.action_dim, device=device)  # (action_dim, )
        all_action_int = action_int.unsqueeze(0).repeat((batch_size, 1))  # (batch_size, action_dim)
        all_action = self.action_emb(all_action_int)  # (batch_size, action_dim, embedding_dim)

        all_state = state.unsqueeze(1).repeat((1, self.action_dim, 1))  # (batch, action_dim, state_dim)
        all_state_action = th.concat((all_state, all_action), dim=2)  # (batch, action_dim, state_dim+embedding)

        all_tensor_sa = self.encoder_sa(all_state_action)
        all_q_values = th.concat([decoder_q(all_tensor_sa) for decoder_q in self.decoder_qs], dim=-1)
        return all_q_values  # (batch, action_dim, num_ensembles)


"""
QR-DQN: Distributional Reinforcement Learning with Quantile Regression
IQN: Implicit Quantile Networks for Distributional Reinforcement Learning
"""
