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
