from typing import Optional

import torch
from tensordict.tensordict import TensorDict
from torchrl.data import Bounded

from rlsolver.methods.eco_s2v.rl4co.envs.common.base import RL4COEnvBase
from rlsolver.methods.eco_s2v.rl4co.utils.pylogger import get_pylogger
from .generator import MaxCutGenerator

log = get_pylogger(__name__)


class MaxCutEnv(RL4COEnvBase):
    name = "maxcut"

    def __init__(
            self,
            generator: MaxCutGenerator = None,
            generator_params: dict = {},
            **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator = MaxCutGenerator(**generator_params)
        self.generator = generator
        self._make_spec(self.generator)

    def _step(self, td: TensorDict) -> TensorDict:
        # action: [batch_size, 1]; the location to be chosen in each instance
        selected = td["action"]
        batch_size = selected.shape[0]

        # Update location selection status
        state = td["state"].clone()  # (batch_size, n_locations)

        state[torch.arange(batch_size).to(td.device), selected] = torch.logical_not(td["state"][torch.arange(batch_size).to(td.device), selected])
        td["state"] = state
        # We are done if we choose enough locations
        done = td["i"] >= (td["to_choose"] - 1)
        action_mask = state.clone()  # 已经选过的动作不允许再次选择

        # The reward is calculated outside via get_reward for efficiency, so we set it to zero here
        reward = torch.zeros_like(done)
        # sim_indices = torch.arange(batch_size,device=td.device)
        state_ = (state * 2 - 1).to(torch.float)
        # Update distances
        obj = ((1 / 4) * (torch.matmul(td['adj'], state_.unsqueeze(-1)).squeeze(-1) * -state_).sum(dim=-1) + (1 / 4)
               * torch.sum(td['adj'], dim=(-1, -2)))
        best_obj_ = td["best_obj"].clone()
        best_obj = torch.where(obj > best_obj_, obj, best_obj_)

        td.update(
            {"state": state,
             # states changed by actions
             "i": td["i"] + 1,  # the number of sets we have chosen
             "reward": reward,
             "done": done,
             "adj": td["adj"],
             "action_mask": action_mask,
             "best_obj": best_obj,
             }
        )
        return td

    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        self.to(td.device)
        state_ = torch.ones((*batch_size, self.generator.n_spins), dtype=torch.bool, device=td.device)
        best_obj = torch.zeros(batch_size, device=td.device)
        return TensorDict(
            {
                # given information
                "adj": td["adj"],  # (batch_size, n_points, dim_loc)
                "to_choose": td["to_choose"],  # 每个环境的交互次数
                "state": state_,
                "i": torch.zeros(
                    *batch_size, dtype=torch.int64, device=td.device
                ),
                "action_mask": torch.ones_like(td["state"], dtype=torch.bool),
                "best_obj": best_obj,
            },
            batch_size=batch_size,
        )

    def _make_spec(self, generator: MaxCutGenerator):
        self.action_spec = Bounded(
            shape=(1),
            dtype=torch.int64,
            low=0,
            high=generator.n_spins,
        )

    def _get_reward(self, td: TensorDict, actions) -> torch.Tensor:
        obj = td['best_obj'] / td['adj'].shape[-1]
        return obj

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor) -> None:
        # TODO: check solution validity
        pass

    @staticmethod
    def local_search(td: TensorDict, actions: torch.Tensor, **kwargs) -> torch.Tensor:
        # TODO: local search
        pass

    @staticmethod
    def get_num_starts(td):
        return td["action_mask"].shape[-1]
