from typing import Tuple

import torch.nn as nn
from tensordict import TensorDict
from torch import Tensor

from rlsolver.methods.eco_s2v.rl4co.envs import RL4COEnvBase
from rlsolver.methods.eco_s2v.rl4co.models.common.constructive.autoregressive.decoder import AutoregressiveDecoder
from rlsolver.methods.eco_s2v.rl4co.models.nn.env_embeddings import env_context_embedding
from rlsolver.methods.eco_s2v.rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class S2VModelDecoder(AutoregressiveDecoder):

    def __init__(
            self,
            env_name: str = "tsp",
            context_embedding: nn.Module = None,
            embed_dim: int = 128,
    ):
        super().__init__()
        if isinstance(env_name, RL4COEnvBase):
            env_name = env_name.name
        self.env_name = env_name
        self.context_embedding = (
            env_context_embedding(self.env_name, {"embed_dim": embed_dim})
            if context_embedding is None
            else context_embedding
        )
        self.embed_dim = embed_dim

    def forward(
            self,
            td: TensorDict,
            cached: None,
            num_starts: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        logits = self.context_embedding(self.embed_dim, td)
        # Compute logits
        mask = td["action_mask"]

        return logits, mask

    def pre_decoder_hook(
            self, td, env, embeddings, num_starts: int = 0
    ) -> Tuple[TensorDict, RL4COEnvBase, int]:
        """Precompute the embeddings cache before the decoder is called"""
        return td, env, 0
