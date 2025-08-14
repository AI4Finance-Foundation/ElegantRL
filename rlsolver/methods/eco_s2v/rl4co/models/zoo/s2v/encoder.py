from typing import Tuple

import torch.nn as nn
from tensordict import TensorDict
from torch import Tensor

from rlsolver.methods.eco_s2v.rl4co.envs import RL4COEnvBase
from rlsolver.methods.eco_s2v.rl4co.models.common.constructive import AutoregressiveEncoder
from rlsolver.methods.eco_s2v.rl4co.models.nn.env_embeddings import env_init_embedding


class S2VModelEncoder(AutoregressiveEncoder):
    """Graph Attention Encoder as in Kool et al. (2019).
    First embed the input and then process it with a Graph Attention Network.

    Args:
        embed_dim: Dimension of the embedding space
        init_embedding: Module to use for the initialization of the embeddings
        env_name: Name of the environment used to initialize embeddings
        num_heads: Number of heads in the attention layers
        num_layers: Number of layers in the attention network
        normalization: Normalization type in the attention layers
        feedforward_hidden: Hidden dimension in the feedforward layers
        net: Graph Attention Network to use
        sdpa_fn: Function to use for the scaled dot product attention
        moe_kwargs: Keyword arguments for MoE
    """

    def __init__(
            self,
            init_embedding: nn.Module = None,
            env_name: str = "tsp",
            embed_dim: int = 128,
    ):
        super(S2VModelEncoder, self).__init__()

        if isinstance(env_name, RL4COEnvBase):
            env_name = env_name.name
        self.env_name = env_name

        self.init_embedding = (
            env_init_embedding(self.env_name, {"embed_dim": embed_dim})
            if init_embedding is None
            else init_embedding
        )

    def forward(
            self, td: TensorDict, mask: Tensor | None = None
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass of the encoder.
        Transform the input TensorDict into a latent representation.

        Args:
            td: Input TensorDict containing the environment state
            mask: Mask to apply to the attention

        Returns:
            h: Latent representation of the input
            init_h: Initial embedding of the input
        """
        # Transfer to embedding space
        init_h = self.init_embedding(td)

        # Process embedding
        h = init_h.clone()

        # Return latent representation and initial embedding
        return h, init_h
