from typing import Callable, Optional

import torch.nn as nn
from torch import Tensor

from rlsolver.methods.eco_s2v.rl4co.models.nn.attention import MultiHeadAttention
from rlsolver.methods.eco_s2v.rl4co.models.nn.mlp import MLP
from rlsolver.methods.eco_s2v.rl4co.models.nn.moe import MoE
from rlsolver.methods.eco_s2v.rl4co.models.nn.ops import Normalization, SkipConnection
from rlsolver.methods.eco_s2v.rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class MultiHeadAttentionLayer(nn.Sequential):
    """Multi-Head Attention Layer with normalization and feed-forward layer

    Args:
        embed_dim: dimension of the embeddings
        num_heads: number of heads in the MHA
        feedforward_hidden: dimension of the hidden layer in the feed-forward layer
        normalization: type of normalization to use (batch, layer, none)
        sdpa_fn: scaled dot product attention function (SDPA)
        moe_kwargs: Keyword arguments for MoE
    """

    def __init__(
            self,
            embed_dim: int,
            num_heads: int = 8,
            feedforward_hidden: int = 512,
            normalization: Optional[str] = "batch",
            bias: bool = True,
            sdpa_fn: Optional[Callable] = None,
            moe_kwargs: Optional[dict] = None,
    ):
        num_neurons = [feedforward_hidden] if feedforward_hidden > 0 else []
        if moe_kwargs is not None:
            ffn = MoE(embed_dim, embed_dim, num_neurons=num_neurons, **moe_kwargs)
        else:
            ffn = MLP(input_dim=embed_dim, output_dim=embed_dim, num_neurons=num_neurons, hidden_act="ReLU")

        super(MultiHeadAttentionLayer, self).__init__(
            SkipConnection(
                MultiHeadAttention(embed_dim, num_heads, bias=bias, sdpa_fn=sdpa_fn)
            ),
            Normalization(embed_dim, normalization),
            SkipConnection(ffn),
            Normalization(embed_dim, normalization),
        )


class GraphAttentionNetwork(nn.Module):
    """Graph Attention Network to encode embeddings with a series of MHA layers consisting of a MHA layer,
    normalization, feed-forward layer, and normalization. Similar to Transformer encoder, as used in Kool et al. (2019).

    Args:
        num_heads: number of heads in the MHA
        embed_dim: dimension of the embeddings
        num_layers: number of MHA layers
        normalization: type of normalization to use (batch, layer, none)
        feedforward_hidden: dimension of the hidden layer in the feed-forward layer
        sdpa_fn: scaled dot product attention function (SDPA)
        moe_kwargs: Keyword arguments for MoE
    """

    def __init__(
            self,
            num_heads: int,
            embed_dim: int,
            num_layers: int,
            normalization: str = "batch",
            feedforward_hidden: int = 512,
            sdpa_fn: Optional[Callable] = None,
            moe_kwargs: Optional[dict] = None,
    ):
        super(GraphAttentionNetwork, self).__init__()

        self.layers = nn.Sequential(
            *(
                MultiHeadAttentionLayer(
                    embed_dim,
                    num_heads,
                    feedforward_hidden=feedforward_hidden,
                    normalization=normalization,
                    sdpa_fn=sdpa_fn,
                    moe_kwargs=moe_kwargs,
                )
                for _ in range(num_layers)
            )
        )

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Forward pass of the encoder

        Args:
            x: [batch_size, graph_size, embed_dim] initial embeddings to process
            mask: [batch_size, graph_size, graph_size] mask for the input embeddings. Unused for now.
        """
        assert mask is None, "Mask not yet supported!"
        h = self.layers(x)
        return h
