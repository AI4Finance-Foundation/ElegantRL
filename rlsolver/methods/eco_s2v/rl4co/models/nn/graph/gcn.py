from typing import Callable, Tuple

import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from torch import Tensor

try:
    from torch_geometric.nn import GCNConv
except ImportError:
    GCNConv = None
from rlsolver.methods.eco_s2v.rl4co.models.nn.env_embeddings import env_init_embedding
from rlsolver.methods.eco_s2v.rl4co.utils.ops import get_full_graph_edge_index
from rlsolver.methods.eco_s2v.rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

EdgeIndexFnSignature = Callable[[TensorDict, int], Tensor]


def edge_idx_fn_wrapper(td: TensorDict, num_nodes: int):
    # self-loop is added by GCNConv layer
    return get_full_graph_edge_index(num_nodes, self_loop=False).to(td.device)


class GCNEncoder(nn.Module):
    """Graph Convolutional Network to encode embeddings with a series of GCN
    layers from the pytorch geometric package

    Args:
        embed_dim: dimension of the embeddings
        num_nodes: number of nodes in the graph
        num_gcn_layer: number of GCN layers
        self_loop: whether to add self loop in the graph
        residual: whether to use residual connection
    """

    def __init__(
            self,
            env_name: str,
            embed_dim: int,
            num_layers: int,
            init_embedding: nn.Module = None,
            residual: bool = True,
            edge_idx_fn: EdgeIndexFnSignature = None,
            dropout: float = 0.5,
            bias: bool = True,
    ):
        super().__init__()

        self.env_name = env_name
        self.embed_dim = embed_dim
        self.residual = residual
        self.dropout = dropout

        self.init_embedding = (
            env_init_embedding(self.env_name, {"embed_dim": embed_dim})
            if init_embedding is None
            else init_embedding
        )

        if edge_idx_fn is None:
            log.warning("No edge indices passed. Assume a fully connected graph")
            edge_idx_fn = edge_idx_fn_wrapper

        self.edge_idx_fn = edge_idx_fn

        # Define the GCN layers
        self.gcn_layers = nn.ModuleList(
            [GCNConv(embed_dim, embed_dim, bias=bias) for _ in range(num_layers)]
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
        bs, num_nodes, emb_dim = init_h.shape
        # (bs*num_nodes, emb_dim)
        update_node_feature = init_h.reshape(-1, emb_dim)
        # shape=(2, num_edges)
        edge_index = self.edge_idx_fn(td, num_nodes)

        for layer in self.gcn_layers[:-1]:
            update_node_feature = layer(update_node_feature, edge_index)
            update_node_feature = F.relu(update_node_feature)
            update_node_feature = F.dropout(
                update_node_feature, training=self.training, p=self.dropout
            )

        # last layer without relu activation and dropout
        update_node_feature = self.gcn_layers[-1](update_node_feature, edge_index)

        # De-batch the graph
        update_node_feature = update_node_feature.view(bs, num_nodes, emb_dim)

        # Residual
        if self.residual:
            update_node_feature = update_node_feature + init_h

        return update_node_feature, init_h
