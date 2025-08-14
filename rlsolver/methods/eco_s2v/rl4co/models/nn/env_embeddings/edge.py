from typing import Callable, Union

import torch
import torch.nn as nn
from torch import Tensor

try:
    from torch_geometric.data import Batch, Data
except ImportError:
    Batch = Data = None

from rlsolver.methods.eco_s2v.rl4co.utils.ops import get_distance_matrix, get_full_graph_edge_index, sparsify_graph
from rlsolver.methods.eco_s2v.rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def env_edge_embedding(env_name: str, config: dict) -> nn.Module:
    """Retrieve the edge embedding module specific to the environment. Edge embeddings are crucial for
    transforming the raw edge features into a format suitable for the neural network, especially in
    graph neural networks where edge features can significantly impact the model's performance.

    Args:
        env: Environment or its name.
        config: A dictionary of configuration options for the environment.
    """
    embedding_registry = {
        "tsp": TSPEdgeEmbedding,
        "atsp": ATSPEdgeEmbedding,
        "cvrp": CVRPEdgeEmbedding,
        "cvrpmvc": CVRPEdgeEmbedding,
        "sdvrp": TSPEdgeEmbedding,
        "pctsp": CVRPEdgeEmbedding,
        "spctsp": TSPEdgeEmbedding,
        "op": CVRPEdgeEmbedding,
        "dpp": TSPEdgeEmbedding,
        "mdpp": TSPEdgeEmbedding,
        "pdp": TSPEdgeEmbedding,
        "mtsp": TSPEdgeEmbedding,
        "smtwtp": NoEdgeEmbedding,
        "shpp": TSPEdgeEmbedding,
    }

    if env_name not in embedding_registry:
        raise ValueError(
            f"Unknown environment name '{env_name}'. Available init embeddings: {embedding_registry.keys()}"
        )

    return embedding_registry[env_name](**config)


class TSPEdgeEmbedding(nn.Module):
    """Edge embedding module for the Traveling Salesman Problem (TSP) and related problems.
    This module converts the cost matrix or the distances between nodes into embeddings that can be
    used by the neural network. It supports sparsification to focus on a subset of relevant edges,
    which is particularly useful for large graphs.
    """

    node_dim = 1

    def __init__(
            self,
            embed_dim,
            linear_bias=True,
            sparsify=True,
            k_sparse: Union[int, Callable[[int], int], None] = None,
    ):
        assert Batch is not None, (
            "torch_geometric not found. Please install torch_geometric using instructions from "
            "https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html."
        )

        super(TSPEdgeEmbedding, self).__init__()

        if k_sparse is None:
            self._get_k_sparse = lambda n: max(n // 5, 10)
        elif isinstance(k_sparse, int):
            self._get_k_sparse = lambda n: k_sparse
        elif callable(k_sparse):
            self._get_k_sparse = k_sparse
        else:
            raise ValueError("k_sparse must be an int or a callable")

        self.sparsify = sparsify
        self.edge_embed = nn.Linear(self.node_dim, embed_dim, linear_bias)

    def forward(self, td, init_embeddings: Tensor):
        cost_matrix = get_distance_matrix(td["locs"])
        batch = self._cost_matrix_to_graph(cost_matrix, init_embeddings)
        return batch

    def _cost_matrix_to_graph(self, batch_cost_matrix: Tensor, init_embeddings: Tensor):
        """Convert batched cost_matrix to batched PyG graph, and calculate edge embeddings.

        Args:
            batch_cost_matrix: Tensor of shape [batch_size, n, n]
            init_embedding: init embeddings
        """
        k_sparse = self._get_k_sparse(batch_cost_matrix.shape[-1])
        graph_data = []
        for index, cost_matrix in enumerate(batch_cost_matrix):
            if self.sparsify:
                edge_index, edge_attr = sparsify_graph(
                    cost_matrix, k_sparse, self_loop=False
                )
            else:
                edge_index = get_full_graph_edge_index(
                    cost_matrix.shape[0], self_loop=False
                ).to(cost_matrix.device)
                edge_attr = cost_matrix[edge_index[0], edge_index[1]].unsqueeze(-1)

            graph = Data(
                x=init_embeddings[index], edge_index=edge_index, edge_attr=edge_attr
            )
            graph_data.append(graph)

        batch = Batch.from_data_list(graph_data)
        batch.edge_attr = self.edge_embed(batch.edge_attr)
        return batch


class CVRPEdgeEmbedding(TSPEdgeEmbedding):
    """Edge embedding module for the Capacitated Vehicle Routing Problem (CVRP).
    Unlike the TSP, all nodes in the CVRP should be connected to the depot,
    so each node will have k_sparse + 1 edges.
    """

    def _cost_matrix_to_graph(self, batch_cost_matrix: Tensor, init_embeddings: Tensor):
        """Convert batched cost_matrix to batched PyG graph, and calculate edge embeddings.

        Args:
            batch_cost_matrix: Tensor of shape [batch_size, n, n]
            init_embedding: init embeddings
        """
        k_sparse = self._get_k_sparse(batch_cost_matrix.shape[-1])
        graph_data = []
        for index, cost_matrix in enumerate(batch_cost_matrix):
            if self.sparsify:
                edge_index, edge_attr = sparsify_graph(
                    cost_matrix[1:, 1:], k_sparse, self_loop=False
                )
                edge_index = edge_index + 1  # because we removed the depot
                # Note here
                edge_index = torch.cat(
                    [
                        edge_index,
                        # All nodes should be connected to the depot
                        torch.stack(
                            [
                                torch.arange(1, cost_matrix.shape[0]),
                                torch.zeros(cost_matrix.shape[0] - 1, dtype=torch.long),
                            ]
                        ).to(edge_index.device),
                        # Depot should be connected to all nodes
                        torch.stack(
                            [
                                torch.zeros(cost_matrix.shape[0] - 1, dtype=torch.long),
                                torch.arange(1, cost_matrix.shape[0]),
                            ]
                        ).to(edge_index.device),
                    ],
                    dim=1,
                )
                edge_attr = torch.cat(
                    [edge_attr, cost_matrix[1:, [0]], cost_matrix[[0], 1:].t()], dim=0
                )

            else:
                edge_index = get_full_graph_edge_index(
                    cost_matrix.shape[0], self_loop=False
                ).to(cost_matrix.device)
                edge_attr = cost_matrix[edge_index[0], edge_index[1]].unsqueeze(-1)

            graph = Data(
                x=init_embeddings[index], edge_index=edge_index, edge_attr=edge_attr
            )
            graph_data.append(graph)

        batch = Batch.from_data_list(graph_data)  # type: ignore
        batch.edge_attr = self.edge_embed(batch.edge_attr)  # type: ignore
        return batch


class VRPPolarEdgeEmbedding(TSPEdgeEmbedding):
    """Edge embedding module for VRPs using polar coordinates, with the depot as the origin.

    This class extends TSPEdgeEmbedding to handle VRP problems by incorporating polar coordinate
    information for edge embeddings.
    """

    node_dim = 2

    def forward(self, td, init_embeddings: Tensor):
        with torch.no_grad():
            if "polar_locs" in td.keys():
                theta = td["polar_locs"][..., 1]
            else:
                shifted_locs = td["locs"] - td["locs"][..., 0:1, :]
                x, y = shifted_locs[..., 0], shifted_locs[..., 1]
                theta = torch.atan2(y, x)

            delta_theta_matrix = theta[..., :, None] - theta[..., None, :]
            edge_attr1 = 1 - torch.cos(delta_theta_matrix)
            edge_attr2 = get_distance_matrix(td["locs"])
            cost_matrix = torch.stack((edge_attr1, edge_attr2), dim=-1)
            del edge_attr1, edge_attr2, delta_theta_matrix

            batch = self._cost_matrix_to_graph(cost_matrix, init_embeddings)
            del cost_matrix

        batch.edge_attr = self.edge_embed(batch.edge_attr)  # type: ignore
        return batch

    @torch.no_grad()
    def _cost_matrix_to_graph(self, batch_cost_matrix: Tensor, init_embeddings: Tensor):
        """Convert batched cost_matrix to batched PyG graph, and calculate edge embeddings.

        Args:
            batch_cost_matrix: Tensor of shape [batch_size, n, n, m]
            init_embedding: init embeddings of shape [batch_size, n, m]
        """
        k_sparse = self._get_k_sparse(batch_cost_matrix.shape[-1])
        graph_data = []
        for index, cost_matrix in enumerate(batch_cost_matrix):
            edge_index, _ = sparsify_graph(cost_matrix[..., 0], k_sparse, self_loop=False)
            edge_index = edge_index.T[torch.all(edge_index != 0, dim=0)].T
            _, depot_edge_index = torch.topk(
                cost_matrix[0, :, 1], k=k_sparse, largest=False, sorted=False
            )
            depot_edge_index = depot_edge_index[depot_edge_index != 0]
            depot_edge_index = torch.stack(
                (torch.zeros_like(depot_edge_index), depot_edge_index), dim=0
            )
            edge_index = torch.concat((depot_edge_index, edge_index), dim=-1).detach()
            edge_attr = cost_matrix[edge_index[0], edge_index[1]].detach()

            graph = Data(
                x=init_embeddings[index],
                edge_index=edge_index,
                edge_attr=edge_attr,
            )  # type: ignore
            graph_data.append(graph)

        batch = Batch.from_data_list(graph_data)  # type: ignore
        return batch


class ATSPEdgeEmbedding(TSPEdgeEmbedding):
    """Edge embedding module for the Asymmetric Traveling Salesman Problem (ATSP).
    Inherits from TSPEdgeEmbedding and adapts the edge embedding process to handle
    asymmetric cost matrices, where the cost from node i to node j may not be the same as from j to i.
    """

    def forward(self, td, init_embeddings: Tensor):
        batch = self._cost_matrix_to_graph(td["cost_matrix"], init_embeddings)
        return batch


class NoEdgeEmbedding(nn.Module):
    """A module for environments that do not require edge embeddings, or where edge features
    are not used. This can be useful for simplifying models in problems where only node
    features are relevant.
    """

    def __init__(self, embed_dim, self_loop=False, **kwargs):
        assert Batch is not None, (
            "torch_geometric not found. Please install torch_geometric using instructions from "
            "https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html."
        )

        super(NoEdgeEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.self_loop = self_loop

    def forward(self, td, init_embeddings: Tensor):
        data_list = []
        n = init_embeddings.shape[1]
        device = init_embeddings.device
        edge_index = get_full_graph_edge_index(n, self_loop=self.self_loop).to(device)
        m = edge_index.shape[1]

        for node_embed in init_embeddings:
            data = Data(
                x=node_embed,
                edge_index=edge_index,
                edge_attr=torch.zeros((m, self.embed_dim), device=device),
            )  # type: ignore
            data_list.append(data)

        batch = Batch.from_data_list(data_list)  # type: ignore
        return batch
