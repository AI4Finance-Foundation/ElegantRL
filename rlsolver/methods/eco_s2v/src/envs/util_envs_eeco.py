import random
from abc import ABC, abstractmethod

import networkx as nx
import numpy as np
import torch

from rlsolver.methods.config import GraphType
from rlsolver.methods.eco_s2v.src.envs.util_envs import (EdgeType)


class GraphGenerator(ABC):

    def __init__(self, n_spins, edge_type, biased=False, n_sims=2 ** 3):
        self.n_spins = n_spins
        self.edge_type = edge_type
        self.biased = biased
        self.n_sims = n_sims

    def pad_matrix(self, matrix):
        dim = matrix.shape[0]
        m = np.zeros((dim + 1, dim + 1))
        m[:-1, :-1] = matrix
        return matrix

    def pad_bias(self, bias):
        return np.concatenate((bias, [0]))

    @abstractmethod
    def get(self, with_padding=False):
        raise NotImplementedError


###################
# Unbiased graphs #
###################
class RandomGraphGenerator(GraphGenerator):

    def __init__(self, n_spins=20, edge_type=EdgeType.DISCRETE, biased=False):
        super().__init__(n_spins, edge_type, biased)

        if self.edge_type == EdgeType.UNIFORM:
            self.get_w = lambda: 1
        elif self.edge_type == EdgeType.DISCRETE:
            self.get_w = lambda: np.random.choice([+1, -1])
        elif self.edge_type == EdgeType.RANDOM:
            self.get_w = lambda: np.random.uniform(-1, 1)
        else:
            raise NotImplementedError()

    def get(self, with_padding=False):

        g_size = self.n_spins

        density = np.random.uniform()
        matrix = np.zeros((g_size, g_size))
        for i in range(self.n_spins):
            for j in range(i):
                if np.random.uniform() < density:
                    w = self.get_w()
                    matrix[i, j] = w
                    matrix[j, i] = w

        matrix = self.pad_matrix(matrix) if with_padding else matrix

        if self.biased:
            bias = np.array([self.get_w() if np.random.uniform() < density else 0 for _ in range(self.n_spins)])
            bias = self.pad_bias(bias) if with_padding else bias
            return matrix, bias
        else:
            return matrix

        m = self.pad_matrix(self.matrix) if with_padding else self.matrix

        if self.biased:
            b = self.pad_bias(self.bias) if with_padding else self.bias
            return m, b
        else:
            return m


class RandomErdosRenyiGraphGenerator(GraphGenerator):
    def __init__(self, n_spins=20, p_connection=0.2, edge_type=EdgeType.DISCRETE, n_sims=8, device="cuda"):
        super().__init__(n_spins, edge_type, False, n_sims)
        self.p_connection = p_connection
        self.device = device

        if self.edge_type == EdgeType.UNIFORM:
            self.get_connection_mask = lambda: torch.ones((self.n_spins, self.n_spins), device=self.device)
        elif self.edge_type == EdgeType.DISCRETE:
            def get_connection_mask():
                mask = 2. * torch.randint(0, 2, (self.n_spins, self.n_spins), device=self.device) - 1.
                mask = torch.tril(mask) + torch.triu(mask.T, 1)
                return mask

            self.get_connection_mask = get_connection_mask
        elif self.edge_type == EdgeType.RANDOM:
            def get_connection_mask():
                mask = 2. * torch.randint(0, 2, (self.n_sims, self.n_spins, self.n_spins), dtype=torch.float32, device=self.device) - 1
                mask = torch.tril(mask, diagonal=0) + torch.triu(mask.transpose(1, 2), diagonal=1)
                return mask

            self.get_connection_mask = get_connection_mask
        else:
            raise NotImplementedError()

    def generate_er_graph(self):

        # 对于每个 batch，生成 n_spins x n_spins 的邻接矩阵
        adj = (torch.rand(self.n_sims, self.n_spins, self.n_spins, device=self.device) < self.p_connection).float()

        # 对角线置零（无自环）
        adj = adj * (1 - torch.eye(self.n_spins, device=self.device).unsqueeze(0))

        # 对称化处理（无向图）
        adj = torch.triu(adj, diagonal=1)
        adj = adj + adj.transpose(1, 2)

        return adj

    def get(self, with_padding=False):
        adj = self.generate_er_graph()
        adj = adj * self.get_connection_mask()
        return self.pad_matrix(adj) if with_padding else adj


class RandomBarabasiAlbertGraphGenerator(GraphGenerator):
    def __init__(self, n_spins=20, m_insertion_edges=4, edge_type=EdgeType.DISCRETE, n_sims=8, device="cuda"):
        super().__init__(n_spins, edge_type, False, n_sims)
        self.m_insertion_edges = m_insertion_edges
        self.device = device
        # self.seed = seed
        # if self.seed is not None:
        #     torch.manual_seed(self.seed)
        if self.edge_type == EdgeType.UNIFORM:
            self.get_connection_mask = lambda: torch.ones((self.n_spins, self.n_spins), device=self.device)
        elif self.edge_type == EdgeType.DISCRETE:
            def get_connection_mask():
                mask = 2. * torch.randint(0, 2, (self.n_spins, self.n_spins), device=self.device) - 1.
                mask = torch.tril(mask) + torch.triu(mask.T, 1)
                return mask

            self.get_connection_mask = get_connection_mask
        elif self.edge_type == EdgeType.RANDOM:
            def get_connection_mask():
                mask = 2. * torch.randint(0, 2, (self.n_sims, self.n_spins, self.n_spins), dtype=torch.float32, device=self.device) - 1
                mask = torch.tril(mask, diagonal=0) + torch.triu(mask.transpose(1, 2), diagonal=1)
                return mask

            self.get_connection_mask = get_connection_mask
        else:
            raise NotImplementedError()

    def generate_barabasi_albert(self):
        """
        直接在 PyTorch 中并行生成 Barabási–Albert 图
        """
        adj = torch.zeros((self.n_sims, self.n_spins, self.n_spins), device=self.device)

        # 初始完全连通子图
        for i in range(self.m_insertion_edges + 1):
            adj[:, i, :i + 1] = 1
            adj[:, :i + 1, i] = 1

        for new_node in range(self.m_insertion_edges + 1, self.n_spins):
            degree = adj.sum(dim=-1)
            prob = degree / degree.sum(dim=-1, keepdim=True)

            chosen_edges = torch.multinomial(prob, num_samples=self.m_insertion_edges, replacement=False)
            batch_indices = torch.arange(self.n_sims, device=self.device).repeat_interleave(self.m_insertion_edges)
            adj[batch_indices, new_node, chosen_edges.view(-1)] = 1
            adj[batch_indices, chosen_edges.view(-1), new_node] = 1

        return adj

    def get(self, with_padding=False):
        adj = self.generate_barabasi_albert()
        # if self.seed is not None:
        adj = adj * self.get_connection_mask()
        return self.pad_matrix(adj) if with_padding else adj
    # class RandomBarabasiAlbertGraphGenerator:


#     def __init__(self, n_spins=20, m_insertion_edges=4, edge_type="DISCRETE", n_sims=8, device="cuda"):
#         self.n_spins = n_spins
#         self.m_insertion_edges = m_insertion_edges
#         self.n_sims = n_sims
#         self.device = device

#     def generate_barabasi_albert(self):
#         """
#         使用 PyTorch 并行实现 Barabási–Albert 过程
#         生成多个 BA 图的邻接矩阵
#         """
#         # 初始化邻接矩阵，全 0
#         adj = torch.zeros((self.n_sims, self.n_spins, self.n_spins), device=self.device)

#         # 初始完全连通子图
#         for i in range(self.m_insertion_edges + 1):
#             adj[:, i, :i + 1] = 1
#             adj[:, :i + 1, i] = 1

#         # 节点加入过程
#         for new_node in range(self.m_insertion_edges + 1, self.n_spins):
#             # 计算度
#             degree = adj.sum(dim=-1)

#             # 计算连接概率 (度数归一化)
#             prob = degree / degree.sum(dim=-1, keepdim=True)

#             # 选择 m 个节点（基于概率）
#             chosen_edges = torch.multinomial(prob, num_samples=self.m_insertion_edges, replacement=False)

#             # 连接新节点
#             batch_indices = torch.arange(self.n_sims, device=self.device).repeat_interleave(self.m_insertion_edges)
#             adj[batch_indices, new_node, chosen_edges.view(-1)] = 1
#             adj[batch_indices, chosen_edges.view(-1), new_node] = 1  # 无向图

#         return adj

#     def get(self, with_padding=False):
#         # 生成邻接矩阵
#         adj = self.generate_barabasi_albert()

#         # 连接掩码
#         connection_mask = torch.randint(0, 2, (self.n_sims, self.n_spins, self.n_spins), device=self.device) * 2 - 1
#         connection_mask = torch.tril(connection_mask) + torch.triu(connection_mask.transpose(1, 2), 1)

#         adj = adj * connection_mask  # 应用掩码
#         return adj


class RandomRegularGraphGenerator(GraphGenerator):

    def __init__(self, n_spins=20, d_node=[2, 0], edge_type=EdgeType.DISCRETE, biased=False):
        super().__init__(n_spins, edge_type, biased)

        if type(d_node) not in [list, tuple]:
            d_node = [d_node, 0]
        assert len(d_node) == 2, "k_neighbours must have length 2"
        self.d_node = d_node

        if self.edge_type == EdgeType.UNIFORM:
            self.get_connection_mask = lambda: np.ones((self.n_spins, self.n_spins))
        elif self.edge_type == EdgeType.DISCRETE:
            def get_connection_mask():
                mask = 2. * np.random.randint(2, size=(self.n_spins, self.n_spins)) - 1.
                mask = np.tril(mask) + np.triu(mask.T, 1)
                return mask

            self.get_connection_mask = get_connection_mask
        elif self.edge_type == EdgeType.RANDOM:
            def get_connection_mask():
                mask = 2. * np.random.rand(self.n_spins, self.n_spins) - 1
                mask = np.tril(mask) + np.triu(mask.T, 1)
                return mask

            self.get_connection_mask = get_connection_mask
        else:
            raise NotImplementedError()

    def get(self, with_padding=False):
        k = np.clip(int(np.random.normal(*self.d_node)), 0, self.n_spins)

        g = nx.random_regular_graph(k, self.n_spins)
        adj = np.multiply(nx.to_numpy_array(g), self.get_connection_mask())

        if not self.biased:
            # No self-connections (this modifies adj in-place).
            np.fill_diagonal(adj, 0)

        return self.pad_matrix(adj) if with_padding else adj


class RandomWattsStrogatzGraphGenerator(GraphGenerator):

    def __init__(self, n_spins=20, k_neighbours=[2, 0], edge_type=EdgeType.DISCRETE, biased=False):
        super().__init__(n_spins, edge_type, biased)

        if type(k_neighbours) not in [list, tuple]:
            k_neighbours = [k_neighbours, 0]
        assert len(k_neighbours) == 2, "k_neighbours must have length 2"
        self.k_neighbours = k_neighbours

        if self.edge_type == EdgeType.UNIFORM:
            self.get_connection_mask = lambda: np.ones((self.n_spins, self.n_spins))
        elif self.edge_type == EdgeType.DISCRETE:
            def get_connection_mask():
                mask = 2. * np.random.randint(2, size=(self.n_spins, self.n_spins)) - 1.
                mask = np.tril(mask) + np.triu(mask.T, 1)
                return mask

            self.get_connection_mask = get_connection_mask
        elif self.edge_type == EdgeType.RANDOM:
            def get_connection_mask():
                mask = 2. * np.random.rand(self.n_spins, self.n_spins) - 1
                mask = np.tril(mask) + np.triu(mask.T, 1)
                return mask

            self.get_connection_mask = get_connection_mask
        else:
            raise NotImplementedError()

    def get(self, with_padding=False):
        k = np.clip(int(np.random.normal(*self.k_neighbours)), 0, self.n_spins)

        g = nx.watts_strogatz_graph(self.n_spins, k, 0)
        adj = np.multiply(nx.to_numpy_array(g), self.get_connection_mask())

        if not self.biased:
            # No self-connections (this modifies adj in-place).
            np.fill_diagonal(adj, 0)

        return self.pad_matrix(adj) if with_padding else adj


################
# Known graphs #
################
class SingleGraphGenerator(GraphGenerator):

    def __init__(self, matrix, bias=None):

        n_spins = matrix.shape[0]

        if np.isin(matrix, [0, 1]).all():
            edge_type = EdgeType.UNIFORM
        elif np.isin(matrix, [0, -1, 1]).all():
            edge_type = EdgeType.DISCRETE
        else:
            edge_type = EdgeType.RANDOM

        super().__init__(n_spins, edge_type, bias is not None)

        self.matrix = matrix
        self.bias = bias

    def get(self, with_padding=False):

        m = self.pad_matrix(self.matrix) if with_padding else self.matrix

        if self.biased:
            b = self.pad_bias(self.bias) if with_padding else self.bias
            return m, b
        else:
            return m


class ValidationGraphGenerator(GraphGenerator):
    def __init__(self, device, n_spins=20, edge_type=EdgeType.DISCRETE, n_sims=2 ** 3, seed=None, graph_type=GraphType.BA):
        super().__init__(n_spins, edge_type, False, n_sims)
        self.seed = seed
        self.device = device
        self.graph_type = graph_type
        self.n_sims = n_sims
        self.n_spins = n_spins
        self.adj = torch.empty((self.n_sims, self.n_spins, self.n_spins), device=self.device, dtype=torch.float)

    def get(self):
        adj = torch.empty((self.n_sims, self.n_spins, self.n_spins), device=self.device, dtype=torch.float)
        seed = self.seed
        for i in range(self.n_sims):
            if self.seed is not None:
                if self.graph_type == GraphType.BA:
                    g = nx.barabasi_albert_graph(self.n_spins, 4, seed=seed)
                elif self.graph_type == GraphType.ER:
                    g = nx.erdos_renyi_graph(self.n_spins, 0.15, seed=seed)
                adj[i] = torch.tensor(nx.to_numpy_array(g), dtype=torch.float32, device=self.device).fill_diagonal_(0)
                seed += 1
        return adj


class SetGraphGenerator(GraphGenerator):

    def __init__(self, matrices, biases=None, ordered=False, device="cuda"):

        if len(set([m.shape[0] - 1 for m in matrices])) == 1:
            n_spins = matrices[0].shape[0]
        else:
            raise NotImplementedError("All graphs in SetGraphGenerator must have the same dimension.")

        if all([torch.isin(m, torch.tensor([0, 1], device=device)).all().item() for m in matrices]):

            edge_type = EdgeType.UNIFORM
        elif all([torch.isin(m, torch.tensor([0, -1, 1], device=device)).all().item() for m in matrices]):
            edge_type = EdgeType.DISCRETE
        else:
            edge_type = EdgeType.RANDOM

        super().__init__(n_spins, edge_type, biases is not None)

        if not self.biased:
            self.graphs = matrices
        else:
            assert len(matrices) == len(biases), "Must pass through the same number of matrices and biases."
            assert all([len(b) == self.n_spins + 1 for b in
                        biases]), "All biases and must have the same dimension as the matrices."
            self.graphs = list(zip(matrices, biases))

        self.ordered = ordered
        if self.ordered:
            self.i = 0

    def get(self, with_padding=False):
        m = self.graphs
        return self.pad_matrix(m) if with_padding else m


class PerturbedGraphGenerator(GraphGenerator):

    def __init__(self, matrices, perturb_mean=0, perturb_std=0.01, biases=None, ordered=False):

        if type(matrices) != list:
            matrices = list(matrices)

        if biases is not None:
            if type(biases) != list:
                biases = list(biases)

        if len(set([m.shape[0] - 1 for m in matrices])) == 1:
            n_spins = matrices[0].shape[0]
        else:
            raise NotImplementedError("All graphs passed to PerturbedGraphGenerator must have the same dimension.")

        super().__init__(n_spins, EdgeType.RANDOM, biases is not None)

        self.perturb_mean = perturb_mean
        self.perturb_std = perturb_std

        if not self.biased:
            self.graphs = matrices
        else:
            raise NotImplementedError("Not implemented PerturbedGraphGenerator for biased graphs yet.")

        self.ordered = ordered
        if self.ordered:
            self.i = 0

    def get(self, with_padding=False):
        if self.ordered:
            m = self.graphs[self.i]
            self.i = (self.i + 1) % len(self.graphs)
            if self.biased:
                m, b = m
        else:
            if not self.biased:
                m = random.sample(self.graphs, k=1)[0]
            else:
                m, b = random.sample(self.graphs, k=1)[0]

        # Sample noise.
        noise = np.random.normal(self.perturb_mean, self.perturb_std, size=m.shape)
        # Set noise to 0 for non-edges in the adjacency matrix.
        np.putmask(noise, m == 0, 0)
        # Ensure noise is symettric.
        noise = np.tril(noise) + np.triu(noise.T, 1)

        m = m + noise

        return self.pad_matrix(m) if with_padding else m


class HistoryBuffer():
    def __init__(self, n_sims, device):
        self.n_sims = n_sims
        self.buffer = None
        self.device = device

    # def pack_spins(self,spins):
    #     n_sims, spin_dim = spins.shape
    #     assert spin_dim % 8 == 0, "spin_dim 必须是 8 的倍数"
    #     return spins.view(n_sims, -1, 8).mul(2**torch.arange(7, -1, -1, device=spins.device)).sum(dim=2).to(torch.uint8)

    def pack_spins(self, spins_0):
        """
        Packs a 0/1 spin tensor into uint8 for efficient storage.
        Handles cases where spin_dim is not a multiple of 8 by padding with zeros.
        """
        n_sims, spin_dim = spins_0.shape

        # Calculate padding if necessary
        remainder = spin_dim % 8
        if remainder != 0:
            padding_needed = 8 - remainder
            # Create a padding tensor of zeros
            padding = torch.zeros(n_sims, padding_needed, dtype=spins_0.dtype, device=spins_0.device)
            # Concatenate padding to the original spins
            spins_padded = torch.cat([spins_0, padding], dim=1)
            # The new dimension is now guaranteed to be a multiple of 8
            packed_dim = (spin_dim + padding_needed) // 8
        else:
            # No padding needed
            spins_padded = spins_0
            packed_dim = spin_dim // 8

        packed_spins = spins_padded.view(n_sims, packed_dim, 8) \
            .mul(2 ** torch.arange(7, -1, -1, device=spins_padded.device)) \
            .sum(dim=2) \
            .to(torch.uint8)
        return packed_spins

    def update(self, spins):
        """
        记录新的 spins，并返回哪些是新状态
        :param spins: (n_sims, spin_dim) 形状的 0/1 Tensor
        :return: (n_sims,) 形状的布尔 Tensor，表示哪些是新状态
        """
        # spins_0的值是0或1，spins的值是1或-1
        spins_0 = (spins + 1) / 2
        spins_packed = self.pack_spins(spins_0)  # 压缩存储

        # 第一次更新时初始化 buffer
        if self.buffer is None:
            self.buffer = spins_packed.unsqueeze(0)  # (1, n_sims, packed_dim)
            return torch.ones(self.n_sims, dtype=torch.bool, device=self.device)  # 全部是新状态

        # 计算 Hamming 距离匹配（更快）
        matches = (self.buffer ^ spins_packed.unsqueeze(0)).sum(dim=2)  # 计算每个环境的 Hamming 距离
        visited = (matches == 0).any(dim=0)  # (n_sims,)

        # 只有 `visited=False` 时，表示是新状态
        updated = ~visited

        # 追加新状态
        self.buffer = torch.cat([self.buffer, spins_packed.unsqueeze(0)], dim=0)

        return updated
