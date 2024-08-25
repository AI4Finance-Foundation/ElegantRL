import os

import networkx as nx
import numpy as np
import pandas as pd
import torch
from rich import print
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DATA_DIR = "./data/raw"


def get_data_files():
    fileList = sorted(
        [
            os.path.join(DATA_DIR, fileName)
            for fileName in os.listdir(DATA_DIR)
            if not fileName.endswith(".txt")
            and not os.path.isdir(os.path.join(DATA_DIR, fileName))
        ]
    )
    return fileList


def get_dataset():
    dataset = parse_files(get_data_files())
    return dataset


def parse_single_file(filePath):
    dataset = pd.read_csv(filePath, sep=" ")
    V, E, _ = dataset.columns.tolist()
    V = int(V)
    E = int(E)
    # Graph here should be undirected (in most cases)
    G = nx.Graph()
    G.add_nodes_from(range(V))
    for i in range(E):
        edge = dataset.iloc[i]
        # node index in data file range from 1..V, convert to 0..V-1
        # Here we don't know what the negative number in the data file means
        # so we just convert it to positive
        G.add_edge(edge[0] - 1, edge[1] - 1, weight=abs(edge[2]))
    return os.path.basename(filePath), G


def parse_files(fileList):
    dataset = [
        (fileName, G) for (fileName, G) in tqdm(map(parse_single_file, fileList))
    ]
    return dataset


def get_dense_adj(graph: nx.Graph):
    adj = np.zeros((graph.number_of_nodes(), graph.number_of_nodes()))
    for (u, v, d) in graph.edges.data("weight"):
        adj[u][v] = d
        adj[v][u] = d
    return adj


def Hamiltonian_MaxCut(graph: nx.Graph):
    # sum_{(i,j) in E} (2*x_i*x_j - x_i - x_j)
    adj = get_dense_adj(graph)
    deg = np.array(graph.degree())[:, 1]
    return np.triu(adj) * 2 - np.diag(deg)


def Hamiltonian_MaxIndSet(graph: nx.Graph, penalty=2):
    # -sum_{i in V} (x_i) + P * sum_{(i,j) in E} (x_i*x_j)
    # P=2 is the default value in the paper
    adj = get_dense_adj(graph)
    return np.triu(adj) * penalty - np.eye(graph.number_of_nodes())


def Hamiltonian_MinVerCover(graph: nx.Graph, penalty=2):
    # sum_{i in V} (x_i) + P * sum_{(i,j) in E} (1 - x_i - x_j + x_i*x_j)
    # sum_{i in V} (x_i) + P * sum_{(i,j) in E} (- x_i - x_j + x_i*x_j)
    adj = get_dense_adj(graph)
    deg = np.array(graph.degree())[:, 1]
    return (
        np.eye(graph.number_of_nodes())
        + np.triu(adj) * penalty
        - np.diag(deg) * penalty
    )


def get_sparse_adj(graph: nx.Graph):
    # convert iterator to list
    sadj = np.array(list(graph.edges.data("weight")))
    edge_index = sadj[:, :2].astype(np.int64)
    edge_attr = sadj[:, 2]
    return edge_index, edge_attr


def preprocess(root_dir="data/dataset"):
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    dataset = get_dataset()
    for inst in dataset:
        torch.save(
            inst[1],
            os.path.join(root_dir, f"{inst[0]}.pt"),
        )


class GraphDataset(Dataset):
    def __init__(
        self,
        root_dir="data/dataset",
        transform=Hamiltonian_MaxCut,
        dtype=torch.float32,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.dtype = dtype
        self.file_paths = [
            os.path.join(self.root_dir, f"{os.path.basename(file_name)}.pt")
            for file_name in get_data_files()
        ]
        if not all(map(os.path.exists, self.file_paths)):
            preprocess(self.root_dir)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        G = torch.load(self.file_paths[index])
        edge_index, edge_attr = get_sparse_adj(G)
        hamiltonian = self.transform(G)
        return (
            os.path.basename(self.file_paths[index]).removesuffix(".pt"),
            torch.from_numpy(edge_index.T).to(torch.int64),
            torch.from_numpy(edge_attr).to(self.dtype),
            torch.from_numpy(hamiltonian).to(self.dtype),
        )


class Logger:
    def __init__(self, file_path, mode="w"):
        self.file_path = file_path
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        if mode == "w":
            with open(file_path, mode) as f:
                f.write("")

    def log(self, message):
        with open(self.file_path, "a") as f:
            f.write(message + "\n")
        print(message)

    def log_no_print(self, message):
        with open(self.file_path, "a") as f:
            f.write(message + "\n")

    def __call__(self, message, print_=True):
        if print_:
            self.log(message)
        else:
            self.log_no_print(message)


if __name__ == "__main__":
    # # example graph from the paper
    # G = nx.Graph()
    # G.add_nodes_from(range(5))
    # G.add_edge(0, 1, weight=1)
    # G.add_edge(0, 2, weight=1)
    # G.add_edge(1, 3, weight=1)
    # G.add_edge(2, 3, weight=1)
    # G.add_edge(2, 4, weight=1)
    # G.add_edge(3, 4, weight=1)
    # print(Hamiltonian_MaxCut(G))
    # print(Hamiltonian_MaxIndSet(G))
    # print(Hamiltonian_MinVerCover(G))

    dataset = GraphDataset(
        transform=Hamiltonian_MaxCut,
    )
    dataset = GraphDataset(
        transform=Hamiltonian_MaxIndSet,
    )
    dataset = GraphDataset(
        transform=Hamiltonian_MinVerCover,
    )
    print(dataset[0])
