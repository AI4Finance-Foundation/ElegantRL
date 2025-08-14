import networkx as nx
import numpy as np
import os
import glob
from typing import List, Tuple


class GraphGenerator:
    """生成不同分布的合成图"""

    @staticmethod
    def generate_ba_graph(n: int, m: int = 4, seed: int = None) -> nx.Graph:
        """生成 Barabási–Albert 图"""
        return nx.barabasi_albert_graph(n, m, seed=seed)

    @staticmethod
    def generate_er_graph(n: int, p: float = None, m: int = None, seed: int = None) -> nx.Graph:
        """生成 Erdős–Rényi 图"""
        if p is not None:
            return nx.erdos_renyi_graph(n, p, seed=seed)
        elif m is not None:
            return nx.gnm_random_graph(n, m, seed=seed)
        else:
            p = np.log(n) / n
            return nx.erdos_renyi_graph(n, p, seed=seed)

    @staticmethod
    def generate_powerlaw_graph(n: int, m: int = 2, p: float = 0.1, seed: int = None) -> nx.Graph:
        """生成幂律分布图"""
        return nx.powerlaw_cluster_graph(n, m, p, seed=seed)

    @staticmethod
    def add_weights(graph: nx.Graph, weight_range: Tuple[float, float] = (1, 10),
                   weight_type: str = 'uniform', seed: int = None) -> nx.Graph:
        """为图添加边权重"""
        if seed is not None:
            np.random.seed(seed)

        weighted_graph = graph.copy()

        for u, v in weighted_graph.edges():
            if weight_type == 'uniform':
                weight = np.random.uniform(weight_range[0], weight_range[1])
            elif weight_type == 'normal':
                mean = (weight_range[0] + weight_range[1]) / 2
                std = (weight_range[1] - weight_range[0]) / 4
                weight = np.random.normal(mean, std)
                weight = np.clip(weight, weight_range[0], weight_range[1])
            else:
                weight = 1.0

            weighted_graph.edges[u, v]['weight'] = weight

        return weighted_graph


class GraphIO:
    """处理图文件的读写"""

    @staticmethod
    def save_graph(graph: nx.Graph, filepath: str):
        """保存图到文件

        文件格式：
        n m           # 节点数和边数
        u v w         # 边 (u, v) 权重 w
        """
        n = graph.number_of_nodes()
        m = graph.number_of_edges()

        with open(filepath, 'w') as f:
            f.write(f"{n} {m}\n")

            for u, v, data in graph.edges(data=True):
                weight = data.get('weight', 1.0)
                f.write(f"{u} {v} {weight:.6f}\n")

    @staticmethod
    def load_graph(filepath: str) -> nx.Graph:
        """从文件加载图"""
        with open(filepath, 'r') as f:
            lines = f.readlines()

        n, m = map(int, lines[0].strip().split())

        graph = nx.Graph()
        graph.add_nodes_from(range(n))

        for i in range(1, m + 1):
            parts = lines[i].strip().split()
            u, v = int(parts[0]), int(parts[1])
            weight = float(parts[2]) if len(parts) > 2 else 1.0
            graph.add_edge(u, v, weight=weight)

        return graph


class MaxCutDataset:
    """Max-Cut 问题的图数据集"""

    def __init__(self, root_dir: str = "./data"):
        self.root_dir = root_dir
        self.graphs = []
        self.graph_info = []

    def generate_synthetic_dataset(self, sizes: List[int], samples_per_size: int = 10,
                                 graph_types: List[str] = ['ba', 'er', 'pl'],
                                 weighted: bool = True, save: bool = True):
        """生成合成数据集"""
        self.graphs = []
        self.graph_info = []

        total_graphs = 0
        for size in sizes:
            for graph_type in graph_types:
                for i in range(samples_per_size):
                    seed = size * 1000 + i

                    try:
                        if graph_type == 'ba':
                            m = max(2, int(np.log(size)))
                            graph = GraphGenerator.generate_ba_graph(size, m=m, seed=seed)
                        elif graph_type == 'er':
                            p = 2 * np.log(size) / size
                            graph = GraphGenerator.generate_er_graph(size, p=p, seed=seed)
                        elif graph_type == 'pl':
                            graph = GraphGenerator.generate_powerlaw_graph(size, seed=seed)
                        else:
                            raise ValueError(f"Unknown graph type: {graph_type}")

                        if weighted:
                            graph = GraphGenerator.add_weights(graph, seed=seed)

                        info = {
                            'type': graph_type,
                            'size': size,
                            'nodes': graph.number_of_nodes(),
                            'edges': graph.number_of_edges(),
                            'weighted': weighted,
                            'index': total_graphs
                        }

                        self.graphs.append(graph)
                        self.graph_info.append(info)
                        total_graphs += 1

                    except Exception as e:
                        print(f"生成图时出错 (type={graph_type}, size={size}): {str(e)}")
                        continue

        if save and len(self.graphs) > 0:
            self.save_to_files()

        print(f"成功生成 {len(self.graphs)} 个图")
        return self.graphs

    def load_from_directory(self, directory: str = None):
        """从目录加载图数据集"""
        if directory is None:
            directory = self.root_dir

        self.graphs = []
        self.graph_info = []

        pattern = os.path.join(directory, "*.txt")
        files = sorted(glob.glob(pattern))
        files = [f for f in files if not f.endswith('dataset_info.txt')]

        for filepath in files:
            try:
                graph = GraphIO.load_graph(filepath)

                filename = os.path.basename(filepath)
                parts = filename.replace('.txt', '').split('_')

                info = {
                    'type': parts[0] if len(parts) > 0 else 'unknown',
                    'nodes': graph.number_of_nodes(),
                    'edges': graph.number_of_edges(),
                    'weighted': any('weight' in data for _, _, data in graph.edges(data=True)),
                    'index': len(self.graphs),
                    'filepath': filepath
                }

                self.graphs.append(graph)
                self.graph_info.append(info)
            except Exception as e:
                print(f"跳过文件 {filepath}: {str(e)}")
                continue

        print(f"从 {directory} 加载了 {len(self.graphs)} 个图")
        return self.graphs

    def save_to_files(self, directory: str = None):
        """保存数据集到文件"""
        if directory is None:
            directory = self.root_dir

        os.makedirs(directory, exist_ok=True)

        for graph, info in zip(self.graphs, self.graph_info):
            graph_type = info.get('type', 'unknown')
            nodes = info.get('nodes', graph.number_of_nodes())
            index = info.get('index', 0)
            filename = f"{graph_type}_{nodes:04d}_{index:04d}.txt"
            filepath = os.path.join(directory, filename)
            GraphIO.save_graph(graph, filepath)

        info_file = os.path.join(directory, "dataset_info.txt")
        with open(info_file, 'w') as f:
            f.write(f"Total graphs: {len(self.graphs)}\n")
            graph_types = set(info['type'] for info in self.graph_info)
            f.write(f"Graph types: {graph_types}\n")
            if self.graph_info:
                min_nodes = min(info['nodes'] for info in self.graph_info)
                max_nodes = max(info['nodes'] for info in self.graph_info)
                f.write(f"Size range: {min_nodes} - {max_nodes}\n")

    def get_graphs_by_type(self, graph_type: str) -> List[nx.Graph]:
        """获取特定类型的所有图"""
        indices = [i for i, info in enumerate(self.graph_info) if info['type'] == graph_type]
        return [self.graphs[i] for i in indices]

    def get_graphs_by_size(self, min_size: int = 0, max_size: int = float('inf')) -> List[nx.Graph]:
        """获取特定大小范围的图"""
        indices = [i for i, info in enumerate(self.graph_info)
                  if min_size <= info['nodes'] <= max_size]
        return [self.graphs[i] for i in indices]

    def split_dataset(self, train_ratio: float = 0.8, val_ratio: float = 0.1):
        """分割数据集"""
        n = len(self.graphs)
        indices = list(range(n))
        np.random.shuffle(indices)

        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        return {
            'train': [self.graphs[i] for i in train_indices],
            'val': [self.graphs[i] for i in val_indices],
            'test': [self.graphs[i] for i in test_indices]
        }


if __name__ == "__main__":
    # 生成数据集
    dataset = MaxCutDataset(root_dir="./maxcut_data")

    # 生成不同大小的图
    sizes = [20, 30, 50, 100, 200]
    dataset.generate_synthetic_dataset(
        sizes=sizes,
        samples_per_size=10,
        graph_types=['ba', 'er', 'pl'],
        weighted=True,
        save=True
    )

    # 展示数据集信息
    print(f"\n数据集统计:")
    for graph_type in ['ba', 'er', 'pl']:
        graphs = dataset.get_graphs_by_type(graph_type)
        print(f"{graph_type.upper()}: {len(graphs)} 个图")

    # 展示大小分布
    print(f"\n大小分布:")
    for size in sizes:
        graphs = dataset.get_graphs_by_size(size-5, size+5)
        print(f"节点数 ~{size}: {len(graphs)} 个图")