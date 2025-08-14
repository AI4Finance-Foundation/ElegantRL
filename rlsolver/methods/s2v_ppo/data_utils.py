# data_utils.py
import torch
import networkx as nx
import numpy as np
import os
import glob
from tqdm import tqdm


def convert_graph_to_data(graph):
    """优化的图转换函数"""
    edge_list = list(graph.edges(data=True))
    if not edge_list:
        return {
            'edge_index': torch.empty((2, 0), dtype=torch.long),
            'edge_weight': torch.empty(0, dtype=torch.float32),
            'num_nodes': graph.number_of_nodes()
        }
    
    # 创建双向边
    edges = []
    weights = []
    for u, v, data in edge_list:
        weight = data.get('weight', 1.0)
        edges.extend([[u, v], [v, u]])
        weights.extend([weight, weight])
    
    edge_index = torch.tensor(edges, dtype=torch.long).T.contiguous()
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    
    return {
        'edge_index': edge_index,
        'edge_weight': edge_weight,
        'num_nodes': graph.number_of_nodes()
    }


def load_graphs_from_directory(directory):
    """从目录加载所有图数据"""
    pattern = os.path.join(directory, "*.txt")
    files = sorted(glob.glob(pattern))
    files = [f for f in files if not f.endswith('dataset_info.txt')]
    
    graph_datas = []
    
    for filepath in tqdm(files, desc="加载图数据"):
        try:
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
            
            # 转换为张量格式
            graph_data = convert_graph_to_data(graph)
            graph_datas.append(graph_data)
        except Exception as e:
            print(f"无法加载 {filepath}: {e}")
            continue
    
    print(f"成功加载 {len(graph_datas)} 个图")
    return graph_datas


def sample_batch_graphs(all_graphs, batch_size):
    """从预加载的图中随机采样一批"""
    indices = np.random.choice(len(all_graphs), batch_size, replace=True)
    return [all_graphs[i] for i in indices]