from rlsolver.methods.iSCO.config.maxcut_config import *
import torch
import networkx as nx

def load_data(filename):
    with open(filename, 'r') as file:
        lines = []
        line = file.readline()  # 读取第一行
        while line is not None and line != '':
            if '//' not in line:
                lines.append(line)
            line = file.readline()  # 读取下一行
        lines = [[int(i1) for i1 in i0.split()] for i0 in lines]
    num_nodes, num_edges = lines[0]
    g = nx.Graph()
    nodes = list(range(num_nodes))
    g.add_nodes_from(nodes)
    for item in lines[1:]:
        g.add_edge(item[0] - 1, item[1] - 1, weight=item[2])
    edge_from = [0] * num_edges
    edge_to = [0] * num_edges
    edge_weight = [0] * num_edges
    for i, e in enumerate(g.edges(data=True)):
        x, y = e[0], e[1]
        edge_from[i] = x
        edge_to[i] = y
        edge_weight[i] = e[2]['weight']

    edge_from = torch.tensor(edge_from, dtype=torch.int32,device=DEVICE).long()
    edge_to = torch.tensor(edge_to, dtype=torch.int32,device=DEVICE).long()
    data = {
        'num_nodes': num_nodes,
        "num_edges": num_edges,
        'edge_from': edge_from,
        'edge_to': edge_to,
    }
    return data

