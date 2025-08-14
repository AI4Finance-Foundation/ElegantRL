from rlsolver.methods.iSCO.config.config_maxcut import *
import torch
import networkx as nx
import torch.nn.functional as F


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
    adj_matrix_numpy = nx.to_numpy_array(g, weight='weight')
    A = torch.tensor(adj_matrix_numpy, dtype=torch.float16,device=DEVICE)
    padded= (num_nodes + 7) // 8 * 8
    pad_rows = padded - num_nodes
    A = F.pad(A, (0, pad_rows, 0, pad_rows), mode='constant', value=0)

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
        'adj_matrix':A
    }
    return data

def record(pre_obj,pre_sample,new_obj,new_sample):
    max_value, max_index = torch.max(new_obj, dim=0)
    if max_value > pre_obj:
        pre_sample = new_sample[max_index]
        pre_obj = max_value
    return pre_obj,pre_sample