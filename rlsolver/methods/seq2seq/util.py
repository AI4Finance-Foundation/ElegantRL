import torch
import networkx as nx
import os
import numpy as np
from config import *

def load_adj_matrix_from_folder(folder_name):
    adj_dict = {}
    for filename in os.listdir(folder_name):
        data_path = os.path.join(folder_name, filename)
        graph_name = (data_path.split('/')[-1]).replace('.txt', '')
        adj_tensor = load_adj(data_path)
        adj_dict[graph_name] = adj_tensor
    return adj_dict


def load_adj(filename):
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
    adj_matrix_numpy = nx.to_numpy_array(g, weight='weight')
    A = torch.tensor(adj_matrix_numpy, dtype=torch.float,device=DEVICE)

    return A

def load_sol_result(sol_path):
    sol_dict = {}
    result_dict = {}
    for filename in os.listdir(sol_path):
        data_path = os.path.join(sol_path, filename)
        graph_name = (data_path.split('/')[-1]).replace('.npy', '')
        np_sol_result = np.load(data_path)
        sol = torch.tensor(np_sol_result[0]).to(DEVICE).float()
        result = torch.from_numpy(np_sol_result[1:31]).to(DEVICE).float()
        sol_dict[graph_name],result_dict[graph_name] = sol,result
    return sol_dict,result_dict


def load_data(data_folder):
    adj_dict = load_adj_matrix_from_folder(data_folder+'/graph')
    sol_dict, result_dict = load_sol_result(data_folder+'/result')
    data_list = [{"sol": sol_dict[key], "adj": adj_dict[key] , "result": result_dict[key],"key":key} for key in adj_dict]
    return data_list

def load_data_from_txt(filename):
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
    A = torch.tensor(adj_matrix_numpy, dtype=torch.float,device=DEVICE)

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

