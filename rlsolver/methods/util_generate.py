import sys
import os
cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../rlsolver')
sys.path.append(os.path.dirname(rlsolver_path))

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import copy
import numpy as np
from typing import Union, Tuple, List
import networkx as nx
import torch as th

from rlsolver.methods.util import (transfer_weightmatrix_to_nxgraph,
                                                      )
from rlsolver.methods.config import GraphType

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from rlsolver.methods.config import GRAPH_TYPES
def generate_graph(num_nodes: int, graph_type: GraphType):
    graph_types = GRAPH_TYPES
    assert graph_type in graph_types

    if graph_type == GraphType.ER:
        g = nx.erdos_renyi_graph(n=num_nodes, p=0.15)
    elif graph_type == GraphType.PL:
        g = nx.powerlaw_cluster_graph(n=num_nodes, m=4, p=0.05)
    elif graph_type == GraphType.BA:
        g = nx.barabasi_albert_graph(n=num_nodes, m=4)
    else:
        raise ValueError(f"g_type {graph_type} should in {graph_types}")

    graph = []
    for node0, node1 in g.edges:
        distance = 1
        graph.append((node0, node1, distance))
    num_nodes = num_nodes
    num_edges = len(graph)
    return graph, num_nodes, num_edges

# genete a graph, and output a symmetric_adjacency_matrix and networkx_graph. The graph will be written to a file.
# weight_low (inclusive) and weight_high (exclusive) are the low and high int values for weight, and should be int.
# If writing the graph to file, the node starts from 1, not 0. The first node index < the second node index. Only the non-zero weight will be written.
# If writing the graph, the file name will be revised, e.g., syn.txt will be revised to syn_n_m.txt, where n is num_nodes, and m is num_edges.
def generate_write_adjacencymatrix_and_nxgraph(num_nodes: int,
                                               num_edges: int,
                                               filename: str = '../data/syn.txt',
                                               weight_low=0,
                                               weight_high=2) -> (List[List[int]], nx.Graph):
    if weight_low == 0:
        weight_low += 1
    adjacency_matrix = []
    # generate adjacency_matrix where each row has num_edges_per_row edges
    num_edges_per_row = int(np.ceil(2 * num_edges / num_nodes))
    for i in range(num_nodes):
        indices = []
        while True:
            all_indices = list(range(0, num_nodes))
            np.random.shuffle(all_indices)
            indices = all_indices[: num_edges_per_row]
            if i not in indices:
                break
        row = [0] * num_nodes
        weights = np.random.randint(weight_low, weight_high, size=num_edges_per_row)
        for k in range(len(indices)):
            row[indices[k]] = weights[k]
        adjacency_matrix.append(row)
    # the num of edges of the generated adjacency_matrix may not be the specified, so we revise it.
    indices1 = []  # num of non-zero weights for i < j
    indices2 = []  # num of non-zero weights for i > j
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adjacency_matrix[i][j] != 0:
                if i < j:
                    indices1.append((i, j))
                else:
                    indices2.append((i, j))
    # if |indices1| > |indices2|, we get the new adjacency_matrix by swapping symmetric elements
    # based on adjacency_matrix so that |indices1| < |indices2|
    if len(indices1) > len(indices2):
        indices1 = []
        indices2 = []
        new_adjacency_matrix = copy.deepcopy(adjacency_matrix)
        for i in range(num_nodes):
            for j in range(num_nodes):
                new_adjacency_matrix[i][j] = adjacency_matrix[j][i]
                if new_adjacency_matrix[i][j] != 0:
                    if i < j:
                        indices1.append((i, j))
                    else:
                        indices2.append((i, j))
        adjacency_matrix = new_adjacency_matrix
    # We first set some elements of indices2 0 so that |indices2| = num_edges,
    # then, fill the adjacency_matrix so that the symmetric elements along diagonal are the same
    if len(indices1) <= len(indices2):
        num_set_0 = len(indices2) - num_edges
        if num_set_0 < 0:
            raise ValueError("wrong num_set_0")
        while True:
            all_ind_set_0 = list(range(0, len(indices2)))
            np.random.shuffle(all_ind_set_0)
            ind_set_0 = all_ind_set_0[: num_set_0]
            indices2_set_0 = [indices2[k] for k in ind_set_0]
            new_indices2 = set([indices2[k] for k in range(len(indices2)) if k not in ind_set_0])
            my_list = list(range(num_nodes))
            my_set: set = set()
            satisfy = True
            # check if all nodes exist in new_indices2. If yes, the condition is satisfied, and iterate again otherwise.
            for i, j in new_indices2:
                my_set.add(i)
                my_set.add(j)
            for item in my_list:
                if item not in my_set:
                    satisfy = False
                    break
            if satisfy:
                break
        for (i, j) in indices2_set_0:
            adjacency_matrix[i][j] = 0
        if len(new_indices2) != num_edges:
            raise ValueError("wrong new_indices2")
        # fill elements of adjacency_matrix based on new_indices2
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if (j, i) in new_indices2:
                    adjacency_matrix[i][j] = adjacency_matrix[j][i]
                else:
                    adjacency_matrix[i][j] = 0
    # create a networkx graph
    graph = nx.Graph()
    nodes = list(range(num_nodes))
    graph.add_nodes_from(nodes)
    num_edges = len(new_indices2)
    # create a new filename, and write the graph to the file.
    new_filename = filename.split('.')[0] + '_' + str(num_nodes) + '_' + str(num_edges) + '.txt'
    with open(new_filename, 'w', encoding="UTF-8") as file:
        file.write(f'{num_nodes} {num_edges} \n')
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                weight = int(adjacency_matrix[i][j])
                graph.add_edge(i, j, weight=weight)
                if weight != 0:
                    file.write(f'{i + 1} {j + 1} {weight}\n')
    return adjacency_matrix, graph

def generate_write_distribution(num_nodess: List[int], num_graphs: int, graph_type: GraphType, directory: str, need_write=True):
    nxgraphs = []
    for num_nodes in num_nodess:
        for i in range(num_graphs):
            weightmatrix, num_nodes, num_edges = generate_graph(num_nodes, graph_type)
            nxgraph = transfer_weightmatrix_to_nxgraph(weightmatrix, num_nodes)
            nxgraphs.append(nxgraph)
            filename = directory + '/' + graph_type.value + '_' + str(num_nodes) + '_ID' + str(i) + '.txt'
            if need_write:
                write_nxgraph(nxgraph, filename)
    return nxgraphs

def write_nxgraph(g: nx.Graph(), filename: str):
    num_nodes = nx.number_of_nodes(g)
    num_edges = nx.number_of_edges(g)
    adjacency_matrix = nx.to_numpy_array(g)
    with open(filename, 'w', encoding="UTF-8") as file:
        file.write(f'{num_nodes} {num_edges} \n')
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                weight = int(adjacency_matrix[i][j])
                g.add_edge(i, j, weight=weight)
                if weight != 0:
                    file.write(f'{i + 1} {j + 1} {weight}\n')



if __name__ == '__main__':
    if_generate_distribution = False
    if if_generate_distribution:
        num_nodess = [20, 40] + list(range(100, 201, 100))
        # num_nodess = list(range(2100, 3001, 100))
        # num_nodess = [1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
        # num_nodess = [20]
        num_graphs = 30
        graph_type = GraphType.BA
        directory = '../data/syn_BA'
        generate_write_distribution(num_nodess, num_graphs, graph_type, directory)

    # generate synthetic data
    generate_data = False
    if generate_data:
        # num_nodes_edges = [(20, 50), (30, 110), (50, 190), (100, 460), (200, 1004), (400, 1109), (800, 2078), (1000, 4368), (2000, 9386), (3000, 11695), (4000, 25654), (5000, 50543), (10000, 100457)]
        # num_nodes_edges = [(3000, 25695), (4000, 38654), (5000, 50543), (6000, 73251), (7000, 79325), (8000, 83647),
        #                    (9000, 96324), (10000, 100457), (13000, 18634), (16000, 19687), (20000, 26358)]
        num_nodes_edges = [(30, 25),]
        # num_nodes_edges = [(100, 460)]
        num_datasets = 1
        for num_nodes, num_edges in num_nodes_edges:
            for n in range(num_datasets):
                generate_write_adjacencymatrix_and_nxgraph(num_nodes, num_edges + n)
        print()

