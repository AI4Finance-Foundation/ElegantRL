import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import copy
from torch.autograd import Variable
import functools
import time
import numpy as np
from typing import Union, Tuple, List
import networkx as nx
from torch import Tensor
import torch as th
from config import *
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
TEN = th.Tensor
INT = th.IntTensor
TEN = th.Tensor
GraphList = List[Tuple[int, int, int]]
IndexList = List[List[int]]
from config import GSET_DIR
DataDir = GSET_DIR

# read graph file, e.g., gset_14.txt, as networkx.Graph
# The nodes in file start from 1, but the nodes start from 0 in our codes.
def read_nxgraph(filename: str) -> nx.Graph():
    graph = nx.Graph()
    with open(filename, 'r') as file:
        # lines = []
        line = file.readline()
        is_first_line = True
        while line is not None and line != '':
            if '//' not in line:
                if is_first_line:
                    strings = line.split(" ")
                    num_nodes = int(strings[0])
                    num_edges = int(strings[1])
                    nodes = list(range(num_nodes))
                    graph.add_nodes_from(nodes)
                    is_first_line = False
                else:
                    node1, node2, weight = line.split(" ")
                    graph.add_edge(int(node1) - 1, int(node2) - 1, weight=weight)
            line = file.readline()
    return graph

# read tsp file,latitude and longitude, horizontal and vertical coordinates
def read_tsp(filename:str) -> nx.Graph():

    graph = nx.Graph()

    with open(filename,'r') as file:
        num_nodes = int(file.readline().strip())
        positions = np.array([list(map(float,line.split(" "))) for line in file if line.strip()])

    for node_id,pos in enumerate(positions):
        graph.add_node(node_id,pos=tuple(pos))

    for i in range(num_nodes):
        dists = np.sqrt(np.sum((positions[i] - positions[i+1:num_nodes]) ** 2,axis=1))
        for j,dist in enumerate(dists,start=i + 1):
            graph.add_edge(i,j,weight = dist)

    return graph

# def read_set_cover(filename: str):
#     with open(filename, 'r') as file:
#         # lines = []
#         line = file.readline()
#         item_matrix = []
#         while line is not None and line != '':
#             if 'p set' in line:
#                 strings = line.split(" ")
#                 num_items = int(strings[-2])
#                 num_sets = int(strings[-1])
#             elif 's' in line:
#                 strings = line.split(" ")
#                 items = [int(s) for s in strings[1:]]
#                 item_matrix.append(items)
#             else:
#                 raise ValueError("error in read_set_cover")
#             line = file.readline()
#     return num_items, num_sets, item_matrix

def read_knapsack_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        N, W = map(int, lines[0].split())
        items = []
        for line in lines[1:]:
            weight, value = map(int, line.split())
            items.append((weight, value))
    return N, W, items


def read_set_cover_data(filename):
    with open(filename, 'r') as file:
        first_line = file.readline()
        total_elements, total_subsets = map(int, first_line.split())
        subsets = []
        for line in file:
            subset = list(map(int, line.strip().split()))
            subsets.append(subset)

    return total_elements, total_subsets, subsets

if __name__ == '__main__':

    read_txt = True
    if read_txt:
        graph1 = read_nxgraph('../data/gset/gset_14.txt')
        graph2 = read_nxgraph('../data/syn_5_5.txt')
