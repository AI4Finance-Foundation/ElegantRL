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

from util import (transfer_nxgraph_to_adjacencymatrix,
                  )

from util_read_data import (read_nxgraph,
                            read_set_cover_data
                            )
from util_generate import generate_write_adjacencymatrix_and_nxgraph
# max total cuts
def obj_maxcut(result: Union[Tensor, List[int], np.array], graph: nx.Graph):
    num_nodes = len(result)
    obj = 0
    adj_matrix = transfer_nxgraph_to_adjacencymatrix(graph)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if result[i] != result[j]:
                obj += adj_matrix[(i, j)]
    return obj

# min total cuts
def obj_graph_partitioning(solution: Union[Tensor, List[int], np.array], graph: nx.Graph):
    num_nodes = len(solution)
    obj = 0
    adj_matrix = transfer_nxgraph_to_adjacencymatrix(graph)
    sum1 = 0
    for i in range(num_nodes):
        if solution[i] == 0:
            sum1 += 1
        for j in range(i + 1, num_nodes):
            if solution[i] != solution[j]:
                obj -= adj_matrix[(i, j)]
    if sum1 != num_nodes / 2:
        return -INF
    return obj

def cover_all_edges(solution: List[int], graph: nx.Graph):
    if graph.number_of_nodes() == 0:
        return False
    cover_all = True
    for node1, node2 in graph.edges:
        if solution[node1] == 0 and solution[node2] == 0:
            cover_all = False
            break
    return cover_all

def obj_minimum_vertex_cover(solution: Union[Tensor, List[int], np.array], graph: nx.Graph, need_check_cover_all_edges=True):
    num_nodes = len(solution)
    obj = 0
    for i in range(num_nodes):
        if solution[i] == 1:
            obj -= 1
    if need_check_cover_all_edges:
        if not cover_all_edges(solution, graph):
                return -INF
    return obj

# make sure solution[i] = 0 or 1
def obj_maximum_independent_set(solution: Union[Tensor, List[int], np.array], graph: nx.Graph):
    sol = set(solution)
    # if len(solution) > 0:
    #     assert len(sol) == 2
    max_elem = max(sol)
    min_elem = min(sol)
    if max_elem == min_elem:
        max_elem += 1
    obj = 0
    edges = list(graph.edges)
    num_nodes = int(graph.number_of_nodes())
    for i, j in edges:
        if solution[i] == max_elem and solution[j] == max_elem:
            return -INF
    for i in range(num_nodes):
        if solution[i] == max_elem:
            obj += 1
    return obj

# the returned score, the higher, the better
def obj_maximum_independent_set_SA(node: int, solution: Union[Tensor, List[int], np.array], graph: nx.Graph):
    def adjacent_to_selected_nodes(node: int, solution: Union[Tensor, List[int], np.array]):
        for i in range(len(solution)):
            if solution[i] == 1:
                min_node = min(node, i)
                max_node = max(node, i)
                if (min_node, max_node) in graph.edges():
                    return True
        return False
    num_edges = graph.number_of_edges()
    if solution[node] == 0:  # 0 -> 1
        if adjacent_to_selected_nodes(node, solution):
            score = -INF
        else:
            score = 1 - graph.degree(node) / num_edges
    else:  # 1 -> 0
        score = 1 + graph.degree(node) / num_edges
    return score

# the ratio of items that covered. 1.0 is the max returned value.
def obj_set_cover_ratio(solution: Union[Tensor, List[int], np.array], num_items: int, item_matrix: List[List[int]]):
    num_sets = len(solution)
    covered_items = set()
    for i in range(num_sets):
        assert solution[i] in [0, 1]
        if solution[i] == 1:
            for j in range(len(item_matrix[i])):
                covered_items.add(item_matrix[i][j])
    num_covered = 0
    items = set(np.array(range(num_items)) + 1)
    for i in covered_items:
        if i in items:
            num_covered += 1
    obj = float(num_covered) / float(num_items)
    return obj

# return negative value. the smaller abs of obj, the better.
def obj_set_cover(solution: Union[Tensor, List[int], np.array], num_items: int, item_matrix: List[List[int]]):
    num_sets = len(solution)
    covered_items = set()
    selected_sets = []
    for i in range(num_sets):
        assert solution[i] in [0, 1]
        if solution[i] == 1:
            selected_sets.append(i + 1)
            for j in range(len(item_matrix[i])):
                covered_items.add(item_matrix[i][j])
    num_covered = 0
    items = set(np.array(range(num_items)) + 1)
    for i in covered_items:
        if i in items:
            num_covered += 1
    if num_covered == num_items:
        obj = -len(selected_sets)
    else:
        obj = -INF
    return obj

def obj_graph_coloring(solution: Union[Tensor, List[int], np.array], graph: nx.Graph) -> int:
    assert None not in solution
    assert len(solution) == graph.number_of_nodes()
    for node1, node2 in graph.edges:
        if solution[node1] == solution[node2]:
            return -INF
    colors = set()
    for node in range(len(solution)):
        color = solution[node]
        colors.add(color)
    num_colors = len(colors)
    return -num_colors

if __name__ == '__main__':
    generate_read = False
    if generate_read:
        adj_matrix, graph3 = generate_write_adjacencymatrix_and_nxgraph(6, 8)
        graph4 = read_nxgraph('../data/syn_6_8.txt')
        result = [0] * 6
        obj_maxcut(result, graph4)

    if_test_read_set_cover = False
    filename = '../data/set_cover/frb45-21-5.msc'
    if if_test_read_set_cover:
        num_items, num_sets, item_matrix = read_set_cover_data(filename)
        print(f'num_items: {num_items}, num_sets: {num_sets}, item_matrix: {item_matrix}')
        solution1 = [1] * num_sets
        obj1 = obj_set_cover_ratio(solution1, num_items, item_matrix)
        print(f'obj1: {obj1}')
        solution2 = [1] * (num_sets // 2) + [0] * (num_sets - num_sets // 2)
        obj2 = obj_set_cover_ratio(solution2, num_items, item_matrix)
        print(f'obj2: {obj2}')

        solution3 = [0] * num_sets
        obj3 = obj_set_cover_ratio(solution3, num_items, item_matrix)
        print(f'obj3: {obj3}')