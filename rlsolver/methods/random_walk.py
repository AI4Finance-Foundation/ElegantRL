import sys
import os
cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../rlsolver')
sys.path.append(os.path.dirname(rlsolver_path))

import copy
import time
import networkx as nx
import numpy as np
from typing import List, Union
import random
from rlsolver.methods.util_read_data import read_nxgraph
from rlsolver.methods.util_obj import obj_maxcut
from rlsolver.methods.util_result import write_graph_result
from rlsolver.methods.util import plot_fig

import sys
sys.path.append('../')

def random_walk_maxcut(init_solution: Union[List[int], np.array], num_steps: int, graph: nx.Graph) -> (int, Union[List[int], np.array], List[int]):
    print('random_walk')
    start_time = time.time()
    curr_solution = copy.deepcopy(init_solution)
    init_score = obj_maxcut(init_solution, graph)
    num_nodes = len(curr_solution)
    scores = []
    for i in range(num_steps):
        # select a node randomly
        node = random.randint(0, num_nodes - 1)
        curr_solution[node] = (curr_solution[node] + 1) % 2
        # calc the obj
        score = obj_maxcut(curr_solution, graph)
        scores.append(score)
    print("score, init_score of random_walk", score, init_score)
    print("scores: ", scores)
    print("solution: ", curr_solution)
    running_duration = time.time() - start_time
    print('running_duration: ', running_duration)
    return score, curr_solution, scores


if __name__ == '__main__':
    # read data
    # graph1 = read_as_networkx_graph('data/gset_14.txt')
    start_time = time.time()
    filename = '../data/syn_BA/BA_100_ID0.txt'
    graph = read_nxgraph(filename)

    # run alg
    # init_solution = [1, 0, 1, 0, 1]
    init_solution = list(np.random.randint(0, 2, graph.number_of_nodes()))
    rw_score, rw_solution, rw_scores = random_walk_maxcut(init_solution=init_solution, num_steps=1000, graph=graph)
    running_duration = time.time() - start_time
    num_nodes = graph.number_of_nodes
    alg_name = "random_walk"
    # write result
    write_graph_result(rw_score, running_duration, num_nodes, alg_name, rw_solution, filename)
    # write_result(rw_solution, '../result/result.txt')
    obj = obj_maxcut(rw_solution, graph)
    print('obj: ', obj)
    alg_name = 'RW'

    # plot fig
    if_plot = False
    if if_plot:
        plot_fig(rw_scores, alg_name)


