import sys
import os
cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../rlsolver')
sys.path.append(os.path.dirname(rlsolver_path))

import cvxpy as cp
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import os
import time
from typing import List

from rlsolver.methods.util_obj import obj_maxcut
from rlsolver.methods.util_read_data import read_nxgraph
from rlsolver.methods.util import (calc_txt_files_with_prefixes,
                  )
from rlsolver.methods.util_result import (write_graph_result,
                                          )
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# approx ratio 1/0.87
# goemans_williamson alg
def sdp_maxcut(filename: str):
    graph = read_nxgraph(filename)
    n = graph.number_of_nodes() # num of nodes
    edges = graph.edges

    x = cp.Variable((n,n), symmetric = True) #construct n x n matrix

    # diagonals must be 1 (unit) and eigenvalues must be postivie
    # semidefinite
    constraints = [x >> 0] + [ x[i,i] == 1 for i in range(n) ]

    #this is function defing the cost of the cut. You want to maximize this function
    #to get heaviest cut
    objective = sum( (0.5)* (1 - x[i,j]) for (i,j) in edges)

    # solves semidefinite program, optimizes linear cost function
    prob = cp.Problem(cp.Maximize(objective), constraints)
    prob.solve()

    # normalizes matrix, makes it applicable in unit sphere
    sqrtProb = scipy.linalg.sqrtm(x.value)

    #generates random hyperplane used to split set of points into two disjoint sets of nodes
    hyperplane = np.random.randn(n)

    #gives value -1 if on one side of plane and 1 if on other
    #returned as a array
    sqrtProb = np.sign( sqrtProb @ hyperplane)
    # print(sqrtProb)

    colors = ["r" if sqrtProb[i] == -1 else "c" for i in range(n)]
    solution = [0 if sqrtProb[i] == -1 else 1 for i in range(n)]

    pos = nx.spring_layout(graph)
    # draw_graph(graph, colors, pos)
    score = obj_maxcut(solution, graph)
    print("obj: ", score, ",solution = " + str(solution))
    return score, solution

def run_sdp_over_multiple_files(alg, alg_name, directory_data: str, prefixes: List[str])-> List[List[float]]:
    scores = []
    files = calc_txt_files_with_prefixes(directory_data, prefixes)
    files.sort()
    for i in range(len(files)):
        start_time = time.time()
        filename = files[i]
        print(f'Start the {i}-th file: {filename}')
        score, solution = alg(filename)
        scores.append(score)
        print(f"score: {score}")
        running_duration = time.time() - start_time
        graph = read_nxgraph(filename)
        num_nodes = int(graph.number_of_nodes())
        write_graph_result(score, running_duration, num_nodes, alg_name, solution, filename)
    return scores


if __name__ == '__main__':
    # n = 5
    # graph = nx.Graph()
    # graph.add_nodes_from(np.arange(0, 4, 1))
    #
    # edges = [(1, 2), (1, 3), (2, 4), (3, 4), (3, 0), (4, 0)]
    # # edges = [(0,1),(1,2),(2,3),(3,4)]#[(1,2),(2,3),(3,4),(4,5)]
    # graph.add_edges_from(edges)


    # filename = '../data/gset/gset_14.txt'
    run_single_file = False
    if run_single_file:
        filename = '../data/syn_BA/BA_100_ID0.txt'
        sdp_maxcut(filename)

    run_multi_files = True
    if run_multi_files:
        alg = sdp_maxcut
        alg_name = 'sdp'
        directory_data = '../data/syn_BA'
        prefixes = ['barabasi_albert_100']
        scores = run_sdp_over_multiple_files(alg, alg_name, directory_data, prefixes)
        print(f"scores: {scores}")

