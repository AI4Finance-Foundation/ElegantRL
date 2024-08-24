import sys
sys.path.append('../')
import cvxpy as cp
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import os
import time
from typing import List

from util_obj import obj_maxcut
from util_read_data import read_nxgraph
from util import (calc_txt_files_with_prefix,
                  )
from util_result import (write_result3,
                         )
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def draw_graph(G, colors, pos):
    default_axes = plt.axes(frameon=True)
    nx.draw_networkx(G, node_color=colors, node_size=400, alpha=0.8, ax=default_axes, pos=pos)
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)
    plt.show()




# approx ratio 0.87
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
    for prefix in prefixes:
        files = calc_txt_files_with_prefix(directory_data, prefix)
        files.sort()
        for i in range(len(files)):
            start_time = time.time()
            filename = files[i]
            print(f'The {i}-th file: {filename}')
            score, solution = alg(filename)
            scores.append(score)
            print(f"score: {score}")
            running_duration = time.time() - start_time
            graph = read_nxgraph(filename)
            num_nodes = int(graph.number_of_nodes())
            write_result3(score, running_duration, num_nodes, alg_name, solution, filename)
    return scores


if __name__ == '__main__':
    # n = 5
    # graph = nx.Graph()
    # graph.add_nodes_from(np.arange(0, 4, 1))
    #
    # edges = [(1, 2), (1, 3), (2, 4), (3, 4), (3, 0), (4, 0)]
    # # edges = [(0,1),(1,2),(2,3),(3,4)]#[(1,2),(2,3),(3,4),(4,5)]
    # graph.add_edges_from(edges)


    # graph = read_nxgraph('../data/syn/syn_50_176.txt')
    # filename = '../data/gset/gset_14.txt'
    filename = '../data/syn/syn_50_176.txt'
    sdp_maxcut(filename)

    alg = sdp_maxcut
    alg_name = 'sdp'
    directory_data = '../data/syn_BA'
    prefixes = ['barabasi_albert_100']
    scores = run_sdp_over_multiple_files(alg, alg_name, directory_data, prefixes)
    print(f"scores: {scores}")

