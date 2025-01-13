import sys
import os
cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../rlsolver')
sys.path.append(os.path.dirname(rlsolver_path))


import copy
import time
from typing import Union, Optional
import numpy as np
import multiprocessing as mp
import networkx as nx
from rlsolver.methods.util import (plot_fig,
                  plot_nxgraph,
                  transfer_nxgraph_to_weightmatrix,
                  calc_txt_files_with_prefixes,
                )
from rlsolver.methods.util_read_data import (read_nxgraph,
                            )
from rlsolver.methods.util_obj import (cover_all_edges,
                  obj_maxcut,
                  obj_graph_partitioning,
                  obj_minimum_vertex_cover,
                  obj_maximum_independent_set,
                  obj_set_cover_ratio,
                  obj_set_cover,
                  obj_graph_coloring,)
from rlsolver.methods.util_result import (write_graph_result,
                                          )
from rlsolver.methods.config import *

# init_solution is useless
def greedy_maxcut(num_steps: Optional[int], graph: nx.Graph, filename) -> (int, Union[List[int], np.array], List[int]):
    print('greedy')
    start_time = time.time()
    num_nodes = int(graph.number_of_nodes())
    nodes = list(range(num_nodes))
    init_solution = [0] * graph.number_of_nodes()
    assert sum(init_solution) == 0
    if num_steps is None:
        num_steps = num_nodes
    curr_solution = copy.deepcopy(init_solution)
    curr_score: int = obj_maxcut(curr_solution, graph)
    init_score = curr_score
    scores = []
    for iteration in range(num_nodes):
        if iteration >= num_steps:
            break
        score = obj_maxcut(curr_solution, graph)
        print(f"iteration: {iteration}, score: {score}")
        traversal_scores = []
        traversal_solutions = []
        # calc the new solution when moving to a new node. Then store the scores and solutions.
        for node in nodes:
            new_solution = copy.deepcopy(curr_solution)
            # search a new solution and calc obj
            new_solution[node] = (new_solution[node] + 1) % 2
            new_score = obj_maxcut(new_solution, graph)
            traversal_scores.append(new_score)
            traversal_solutions.append(new_solution)
        best_score = max(traversal_scores)
        index = traversal_scores.index(best_score)
        best_solution = traversal_solutions[index]
        if best_score > curr_score:
            scores.append(best_score)
            curr_score = best_score
            curr_solution = best_solution
        else:
            break
    print("init_score, final score of greedy", init_score, curr_score, )
    print("scores: ", traversal_scores)
    print("solution: ", curr_solution)
    running_duration = time.time() - start_time
    print('running_duration: ', running_duration)
    alg_name = "greedy"
    write_graph_result(score, running_duration, num_nodes, alg_name, curr_solution, filename)
    return curr_score, curr_solution, scores

def greedy_graph_partitioning(num_steps:Optional[int], graph: nx.Graph) -> (int, Union[List[int], np.array], List[int]):
    print('greedy')
    init_solution = [0] * int(graph.number_of_nodes() / 2) + [1] * int(graph.number_of_nodes() / 2)
    num_nodes = int(graph.number_of_nodes())
    if num_steps is None:
        num_steps = num_nodes
    start_time = time.time()
    nodes = list(range(num_nodes))
    curr_solution = copy.deepcopy(init_solution)
    curr_score: int = obj_graph_partitioning(curr_solution, graph)
    init_score = curr_score
    scores = []
    for i in range(num_steps):
        node1 = nodes[i]
        traversal_scores = []
        traversal_solutions = []
        for j in range(i + 1, num_steps):
            node2 = nodes[j]
            if curr_solution[node1] == curr_solution[node2]:
                continue
            new_solution = copy.deepcopy(curr_solution)
            tmp = new_solution[node1]
            new_solution[node1] = new_solution[node2]
            new_solution[node2] = tmp
            new_score = obj_graph_partitioning(new_solution, graph)
            traversal_scores.append(new_score)
            traversal_solutions.append(new_solution)
        if len(traversal_scores) == 0:
            continue
        best_score = max(traversal_scores)
        index = traversal_scores.index(best_score)
        best_solution = traversal_solutions[index]
        if best_score > curr_score:
            scores.append(best_score)
            curr_score = best_score
            curr_solution = best_solution
        else:
            break
    print("init_score, final score of greedy", init_score, curr_score, )
    print("scores: ", scores)
    print("solution: ", curr_solution)
    running_duration = time.time() - start_time
    print('running_duration: ', running_duration)
    return curr_score, curr_solution, scores


def greedy_minimum_vertex_cover(num_steps: int, graph: nx.Graph) -> (int, Union[List[int], np.array], List[int]):
    print('greedy')
    init_solution = [0] * graph.number_of_nodes()
    assert sum(init_solution) == 0
    assert num_steps is None
    start_time = time.time()
    num_nodes = int(graph.number_of_nodes())
    nodes = list(range(num_nodes))
    curr_solution = copy.deepcopy(init_solution)
    curr_score: int = obj_minimum_vertex_cover(curr_solution, graph)
    init_score = curr_score
    scores = []
    iter = 0
    unselected_nodes = list(graph.nodes())
    while True:
        cover_all = cover_all_edges(curr_solution, graph)
        if cover_all:
            break
        max_degree = 0
        best_node = -INF
        for node in unselected_nodes:
            degree = graph.degree(node)
            if degree > max_degree:
                max_degree = degree
                best_node = node
        if max_degree > 0:
            curr_solution[best_node] = 1
            unselected_nodes.remove(best_node)
        iter += 1
        if iter > num_nodes:
            break
    curr_score = obj_minimum_vertex_cover(curr_solution, graph)
    print("score, init_score", curr_score, init_score)
    print("solution: ", curr_solution)
    running_duration = time.time() - start_time
    print('running_duration: ', running_duration)
    return curr_score, curr_solution, scores

def greedy_maximum_independent_set(num_steps: Optional[int], graph: nx.Graph) -> (int, Union[List[int], np.array], List[int]):
    def calc_candidate_nodes(unselected_nodes: List[int], selected_nodes: List[int], graph: nx.Graph):
        candidate_nodes = []
        remove_nodes = set()
        for node1, node2 in graph.edges():
            if node1 in selected_nodes:
                remove_nodes.add(node2)
            elif node2 in selected_nodes:
                remove_nodes.add(node1)
        for node in unselected_nodes:
            if node not in remove_nodes:
                candidate_nodes.append(node)
        return candidate_nodes
    print('greedy')
    num_nodes = int(graph.number_of_nodes())
    nodes = list(range(num_nodes))
    init_solution = [0] * num_nodes
    if num_steps is None:
        num_steps = num_nodes
    start_time = time.time()
    curr_solution = copy.deepcopy(init_solution)
    curr_score: int = obj_maximum_independent_set(curr_solution, graph)
    init_score = curr_score
    scores = []
    selected_nodes = []
    unselected_nodes = copy.deepcopy(nodes)
    candidate_graph = copy.deepcopy(graph)
    # extend_candidate_graph = copy.deepcopy(graph)
    step = 0
    while True:
        step += 1
        candidate_nodes = calc_candidate_nodes(unselected_nodes, selected_nodes, graph)
        if len(candidate_nodes) == 0:
            break
        min_degree = num_nodes
        selected_node = None
        for node in candidate_nodes:
            degree = candidate_graph.degree(node)
            if degree < min_degree:
                min_degree = degree
                selected_node = node
        if selected_node is None:
            break
        else:
            selected_nodes.append(selected_node)
            unselected_nodes.remove(selected_node)
            candidate_graph.remove_node(selected_node)
            curr_solution[selected_node] = 1
            # curr_score2 = obj_maximum_independent_set(curr_solution, graph)
            curr_score += 1
            # assert curr_score == curr_score2
            scores.append(curr_score)
        if step > num_steps:
            break
    curr_score2 = obj_maximum_independent_set(curr_solution, graph)
    assert curr_score == curr_score2
    print("init_score, final score of greedy", init_score, curr_score, )
    print("scores: ", scores)
    print("solution: ", curr_solution)
    running_duration = time.time() - start_time
    print('running_duration: ', running_duration)
    return curr_score, curr_solution, scores

def greedy_set_cover(num_items: int, num_sets: int, item_matrix: List[List[int]]) -> (int, Union[List[int], np.array], List[int]):
    print('greedy')
    start_time = time.time()
    curr_solution = [0] * num_sets
    init_score = 0.0
    curr_score = 0.0
    scores = []
    selected_sets = []
    unselected_sets = set(np.array(range(num_sets)) + 1)
    unselected_items = set(np.array(range(num_items)) + 1)
    while len(unselected_items) > 0:
        max_intersection_num = 0
        selected_set = None
        for i in unselected_sets:
            intersection_num = 0
            for j in item_matrix[i - 1]:
                if j in unselected_items:
                    intersection_num += 1
            if intersection_num > max_intersection_num:
                max_intersection_num = intersection_num
                selected_set = i
        if selected_set is not None:
            selected_sets.append(selected_set)
            unselected_sets.remove(selected_set)
            for j in item_matrix[selected_set - 1]:
                if j in unselected_items:
                    unselected_items.remove(j)
            curr_score += max_intersection_num / num_items
            scores.append(curr_score)
            curr_solution[selected_set - 1] = 1
    real_score = obj_set_cover(curr_solution, num_items, item_matrix)
    print("real score of greedy:", real_score)
    print(f'num_sets: {num_sets}, num_items: {num_items}')
    print("init_score, final score of greedy", init_score, curr_score)
    print("scores: ", scores)
    print("solution: ", curr_solution)
    running_duration = time.time() - start_time
    print('running_duration: ', running_duration)
    return real_score, curr_solution, scores

def greedy_graph_coloring(num_steps: Optional[int], graph: nx.Graph) -> (int, Union[List[int], np.array], List[int]):
    print('greedy')
    start_time = time.time()
    num_nodes = int(graph.number_of_nodes())
    nodes = list(range(num_nodes))
    solution = [None] * graph.number_of_nodes()
    num_used_colors = 0
    assert num_steps is None
    # color ID: start from 1, not 0
    for node in nodes:
        if node == 0:
            solution[node] = num_used_colors + 1
            num_used_colors += 1
        else:
            if node == 10:
                aaa = 1
            neighbor_colors = set()
            for i in graph.neighbors(node):
                if solution[i] is not None:
                    neighbor_colors.add(solution[i])
            if len(neighbor_colors) == num_used_colors:
                solution[node] = num_used_colors + 1
                num_used_colors += 1
            else:
                used_colors = list(range(1, num_used_colors + 1))
                dic = {}  # key: color ID, value: used times
                for i in range(node):
                    color = solution[i]
                    if i not in graph.neighbors(node) and color not in neighbor_colors:
                        if color in dic:
                            dic[color] += 1
                        else:
                            dic[color] = 1
                min_used_times = num_nodes
                color_id = None
                for key, value in dic.items():
                    if value < min_used_times:
                        min_used_times = value
                        color_id = int(key)

                solution[node] = color_id

    print("solution: ", solution)
    running_duration = time.time() - start_time
    print('running_duration: ', running_duration)
    curr_score = obj_graph_coloring(solution, graph)
    curr_solution = solution
    scores = [curr_score]
    return curr_score, curr_solution, scores

# def run_greedy_over_multiple_files(alg, alg_name, num_steps, set_init_0: Optional[bool], directory_data: str, prefixes: List[str])-> List[List[float]]:
def run_greedy_over_multiple_files(alg, alg_name, num_steps, directory_data: str, prefixes: List[str])-> List[List[float]]:
    from util_read_data import (read_set_cover_data, read_nxgraph)
    from util_result import write_result_set_cover
    scoress = []
    files = calc_txt_files_with_prefixes(directory_data, prefixes)
    for i in range(len(files)):
        start_time = time.time()
        filename = files[i]
        print(f'Start the {i}-th file: {filename}')
        if PROBLEM == Problem.set_cover:
            from greedy import greedy_set_cover
            num_items, num_sets, item_matrix = read_set_cover_data(filename)
            score, solution, scores = greedy_set_cover(num_items, num_sets, item_matrix)
            scoress.append(scores)
            running_duration = time.time() - start_time
            write_result_set_cover(score, running_duration, num_items, num_sets, alg_name, filename)
        else:
            graph = read_nxgraph(filename)
            score, solution, scores = alg(num_steps, graph, filename)
            scoress.append(scores)
            running_duration = time.time() - start_time
            write_graph_result(score, running_duration, graph.number_of_nodes(), alg_name, solution, filename)
    return scoress

if __name__ == '__main__':
    # read data
    print(f'problem: {PROBLEM}')
    filename = '../data/syn_BA/BA_100_ID0.txt'
    graph = read_nxgraph(filename)
    weightmatrix = transfer_nxgraph_to_weightmatrix(graph)
    # run alg
    alg_name = 'GR'

    run_one_file = False
    if run_one_file:
        # maxcut
        if PROBLEM == Problem.maxcut:
            num_steps = None
            gr_score, gr_solution, gr_scores = greedy_maxcut(num_steps, graph, filename)

        # graph_partitioning
        elif PROBLEM == Problem.graph_partitioning:
            num_steps = None
            gr_score, gr_solution, gr_scores = greedy_graph_partitioning(num_steps, graph)

        elif PROBLEM == Problem.minimum_vertex_cover:
            num_steps = None
            gr_score, gr_solution, gr_scores = greedy_minimum_vertex_cover(num_steps, graph)
            obj = obj_minimum_vertex_cover(gr_solution, graph)
            print('obj: ', obj)

        elif PROBLEM == Problem.maximum_independent_set:
            num_steps = None
            gr_score, gr_solution, gr_scores = greedy_maximum_independent_set(num_steps, graph)
            obj = obj_maximum_independent_set(gr_solution, graph)
            print('obj: ', obj)

        elif PROBLEM == Problem.set_cover:
            from util import read_set_cover_data
            filename = '../data/set_cover/frb30-15-1.msc'
            num_items, num_sets, item_matrix = read_set_cover_data(filename)
            print(f'num_items: {num_items}, num_sets: {num_sets}, item_matrix: {item_matrix}')
            solution1 = [1] * num_sets
            obj1_ratio = obj_set_cover_ratio(solution1, num_items, item_matrix)
            print(f'obj1_ratio: {obj1_ratio}')
            curr_score, curr_solution, scores = greedy_set_cover(num_items, num_sets, item_matrix)
            print(f'curr_score: {curr_score}, curr_solution:{curr_solution}, scores:{scores}')

        elif PROBLEM == Problem.graph_coloring:
            num_steps = None
            gr_score, gr_solution, gr_scores = greedy_graph_coloring(num_steps, graph)
            fig_filename = '../result/fig.png'
            plot_nxgraph(graph, fig_filename)

    run_multi_files = True
    if run_multi_files:
        if PROBLEM == Problem.maxcut:
            alg = greedy_maxcut
        elif PROBLEM == Problem.graph_partitioning:
            alg = greedy_graph_partitioning
        elif PROBLEM == Problem.minimum_vertex_cover:
            alg = greedy_minimum_vertex_cover
        elif PROBLEM == Problem.maximum_independent_set:
            alg = greedy_maximum_independent_set
        elif PROBLEM == Problem.set_cover:
            alg = greedy_set_cover
        elif PROBLEM == Problem.graph_coloring:
            alg = greedy_graph_coloring

        alg_name = "greedy"
        num_steps = None
        directory_data = '../data/syn_BA'
        # directory_data = '../data/syn_ER'
        prefixes = ['BA_100_']
        # prefixes = ['ER_100_']

        if_run_set_cover = False
        if if_run_set_cover:
            directory_data = '../data/set_cover'
            prefixes = ['frb30-15-1']

        scoress = run_greedy_over_multiple_files(alg, alg_name, num_steps, directory_data, prefixes)
        print(f"scoress: {scoress}")

        # plot fig
        plot_fig_ = False
        if plot_fig_:
            for scores in scoress:
                plot_fig(scores, alg_name)




