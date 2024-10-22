import sys
import os
cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../rlsolver')
sys.path.append(os.path.dirname(rlsolver_path))

import random
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import copy
import time
import numpy as np
import networkx as nx
from rlsolver.methods.config import *

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from rlsolver.methods.util import (calc_txt_files_with_prefixes,
                                    )
from rlsolver.methods.util_read_data import (read_nxgraph,
                                            )
from rlsolver.methods.util_obj import (obj_maxcut,
                                        )
from rlsolver.methods.util_result import (write_graph_result,
                                         )

# constants for tabuSearch
P_iter = 100
MaxIter = 10000
gamma = 65


def generate_random(graph):
    nodes = list(graph.nodes())
    binary_vector = [random.randint(0, 1) for _ in range(len(nodes))]
    return tabu_search(binary_vector, graph)


def generate_random_population(graph, pop_size):
    count = 1
    Pop = []
    best_binary_vector = []
    score_list = []
    best_score = 0
    while len(Pop) < pop_size:
        binary_vector, result = generate_random(graph)
        score = result

        if binary_vector not in Pop:
            score_list.append(score)
            Pop.append(binary_vector)
            print(count, "Score: ", score)
            count += 1
            if score > best_score:
                best_score = score
                best_binary_vector = binary_vector
    return Pop, best_binary_vector, best_score, score_list


def tenure(iteration, maxT):
    # define the sequence of values for the tenure function
    a = [maxT * bi for bi in [1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1]]
    # define the sequence of interval margins
    x = [1] + [x + 4 * maxT * bi for x, bi in zip([1] * 15, [1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1])]
    # find the interval to determine the tenure
    interval = next(i for i, xi in enumerate(x) if xi > iteration) - 1
    return a[interval]


def compute_move_gains(graph, vector, tabu_list):
    move_gains = []
    for i in range(len(vector)):
        delta_v = 0
        neighbor_nodes = list(graph.neighbors(i))
        for j in neighbor_nodes:
            if vector[i] == vector[j]:
                delta_v += float(graph[i][j]["weight"])
            else:
                delta_v -= float(graph[i][j]["weight"])
        move_gains.append(delta_v)

    return move_gains


def update_move_gains(node_flipped, move_gains, vector, graph):
    neighbors = list(graph.neighbors(node_flipped))
    for i in neighbors:
        if vector[i] == vector[node_flipped]:
            move_gains[i] += 2 * float(graph[i][node_flipped]["weight"])
        else:
            move_gains[i] -= 2 * float(graph[i][node_flipped]["weight"])
    move_gains[node_flipped] = -move_gains[node_flipped]
    return move_gains


def perturb(binary_vector):
    # randomly select gamma vertices to move
    vertices_to_move = random.sample(range(len(binary_vector)), gamma)

    # flip the subsets for the selected vertices
    for vertex in vertices_to_move:
        binary_vector[vertex] = 1 - binary_vector[vertex]

    return binary_vector


def tabu_search(initial_solution, graph):
    # initialize best solution and its score
    best_solution = initial_solution
    best_score = obj_maxcut(initial_solution, graph)
    curr_solution = copy.deepcopy(initial_solution)
    curr_score = best_score
    # initialize iteration counter
    Iter = 0
    pit = 0

    # initialize tabu list and tabu tenure
    tabu_list = [0] * len(curr_solution)
    maxT = 150

    # compute move gains
    move_gains = compute_move_gains(graph, curr_solution, tabu_list)
    while Iter < MaxIter:
        v = 0
        delta_v = -999999
        for i in range(0, len(move_gains)):
            if delta_v < move_gains[i] and tabu_list[i] <= Iter:
                delta_v = move_gains[i]
                v = i

        # move v from its original subset to the opposite set
        curr_solution[v] = 1 - curr_solution[v]
        curr_score += delta_v
        # print("Current ",curr_score)
        # print("Actual ",obj_maxcut(curr_solution,graph))
        # update tabu list and move gains for each vertex v ∈ V
        tabu_list[v] = maxT + Iter
        move_gains = update_move_gains(v, move_gains, curr_solution, graph)

        # update best solution if current solution is better
        if curr_score > best_score:
            best_solution = copy.deepcopy(curr_solution)
            best_score = curr_score
            pit = 0

        # increment iteration counter
        Iter += 1
        pit += 1
        # check if best solution hasn't improved after P_iter iterations
        if pit == P_iter and curr_score <= best_score:
            pit = 0
            curr_solution = perturb(curr_solution)
            curr_score = obj_maxcut(curr_solution, graph)
            tabu_list = [0] * len(curr_solution)
            move_gains = compute_move_gains(graph, curr_solution, tabu_list)

    return best_solution, best_score



def cross_over(population, graph):
    selected_parents = random.sample(population, num_parents)

    child = []
    for node in range(0, len(selected_parents[0])):
        node_in_same_set = all(parent[node] == selected_parents[0][node] for parent in selected_parents)

        if node_in_same_set:
            child.append(selected_parents[0][node])
        else:
            child.append(random.randint(0, 1))
    child, child_score = tabu_search(child, graph)
    return child


def genetic_maxcut(graph: nx.Graph(), filename):
    start_time = time.time()
    population, best_binary_vector, best_score, population_scores = generate_random_population(graph, 10)
    c_iter = 0
    print("Start Genetic Crossover")
    while c_iter < c_itMax:
        child = cross_over(population, graph)
        if (child not in population):
            child_score = obj_maxcut(child, graph)

            print(c_iter + 1, " Childs Score: ", child_score)

            # Finding the min score in the list and replacing it with the child if smaller than child cut
            min_score_index = np.argmin(population_scores)
            if (population_scores[min_score_index] < child_score):
                population_scores[min_score_index] = child_score
                population[min_score_index] = child
            c_iter += 1

    max_score_index = np.argmax(population_scores)
    obj = population_scores[max_score_index]
    best_solution = population[max_score_index]
    print("solution : ", best_solution)
    print("obj: ", obj)

    running_duration = time.time() - start_time
    num_nodes = graph.number_of_nodes()
    alg_name = "genetic algorithm"
    write_graph_result(obj, running_duration, num_nodes, alg_name, best_solution, filename)

    print("Genetic Search Complete")



def run_genetic_over_multiple_files(directory_data: str, prefixes: List[str])-> List[List[float]]:
    assert PROBLEM == Problem.maxcut
    scoress = []
    files = calc_txt_files_with_prefixes(directory_data, prefixes)
    files.sort()
    for i in range(len(files)):
        filename = files[i]
        print(f'Start the {i}-th file: {filename}')
        graph = read_nxgraph(filename)
        print("Genetic Search Start")
        genetic_maxcut(graph, filename)
    return scoress


if __name__ == '__main__':
    # Constants
    num_parents = 5
    c_itMax = 5

    if_run_one_case = False
    if if_run_one_case:
        # read data
        filename = '../data/syn_BA/BA_100_ID0.txt'
        graph = read_nxgraph(filename)
        print("Genetic Search Start")
        genetic_maxcut(graph, filename)

    else:
        directory_data = '../data/syn_BA'
        # directory_data = '../data/syn_ER'
        # directory_data = '../data/syn'
        prefixes = ['BA_100_']
        run_genetic_over_multiple_files(directory_data, prefixes)

    # Cut checker
    # vector = [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1]
    # print(obj_maxcut(vector,graph))
