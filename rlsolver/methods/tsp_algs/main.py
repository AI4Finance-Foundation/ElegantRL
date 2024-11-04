import sys
import os
cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../../rlsolver')
sys.path.append(os.path.dirname(rlsolver_path))


import time
from typing import Union, Tuple, List
import pandas as pd
import util
from config import *
from util import plot_tour
from util import read_tsp_file
from christofides import christofides_algorithm
from ga import genetic_algorithm
from gksp import greedy_karp_steele_patching
from ins_c import cheapest_insertion
from ins_f import farthest_insertion
from nn import nearest_neighbour
from opt_2 import local_search_2_opt
from opt_3 import local_search_3_opt
from s_tabu import tabu_search
from sa import simulated_annealing_tsp
from rlsolver.methods.util_result import write_graph_result
from rlsolver.methods.util import (calc_txt_files_with_prefixes,
                                    transfer_nxgraph_to_adjacencymatrix
                                   )

from rlsolver.methods.tsp_algs.util import build_distance_matrix

def run_multi_instances(dir: str, prefixes: List[str]):
    files = calc_txt_files_with_prefixes(dir, prefixes)
    for i in range(len(files)):
        file = files[i]
        run_one_instance(file)

def run_one_instance(file_name):
    print("file_name: ", )
    # Loading Coordinates # Berlin 52 (Minimum Distance = 7544.3659)
    # if 'Coordinates' in file_name:
    #     coordinates = pd.read_csv(file_name, sep='\t')
    #     coordinates = coordinates.values

    graph, coordinates = read_tsp_file(file_name)
    distance_matrix = transfer_nxgraph_to_adjacencymatrix(graph)
    # Obtaining the Distance Matrix
    # distance_matrix = build_distance_matrix(coordinates)

    start_time = time.time()
    if ALG == Alg.cheapest_insertion:
        route, distance = cheapest_insertion(distance_matrix, **PARAMETERS)
    elif ALG == Alg.christofides_algorithm:
        route, distance = christofides_algorithm(distance_matrix, local_search=True, verbose=True)
    elif ALG == Alg.farthest_insertion:
        route, distance = farthest_insertion(distance_matrix, **PARAMETERS)
    elif ALG == Alg.genetic_algorithm:
        route, distance = genetic_algorithm(distance_matrix, **PARAMETERS)
    elif ALG == Alg.greedy_karp_steele_patching:
        route, distance = greedy_karp_steele_patching(distance_matrix, **PARAMETERS)
    elif ALG == Alg.lkh:
        import requests
        import lkh
        problem_str = requests.get('http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/A/A-n32-k5.vrp').text
        problem = lkh.LKHProblem.parse(problem_str)
        solver_path = 'LKH-3.0.6/LKH_mac'
        lkh.solve(solver_path, problem=problem, **PARAMETERS)
    elif ALG == Alg.local_search_2_opt:
        seed = util.seed_function(distance_matrix)
        route, distance = local_search_2_opt(distance_matrix, seed, **PARAMETERS)
    elif ALG == Alg.local_search_3_opt:
        seed = util.seed_function(distance_matrix)
        route, distance = local_search_3_opt(distance_matrix, seed, **PARAMETERS)
    elif ALG == Alg.nearest_insertion:
        route, distance = nearest_neighbour(distance_matrix, **PARAMETERS)
    elif ALG == Alg.nearest_neighbour:
        route, distance = nearest_neighbour(distance_matrix, **PARAMETERS)
    elif ALG == Alg.simulated_annealing:
        route, distance = simulated_annealing_tsp(distance_matrix, **PARAMETERS)
    elif ALG == Alg.tabu_search:
        seed = util.seed_function(distance_matrix)
        route, distance = tabu_search(distance_matrix, seed, **PARAMETERS)


    if_plot = False
    if if_plot and ALG != Alg.lkh:
        # Plot Locations and Tour
        print('Total Distance: ', round(distance, 2))
        plot_tour(coordinates, city_tour=route, view='notebook', size=10)

    running_duration = time.time() - start_time
    print("running_duration: ", running_duration)
    write_graph_result(obj=distance, running_duration=running_duration, num_nodes=len(route)-1, alg_name=ALG, solution=route, filename=file_name, plus1=False)

    aaa = 1


def main():

    run_one_file = False
    if run_one_file:
        file_name = '../../data/tsplib/berlin52.tsp'
        run_one_instance(file_name)

    run_multi_files = True
    if run_multi_files:
        dir = '../../data/tsplib'
        prefixes = ['a5']

        #prefixes = ['l', 'p', 'r', 's', 't', 'u']
        #prefixes = ['a', 'b', 'c', 'd', 'e', 'g', 'k']

        #prefixes = ['a', 'b', 'c']
        #prefixes = ['d']
        #prefixes = ['e']
        #prefixes = ['g', 'k']
        #prefixes = ['l']
        #prefixes = ['p']
        #prefixes = ['r']
        #prefixes = ['s', 't', 'u']
        #prefixes = ['berlin52']

        run_multi_instances(dir, prefixes)

    pass

if __name__ == '__main__':
    main()