############################################################################



# Lesson: Christofides Algorithm
 


############################################################################

# Required Libraries
import copy
import networkx as nx
import numpy as np

from scipy.sparse.csgraph import minimum_spanning_tree

############################################################################

# Function: Tour Distance
def distance_calc(distance_matrix, city_tour):
    distance = 0
    for k in range(0, len(city_tour[0])-1):
        m        = k + 1
        distance = distance + distance_matrix[city_tour[0][k]-1, city_tour[0][m]-1]            
    return distance

# Function: 2_opt
def local_search_2_opt(distance_matrix, city_tour, recursive_seeding = -1, verbose = True):
    if (recursive_seeding < 0):
        count = -2
    else:
        count = 0
    city_list = copy.deepcopy(city_tour)
    distance  = city_list[1]*2
    iteration = 0
    while (count < recursive_seeding):
        if (verbose == True):
            print('Iteration = ', iteration, 'Distance = ', round(city_list[1], 2))  
        best_route = copy.deepcopy(city_list)
        seed       = copy.deepcopy(city_list)        
        for i in range(0, len(city_list[0]) - 2):
            for j in range(i+1, len(city_list[0]) - 1):
                best_route[0][i:j+1] = list(reversed(best_route[0][i:j+1]))           
                best_route[0][-1]    = best_route[0][0]     
                best_route[1]        = distance_calc(distance_matrix, best_route)                    
                if (city_list[1] > best_route[1]):
                    city_list = copy.deepcopy(best_route)         
                best_route = copy.deepcopy(seed)
        count     = count + 1
        iteration = iteration + 1  
        if (distance > city_list[1] and recursive_seeding < 0):
             distance          = city_list[1]
             count             = -2
             recursive_seeding = -1
        elif(city_list[1] >= distance and recursive_seeding < 0):
            count              = -1
            recursive_seeding  = -2
    return city_list[0], city_list[1]

############################################################################

# Function: Christofides Algorithm
def christofides_algorithm(distance_matrix, local_search = True, verbose = True):
    # Minimum Spanning Tree T
    graph_T = minimum_spanning_tree(distance_matrix)
    graph_T = graph_T.toarray().astype(int)
    # Induced Subgraph G
    graph_O = np.array(graph_T, copy = True) 
    count_r = np.count_nonzero(graph_T  > 0, axis = 1)
    count_c = np.count_nonzero(graph_T  > 0, axis = 0)
    degree  = count_r + count_c
    graph_G = np.zeros((graph_O.shape))
    for i in range(0, degree.shape[0]):
        if (degree[i] % 2 != 0):
            graph_G[i,:] = 1
            graph_G[:,i] = 1  
    for i in range(0, degree.shape[0]):
        if (degree[i] % 2 == 0):
            graph_G[i,:] = 0
            graph_G[:,i] = 0
    np.fill_diagonal(graph_G, 0)
    for i in range(0, graph_G.shape[0]):
        for j in range(0, graph_G.shape[1]):
            if (graph_G[i, j] > 0):
                graph_G[i, j] = distance_matrix[i, j] 
    # Minimum-Weight Perfect Matching M
    graph_G_inv = np.array(graph_G, copy = True) 
    graph_G_inv = -graph_G_inv
    min_w_pm    = nx.algorithms.matching.max_weight_matching(nx.from_numpy_array(graph_G_inv), maxcardinality = True)
    graph_M     = np.zeros((graph_G.shape)) 
    for item in min_w_pm:
        i, j          = item
        graph_M[i, j] = distance_matrix[i, j] 
    # Eulerian Multigraph H
    graph_H = np.array(graph_T, copy = True) 
    for i in range(0, graph_H.shape[0]):
        for j in range(0, graph_H.shape[1]):
            if (graph_M[i, j] > 0 and graph_T[i, j] == 0):
                graph_H[i, j] = 1 #distance_matrix[i, j]  
            elif (graph_M[i, j] > 0 and graph_T[i, j] > 0):
                graph_H[j, i] = 1 #distance_matrix[i, j]    
    # Eulerian Path
    H = nx.from_numpy_array(graph_H)
    if (nx.is_eulerian(H)):
        euler = list(nx.eulerian_path(H))
    else:
        H     = nx.eulerize(H)
        euler = list(nx.eulerian_path(H))
    # Shortcutting
    route = []
    for path in euler:
        i, j = path
        if (i not in route):
            route.append(i)
        if (j not in route):
            route.append(j)
    route    = route + [route[0]]
    route    = [item + 1 for item in route]
    distance = distance_calc(distance_matrix, [route, 1])
    seed     = [route, distance]
    if (local_search == True):
        route, distance = local_search_2_opt(distance_matrix, seed, recursive_seeding = -1, verbose = verbose)
    return route, distance

############################################################################
   