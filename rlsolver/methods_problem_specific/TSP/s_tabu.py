############################################################################



# Lesson: Local Search-Tabu Search
 


############################################################################

# Required Libraries
import random
import numpy  as np
import copy 

############################################################################

# Function: Tour Distance
def distance_calc(distance_matrix, city_tour):
    distance = 0
    for k in range(0, len(city_tour[0])-1):
        m        = k + 1
        distance = distance + distance_matrix[city_tour[0][k]-1, city_tour[0][m]-1]            
    return distance

############################################################################

# Function: Swap
def local_search_2_swap(distance_matrix, city_tour):
    best_route = copy.deepcopy(city_tour)      
    i, j       = random.sample(range(0, len(city_tour[0])-1), 2)
    if (i > j):
        i, j = j, i
    best_route[0][i:j+1] = list(reversed(best_route[0][i:j+1]))           
    best_route[0][-1]    = best_route[0][0]              
    best_route[1]        = distance_calc(distance_matrix, best_route)                     
    return best_route

# Function: 2_opt
def local_search_2_opt(distance_matrix, city_tour):
    city_list  = copy.deepcopy(city_tour)
    best_route = copy.deepcopy(city_list)
    seed       = copy.deepcopy(city_list)        
    for i in range(0, len(city_list[0]) - 2):
        for j in range(i+1, len(city_list[0]) - 1):
            best_route[0][i:j+1] = list(reversed(best_route[0][i:j+1]))           
            best_route[0][-1]    = best_route[0][0]                          
            best_route[1]        = distance_calc(distance_matrix, best_route)    
            if (best_route[1] < city_list[1]):
                city_list[1] = copy.deepcopy(best_route[1])
                for n in range(0, len(city_list[0])): 
                    city_list[0][n] = best_route[0][n]          
            best_route = copy.deepcopy(seed) 
    return city_list

# Function: 4_opt
def local_search_4_opt_stochastic(distance_matrix, city_tour):
    count        = 0
    city_list    = [city_tour[0][:-1], city_tour[1]]
    best_route   = copy.deepcopy(city_list)
    best_route_1 = [[], 1]
    seed         = copy.deepcopy(city_list)     
    i, j, k, L   = random.sample(list(range(0, len(city_list[0]))), 4)
    idx          = [i, j, k, L]
    idx.sort()
    i, j, k, L   = idx   
    A            = best_route[0][:i+1] + best_route[0][i+1:j+1]
    B            = best_route[0][j+1:k+1]
    b            = list(reversed(B))
    C            = best_route[0][k+1:L+1]
    c            = list(reversed(C))
    D            = best_route[0][L+1:]
    d            = list(reversed(D))
    trial        = [          
                      # 4-opt: Sequential
                      [A + b + c + d], [A + C + B + d], [A + C + b + d], [A + c + B + d], [A + D + B + c], 
                      [A + D + b + C], [A + d + B + c], [A + d + b + C], [A + d + b + c], [A + b + D + C], 
                      [A + b + D + c], [A + b + d + C], [A + C + d + B], [A + C + d + b], [A + c + D + B], 
                      [A + c + D + b], [A + c + d + b], [A + D + C + b], [A + D + c + B], [A + d + C + B],
                      
                      # 4-opt: Non-Sequential
                      [A + b + C + d], [A + D + b + c], [A + c + d + B], [A + D + C + B], [A + d + C + b]  
                   ]    
    for item in trial:   
        best_route_1[0] = item[0]
        best_route_1[1] = distance_calc(distance_matrix, [best_route_1[0] + [best_route_1[0][0]], 1])
        if (best_route_1[1]  < best_route[1]):
            best_route = [best_route_1[0], best_route_1[1]]
        if (best_route[1] < city_list[1]):
            city_list = [best_route[0], best_route[1]]              
    best_route = copy.deepcopy(seed) 
    count      = count + 1    
    city_list  = [city_list[0] + [city_list[0][0]], city_list[1]]
    return city_list

############################################################################

# Function:  Build Recency Based Memory and Frequency Based Memory (STM and LTM)
def build_stm_and_ltm(distance_matrix):
    n           = int((distance_matrix.shape[0]**2 - distance_matrix.shape[0])/2)
    stm_and_ltm = np.zeros((n, 5)) # ['City 1','City 2','Recency', 'Frequency', 'Distance']
    count       = 0
    for i in range (0, int((distance_matrix.shape[0]**2))):
        city_1 = i // (distance_matrix.shape[1])
        city_2 = i %  (distance_matrix.shape[1])
        if (city_1 < city_2):
            stm_and_ltm[count, 0] = city_1 + 1
            stm_and_ltm[count, 1] = city_2 + 1
            count = count + 1
    return stm_and_ltm

# Function: Diversification
def ltm_diversification (distance_matrix, stm_and_ltm, city_list):
    stm_and_ltm = stm_and_ltm[stm_and_ltm[:,3].argsort()]
    stm_and_ltm = stm_and_ltm[stm_and_ltm[:,4].argsort()]
    lenght      = random.sample((range(1, int(distance_matrix.shape[0]/3))), 1)[0]
    for i in range(0, lenght):
        city_list         = local_search_2_swap(distance_matrix, city_list)
        stm_and_ltm[i, 3] = stm_and_ltm[i, 3] + 1
        stm_and_ltm[i, 2] = 1
    return stm_and_ltm, city_list
	
# Function: Tabu Update
def tabu_update(distance_matrix, stm_and_ltm, city_list, best_distance, tabu_list, tabu_tenure = 20, diversify = False):
    m_list    = []
    n_list    = []
    city_list = local_search_2_opt(distance_matrix, city_list) # itensification
    for i in range(0, stm_and_ltm.shape[0]):
        stm_and_ltm[i, -1] = local_search_2_swap(distance_matrix, city_list)[1] 
    stm_and_ltm = stm_and_ltm[stm_and_ltm[:,4].argsort()] # Distance
    m           = int(stm_and_ltm[0,0]-1)
    n           = int(stm_and_ltm[0,1]-1)
    recency     = int(stm_and_ltm[0,2])
    distance    = stm_and_ltm[0,-1]     
    if (distance < best_distance): # Aspiration Criterion -> by Objective
        city_list = local_search_2_swap(distance_matrix, city_list)
        i         = 0
        while (i < stm_and_ltm.shape[0]):
            if (stm_and_ltm[i, 0] == m + 1 and stm_and_ltm[i, 1] == n + 1):
                stm_and_ltm[i, 2]  = 1
                stm_and_ltm[i, 3]  = stm_and_ltm[i, 3] + 1
                stm_and_ltm[i, -1] = distance
                if (stm_and_ltm[i, 3] == 1):
                    m_list.append(m + 1)
                    n_list.append(n + 1)
                i = stm_and_ltm.shape[0]
            i = i + 1
    else:
        i = 0
        while (i < stm_and_ltm.shape[0]):
            recency  = int(stm_and_ltm[i,2]) 
            distance = local_search_2_swap(distance_matrix, city_list)[1]
            if (distance < best_distance):
                city_list = local_search_2_swap(distance_matrix, city_list)
            if (recency == 0):
                city_list = local_search_2_swap(distance_matrix, city_list)
                stm_and_ltm[i, 2]  = 1
                stm_and_ltm[i, 3]  = stm_and_ltm[i, 3] + 1
                stm_and_ltm[i, -1] = distance
                if (stm_and_ltm[i, 3] == 1):
                    m_list.append(m + 1)
                    n_list.append(n + 1)
                i = stm_and_ltm.shape[0]
            i = i + 1
    if (len(m_list) > 0): 
        tabu_list[0].append(m_list[0])
        tabu_list[1].append(n_list[0])
    if (len(tabu_list[0]) > tabu_tenure):
        i = 0
        while (i < stm_and_ltm.shape[0]):
            if (stm_and_ltm[i, 0] == tabu_list[0][0] and stm_and_ltm[i, 1] == tabu_list[1][0]):
                del tabu_list[0][0]
                del tabu_list[1][0]
                stm_and_ltm[i, 2] = 0
                i = stm_and_ltm.shape[0]
            i = i + 1          
    if (diversify == True):
        stm_and_ltm, city_list = ltm_diversification(distance_matrix, stm_and_ltm, city_list) # diversification
        if (distance_matrix.shape[0] > 4):
            city_list = local_search_4_opt_stochastic(distance_matrix, city_list) # diversification
    return stm_and_ltm, city_list, tabu_list

############################################################################

# Function: Tabu Search
def tabu_search(distance_matrix, city_tour, iterations = 150, tabu_tenure = 20, verbose = True):
    count          = 0
    best_solution  = copy.deepcopy(city_tour)
    stm_and_ltm    = build_stm_and_ltm(distance_matrix)
    tabu_list      = [[],[]]
    diversify      = False
    no_improvement = 0
    while (count < iterations):
        if (verbose == True):
            print('Iteration = ', count, 'Distance = ', round(best_solution[1], 2))
        stm_and_ltm, city_tour, tabu_list = tabu_update(distance_matrix, stm_and_ltm, city_tour, best_solution[1], tabu_list, tabu_tenure, diversify)
        if (city_tour[1] < best_solution[1]):
            best_solution  = copy.deepcopy(city_tour)
            no_improvement = 0
            diversify      = False
        else:
            if (no_improvement > 0 and no_improvement % int(iterations/5) == 0):
                diversify      = True
                no_improvement = 0
            else:
                diversify  = False
            no_improvement = no_improvement + 1
        count = count + 1
    route, distance = best_solution
    return route, distance

############################################################################
