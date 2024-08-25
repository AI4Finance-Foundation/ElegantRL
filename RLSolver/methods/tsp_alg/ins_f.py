############################################################################



# Lesson: Farthest Insertion
 


############################################################################

# Required Libraries
import copy
import numpy as np

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

# Function: Best Insertion
def best_insertion(distance_matrix, temp):
    temp_    = [item+1 for item in temp]
    temp_    = temp_ + [temp_[0]]
    d        = distance_calc(distance_matrix, [temp_, 1])
    seed     = [temp_, d]
    temp_, _ = local_search_2_opt(distance_matrix, seed, recursive_seeding = -1, verbose = False)
    temp     = [item-1 for item in temp_[:-1]]
    return temp

############################################################################

# Function: Farthest Insertion
def farthest_insertion(distance_matrix, initial_location = -1, verbose = True):
    maximum    = float('+inf')
    distance   = float('+inf')
    route      = []
    for i1 in range(0, distance_matrix.shape[0]):
        if (initial_location != -1):
            i1 = initial_location-1
        temp       = []
        dist       = np.copy(distance_matrix)
        dist       = dist.astype(float)
        np.fill_diagonal(dist, float('-inf'))
        idx        = dist[i1,:].argmax()
        dist[i1,:] = float('-inf')
        dist[:,i1] = float('-inf')
        temp.append(i1)
        temp.append(idx)
        for j in range(0, distance_matrix.shape[0]-2):
            i2         = idx
            idx        = dist[i2,:].argmax()  
            dist[i2,:] = float('-inf')
            dist[:,i2] = float('-inf')
            temp.append(idx)
            temp       = best_insertion(distance_matrix, temp)
        temp = temp + [temp[0]]
        temp = [item + 1 for item in temp]
        val  = distance_calc(distance_matrix, [temp, 1])
        if (val < maximum):
            maximum  = val
            distance = val
            route    = [item for item in temp]
        if (verbose == True):
            print('Iteration = ', i1, 'Distance = ', round(distance, 2))
        if (initial_location == -1):
            continue
        else: 
            break
    return route, distance

############################################################################
