############################################################################



# Lesson: Cheapest Insertion
 


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

# Function: Cheapest Insertion
def cheapest_insertion(distance_matrix, verbose = True):
    route    = []
    temp     = []
    i, idx   = np.unravel_index(np.argmax(distance_matrix, axis = None), distance_matrix.shape)
    temp.append(i)
    temp.append(idx)
    count    = 0
    while (len(temp) < distance_matrix.shape[0]):
        temp_    = [item+1 for item in temp]
        temp_    = temp_ + [temp_[0]]
        d        = distance_calc(distance_matrix, [temp_, 1])
        seed     = [temp_, d]
        temp_, _ = local_search_2_opt(distance_matrix, seed, recursive_seeding = -1, verbose = False)
        temp     = [item-1 for item in temp_[:-1]]
        idx      = [i for i in range(0, distance_matrix.shape[0]) if i not in temp]
        best_d   = []
        best_r   = []
        for i in idx:
            temp_    = [item for item in temp]
            temp_.append(i)
            temp_    = [item+1 for item in temp_]
            temp_    = temp_ + [temp_[0]]
            d        = distance_calc(distance_matrix, [temp_, 1])
            seed     = [temp_, d]
            temp_, d = local_search_2_opt(distance_matrix, seed, recursive_seeding = -1, verbose = False)
            temp_    = [item-1 for item in temp_[:-1]]
            best_d.append(d)
            best_r.append(temp_)
        temp = [item for item in best_r[best_d.index(min(best_d))]]
        if (verbose == True):
            print('Iteration = ', count)
        count = count + 1
    route    = temp + [temp[0]]
    route    = [item + 1 for item in route]
    distance = distance_calc(distance_matrix, [route, 1])
    print("distance: ", distance)
    return route, distance

############################################################################