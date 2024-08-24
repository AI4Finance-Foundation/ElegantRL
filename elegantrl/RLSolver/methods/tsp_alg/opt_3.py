############################################################################



# Lesson: Local Search-3-opt
 


############################################################################

# Required Libraries
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

# Function: Possible Segments
def segments_3_opt(n):
    x       = []
    a, b, c = 0, 0, 0
    for i in range(0, n):
        a = i
        for j in range(i + 1, n):
            b = j
            for k in range(j + 1, n + (i > 0)):
                c = k
                x.append((a, b, c))    
    return x

############################################################################

# Function: 3_opt
def local_search_3_opt(distance_matrix, city_tour, recursive_seeding = -1, verbose = True):
    if (recursive_seeding < 0):
        count = recursive_seeding - 1
    else:
        count = 0
    city_list     = [city_tour[0][:-1], city_tour[1]]
    city_list_old = city_list[1]*2
    iteration     = 0
    while (count < recursive_seeding):
        if (verbose == True):
            print('Iteration = ', iteration, 'Distance = ', round(city_list[1], 2))  
        best_route   = copy.deepcopy(city_list)
        best_route_1 = [[], 1]
        seed         = copy.deepcopy(city_list)     
        x            = segments_3_opt(len(city_list[0]))
        for item in x:
            i, j, k = item   
            A       = best_route[0][:i+1] + best_route[0][i+1:j+1]
            a       = best_route[0][:i+1] + list(reversed(best_route[0][i+1:j+1]))
            B       = best_route[0][j+1:k+1]
            b       = list(reversed(B))
            C       = best_route[0][k+1:]
            c       = list(reversed(C))
            trial   = [ 
                        # Original Tour
                        #[A + B + C], 
                        
                        # 1
                        [a + B + C], 
                        [A + b + C], 
                        [A + B + c],
                        
                        
                        # 2
                        [A + b + c], 
                        [a + b + C], 
                        [a + B + c], 

                        
                        # 3
                        [a + b + c]
                        
                      ] 
            # Possibly, there is a sequence of 2-opt moves that decreases the total distance but it begins 
            # with a move that first increases it
            for item in trial:   
                best_route_1[0] = item[0]
                best_route_1[1] = distance_calc(distance_matrix, [best_route_1[0] + [best_route_1[0][0]], 1])
                if (best_route_1[1]  < best_route[1]):
                    best_route = [best_route_1[0], best_route_1[1]]
                if (best_route[1] < city_list[1]):
                    city_list = [best_route[0], best_route[1]]              
            best_route = copy.deepcopy(seed) 
        count     = count + 1  
        iteration = iteration + 1  
        if (city_list_old > city_list[1] and recursive_seeding < 0):
             city_list_old     = city_list[1]
             count             = -2
             recursive_seeding = -1
        elif(city_list[1] >= city_list_old and recursive_seeding < 0):
            count              = -1
            recursive_seeding  = -2
    city_list = [city_list[0] + [city_list[0][0]], city_list[1]]
    return city_list[0], city_list[1]

############################################################################
