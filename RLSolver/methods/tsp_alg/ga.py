############################################################################



# Lesson: pyCombinatorial - Genetic Algorithm
  


############################################################################

# Required Libraries
import copy
import numpy  as np
import random
import os

############################################################################

# Function: Tour Distance
def distance_calc(distance_matrix, city_tour):
    distance = 0
    for k in range(0, len(city_tour[0])-1):
        m        = k + 1
        distance = distance + distance_matrix[city_tour[0][k]-1, city_tour[0][m]-1]            
    return distance

# Function: 2_opt
def local_search_2_opt(distance_matrix, city_tour, recursive_seeding = -1):
    if (recursive_seeding < 0):
        count = -2
    else:
        count = 0
    city_list = copy.deepcopy(city_tour)
    distance  = city_list[1]*2
    while (count < recursive_seeding):
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
        count = count + 1
        if (distance > city_list[1] and recursive_seeding < 0):
             distance          = city_list[1]
             count             = -2
             recursive_seeding = -1
        elif(city_list[1] >= distance and recursive_seeding < 0):
            count             = -1
            recursive_seeding = -2
    return city_list

############################################################################

# Function: Initial Seed
def seed_function(distance_matrix):
    seed     = [[],float("inf")]
    sequence = random.sample(list(range(1, distance_matrix.shape[0]+1)), distance_matrix.shape[0])
    sequence.append(sequence[0])
    seed[0]  = sequence
    seed[1]  = distance_calc(distance_matrix, seed)
    return seed

# Function: Initial Population
def initial_population(population_size, distance_matrix):
    population = []
    for i in range(0, population_size):
        seed = seed_function(distance_matrix)
        population.append(seed)
    return population

############################################################################

# Function: Fitness
def fitness_function(cost, population_size): 
    fitness = np.zeros((population_size, 2))
    for i in range(0, fitness.shape[0]):
        fitness[i,0] = 1/(1 + cost[i] + abs(np.min(cost)))
    fit_sum      = fitness[:,0].sum()
    fitness[0,1] = fitness[0,0]
    for i in range(1, fitness.shape[0]):
        fitness[i,1] = (fitness[i,0] + fitness[i-1,1])
    for i in range(0, fitness.shape[0]):
        fitness[i,1] = fitness[i,1]/fit_sum
    return fitness

# Function: Selection
def roulette_wheel(fitness): 
    ix     = 0
    random = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
    for i in range(0, fitness.shape[0]):
        if (random <= fitness[i, 1]):
          ix = i
          break
    return ix

# Function: TSP Crossover - BCR (Best Cost Route Crossover)
def crossover_tsp_bcr(distance_matrix, parent_1, parent_2):
    individual = copy.deepcopy(parent_2)
    cut        = random.sample(list(range(0, len(parent_1[0]))), int(len(parent_1)/2))
    cut        = [ parent_1[0][cut[i]] for i in range(0, len(cut)) if parent_1[0][cut[i]] != parent_2[0][0]]
    d_1        = float('+inf')
    for i in range(0, len(cut)):
        best      = []
        A         = cut[i]
        parent_2[0].remove(A)
        dist_list = [distance_calc(distance_matrix, [ parent_2[0][:n] + [A] + parent_2[0][n:], parent_2[1]] ) for n in range(1, len(parent_2[0]))]
        d_2       = min(dist_list)
        if (d_2 <= d_1):
            d_1  = d_2
            n    = dist_list.index(d_1)
            best = parent_2[0][:n] + [A] + parent_2[0][n:]
        if (best[0] == best[-1]):
            parent_2[0] = best
            parent_2[1] = d_1
            individual  = copy.deepcopy(parent_2)
    return individual

# Function: TSP Crossover - ER (Edge Recombination)
def crossover_tsp_er(distance_matrix, parent_1, parent_2):
    ind_list   = [item for item in parent_2[0]]
    ind_list.sort()
    ind_list   = list(dict.fromkeys(ind_list))
    edg_list   = [ [item, []] for item in ind_list]
    for i in range(0, len(edg_list)):
        edges = []
        idx_c = parent_2[0].index(i+1)
        idx_l = np.clip(idx_c - 1, 0, len(parent_2[0])-1)
        idx_r = np.clip(idx_c + 1, 0, len(parent_2[0])-1)
        if (parent_2[0][idx_l] not in edges):
            edges.append(parent_2[0][idx_l])
        if (parent_2[0][idx_r] not in edges):
            edges.append(parent_2[0][idx_r])
        idx_c = parent_1[0].index(i+1)
        idx_l = np.clip(idx_c - 1, 0, len(parent_1[0])-1)
        idx_r = np.clip(idx_c + 1, 0, len(parent_1[0])-1)
        if (parent_1[0][idx_l] not in edges):
            edges.append(parent_1[0][idx_l])
        if (parent_1[0][idx_r] not in edges):
            edges.append(parent_1[0][idx_r])
        for edge in edges:
            edg_list[i][1].append(edge)
    start      = parent_1[0][0]
    individual = [[start], 1]
    target     = start
    del edg_list[start-1]
    while len(individual[0]) != len(parent_2[0])-1:
        limit      = float('+inf')
        candidates =  [ [[], []] for item in edg_list]
        for i in range(0, len(edg_list)):
            if (target in edg_list[i][1]):
                candidates[i][0].append(edg_list[i][0])
                candidates[i][1].append(len(edg_list[i][1]))
                if (len(edg_list[i][1]) < limit):
                    limit = len(edg_list[i][1])
                edg_list[i][1].remove(target)
        for i in range(len(candidates)-1, -1, -1):
            if (len(candidates[i][0]) == 0 or candidates[i][1] != [limit]):
                del candidates[i]
        if len(candidates) > 1:
            k = random.sample(list(range(0, len(candidates))), 1)[0]
        else:
            k = 0
        if (len(candidates) > 0):
            target = candidates[k][0][0]
            individual[0].append(target)
        else:
            if (len(edg_list) > 0):
                target = edg_list[0][0]
                individual[0].append(target)
            else:
                last_edges = [item for item in ind_list if item not in individual[0]]
                for edge in last_edges:
                    individual[0].append(edge)
        for i in range(len(edg_list)-1, -1, -1):
            if (edg_list[i][0] == target):
                del edg_list[i]
                break
        if (len(edg_list) == 1):
            individual[0].append(edg_list[0][0])
    individual[0].append(individual[0][0])
    individual[1] = distance_calc(distance_matrix, individual)
    return individual

# Function: Breeding
def breeding(distance_matrix, population, fitness, elite):
    cost = [item[1] for item in population]
    if (elite > 0):
        cost, offspring = (list(t) for t in zip(*sorted(zip(cost, population))))
    for i in range (elite, len(offspring)):
        parent_1, parent_2 = roulette_wheel(fitness), roulette_wheel(fitness)
        while parent_1 == parent_2:
            parent_2 = random.sample(range(0, len(population) - 1), 1)[0]
        parent_1 = copy.deepcopy(population[parent_1])  
        parent_2 = copy.deepcopy(population[parent_2])
        rand     = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)  
        if (rand > 0.5):
            rand_1 = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)  
            if (rand_1 > 0.5):
                offspring[i] = crossover_tsp_bcr(distance_matrix, parent_1, parent_2)
            else:
                offspring[i] = crossover_tsp_bcr(distance_matrix, parent_2, parent_1)
        else: 
            offspring[i] = crossover_tsp_er(distance_matrix, parent_1, parent_2)
    return offspring

# Function: Mutation - Swap with 2-opt Local Search
def mutation_tsp_swap(distance_matrix, individual, mutation_search):
    k  = random.sample(list(range(1, len(individual[0])-1)), 2)
    k1 = k[0]
    k2 = k[1]  
    A  = individual[0][k1]
    B  = individual[0][k2]
    individual[0][k1] = B
    individual[0][k2] = A
    individual[1]     = distance_calc(distance_matrix, individual)
    individual        = local_search_2_opt(distance_matrix, individual, mutation_search)
    return individual

# Function: Mutation
def mutation(distance_matrix, offspring, mutation_rate, mutation_search, elite):
    for i in range(elite, len(offspring)):
        probability = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
        if (probability <= mutation_rate):
            offspring[i] = mutation_tsp_swap(distance_matrix, offspring[i], mutation_search)
    return offspring

############################################################################

# Function: GA TSP
def genetic_algorithm(distance_matrix, population_size = 5, elite = 1, mutation_rate = 0.1, mutation_search = -1, generations = 100, verbose = True):
    population       = initial_population(population_size, distance_matrix)
    cost             = [item[1] for item in population]
    cost, population = (list(t) for t in zip(*sorted(zip(cost, population))))
    elite_ind        = population[0] 
    fitness          = fitness_function(cost, population_size)
    count            = 0
    while (count <= generations): 
        if (verbose == True):
            print('Generation = ', count, 'Distance = ', round(elite_ind[1], 2))
        offspring        = breeding(distance_matrix, population, fitness, elite)  
        offspring        = mutation(distance_matrix, offspring, mutation_rate, mutation_search, elite)
        cost             = [item[1] for item in offspring]
        cost, population = (list(t) for t in zip(*sorted(zip(cost, offspring ))))
        elite_child      = population[0]
        fitness          = fitness_function(cost, population_size)
        if(elite_ind[1] > elite_child[1]):
            elite_ind = elite_child 
        count = count + 1  
    route, distance = elite_ind
    return route, distance

############################################################################