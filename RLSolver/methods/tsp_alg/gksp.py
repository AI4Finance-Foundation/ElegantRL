############################################################################



# Lesson: Greedy Karp-Steele Patching Algorithm
 


############################################################################

# Required Libraries
import copy
import numpy as np

from collections import defaultdict
from scipy import optimize

###############################################################################

# Function: Cycle Finder (Adapted from:  https://gist.github.com/qpwo/272df112928391b2c83a3b67732a5c25; Author: Luke Harold Miles; email: luke@cs.uky.edu; Site: https://lukemiles.org
def simple_cycles(G):
    def _unblock(thisnode, blocked, B):
        stack = set([thisnode])
        while stack:
            node = stack.pop()
            if node in blocked:
                blocked.remove(node)
                stack.update(B[node])
                B[node].clear()
    G    = {v: set(nbrs) for (v,nbrs) in G.items()}
    sccs = strongly_connected_components(G)
    while sccs:
        scc       = sccs.pop()
        startnode = scc.pop()
        path      = [startnode]
        blocked   = set()
        closed    = set()
        blocked.add(startnode)
        B         = defaultdict(set)
        stack     = [ (startnode,list(G[startnode])) ]
        while stack:
            thisnode, nbrs = stack[-1]
            if nbrs:
                nextnode = nbrs.pop()
                if nextnode == startnode:
                    yield path[:]
                    closed.update(path)
                elif nextnode not in blocked:
                    path.append(nextnode)
                    stack.append( (nextnode, list(G[nextnode])) )
                    closed.discard(nextnode)
                    blocked.add(nextnode)
                    continue
            if not nbrs:
                if thisnode in closed:
                    _unblock(thisnode, blocked, B)
                else:
                    for nbr in G[thisnode]:
                        if thisnode not in B[nbr]:
                            B[nbr].add(thisnode)
                stack.pop()
                path.pop()
        remove_node(G, startnode)
        H = subgraph(G, set(scc))
        sccs.extend(strongly_connected_components(H))

# Function: SCC       
def strongly_connected_components(graph):
    index_counter = [0]
    stack         = []
    lowlink       = {}
    index         = {}
    result        = []   
    def _strong_connect(node):
        index[node]      = index_counter[0]
        lowlink[node]    = index_counter[0]
        index_counter[0] = index_counter[0] + 1
        stack.append(node) 
        successors       = graph[node]
        for successor in successors:
            if successor not in index:
                _strong_connect(successor)
                lowlink[node] = min(lowlink[node],lowlink[successor])
            elif successor in stack:
                lowlink[node] = min(lowlink[node],index[successor])
        if lowlink[node] == index[node]:
            connected_component = []
            while True:
                successor = stack.pop()
                connected_component.append(successor)
                if successor == node: break
            result.append(connected_component[:])
    for node in graph:
        if node not in index:
            _strong_connect(node)
    return result

# Function: Remove Node
def remove_node(G, target):
    del G[target]
    for nbrs in G.values():
        nbrs.discard(target)

# Function: Subgraph
def subgraph(G, vertices):
    return {v: G[v] & vertices for v in vertices}

###############################################################################

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

# Function: Greedy Karp Steele Patching
def greedy_karp_steele_patching(distance_matrix, verbose = True):
    count      = 0
    route      = []
    dist       = np.copy(distance_matrix)
    np.fill_diagonal(dist, np.sum(dist))
    r, c       = optimize.linear_sum_assignment(dist)
    adj_m      = np.zeros((dist.shape))
    adj_m[r,c] = 1
    graph      = {}
    value      = [[] for i in range(adj_m.shape[0])]
    keys       = range(adj_m.shape[0])
    for i in range(0, adj_m.shape[0]):
        for j in range(0, adj_m.shape[0]):
            if (adj_m[i,j] == 1):
                value[i].append(j)
    for i in keys:
        graph[i] = value[i]  
    cycles = list(simple_cycles(graph))
    while len(route) < distance_matrix.shape[0]:
        r_dict   = {}
        r_matrix = np.zeros((len(cycles), len(cycles)))
        r_matrix.fill(float('+inf'))
        for i in range(0, r_matrix.shape[0]):
            for j in range(i+1, r_matrix.shape[0]):
                a              = cycles[i]
                b              = cycles[j]
                c              = list(set(a + b))
                if (len(route) > 0):
                    c = route + c + [route[0]]
                else:
                    c = route + c + [c[0]]
                c              = [item+1 for item in c]
                d              = distance_calc(distance_matrix, [c, 1])
                seed           = [c, d]
                temp, val      = local_search_2_opt(distance_matrix, seed, recursive_seeding = -1, verbose = False)
                r_matrix[i, j] = val
                r_dict[(str(a),str(b))] = [item-1 for item in temp[:-1]]
        m, n  = np.argwhere(r_matrix == np.min(r_matrix))[0]
        if (str(cycles[m]) != str(cycles[n])):
            route = r_dict[(str(cycles[m]), str(cycles[n]))]
            remv1 = [item for item in cycles[m]]
            remv2 = [item for item in cycles[n]]
            cycles.remove(remv1)
            cycles.remove(remv2)
        else:
            route = route + cycles[m]
            remv1 = [item for item in cycles[m]]
            cycles.remove(remv1)
        if (verbose == True):
            print('Iteration = ', count, ' Visited Nodes = ', len(route))
        count = count + 1
    route           = route + [route[0]]
    route           = [item + 1 for item in route]
    distance        = distance_calc(distance_matrix, [route, 1])
    seed            = [route, distance]
    route, distance = local_search_2_opt(distance_matrix, seed, recursive_seeding = -1, verbose = False)
    return route, distance

############################################################################
