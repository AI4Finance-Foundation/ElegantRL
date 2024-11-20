import sys
import os
cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../../rlsolver')
sys.path.append(os.path.dirname(rlsolver_path))

import copy
import operator

import numpy as np
from sys import exit
import os
from typing import List, Dict
import networkx as nx

from rlsolver.methods.VRPTW_algs.config import (Config, update_config)
from rlsolver.methods.VRPTW_algs.Vehicle import Vehicle
from rlsolver.methods.VRPTW_algs.Customer import Customer

def read_data(filename, num_pure_customers):
    with open(filename, "r") as file:
        stream = file.readlines()
    if stream == "":
        print("Error in reading file")
    else:
        print("Read file", filename)

    vehicle_number, capacity = [int(i) for i in stream[4].split()]
    fields = ("CUST-NO.", "XCOORD.", "YCOORD.", "DEMAND", "READY-TIME", "DUE-DATE", "SERVICE-TIME")
    data = list()
    for i in range(9, len(stream)):
        if stream[i] == "\n":
            continue
        val = stream[i].split()
        if len(val) != len(fields):
            print("Error in reading data")
            continue
        customer = dict(zip(fields, val))
        data.append(customer)

    # Consider only depot + 50 customers
    data = data[0: num_pure_customers + 1]
    # data.append(data[0]) # The depot is represented by two identical
    #                      # nodes: 0 and n+1
    # data[-1]["CUST-NO."] = "51"
    x = []; y = []; demands = []; time_window_start = []; time_window_end = []; service_time = []
    for customer in data:
        x.append(int(customer["XCOORD."]))
        y.append(int(customer["YCOORD."]))
        demands.append(int(customer["DEMAND"]))
        time_window_start.append(int(customer["READY-TIME"]))
        time_window_end.append(int(customer["DUE-DATE"]))
        service_time.append(int(customer["SERVICE-TIME"]))
    x.append(x[0])
    y.append(y[0])
    demands.append(demands[0])
    time_window_start.append(time_window_start[0])
    time_window_end.append(time_window_end[0])
    service_time.append(service_time[0])
    return vehicle_number, capacity, x, y, demands, time_window_start, time_window_end, service_time

def read_data_as_nxdigraph2(filename, num_pure_customers) -> nx.DiGraph:
    num_vehicles, vehicle_capacity, x, y, demands, time_window_start, time_window_end, service_time = read_data(filename, num_pure_customers)
    update_config(num_vehicles, vehicle_capacity, x, y, demands, time_window_start, time_window_end, service_time)
    graph = nx.DiGraph()
    nodes = {}
    edges = {}
    for i in range(Config.NUM_PURE_CUSTOMERS + 1):
        nodes[i] = (Config.TIME_WINDOW_START[i], Config.TIME_WINDOW_END[i], Config.DEMANDS[i], Config.SERVICE_DURATION[i])
        for j in range(i + 1, Config.NUM_PURE_CUSTOMERS + 1):
            if i == Config.ORIG_ID and j == Config.DEST_ID:
                continue
            edges[(i, j)] = (Config.TRAVEL_DURATION_MATRIX[i][j], Config.TRAVEL_DIST_MATRIX[i][j])
            edges[(j, i)] = (Config.TRAVEL_DURATION_MATRIX[i][j], Config.TRAVEL_DIST_MATRIX[i][j])
    # add nodes into the graph
    for name in nodes.keys():
        (time_window_start, time_window_end, demand, service_duration) = nodes[name]
        graph.add_node(name, time_window=(time_window_start, time_window_end), demand=demand, service_duration=service_duration)

    # add edges into the graph
    for key in edges.keys():
        (duration, dist) = edges[key]
        graph.add_edge(key[0], key[1], duration=duration, dist=dist)
    return graph

# the weight of edge is duration
def read_data_as_nxdigraph(filename, num_pure_customers) -> (List[Customer], nx.DiGraph):
    num_vehicles, vehicle_capacity, x, y, demands, time_window_start, time_window_end, service_time = read_data(filename, num_pure_customers)
    update_config(num_vehicles, vehicle_capacity, x, y, demands, time_window_start, time_window_end, service_time)
    graph = nx.DiGraph()
    nodes = {}
    edges = {}

    customers = []
    orig: Customer = Customer(demand=Config.DEMANDS[0],
                              time_window_start=Config.TIME_WINDOW_START_OF_DEPOT,
                              time_window_end=Config.TIME_WINDOW_END_OF_DEPOT,
                              service_duration=Config.SERVICE_DURATION_OF_DEPOT)
    orig.id = Config.ORIG_ID
    # orig.name = str(orig.id)
    orig.name = Config.ORIG_NAME
    orig.is_depot = True
    orig.is_forward_path_planned = True
    orig.is_backward_path_planned = True
    orig.is_orig = True
    orig.is_dest = False
    orig.is_visited_in_forward_path = True
    orig.is_visited_in_backward_path = False
    customers.append(orig)
    nodes[orig.name] = orig
    for i in range(orig.id + 1, Config.NUM_PURE_CUSTOMERS + 1):
        customer: Customer = Customer(demand=Config.DEMANDS[i],
                                      time_window_start=Config.TIME_WINDOW_START[i],
                                      time_window_end=Config.TIME_WINDOW_END[i],
                                      service_duration=Config.SERVICE_DURATION[i])
        customer.id = i
        customer.name = str(customer.id)
        customers.append(customer)
        nodes[customer.name] = customer

    if Config.ADD_DEST_AS_SAME_ORIG:
        x_dest = Config.X[Config.ORIG_ID] if Config.ADD_DEST_AS_SAME_ORIG else Config.X[Config.DEST_ID]
        y_dest = Config.Y[Config.ORIG_ID] if Config.ADD_DEST_AS_SAME_ORIG else Config.Y[Config.DEST_ID]
        demand_dest = Config.DEMANDS[Config.ORIG_ID] if Config.ADD_DEST_AS_SAME_ORIG else Config.DEMANDS[Config.DEST_ID]
        time_window_start_dest = Config.TIME_WINDOW_START_OF_DEPOT if Config.ADD_DEST_AS_SAME_ORIG else Config.TIME_WINDOW_START[Config.DEST_ID]
        time_window_end_dest = Config.TIME_WINDOW_END_OF_DEPOT if Config.ADD_DEST_AS_SAME_ORIG else Config.TIME_WINDOW_END[Config.DEST_ID]
        service_duration_dest = Config.SERVICE_DURATION_OF_DEPOT if Config.ADD_DEST_AS_SAME_ORIG else Config.SERVICE_DURATION[Config.DEST_ID]
        dest: Customer = Customer(demand=demand_dest,
                                  time_window_start=time_window_start_dest,
                                  time_window_end=time_window_end_dest,
                                  service_duration=service_duration_dest)
        dest.id = Config.DEST_ID
        # dest.name = str(dest.id)
        dest.name = Config.DEST_NAME
        dest.is_depot = True
        dest.is_forward_path_planned = True
        dest.is_backward_path_planned = True
        dest.is_orig = False
        dest.is_dest = True
        dest.is_visited_in_forward_path = False
        dest.is_visited_in_backward_path = True
        customers.append(dest)
        nodes[dest.name] = dest
    else:
        j = Customer.obtain_index_by_id(Config.DEST_ID, customers)
        dest = customers[j]
        dest.name = Config.DEST_NAME
        dest.is_depot = False
        dest.is_forward_path_planned = True
        dest.is_backward_path_planned = True
        dest.is_orig = False
        dest.is_dest = True
        dest.is_visited_in_forward_path = False
        dest.is_visited_in_backward_path = True
        nodes[dest.name] = dest
        del nodes[str(Config.DEST_ID)]

    for name_i in nodes.keys():
        # info = Customer.obtain_by_id(i, customers)
        for name_j in nodes.keys():
            if (name_i == Config.ORIG_NAME and name_j == Config.DEST_NAME) or (name_i == Config.DEST_NAME and name_j == Config.ORIG_NAME):
                if not Config.CONNECT_ORIG_DEST:
                    continue
            if name_i == name_j or name_i == Config.DEST_NAME or name_j == Config.ORIG_NAME:
                continue
            i = Config.NAMES_OF_CUSTOMERS.index(name_i)
            j = Config.NAMES_OF_CUSTOMERS.index(name_j)
            edges[(name_i, name_j)] = (Config.TRAVEL_DURATION_MATRIX[i][j], Config.TRAVEL_DIST_MATRIX[i][j])
            # edges[(name_j, name_i)] = (Config.TRAVEL_DURATION_MATRIX[i][j], Config.TRAVEL_DIST_MATRIX[i][j])

    # add nodes into the graph
    for name in nodes.keys():
        # node: Customer = nodes[name]
        # graph.add_node(name, info=node)
        graph.add_node(name)

    # add edges into the graph
    for key in edges.keys():
        (duration, dist) = edges[key]
        graph.add_edge(key[0], key[1], duration=duration, cost=dist, dist=dist)

    return graph, customers


def obtain_paths_based_on_vehicles(vehicles: List[Vehicle]):
    paths = []
    for vehicle in vehicles:
        path = [i.name for i in vehicle.path_denoted_by_customers]
        if len(path) >= 2:
            paths.append(path)
    return paths

# demands: demand on each path
# durations: duration on each path
def write_result(filename, alg_name: str, paths: List[int], running_duration: float, dists: List[float], demands: List[float]=[], durations: List[float]=[]):
    with open(filename, "w") as f:
        f.write(f'alg_name: {alg_name}\n')
        f.write(f'running_duration: {running_duration}\n')
        dist = sum(dists)
        f.write(f'dist: {dist}\n')
        f.write(f'dists: {dists}\n')
        f.write(f'demands: {demands}\n')
        f.write(f'durations: {durations}\n')
        if len(paths) == 0:
            f.write(f'no paths are satisfied.\n')
        f.write(f'paths:\n')
        for path in paths:
            f.write(str(path) + '\n')

def write_result_based_on_vehicles(vehicles, alg_name, running_duration, result_filename):
    paths = obtain_paths_based_on_vehicles(vehicles)
    print("paths: ", paths)
    print("running_duration: ", running_duration)
    dists = []
    write_result(result_filename, alg_name, paths, running_duration, dists)


def generate_vehicles(num_vehicles: int, customers):
    vehicles = []
    for i in range(num_vehicles):
        vehicle = Vehicle(i)
        vehicle.arrival_time_dict[Config.ORIG_NAME] = Config.TIME_WINDOW_START_OF_DEPOT
        vehicle.departure_time_dict[Config.ORIG_NAME] = Config.TIME_WINDOW_START_OF_DEPOT
        vehicle.arrival_time_dict[Config.DEST_NAME] = Config.TIME_WINDOW_END_OF_DEPOT  # not accurate for dest
        vehicle.departure_time_dict[Config.DEST_NAME] = Config.TIME_WINDOW_END_OF_DEPOT  # not accurate for dest
        orig = Customer.obtain_by_name(Config.ORIG_NAME, customers)
        dest = Customer.obtain_by_name(Config.DEST_NAME, customers)
        vehicle.path_denoted_by_customers = [orig, dest]
        vehicles.append(vehicle)
    return vehicles

# the depot is customer 0. add it.
# there are totally NUM_PURE_CUSTOMERS + 2 customers.
def generate_customers_including_orig_dest():
    customers = []
    orig: Customer = Customer(demand=Config.DEMANDS[0],
                              time_window_start=Config.TIME_WINDOW_START_OF_DEPOT,
                              time_window_end=Config.TIME_WINDOW_END_OF_DEPOT,
                              service_duration=Config.SERVICE_DURATION_OF_DEPOT)
    orig.id = Config.ORIG_ID
    orig.name = Config.ORIG_NAME
    orig.is_depot = True
    orig.is_forward_path_planned = True
    orig.is_backward_path_planned = True
    orig.is_orig = True
    orig.is_dest = False
    orig.is_visited_in_forward_path = True
    orig.is_visited_in_backward_path = False
    customers.append(orig)
    for i in range(1, Config.NUM_PURE_CUSTOMERS + 1):
        customer: Customer = Customer(demand=Config.DEMANDS[i],
                                      time_window_start=Config.TIME_WINDOW_START[i],
                                      time_window_end=Config.TIME_WINDOW_END[i],
                                      service_duration=Config.SERVICE_DURATION[i])
        customer.id = i
        customer.name = str(customer.id)
        customers.append(customer)

    if Config.ADD_DEST_AS_SAME_ORIG:
        x_dest = Config.X[Config.ORIG_ID] if Config.ADD_DEST_AS_SAME_ORIG else Config.X[Config.DEST_ID]
        y_dest = Config.Y[Config.ORIG_ID] if Config.ADD_DEST_AS_SAME_ORIG else Config.Y[Config.DEST_ID]
        demand_dest = Config.DEMANDS[Config.ORIG_ID] if Config.ADD_DEST_AS_SAME_ORIG else Config.DEMANDS[Config.DEST_ID]
        time_window_start_dest = Config.TIME_WINDOW_START_OF_DEPOT if Config.ADD_DEST_AS_SAME_ORIG else Config.TIME_WINDOW_START[Config.DEST_ID]
        time_window_end_dest = Config.TIME_WINDOW_END_OF_DEPOT if Config.ADD_DEST_AS_SAME_ORIG else Config.TIME_WINDOW_END[Config.DEST_ID]
        service_duration_dest = Config.SERVICE_DURATION_OF_DEPOT if Config.ADD_DEST_AS_SAME_ORIG else Config.SERVICE_DURATION[Config.DEST_ID]
        dest: Customer = Customer(demand=demand_dest,
                                  time_window_start=time_window_start_dest,
                                  time_window_end=time_window_end_dest,
                                  service_duration=service_duration_dest)
        dest.id = Config.DEST_ID
        dest.name = Config.DEST_NAME
        dest.is_depot = False
        dest.is_forward_path_planned = True
        dest.is_backward_path_planned = True
        dest.is_orig = False
        dest.is_dest = True
        dest.is_visited_in_forward_path = False
        dest.is_visited_in_backward_path = True
        customers.append(dest)
    else:
        dest = Customer.obtain_by_name(Config.DEST_NAME, customers)
        dest.is_depot = False
        dest.is_forward_path_planned = True
        dest.is_backward_path_planned = True
        dest.is_orig = False
        dest.is_dest = True
        dest.is_visited_in_forward_path = False
        dest.is_visited_in_backward_path = True

    return customers



def createDistanceMatrix(x, y):
    n = len(x)
    d = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            p1 = np.array([x[i], y[i]])
            p2 = np.array([x[j], y[j]])
            d[i,j] = d[j,i] = int(round(np.linalg.norm(p1-p2)))
    return d


def addRoutesToMaster(routes, mat, costs, d):
    for i in range(len(routes)):
        cost = d[routes[i][0],routes[i][1]]
        for j in range(1,len(routes[i])-1):
            cost += d[routes[i][j], routes[i][j+1]]
            mat[routes[i][j]-1,i] += 1
        costs[i] = cost

def calc_dist_of_path(path: List[int], graph: nx.DiGraph) -> float:
    dist = 0
    for i in range(len(path) - 1):
        dist += graph.edges[(path[i], path[i + 1])]["dist"]
    return dist

def calc_dists_of_paths(paths: List[List[int]], graph: nx.DiGraph) -> List[float]:
    dists = []
    for path in paths:
        dist = 0
        for i in range(len(path) - 1):
            dist += graph.edges[(path[i], path[i + 1])]["dist"]
        dists.append(dist)
    return dists

def calc_demands_of_paths(vehicles: List[Vehicle]) -> List[float]:
    demands = []
    for veh in vehicles:
        demand = 0
        for cust in veh.path_denoted_by_customers:
            demand += cust.demand
        demands.append(demand)
    return demands

# departure time at dest
def calc_durations_of_paths(vehicles: List[Vehicle]) -> List[float]:
    durations = []
    for veh in vehicles:
        duration = veh.departure_time_dict[Config.DEST_NAME]
        durations.append(duration)
    return durations

# sort the vehicles by cumulative_travel_cost
def generate_vehicles_and_assign_paths(num_vehicles, customers: List[Customer]):
    vehicles = generate_vehicles(num_vehicles, customers)
    res = []
    dest = Customer.obtain_by_name(Config.DEST_NAME, customers)
    if dest is None:
        return []
    dest.forward_labels.sort(key=operator.attrgetter('cumulative_travel_cost'))
    num = min(len(dest.forward_labels), len(vehicles))
    for i in range(num):
        label = dest.forward_labels[i]
        vehicle = vehicles[i]
        path_denoted_by_names = label.path_denoted_by_names
        vehicle.departure_time_list = label.departure_time_list
        vehicle.arrival_time_list = label.arrival_time_list
        vehicle.departure_time_dict = {}
        vehicle.arrival_time_dict = {}
        vehicle.path_denoted_by_customers = []
        for k in range(len(path_denoted_by_names)):
            this_name = path_denoted_by_names[k]
            customer = Customer.obtain_by_name(this_name, customers)
            vehicle.path_denoted_by_customers.append(customer)
            vehicle.departure_time_dict[this_name] = vehicle.departure_time_list[k]
            vehicle.arrival_time_dict[this_name] = vehicle.arrival_time_list[k]
        res.append(vehicle)
    return res


# sort the vehicles by cumulative_travel_cost
def filter_vehicles_based_on_paths(vehicles2: List[Vehicle], paths: List[str], customers: List[Customer], graph: nx.DiGraph):
    num = min(len(vehicles2), len(paths))
    vehicles = copy.deepcopy(vehicles2)[:num]
    orig: Customer = Customer.obtain_by_name(Config.ORIG_NAME, customers)
    orig_arrival = orig.forward_time_window[0]
    orig_departure = orig_arrival + orig.service_duration
    for vehicle in vehicles:
        vehicle.arrival_time_dict[orig.name] = orig_arrival
        vehicle.departure_time_dict[orig.name] = orig_departure
    for i in range(num):
        vehicle = vehicles[i]
        path = paths[i]
        for k in range(1, len(path)):
            prev_name = path[k - 1]
            this_name = path[k]
            prev: Customer = Customer.obtain_by_name(prev_name, customers)
            this: Customer = Customer.obtain_by_name(this_name, customers)
            if k == 1:
                vehicle.path_denoted_by_customers = []
                vehicle.path_denoted_by_customers.append(prev)
            feasible, arrival, departure = Customer.calc_arrival_departure_time(vehicle.departure_time_dict[prev_name], prev, this, graph)
            assert feasible is True
            vehicle.arrival_time_dict[this_name] = arrival
            vehicle.departure_time_dict[this_name] = departure
            vehicle.path_denoted_by_customers.append(this)
    return vehicles

    # vehicles = generate_vehicles(num_vehicles, customers)
    # res = []
    # dest = Customer.obtain_by_name(Config.DEST_NAME, customers)
    # if dest is None:
    #     return []
    # dest.labels.sort(key=operator.attrgetter('cumulative_travel_cost'))
    # num = min(len(dest.labels), len(vehicles))
    # for i in range(num):
    #     label = dest.labels[i]
    #     vehicle = vehicles[i]
    #     vehicle.path_denoted_by_names = label.path_denoted_by_names
    #     vehicle.departure_time_list = label.departure_time_list
    #     vehicle.arrival_time_list = label.arrival_time_list
    #     vehicle.departure_time_dict = {}
    #     vehicle.arrival_time_dict = {}
    #     vehicle.path_denoted_by_customers = []
    #     for k in range(len(vehicle.path_denoted_by_names)):
    #         this_name = vehicle.path_denoted_by_names[k]
    #         customer = Customer.obtain_by_name(this_name, customers)
    #         vehicle.path_denoted_by_customers.append(customer)
    #         vehicle.departure_time_dict[this_name] = vehicle.departure_time_list[k]
    #         vehicle.arrival_time_dict[this_name] = vehicle.arrival_time_list[k]
    #     res.append(vehicle)
    # return res

def obtain_var_vals(model, var_name: str):
    theta_vals = []
    num_vars = model.NumVars
    for i in range(num_vars):
        var = model.getVarByName(f"{var_name}[{str(i)}]")
        theta_val = round(var.x, 2)
        theta_vals.append(theta_val)
    return theta_vals