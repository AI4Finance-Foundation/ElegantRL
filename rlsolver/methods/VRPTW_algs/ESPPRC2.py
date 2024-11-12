import sys
import os

cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../../rlsolver')
sys.path.append(os.path.dirname(rlsolver_path))

from gurobipy import *
import copy
import time
import networkx as nx
import operator
from Customer import Customer
from Vehicle import Vehicle
from typing import List, Tuple, Union, Dict
from rlsolver.methods.VRPTW_algs.config import (Config,
                                                )
from typing import Dict, List
from rlsolver.methods.VRPTW_algs.Customer import (Customer,
                                                  )
from rlsolver.methods.VRPTW_algs.Vehicle import Vehicle
from rlsolver.methods.VRPTW_algs.util import (read_data,
                                              generate_vehicles,
                                              generate_customers_including_orig_dest,
                                              generate_vehicles_and_assign_paths,
                                              write_result_based_on_vehicles,
                                              calc_dist_of_path,
                                              obtain_paths_based_on_vehicles,
                                              calc_demands_of_paths,
                                              obtai_var_vals,
                                              calc_durations_of_paths, )
from Label import Label
from util import (read_data_as_nxdigraph,
                  write_result,
                  calc_dists_of_paths)


def check(customers):
    for cust in customers:
        # Label.check(cust.labels)
        for i in range(len(cust.forward_labels)):
            li = cust.forward_labels[i]
            if cust.id not in li.path_denoted_by_names:
                aaa = 1
            for j in range(i + 1, len(cust.forward_labels)):
                lj = cust.forward_labels[j]
                if li == lj:
                    aaa = 1


def forward_loop(forward_customers_will_be_treated: List[Customer], customers: List[Customer], graph: nx.DiGraph, use_arc_bounding=False, use_resource_bounding=False):
    iter = 0
    paths_set_denoted_by_names_been_treated = set()
    while len(forward_customers_will_be_treated) > 0:
        iter += 1
        print(f"iter: {iter}")
        customer_i = forward_customers_will_be_treated.pop()
        successors = list(graph.successors(customer_i.name))
        for name_j in successors:
            customer_j = Customer.obtain_by_name(name_j, customers)
            labels_i_to_j = []  # labels extended from i to j
            for label_i in customer_i.forward_labels:
                # if this path has been treated or extended, continue
                if Config.USE_PATHS_SET_IN_CG and tuple(label_i.path_denoted_by_names) in paths_set_denoted_by_names_been_treated:
                    continue
                # arc bounding
                if use_arc_bounding:
                    reach_halfway_point = arc_bounding(customer_i, label_i, customers, graph)
                    if reach_halfway_point:
                        continue
                label_i_to_j = Customer.extend_forward(customer_i, customer_j, label_i, graph)
                # if customer_i can reach customer_j, label_i_to_j is not None
                if label_i_to_j is not None:
                    labels_i_to_j.append(label_i_to_j)
            labels_j = copy.deepcopy(customer_j.forward_labels)
            labels_j.extend(labels_i_to_j)
            filtered_labels_j = Label.EFF(labels_j, True)
            change = Label.change(filtered_labels_j, customer_j.forward_labels)
            if change:
                customer_j.forward_labels = copy.deepcopy(filtered_labels_j)
                if customer_j not in forward_customers_will_be_treated:
                    forward_customers_will_be_treated.append(customer_j)
        for label_i in customer_i.forward_labels:
            paths_set_denoted_by_names_been_treated.add(tuple(label_i.path_denoted_by_names))


def backward_loop(backward_customers_will_be_treated: List[Customer], customers: List[Customer], graph: nx.DiGraph):
    iter = 0
    paths_set_denoted_by_names_been_treated = set()
    while len(backward_customers_will_be_treated) > 0:
        iter += 1
        print(f"iter: {iter}")
        customer_j = backward_customers_will_be_treated.pop()
        paths_set_denoted_by_names_been_treated.add(customer_j.name)
        predecessors = list(graph.predecessors(customer_j.name))
        for name_i in predecessors:
            customer_i: Customer = Customer.obtain_by_name(name_i, customers)
            labels_j_to_i = []  # labels extended from j to i
            for label_j in customer_j.backward_labels:
                # if this path has been treated or extended, continue
                if Config.USE_PATHS_SET_IN_CG and tuple(label_j.path_denoted_by_names) in paths_set_denoted_by_names_been_treated:
                    continue
                label_j_to_i = Customer.extend_backward(customer_j, customer_i, label_j, customers, graph)
                # if customer_i can reach customer_j, label_i_to_j is not None
                if label_j_to_i is not None:
                    labels_j_to_i.append(label_j_to_i)
            labels_i = copy.deepcopy(customer_i.backward_labels)
            labels_i.extend(labels_j_to_i)
            filtered_labels_i = Label.EFF(labels_i, False)
            change = Label.change(filtered_labels_i, customer_i.backward_labels)
            if change:
                customer_i.backward_labels = copy.deepcopy(filtered_labels_i)
                if customer_i not in backward_customers_will_be_treated:
                    backward_customers_will_be_treated.append(customer_i)
        for label_j in customer_j.backward_labels:
            paths_set_denoted_by_names_been_treated.add(tuple(label_j.path_denoted_by_names))


def obtain_forward_paths(dest: Customer, graph: nx.DiGraph):
    # calc forward paths from orig to dest
    # dest = Customer.obtain_by_name(Config.DEST_NAME, customers)
    if dest is None:
        return [], []
    forward_paths = []
    if Config.SORT_BY_CUMULATIVE_TRAVEL_COST:
        dest.forward_labels.sort(key=operator.attrgetter('cumulative_travel_cost'))
    for i in range(len(dest.forward_labels)):
        label = dest.forward_labels[i]
        path = []
        for k in range(len(label.path_denoted_by_names)):
            this_name = label.path_denoted_by_names[k]
            path.append(this_name)
        if len(path) >= 2:
            forward_paths.append(path)
    forward_dists = calc_dists_of_paths(forward_paths, graph)
    return forward_paths, forward_dists


def obtain_backward_paths(orig: Customer, graph: nx.DiGraph):
    if orig is None:
        return [], []
    backward_paths = []
    if Config.SORT_BY_CUMULATIVE_TRAVEL_COST:
        orig.backward_labels.sort(key=operator.attrgetter('cumulative_travel_cost'))
    for i in range(len(orig.backward_labels)):
        label = orig.backward_labels[i]
        path = []
        for k in range(len(label.path_denoted_by_names)):
            this_name = label.path_denoted_by_names[k]
            path.append(this_name)
        if len(path) >= 2:
            backward_paths.append(path)
    backward_dists = calc_dists_of_paths(backward_paths, graph)
    return backward_paths, backward_dists


def opt_arc_bounding_model(duration_lowerbound_dict: dict, duration_to_depot: float, demand_lowerbound_dict: dict, demand_to_depot: float, label_i: Label, customers: List[Customer]):
    model = Model("arc_bounding_model")
    nonvisited_names = duration_lowerbound_dict.keys()
    num_variables = len(nonvisited_names)
    print(f"nonvisited_names: {nonvisited_names}")
    print(f"num_variables: {num_variables}")


    dest: Customer = Customer.obtain_by_name(Config.DEST_NAME, customers)

    print(f"dest.forward_time_window[1]: {dest.forward_time_window[1]}")
    print(f"Config.VEHICLE_CAPACITY: {Config.VEHICLE_CAPACITY}")
    # variables
    y = model.addVars(num_variables, vtype=GRB.INTEGER, lb=0, ub=1, name="y")

    # objective
    model.setObjective(quicksum(y[i] for i in range(num_variables)) + 1, GRB.MAXIMIZE)

    # constraints

    # duration
    model.addConstr(label_i.cumulative_duration + quicksum(duration_lowerbound_dict[name] for name in nonvisited_names) + duration_to_depot <= dest.forward_time_window[1], name="duration")

    # demand
    model.addConstr(label_i.cumulative_demand + quicksum(demand_lowerbound_dict[name] for name in nonvisited_names) + demand_to_depot <= Config.VEHICLE_CAPACITY, name="duration")


    model.optimize()

    feasible = True
    if model.status == GRB.INFEASIBLE:
        feasible = False
        model.computeIIS()
        infeasibleConstrName = [c.getAttr('ConstrName') for c in model.getConstrs() if
                                c.getAttr(GRB.Attr.IISConstr) > 0]
        print('infeasibleConstrName: {}'.format(infeasibleConstrName))
        model.write('../../result/model.ilp')

    if feasible:
        print(f"Optimal objective: {model.ObjVal}")

    obj = model.ObjVal if feasible else None

    var_name: str = "y"
    y_vals = obtai_var_vals(model, var_name) if feasible else None
    return y_vals, obj, feasible



def calc_duration_demand_lowerbound(customer_i: Customer, customer_j: Customer, label_i: Label, customers: List[Customer], graph: nx.DiGraph):
    duration_lowerbound_ij = Config.INF
    demand_lowerbound_ij = Config.INF
    for customer_k in customers:
        if customer_k == customer_j:
            continue
        if customer_k.name not in label_i.path_denoted_by_names or customer_k == customer_i:
            if (customer_k.name, customer_j.name) not in graph.edges.keys():
                continue
            duration1 = graph.edges[(customer_k.name, customer_j.name)]["duration"] + customer_k.service_duration
            if duration1 < duration_lowerbound_ij:
                duration_lowerbound_ij = duration1
            demand1 = customer_k.demand
            if demand1 < demand_lowerbound_ij:
                demand_lowerbound_ij = demand1
    return duration_lowerbound_ij, demand_lowerbound_ij


def calc_duration_demand_to_depot(customer_i: Customer, label_i: Label, customers: List[Customer], graph: nx.DiGraph):
    duration_to_depot = Config.INF
    demand_to_depot = Config.INF
    dest: Customer = Customer.obtain_by_name(Config.DEST_NAME, customers)
    for customer_k in customers:
        if customer_k == dest:
            continue
        if customer_k.name not in label_i.path_denoted_by_names or customer_k == customer_i:
            duration2 = graph.edges[(customer_k.name, dest.name)]["duration"] + dest.service_duration
            if duration2 < duration_to_depot:
                duration_to_depot = duration2
            demand2 = dest.demand
            if demand2 < demand_to_depot:
                demand_to_depot = demand2
    return duration_to_depot, demand_to_depot

# return if reach_halfway_point
def arc_bounding(customer_i: Customer, label_i: Label, customers: List[Customer], graph: nx.DiGraph):
    duration_lowerbound_dict = {}
    demand_lowerbound_dict = {}
    duration_to_depot, demand_to_depot = calc_duration_demand_to_depot(customer_i, label_i, customers, graph)
    names = []
    min_obj = Config.INF
    for customer_j in customers:
        name = customer_j.name
        if name in label_i.path_denoted_by_names:
            continue
        names.append(name)
        duration_lowerbound_ij, demand_lowerbound_ij = calc_duration_demand_lowerbound(customer_i, customer_j, label_i, customers, graph)
        duration_lowerbound_dict[name] = duration_lowerbound_ij
        demand_lowerbound_dict[name] = demand_lowerbound_ij
        y_vals, obj, feasible = opt_arc_bounding_model(duration_lowerbound_dict, duration_to_depot, demand_lowerbound_dict, demand_to_depot, label_i, customers)
        if not feasible:
            continue
        if obj < min_obj:
            min_obj = obj
    max_num_nodes_to_visit = min_obj
    num_nodes_visited = len(label_i.path_denoted_by_names)
    reach_halfway_point = True if max_num_nodes_to_visit < num_nodes_visited else False
    return reach_halfway_point



# vehicle_capacity: int, result_filename: str
# An exact algorithm for the elementary shortest path problem with resource constraints: Application to some vehicle routing problems
# return paths, each path is a list of customers' names
# paths are sorted by label's cumulative_travel_cost
def ESPPRC2_bidirectional(orig_name: str, dest_name: str, customers: List[Customer], graph: nx.DiGraph()) -> List[List[str]]:
    orig = Customer.obtain_by_name(orig_name, customers)
    orig_label: Label = Label.create_label_for_orig(True)
    orig.forward_labels = [orig_label]

    dest = Customer.obtain_by_name(dest_name, customers)
    dest_label: Label = Label.create_label_for_dest(False)
    dest.backward_labels = [dest_label]

    for customer in customers:
        if customer != orig:
            customer.forward_labels = []
        if customer != dest:
            customer.backward_labels = []
    forward_customers_will_be_treated = [orig]
    backward_customers_will_be_treated = [dest]

    # forward
    forward_loop(forward_customers_will_be_treated, customers, graph, use_arc_bounding=True, use_resource_bounding=False)
    dest = Customer.obtain_by_name(dest_name, customers)
    forward_paths, forward_dists = obtain_forward_paths(dest, graph)

    # backward
    backward_loop(backward_customers_will_be_treated, customers, graph)
    orig = Customer.obtain_by_name(Config.ORIG_NAME, customers)
    backward_paths, backward_dists = obtain_backward_paths(orig, graph)

    # join forward_paths and backward_paths
    paths = []
    dists = []

    return paths, dists


def check_nxgraph():
    g = nx.Graph()  # or MultiDiGraph, etc
    # nx.add_path(G, [0, 1, 2])
    g.add_edge(0, 1, weight=1)
    g.add_edge(1, 2, weight=1)
    g.add_edge(2, 3, weight=5)

    # G.edges.data()  # default data is {} (empty dict)
    # G.edges.data("weight", default=1)
    # G.edges([0, 2])  # only edges originating from these nodes
    edges0 = g.edges(1)  # only edges from node 0
    weights = nx.get_edge_attributes(g, 'weight')
    for i, j in weights.keys():
        weight = weights[(i, j)]
        print(f'weight of ({i},{j}): {weight}')
    print()


def demo():
    start_time = time.time()
    # 点属性: (demand, time_window_start, time_window_end, service_duration)
    nodes = {'s': (0, 0, 0, 0),
             '1': (1, 6, 14, 2),
             '2': (1, 9, 12, 2),
             '3': (1, 8, 12, 2),
             't': (1, 9, 15, 2)}
    # Nodes = {0: (0, 0, 0, 0)
    #     , 1: (1, 6, 14, 2)
    #     , 2: (1, 9, 12, 2)
    #     , 3: (1, 8, 12, 2)
    #     , 4: (1, 9, 15, 2)
    #          }
    # 弧的属性: (duration, dist)
    arcs = {('s', '1'): (8, 3),
            ('s', '2'): (5, 5),
            ('s', '3'): (8, 1),
            ('1', 't'): (4, 7),
            ('2', 't'): (2, 6),
            ('3', 't'): (4, 8)}

    # create the directed Graph
    graph = nx.DiGraph()
    cnt = 0
    customers = []
    # add nodes into the graph
    for name in nodes.keys():
        graph.add_node(name, time_window=(nodes[name][0], nodes[name][1]))
        node = nodes[name]
        customer: Customer = Customer(node[0], node[1], node[2], node[3])
        customer.id = cnt
        customer.name = name
        customers.append(customer)
        cnt += 1

    # add edges into the graph
    for key in arcs.keys():
        graph.add_edge(key[0], key[1], duration=arcs[key][0], cost=arcs[key][1], dist=arcs[key][1])

    # graph, filtered_labels, opt_labels = labeling_SPPRC(graph, org, des)
    Config.NUM_VEHICLES = len(nodes)
    Config.ORIG_NAME = "s"
    Config.DEST_NAME = "t"
    orig_name = Config.ORIG_NAME
    dest_name = Config.DEST_NAME
    ESPPRC2_bidirectional(orig_name, dest_name, customers, graph)
    vehicles = generate_vehicles_and_assign_paths(len(customers), customers)
    filtered_vehicles = vehicles[:Config.NUM_VEHICLES]
    running_duration = time.time() - start_time
    result_filename = "./result/demo.txt"
    alg_name = "ESPPRC1"
    paths = obtain_paths_based_on_vehicles(vehicles)
    dists = calc_dists_of_paths(paths, graph)
    demands = calc_demands_of_paths(vehicles)
    durations = calc_durations_of_paths(vehicles)
    write_result(result_filename, alg_name, paths, running_duration, dists, demands, durations)
    # write_result_based_on_vehicles(vehicles, alg_name, running_duration, result_filename)
    print("paths: ", paths)
    print("running_duration: ", running_duration)
    print()


def main():
    # assert Config.CONNECT_ORIG_DEST is False
    start_time = time.time()
    graph, customers = read_data_as_nxdigraph(Config.INSTANCE_FILENAME, Config.NUM_PURE_CUSTOMERS)
    orig_name = Config.ORIG_NAME
    dest_name = Config.DEST_NAME
    ESPPRC2_bidirectional(orig_name, dest_name, customers, graph)
    vehicles = generate_vehicles_and_assign_paths(len(customers), customers)
    filtered_vehicles = vehicles[:Config.NUM_VEHICLES]
    running_duration = time.time() - start_time
    result_filename = Config.RESULT_FILENAME
    alg_name = "ESPPRC1"
    paths = obtain_paths_based_on_vehicles(filtered_vehicles)
    dists = calc_dists_of_paths(paths, graph)
    demands = calc_demands_of_paths(filtered_vehicles)
    durations = calc_durations_of_paths(filtered_vehicles)
    write_result(result_filename, alg_name, paths, running_duration, dists, demands, durations)
    print("paths: ", paths)
    print("running_duration: ", running_duration)
    print()


if __name__ == '__main__':
    run_main = True
    if run_main:
        main()

    run_demo = False
    if run_demo:
        demo()
