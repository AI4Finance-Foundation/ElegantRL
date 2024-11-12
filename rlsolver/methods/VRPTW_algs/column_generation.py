import sys
import os

cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../../rlsolver')
sys.path.append(os.path.dirname(rlsolver_path))

import gurobipy as gp
from gurobipy import *
import copy
import time
import networkx as nx

from Vehicle import Vehicle
from rlsolver.methods.VRPTW_algs.impact_heuristic import impact_heuristic, run_impact_heuristic
from typing import Dict, List
from rlsolver.methods.VRPTW_algs.Customer import (Customer,
                                                  )

from rlsolver.methods.VRPTW_algs.util import (read_data,
                                              write_result,
                                              generate_vehicles,
                                              generate_customers_including_orig_dest,
                                              read_data_as_nxdigraph,
                                              calc_dist_of_path,
                                              calc_dists_of_paths,
                                              filter_vehicles_based_on_paths,
                                              calc_demands_of_paths,
                                              calc_durations_of_paths,
                                              obtain_paths_based_on_vehicles,
                                              obtain_var_vals,
                                              )
from rlsolver.methods.VRPTW_algs.ESPPRC1 import ESPPRC1_unidirectional
from rlsolver.methods.VRPTW_algs.config import Config


# each vehicle only serves one customer, and then return to the depot
def calc_init_paths(customers: List[Customer]):
    paths = []
    count = 0
    for cust in customers:
        if count >= Config.NUM_VEHICLES:
            break
        if cust.name != Config.ORIG_NAME and cust.name != Config.DEST_NAME:
            path = [Config.ORIG_NAME, cust.name, Config.DEST_NAME]
            paths.append(path)
    return paths


def calc_reduced_cost(path: List[int], duals: Dict[str, float], graph: nx.DiGraph):
    reduced_cost = 0
    for i in range(len(path) - 1):
        edge = (path[i], path[i + 1])
        reduced_cost += graph.edges[edge]["dist"] - duals[path[i]]
    return reduced_cost


def calc_reduced_costs(paths: List[List[int]], duals: Dict[str, float], graph: nx.DiGraph):
    reduced_costs = []
    for i in range(len(paths)):
        rc = calc_reduced_cost(paths[i], duals, graph)
        reduced_costs.append(rc)
    return reduced_costs


def calc_best_reduced_cost_and_path(routes: List[List[int]], duals: List[List[float]]) -> (float, List[int]):
    best_reduced_cost = None
    best_route = None
    for i in range(len(routes) - 1):
        reduced_cost = calc_reduced_cost(routes[i], duals[i])
        if reduced_cost < 0 and reduced_cost < best_reduced_cost:
            best_reduced_cost = reduced_cost
            best_route = routes[i]
    return best_reduced_cost, best_route


# the depot is excluded, i.e., the visit[0] == 0
def calc_if_path_visits_pure_customers(path: List[str]) -> List[int]:
    visit = {}
    for i in range(len(Config.NAMES_OF_CUSTOMERS)):
        name = Config.NAMES_OF_CUSTOMERS[i]
        if name in [Config.ORIG_NAME, Config.DEST_NAME]:
            visit[name] = 1
        else:
            if name in path:
                visit[name] = 1
            else:
                visit[name] = 0
    return visit


def calc_reduced_cost_of_path(path: List[str], dual: float, graph: nx.DiGraph) -> float:
    reduced_cost = 0
    for i in range(len(path) - 1):
        edge = (path[i], path[i + 1])
        rc = graph.edges[edge]["dist"] - dual
        reduced_cost += rc
    return reduced_cost


def calc_if_paths_visit_pure_customers(paths: List[List[int]]) -> List[List[int]]:
    visit = []
    for path in paths:
        row = calc_if_path_visits_pure_customers(path)
        visit.append(row)
    return visit


def create_restricted_master_problem(paths: List[List[int]], dists: List[int], set_variables_int):
    model = Model("restricted_master_problem")
    num_paths = len(paths)

    # i.e., a in paper. a matrix. if path k visits node i
    visit_matrix = calc_if_paths_visit_pure_customers(paths)

    # variables
    vtype = GRB.INTEGER if set_variables_int else GRB.CONTINUOUS
    theta = model.addVars(num_paths, vtype=vtype, lb=0, ub=GRB.INFINITY, name="theta")

    print(f"num_paths: {num_paths}")
    print(f"len(dists): {len(dists)}")
    print(f"dists: {dists}")

    # objective
    model.setObjective(quicksum(dists[i] * theta[i] for i in range(num_paths)), GRB.MINIMIZE)

    # constraints
    for name in Config.NAMES_OF_CUSTOMERS:
        if name in [Config.ORIG_NAME, Config.DEST_NAME]:
            continue
        model.addConstr(quicksum(visit_matrix[k][name] * theta[k] for k in range(num_paths)) >= 1, name="a_" + str(name))

    model.addConstr(quicksum(theta[k] for k in range(num_paths)) <= Config.NUM_VEHICLES, name="b")

    model.write("restricted_master_problem.lp")

    return model


def check_paths_cover_all_customers(paths):
    uncover_names = []
    for name in Config.NAMES_OF_CUSTOMERS:
        in_path = False
        for path in paths:
            if name in path:
                in_path = True
                break
        if not in_path:
            uncover_names.append(name)
    if len(uncover_names) == 0:
        return True, uncover_names
    else:
        return False, uncover_names


def calc_num_duplicates(paths: List[str]):
    num_duplicates = 0
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            if paths[i] == paths[j]:
                num_duplicates += 1
    return num_duplicates


def transfer_list_to_tuple_store_in_set(paths: List[List[str]]):
    res = set()
    for path in paths:
        path2 = tuple(path)
        res.add(path2)
    return res


def obtain_dict_from_paths_dists(paths: List[List[str]], dists: List[float]):
    dic = {}
    for i in range(len(paths)):
        path2 = tuple(paths[i])
        dic[path2] = dists[i]
    return dic


def extract_paths_dists_from_dict(path_dist_dict: Dict) -> (List[List[str]], List[float]):
    paths = []
    dists = []
    for path in path_dist_dict.keys():
        dist = path_dist_dict[path]
        paths.append(path)
        dists.append(dist)
    return paths, dists


# inpu paths, dists are all paths, dists. model select some of them.
def obtain_selected_paths_dists_from_model(model, paths, dists):
    selected_paths = []
    selected_dists = []
    for i in range(len(paths)):
        var = model.getVarByName(f"theta[{str(i)}]")
        theta_val = round(var.x, 3)
        if theta_val > 0.01:
            selected_paths.append(paths[i])
            selected_dists.append(dists[i])
    return selected_paths, selected_dists


def obtain_min_max(dists_of_iterations: List[float], width: int) -> (float, float):
    length = len(dists_of_iterations)
    if length < Config.CHECK_WIDTH_IN_CG:
        return -Config.INF, Config.INF
    min_dist = Config.INF
    max_dist = -Config.INF
    start = max(0, length - width)
    for i in range(start, length):
        dist = dists_of_iterations[i]
        if dist < min_dist:
            min_dist = dist
        if dist > max_dist:
            max_dist = dist
    return min_dist, max_dist





def run_column_generation():
    start_time = time.time()
    graph, original_customers = read_data_as_nxdigraph(Config.INSTANCE_FILENAME, Config.NUM_PURE_CUSTOMERS)
    customers = copy.deepcopy(original_customers)
    # customers = generate_customers_including_orig_dest()
    orig_name = Config.ORIG_NAME

    if Config.NUM_VEHICLES >= Config.NUM_PURE_CUSTOMERS:
        init_paths = calc_init_paths(customers)
        init_dists = calc_dists_of_paths(init_paths, graph)
    else:
        num_vehicles = Config.NUM_VEHICLES
        vehicles, updated_customers = impact_heuristic(num_vehicles, customers, graph)
        init_paths = obtain_paths_based_on_vehicles(vehicles)
        init_dists = calc_dists_of_paths(init_paths, graph)

    paths = init_paths
    dists = init_dists
    path_dist_dict = obtain_dict_from_paths_dists(paths, dists)

    cover, uncover_names = check_paths_cover_all_customers(paths)
    if not cover:
        print(f"uncover_names: {uncover_names}")
        raise ValueError

    num_iterations = 0
    dists_upperbound_of_iterations = []
    running_duration_in_ESPPRC = 0
    theta_valss = []
    while True:
        if num_iterations > Config.MAX_NUM_ITERATIONS_IN_CG:
            break
        num_iterations += 1
        print(f"iteration: {num_iterations}")
        paths, dists = extract_paths_dists_from_dict(path_dist_dict)
        model = create_restricted_master_problem(paths=paths, dists=dists, set_variables_int=False)
        model.optimize()

        print(f"Optimal objective: {model.ObjVal}")

        filtered_paths, filtered_dists = obtain_selected_paths_dists_from_model(model, paths, dists)
        dist_upperbound = sum(filtered_dists)
        dists_upperbound_of_iterations.append(dist_upperbound)
        # print(f"dists_of_iterations: {dists_of_iterations}")
        if Config.USE_CHECK_WIDTH_IN_CG:
            min_dist, max_dist = obtain_min_max(dists_upperbound_of_iterations, Config.CHECK_WIDTH_IN_CG)
            if max_dist - min_dist <= Config.CHECK_DIFF_THRESHOLD_IN_CG:
                break

        theta_vals_this_iteration = obtain_var_vals(model, "theta")
        theta_valss.append(theta_vals_this_iteration)
        print(f"theta_vals_this_iteration: {theta_vals_this_iteration}")

        constrs = model.getConstrs()
        duals = {}
        for constr in constrs:
            constr_name = constr.getAttr(GRB.Attr.ConstrName)
            dual = constr.getAttr(GRB.Attr.Pi)
            # print(f"constr.name: {constr_name}, dual: {dual}")
            if constr_name.startswith("a_"):
                consumer_name = constr_name.split("_")[1]
                duals[consumer_name] = dual
            else:
                duals[Config.ORIG_NAME] = dual

        # update edges' cost on graph
        for (i, j) in graph.edges:
            graph.edges[(i, j)]["cost"] = graph.edges[(i, j)]["dist"] - duals[i]

        start = time.time()
        if Config.USE_ESPPRC_IMPACT_AS_INIT_IN_CG == 0:
            new_paths, new_dists = ESPPRC1_unidirectional(orig_name, customers, graph)
        elif Config.USE_ESPPRC_IMPACT_AS_INIT_IN_CG == 1:
            for customer in customers:
                customer.is_forward_path_planned = False
            filtered_vehicles, updated_customers = impact_heuristic(num_vehicles, original_customers, graph)
            new_paths = obtain_paths_based_on_vehicles(filtered_vehicles)
            new_dists = calc_dists_of_paths(paths, graph)

        reduced_costs = calc_reduced_costs(new_paths, duals, graph)
        duration = time.time() - start
        running_duration_in_ESPPRC += duration

        best_reduced_cost = Config.INF
        best_index = None
        for i in range(len(new_paths)):
            if reduced_costs[i] < best_reduced_cost:
                best_index = i
                best_reduced_cost = reduced_costs[i]
        print(f"best_index: {best_index}")

        if len(new_paths) == 0 or best_reduced_cost >= 0:
            break

        if Config.ADD_ONE_PATH_EACH_ITERATION_IN_CG:
            new_paths = [new_paths[best_index]]
            new_dists = [new_dists[best_index]]
        new_dic = obtain_dict_from_paths_dists(new_paths, new_dists)
        path_dist_dict = path_dist_dict | new_dic

    print(f"num_iterations: {num_iterations}")
    # print(f"theta_vals: {theta_valss}")

    model2 = create_restricted_master_problem(paths=filtered_paths, dists=filtered_dists, set_variables_int=True)
    model2.optimize()
    if model2.Status == GRB.OPTIMAL:
        print(f"Optimal objective: {model.ObjVal}")
    else:
        raise ValueError

    theta_vals = obtain_var_vals(model, "theta")
    theta_vals_in_set = set(theta_vals)
    print(f"theta_vals: {theta_vals}")
    print(f"theta_vals_in_set: {theta_vals_in_set}")

    filtered_paths2, filtered_dists2 = obtain_selected_paths_dists_from_model(model2, filtered_paths, filtered_dists)
    print(f"filtered_paths2: {filtered_paths2}")

    running_duration = time.time() - start_time
    result_filename = Config.RESULT_FILENAME
    alg_name = "column_generation"
    dists = calc_dists_of_paths(filtered_paths2, graph)
    original_vehicles = generate_vehicles(Config.NUM_VEHICLES, original_customers)
    vehicles = filter_vehicles_based_on_paths(original_vehicles, filtered_paths2, original_customers, graph)
    Vehicle.check_update_arrival_departure_time_by_path(vehicles, graph)
    demands = calc_demands_of_paths(vehicles)
    durations = calc_durations_of_paths(vehicles)
    write_result(result_filename, alg_name, filtered_paths2, running_duration, dists, demands, durations)
    num_duplicates = calc_num_duplicates(paths)
    print(f"num_duplicates: {num_duplicates}")
    print(f"paths: {filtered_paths2}")
    print(f"dists_upperbound_of_iterations: {dists_upperbound_of_iterations}")
    print(f"running_duration_in_ESPPRC: {running_duration_in_ESPPRC}")
    print(f"running_duration: {running_duration}")

    print()

    pass


if __name__ == '__main__':
    assert Config.ADD_DEST_AS_SAME_ORIG is True
    run_column_generation()
