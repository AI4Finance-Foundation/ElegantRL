# The impact heuristic is implemented based on the paper "A greedy look-ahead heuristic for the vehicle routing problem with time windows"
import sys
import os
cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../../rlsolver')
sys.path.append(os.path.dirname(rlsolver_path))

import copy
import time
from typing import Dict, List
import numpy as np
import networkx as nx

from rlsolver.methods.VRPTW_algs.Customer import (Customer,
                      )
from rlsolver.methods.VRPTW_algs.Vehicle import Vehicle
from rlsolver.methods.VRPTW_algs.util import (read_data,
                  read_data_as_nxdigraph,
                  generate_vehicles,
                  generate_vehicles_and_assign_paths,
                  obtain_paths_based_on_vehicles,
                  calc_demands_of_paths,
                  calc_durations_of_paths,
                  calc_dists_of_paths,
                  write_result,
                  write_result_based_on_vehicles,
                  )

from rlsolver.methods.VRPTW_algs.config import Config


# from my_config import Config


def calc_own_impact_IS(cust: Customer, vehicle: Vehicle) -> List[float]:
    IS = []
    for i in range(len(vehicle.paths_denoted_by_customers_if_insert)):
        if cust.name in vehicle.arrival_time_list_if_insert[i].keys():
            Is = vehicle.arrival_time_list_if_insert[i][cust.name] - cust.forward_time_window[0]
            IS.append(Is)
    return IS


def calc_external_impact_IU(cust: Customer, nonrouted_customers: List[Customer], graph: nx.DiGraph) -> float:
    num_nonrouted_customers = len(nonrouted_customers)
    sum_Iu = 0
    for customer in nonrouted_customers:
        if cust.name != customer.name:
            edge = (cust.name, customer.name)
            Iu = max(customer.forward_time_window[1] - cust.forward_time_window[0] - graph.edges[edge]["duration"], cust.forward_time_window[1] - customer.forward_time_window[0] - graph.edges[edge]["duration"])
            sum_Iu += Iu
    IU = sum_Iu / (num_nonrouted_customers - 1 + 1e-8)
    return IU


def calc_local_disturbance_LD(this_cust: Customer, vehicle: Vehicle, graph: nx.DiGraph) -> List[float]:
    LD = []
    this_name = this_cust.name
    for m in range(len(vehicle.paths_denoted_by_customers_if_insert)):
        path = vehicle.paths_denoted_by_customers_if_insert[m]
        for k in range(len(path)):
            customer = path[k]
            if customer.name == this_name:
                i = k - 1
                j = k + 1
                customer_i = path[i]
                customer_j = path[j]
                name_i = customer_i.name
                name_j = customer_j.name

                edge_i_this = (name_i, this_name)
                edge_this_j = (this_name, name_j)
                edge_i_j = (name_i, name_j)
                # in paper, c1 uses dist.
                # c1: use cost, or dist???
                c1 = graph.edges[edge_i_this]["cost"] + graph.edges[edge_this_j]["cost"] - graph.edges[edge_i_j]["cost"]
                c2 = (customer_j.forward_time_window[1] - (vehicle.arrival_time_list_if_insert[m][name_i] + customer_i.service_duration + graph.edges[edge_i_j]["duration"])) \
                     - (customer_j.forward_time_window[1] - (vehicle.arrival_time_list_if_insert[m][this_name] + this_cust.service_duration + graph.edges[edge_this_j]["duration"]))
                c3 = this_cust.forward_time_window[1] - (vehicle.arrival_time_list_if_insert[m][name_i] + customer_i.service_duration + graph.edges[edge_i_this]["duration"])

                Ld = Config.B1 * c1 + Config.B2 * c2 + Config.B3 * c3
                LD.append(Ld)
                break
    return LD


def calc_global_disturbance_IR(cust, vehicle: Vehicle, graph: nx.DiGraph) -> float:
    LD = calc_local_disturbance_LD(cust, vehicle, graph)
    IR = np.average(LD)
    return IR


def calc_internal_impact_accessibility_ACC(cust, vehicle: Vehicle):
    IR = calc_global_disturbance_IR(cust, vehicle)
    ACC = 1 / IR
    return ACC


def calc_impact(cust: Customer, vehicle: Vehicle, nonrouted_customers: List, graph: nx.DiGraph):
    success = False
    for i in range(1, len(vehicle.path_denoted_by_customers)):
        succeed = vehicle.succeed_insert_customer(i, cust, graph)
        if succeed:
            success = True
            vehicle.arrival_time_list_if_insert.append(vehicle.arrival_time_dict_if_insert)
            vehicle.departure_time_list_if_insert.append(vehicle.departure_time_dict_if_insert)
            path_if_insert = copy.deepcopy(vehicle.path_denoted_by_customers_if_insert)
            vehicle.paths_denoted_by_customers_if_insert.append(path_if_insert)
            vehicle.clear_if_insert3()
    if not success:
        return Config.INF
    Is = calc_own_impact_IS(cust, vehicle)
    IS = np.average(Is)
    IU = calc_external_impact_IU(cust, nonrouted_customers, graph)
    IR = calc_global_disturbance_IR(cust, vehicle, graph)
    impact = float(Config.Bs * IS + Config.Be * IU + Config.Br * IR)
    return impact


def calc_vehicle_with_min_impact(cust: Customer, vehicles: List[Vehicle], nonrouted_customers: List[Customer], graph: nx.DiGraph):
    min_impact = Config.INF
    selected_vehicle = None
    for vehicle in vehicles:
        vehicle.clear_if_insert6()
        impact = calc_impact(cust, vehicle, nonrouted_customers, graph)
        if impact < min_impact:
            min_impact = impact
            selected_vehicle = vehicle
    return selected_vehicle, min_impact


# select the vehicles with the route having at least 3 points (the first is orig, the final is dest)
def filter_vehicles_by_path_length(vehicles: List[Vehicle]) -> List[Vehicle]:
    res = []
    for vehicle in vehicles:
        if len(vehicle.path_denoted_by_customers) >= 2:
            res.append(vehicle)
    return res


def impact_heuristic(num_vehicles: int, customers: List[Customer], graph: nx.DiGraph) -> (List[Vehicle], List[Customer]):
    vehicles = generate_vehicles(num_vehicles, customers)
    # calc nonrouted_customers
    nonrouted_customers = []
    for customer in customers:
        if not customer.is_forward_path_planned:
            nonrouted_customers.append(customer)

    max_dist_from_depot = 0
    furthest_customer = None
    for customer in nonrouted_customers:
        edge = (Config.ORIG_NAME, customer.name)
        dist = graph.edges[edge]["duration"]
        if dist > max_dist_from_depot:
            max_dist_from_depot = dist
            furthest_customer = customer
    seed_customer = furthest_customer  # set the seed_customer as the furthest

    # find a feasible route for seed_customer, if fail, terminate
    for vehicle in vehicles:
        succeed = vehicle.succeed_insert_customer(1, seed_customer, graph)
        if succeed:
            seed_customer.is_forward_path_planned = True
            seed_customer.is_visited_in_forward_path = True
            vehicle.update_use_if_insert3()
            vehicle.clear_if_insert6()
            nonrouted_customers.remove(seed_customer)
            break
        else:
            return [], []

    # filtered_vehicles = []
    while True:
        if len(nonrouted_customers) == 0:
            filtered_vehicles = filter_vehicles_by_path_length(vehicles)
            return filtered_vehicles, customers

        min_impact = Config.INF
        selected_customer = None
        selected_vehicle = None
        for customer in nonrouted_customers:
            vehicle, impact = calc_vehicle_with_min_impact(customer, vehicles, nonrouted_customers, graph)
            if impact < min_impact:
                selected_customer = customer
                selected_vehicle = vehicle
        if selected_customer is not None and selected_vehicle is not None:
            LD = calc_local_disturbance_LD(selected_customer, selected_vehicle, graph)
            min_LD = min(LD)
            index = LD.index(min_LD)
            selected_customer.is_forward_path_planned = True
            selected_customer.is_visited_in_forward_path = True
            selected_vehicle.path_denoted_by_customers = selected_vehicle.paths_denoted_by_customers_if_insert[index]
            selected_vehicle.arrival_time_dict = selected_vehicle.arrival_time_list_if_insert[index]
            selected_vehicle.departure_time_dict = selected_vehicle.departure_time_list_if_insert[index]
            selected_vehicle.clear_if_insert6()
            # if selected_vehicle not in filtered_vehicles:
            #     filtered_vehicles.append(selected_vehicle)
            nonrouted_customers.remove(selected_customer)


def run_impact_heuristic() -> (List[Vehicle], List[Customer]):
    start_time = time.time()
    graph, customers = read_data_as_nxdigraph(Config.INSTANCE_FILENAME, Config.NUM_PURE_CUSTOMERS)
    num_vehicles = Config.NUM_VEHICLES
    filtered_vehicles, updated_customers = impact_heuristic(num_vehicles, customers, graph)
    Vehicle.check_update_arrival_departure_time_by_path(filtered_vehicles, graph)

    running_duration = time.time() - start_time
    alg_name = "impact_heuristic"
    # result_filename = Config.RESULT_DIR + "/" + Config.RESULT_FILENAME
    paths = obtain_paths_based_on_vehicles(filtered_vehicles)
    print("paths: ", paths)
    print("running_duration: ", running_duration)
    dists = calc_dists_of_paths(paths, graph)
    demands = calc_demands_of_paths(filtered_vehicles)
    durations = calc_durations_of_paths(filtered_vehicles)
    write_result(Config.RESULT_FILENAME, alg_name, paths, running_duration, dists, demands, durations)

    # write_result_based_on_vehicles(vehicles,  alg_name, running_duration, result_filename)
    return filtered_vehicles, updated_customers, paths, dists


if __name__ == '__main__':
    assert Config.CONNECT_ORIG_DEST is True
    vehicles, customers, paths, dists = run_impact_heuristic()
    print()
