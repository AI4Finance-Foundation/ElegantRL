import sys
import os
cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../../rlsolver')
sys.path.append(os.path.dirname(rlsolver_path))

from typing import Dict, List
import copy

import networkx as nx

from rlsolver.methods.VRPTW_algs.Customer import Customer
from rlsolver.methods.VRPTW_algs.config import Config


class Vehicle:
    def __init__(self, id2):
        self.id = id2

        self.arrival_time_dict: Dict = {}  # key is name of customer, value is arrival_time
        self.departure_time_dict: Dict = {}  # key is name of customer, value is departure_time
        self.arrival_time_list: List = []
        self.departure_time_list: List = []
        self.path_denoted_by_customers: List = []  # customers are stored
        # self.path_denoted_by_names: List[str] = []  # only customers' names

        self.arrival_time_dict_if_insert: Dict = {}
        self.departure_time_dict_if_insert: Dict = {}
        self.path_denoted_by_customers_if_insert: List = []

        self.arrival_time_list_if_insert: List = []
        self.departure_time_list_if_insert: List = []
        self.paths_denoted_by_customers_if_insert: List = []

        # self.insertion_position = None

    def __eq__(self, other):
        return self.id == other.id

    @staticmethod
    def check_update_arrival_departure_time_by_path(vehicles, graph: nx.DiGraph):
        for veh in vehicles:
            veh.clear_if_insert6()
            veh.update_arrival_departure_time_by_path(graph)

    def update_arrival_departure_time_by_path(self, graph: nx.DiGraph):
        for j in range(1, len(self.path_denoted_by_customers)):
            i = j - 1
            prev: Customer = self.path_denoted_by_customers[i]
            this: Customer = self.path_denoted_by_customers[j]
            prev_name = prev.name
            this_name = this.name
            prev_departure = self.departure_time_dict[prev_name]
            feasible, arrival, departure = Customer.calc_arrival_departure_time(prev_departure, prev, this, graph)
            assert feasible is True
            if this_name == Config.DEST_NAME:
                self.arrival_time_dict[this_name] = arrival
                self.departure_time_dict[this_name] = departure
            else:
                assert (arrival == self.arrival_time_dict[this_name])
                assert (departure == self.departure_time_dict[this_name])
                start_service = max(arrival, this.forward_time_window[0])
                departure2 = start_service + this.service_duration
                assert departure2 == departure

    def update_use_if_insert3(self):
        self.arrival_time_dict = self.arrival_time_dict_if_insert
        self.departure_time_dict = self.departure_time_dict_if_insert
        self.path_denoted_by_customers = self.path_denoted_by_customers_if_insert

    def clear_if_insert6(self):
        self.arrival_time_dict_if_insert = {}
        self.departure_time_dict_if_insert = {}
        self.path_denoted_by_customers_if_insert = []
        self.arrival_time_list_if_insert = []
        self.departure_time_list_if_insert = []
        self.paths_denoted_by_customers_if_insert = []

    def clear_if_insert3(self):
        self.arrival_time_dict_if_insert = {}
        self.departure_time_dict_if_insert = {}
        self.path_denoted_by_customers_if_insert = []

    # index: insertion position
    # if fail, do nothing
    # if succeed, update xxx_if_insert.
    def succeed_insert_customer(self, index: int, customer, graph: nx.DiGraph) -> bool:
        assert len(self.path_denoted_by_customers) >= 2
        sum_demand = 0
        for cust in self.path_denoted_by_customers:
            sum_demand += cust.demand
        if sum_demand > Config.VEHICLE_CAPACITY:
            return False
        path_denoted_by_customers = copy.deepcopy(self.path_denoted_by_customers)
        path_denoted_by_customers.insert(index, customer)
        arrival_time_dict = copy.deepcopy(self.arrival_time_dict)
        departure_time_dict = copy.deepcopy(self.departure_time_dict)
        # calc arrival and departure time from index to the end
        feasible = True
        for i in range(index, len(path_denoted_by_customers)):
            prev = path_denoted_by_customers[i - 1]
            this = path_denoted_by_customers[i]
            prev_departure = departure_time_dict[prev.name]
            feasible, arrival, departure = Customer.calc_arrival_departure_time(prev_departure, prev, this, graph)
            if not feasible:
                break
            arrival_time_dict[this.name] = arrival
            departure_time_dict[this.name] = departure
            ## if use the following condition, the arrival_time_dict_if_insert and departure_time_dict_if_insert may be not accurate.
            # if i >= index + 1 and arrival_time_dict[this.name] <= self.arrival_time_dict[this.name]:
            #     break
        if feasible:
            self.arrival_time_dict_if_insert = arrival_time_dict
            self.departure_time_dict_if_insert = departure_time_dict
            self.path_denoted_by_customers_if_insert = path_denoted_by_customers
            return True
        else:
            self.clear_if_insert3()
            return False
