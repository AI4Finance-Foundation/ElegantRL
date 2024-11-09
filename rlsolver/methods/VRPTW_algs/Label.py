import sys
import os
cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../../rlsolver')
sys.path.append(os.path.dirname(rlsolver_path))

import copy
from typing import Dict, List
from rlsolver.methods.VRPTW_algs.config import Config


class Label:
    count = 0

    def __init__(self,
                 arrival_time_list: List[float],
                 departure_time_list: List[float],
                 cumulative_travel_cost: float,
                 cumulative_demand: float,
                 num_visited_nodes: int,
                 visitation_vector: List[bool],
                 path_denoted_by_name: List[str]):
        self.id = Label.count
        Label.count += 1
        self.name = str(self.id)
        self.forward = True
        self.cumulative_travel_cost = cumulative_travel_cost
        self.cumulative_duration = departure_time_list[-1]  # arrival time, consumed time
        self.cumulative_demand = cumulative_demand
        self.num_visited_nodes = num_visited_nodes
        self.visitation_vector = visitation_vector
        self.path_denoted_by_names = path_denoted_by_name
        self.arrival_time_list = arrival_time_list
        self.departure_time_list = departure_time_list
        # self.ids_of_successors: List[int] = []

    def __init__(self, ):
        self.id = Label.count
        Label.count += 1
        self.name = str(self.id)
        self.forward = True
        self.cumulative_travel_cost = 0
        self.cumulative_duration = 0  # arrival time, consumed time
        self.cumulative_demand = 0
        self.num_visited_nodes = 0
        self.visitation_vector = [False] * Config.NUM_CUSTOMERS
        self.path_denoted_by_names = []
        self.arrival_time_list = []
        self.departure_time_list = []

    def __eq__(self, another):
        return self.path_denoted_by_names == another.path_denoted_by_names

    @staticmethod
    def create_label_for_orig(forward: bool):
        label = Label()
        label.id = Label.count
        label.name = Config.ORIG_NAME
        label.forward = forward
        Label.count += 1
        label.cumulative_travel_cost = 0
        label.cumulative_duration = 0
        label.cumulative_demand = 0
        label.num_visited_nodes = 0
        label.visitation_vector = [False] * Config.NUM_CUSTOMERS
        label.visitation_vector[0] = True
        label.path_denoted_by_names = [Config.ORIG_NAME]
        label.arrival_time_list = [Config.TIME_WINDOW_START_OF_DEPOT]
        label.departure_time_list = [Config.TIME_WINDOW_START_OF_DEPOT]
        # label.ids_of_successors = copy.deepcopy(Config.IDS_OF_CUSTOMERS)
        # label.ids_of_successors.remove(Config.ID_OF_ORIG)
        return label

    @staticmethod
    def create_label_for_dest(forward: bool):
        label = Label()
        label.id = Label.count
        label.name = Config.DEST_NAME
        label.forward = forward
        Label.count += 1
        label.cumulative_travel_cost = 0
        label.cumulative_duration = 0
        label.cumulative_demand = 0
        label.num_visited_nodes = 0
        label.visitation_vector = [False] * Config.NUM_CUSTOMERS
        label.visitation_vector[-1] = True
        label.path_denoted_by_names = [Config.DEST_NAME]
        label.arrival_time_list = [Config.TIME_WINDOW_END_OF_DEPOT]
        label.departure_time_list = [Config.TIME_WINDOW_END_OF_DEPOT]
        # label.ids_of_successors = copy.deepcopy(Config.IDS_OF_CUSTOMERS)
        # label.ids_of_successors.remove(Config.ID_OF_ORIG)
        return label

    def dominate(self, another, forward: bool):
        if forward and self.path_denoted_by_names[-1] != another.path_denoted_by_names[-1]:
            return False
        if not forward and self.path_denoted_by_names[0] != another.path_denoted_by_names[0]:
            return False
        if Config.ADD_NUM_VISITED_NODES_FOR_DOMINATE_IN_CG:
            if self.cumulative_duration <= another.cumulative_duration \
                    and self.cumulative_travel_cost <= another.cumulative_travel_cost \
                    and self.cumulative_demand <= another.cumulative_demand \
                    and self.num_visited_nodes <= another.num_visited_nodes:
                if self.cumulative_duration < another.cumulative_duration \
                        or self.cumulative_travel_cost < another.cumulative_travel_cost \
                        or self.cumulative_demand < another.cumulative_demand \
                        or self.num_visited_nodes < another.num_visited_nodes:
                    return True
                else:
                    return False
            else:
                return False
        else:
            if self.cumulative_duration <= another.cumulative_duration \
                    and self.cumulative_travel_cost <= another.cumulative_travel_cost \
                    and self.cumulative_demand <= another.cumulative_demand \
                    :
                if self.cumulative_duration < another.cumulative_duration \
                        or self.cumulative_travel_cost < another.cumulative_travel_cost \
                        or self.cumulative_demand < another.cumulative_demand \
                        :
                    return True
                else:
                    return False
            else:
                return False

    @staticmethod
    def EFF(labels2: List, forward: bool) -> List:
        labels = Label.make_unique(labels2)
        if len(labels) <= 1:
            return labels
        indices_will_remove = set()
        while True:
            new_indices_will_remove = copy.deepcopy(indices_will_remove)
            for i in range(len(labels)):
                if i in new_indices_will_remove:
                    continue
                labeli = labels[i]
                for j in range(i + 1, len(labels)):
                    if j in new_indices_will_remove:
                        continue
                    labelj = labels[j]
                    if labeli.dominate(labelj, forward):
                        new_indices_will_remove.add(j)
                        # print(f"compare: {labeli.path_denoted_by_id} better than {labelj.path_denoted_by_id}")
                    elif labelj.dominate(labeli, forward):
                        new_indices_will_remove.add(i)
                        # print(f"compare: {labelj.path_denoted_by_id} better than {labeli.path_denoted_by_id}")
            if len(new_indices_will_remove) == len(indices_will_remove):
                break
            indices_will_remove = new_indices_will_remove
        filtered_labels = []
        for i in range(len(labels)):
            if i not in indices_will_remove:
                filtered_labels.append(labels[i])
        # res = Label.make_unique(filtered_labels)
        return filtered_labels

    @staticmethod
    def change(labels1, labels2b) -> bool:
        if len(labels1) != len(labels2b):
            return True
        labels2 = copy.deepcopy(labels2b)
        for label1 in labels1:
            change = True
            for label2 in labels2:
                if label1 == label2:
                    change = False
                    labels2.remove(label2)
                    break
            if change:
                return True
        return False

    @staticmethod
    def check(labels):
        for i in range(len(labels)):
            li = labels[i]
            for j in range(i + 1, len(labels)):
                lj = labels[j]
                if li == lj:
                    aaa = 1

    @staticmethod
    def make_unique(labels) -> List:
        if len(labels) <= 1:
            return labels
        res = []
        indices_will_remove = set()
        for i in range(len(labels)):
            li = labels[i]
            for j in range(i + 1, len(labels)):
                lj = labels[j]
                if li == lj:
                    indices_will_remove.add(j)
        for i in range(len(labels)):
            if i not in indices_will_remove:
                res.append(labels[i])
        if len(labels) != len(res):
            aaa = 1
        return res
