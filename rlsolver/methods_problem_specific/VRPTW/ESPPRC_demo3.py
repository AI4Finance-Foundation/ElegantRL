import sys
import os
cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../..')
sys.path.append(os.path.dirname(rlsolver_path))

import networkx as nx
import copy
from typing import List, Union
import time
from config import Config
from rlsolver.methods_problem_specific.VRPTW.util import (read_data_as_nxdigraph2,
                                                               write_result,
                                                               calc_dists_of_paths)

class Label:
    path_denoted_by_name = []
    demand = 0  # resource
    arrival_time = 0
    departure_time = 0  # resource. the total consumed time
    # service_duration = 0
    dist = 0  # resource.

    def __eq__(self, other):
        return self.path_denoted_by_name == other.path_denoted_by_names

    def dominate(self, another):
        if self.path_denoted_by_name[-1] != another.path_denoted_by_names[-1]:
            return False
        if len(self.path_denoted_by_name) <= len(another.path_denoted_by_names) \
            and self.demand <= another.demand \
                and self.departure_time <= another.departure_time \
                and self.dist <= another.cost:
            if len(self.path_denoted_by_name) < len(another.path_denoted_by_names) \
                or self.demand < another.demand \
                    or self.departure_time < another.departure_time \
                    or self.dist < another.cost:
                return True
            else:
                return False
        else:
            return False

    @staticmethod
    def obtain_unique(labels) -> List:
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
        return res


    # obtain only good labels. remove the bad labels by compare them, e.g., if a < b, remove b
    @staticmethod
    def EFF(labels2: List) -> List:
        labels = Label.obtain_unique(labels2)
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
                    if labeli.dominate(labelj):
                        new_indices_will_remove.add(j)
                        print(f"compare: {labeli.path_denoted_by_names} better than {labelj.path_denoted_by_names}")
                    elif labelj.dominate(labeli):
                        new_indices_will_remove.add(i)
                        print(f"compare: {labelj.path_denoted_by_names} better than {labeli.path_denoted_by_names}")
            if len(new_indices_will_remove) == len(indices_will_remove):
                break
            indices_will_remove = new_indices_will_remove
        filtered_labels = []
        for i in range(len(labels)):
            if i not in indices_will_remove:
                filtered_labels.append(labels[i])
        return filtered_labels

# select optimal solution
def select_optimal_by_dist(filtered_labels: List[Label]) -> List[Label]:
    opt_labels = [filtered_labels[0]]
    for i in range(1, len(filtered_labels)):
        label = filtered_labels[i]
        if label.dist < opt_labels[-1].dist:
            opt_labels = [label]
        elif label.dist == opt_labels[-1].dist:
            opt_labels.append(label)
    return opt_labels


# labeling algorithm
def ESPPRC(graph: nx.DiGraph(), orig: Union[str, int], dest: Union[str, int], vehicle_capacity: int, result_filename: str):
    start_time = time.time()
    # initial Q
    labels: List[Label] = []

    # creat initial label, from orig
    label = Label()
    label.path_denoted_by_name = [orig]
    label.demand = 0
    label.arrival_time = 0
    label.departure_time = 0
    # label.service_duration = 0
    label.dist = 0
    
    labels.append(label)

    stored_labels = []
    while len(labels) > 0:
        labels = Label.EFF(labels)
        cur_label = labels.pop()

        # extend the current label
        last_node = cur_label.path_denoted_by_name[-1]
        if last_node == dest:
            stored_labels.append(cur_label)
        for succ in graph.successors(last_node):
            # avoid circle
            if succ in cur_label.path_denoted_by_name:
                continue
            extended_label = Label()
            arc = (last_node, succ)

            arrival_time = cur_label.departure_time + graph.edges[arc]["duration"]
            time_window = graph.nodes[succ]["time_window"]
            demand = cur_label.demand + graph.nodes[succ]["demand"]
            service_duration = graph.nodes[succ]["service_duration"]
            # check the feasibility
            if last_node != dest and arrival_time <= time_window[1] and demand <= vehicle_capacity:
                extended_label.path_denoted_by_name = cur_label.path_denoted_by_name + [succ]
                extended_label.demand = demand
                extended_label.arrival_time = cur_label.departure_time + graph.edges[arc]["duration"]
                extended_label.departure_time = max(extended_label.arrival_time, time_window[0]) + service_duration
                extended_label.dist = cur_label.dist + graph.edges[arc]["cost"]
                labels.append(extended_label)

    filtered_labels = Label.EFF(stored_labels)

    if len(filtered_labels) == 0:
        return graph, [], []

    opt_labels_by_dist = select_optimal_by_dist(filtered_labels)

    running_duration = time.time() - start_time

    paths = []
    for i in range(len(filtered_labels)):
        label = filtered_labels[i]
        paths.append(label.path_denoted_by_names)
    dists = calc_dists_of_paths(paths, graph)
    alg_name = "ESPPRC2"
    write_result(result_filename, alg_name, paths, running_duration, dists)

    print("running duration: ", running_duration)

    print("filtered_labels:")
    for i in range(len(filtered_labels)):
        label = filtered_labels[i]
        paths.append(label.path_denoted_by_names)
        print(f'The {i + 1}-th path: {label.path_denoted_by_names}')
        print(f'The {i + 1}-th path (duration): {label.departure_time}')
        print(f'The {i + 1}-th path (dist): {label.cost}')
        print(f'The {i + 1}-th path (demand): {label.demand}')

    print("\n")

    print("opt_labels_by_dist:")
    for i in range(len(opt_labels_by_dist)):
        opt_label = opt_labels_by_dist[i]
        print(f'The {i + 1}-th optimal path: {opt_label.path_denoted_by_name}')
        print(f'The {i + 1}-th optimal path (duration): {opt_label.departure_time}')
        print(f'The {i + 1}-th optimal path (dist): {opt_label.dist}')
        print(f'The {i + 1}-th optimal path (demand): {opt_label.demand}')

    print()


    return graph, filtered_labels, opt_labels_by_dist


def demo():
    # (time_window_start, time_window_end, demand, service_duration)
    # orig = 0
    # dest = 100
    orig = 'orig'
    dest = 'dest'
    nodes = {orig: (0, 0, 0, 0),
             1: (6, 14, 1, 1),
             2: (9, 12, 2, 1),
             3: (8, 12, 1, 1),
             dest: (9, 15, 0, 1)}

    # 边的属性包括duration, dist
    edges = {(orig, 1): (8, 3),
            (orig, 2): (5, 5),
            (orig, 3): (8, 1),
            (1, dest): (4, 7),
            (2, dest): (2, 6),
            (3, dest): (4, 8)}

    # create the directed Graph
    graph = nx.DiGraph()

    # add nodes into the graph
    for name in nodes.keys():
        (time_window_start, time_window_end, demand, service_duration) = nodes[name]
        graph.add_node(name, time_window=(time_window_start, time_window_end), demand=demand, service_duration=service_duration)

    # add edges into the graph
    for key in edges.keys():
        (duration, dist) = edges[key]
        graph.add_edge(key[0], key[1], duration=duration, dist=dist, cost=dist)

    vehicle_capacity = 6
    result_filename = './result/demo.txt'
    graph, filtered_labels, opt_labels_by_dist = ESPPRC(graph, orig, dest, vehicle_capacity, result_filename)

    print()

def main():
    graph = read_data_as_nxdigraph2(Config.INSTANCE_FILENAME, Config.NUM_PURE_CUSTOMERS)
    orig = Config.ORIG_ID
    dest = Config.DEST_ID
    vehicle_capacity = Config.VEHICLE_CAPACITY
    result_filename = Config.RESULT_FILENAME
    graph, filtered_labels, opt_labels_by_dist = ESPPRC(graph, orig, dest, vehicle_capacity, result_filename)
    print()

if __name__ == '__main__':
    run_demo = False
    if run_demo:
        demo()

    run_main = True
    if run_main:
        main()

