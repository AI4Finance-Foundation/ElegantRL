import numpy as np
import networkx as nx
from time import *
import copy
from typing import List, Tuple, Set, Dict

class Label:
    path = []
    duration = 0
    cost = 0

    def __eq__(self, other):
        return self.path == other.path

    def dominate(self, another):
        if len(self.path) <= len(another.path) \
                and self.duration <= another.duration \
                and self.cost <= another.cost:
            if len(self.path) < len(another.path) \
                    or self.duration < another.duration \
                    or self.cost < another.cost:
                return True
            else:
                return False
        else:
            return False

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
        return res

# dominance rule
def EFF(labels2: List[Label]):
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
                if labeli.dominate(labelj):
                    new_indices_will_remove.add(j)
                    # print(f"compare: {labeli.path_denoted_by_id} better than {labelj.path_denoted_by_id}")
                elif labelj.dominate(labeli):
                    new_indices_will_remove.add(i)
                    print(f"compare: {labelj.path} better than {labeli.path}")
        if len(new_indices_will_remove) == len(indices_will_remove):
            break
        indices_will_remove = new_indices_will_remove
    filtered_labels = []
    for i in range(len(labels)):
        if i not in indices_will_remove:
            filtered_labels.append(labels[i])
    return filtered_labels




# labeling algorithm
def labeling_SPPRC(graph, orig, dest):
    # initial Q
    labels: List[Label] = []
    path_dict: Dict = {}

    # creat initial label
    label = Label()
    label.path = [orig]
    label.duration = 0
    label.cost = 0
    labels.append(label)

    count = 0

    stored_labels = []
    while (len(labels) > 0):
        count += 1
        cur_label = labels.pop()

        # extend the current label
        last_node = cur_label.path[-1]
        if last_node == 't':
            stored_labels.append(cur_label)
        for child in graph.successors(last_node):
            extended_label = copy.deepcopy(cur_label)
            arc = (last_node, child)

            # check the feasibility
            arrive_time = cur_label.duration + graph.edges[arc]["duration"]
            time_window = graph.nodes[child]["time_window"]
            if len(extended_label.path) >= 2 and extended_label.path[:2] == ['s', '3']:
                aaa = 1
            if arrive_time >= time_window[0] and arrive_time <= time_window[1] and last_node != dest:
                extended_label.path.append(child)
                extended_label.duration += graph.edges[arc]["duration"]
                extended_label.cost += graph.edges[arc]["cost"]
                labels.append(extended_label)

    filtered_labels = EFF(stored_labels)

    if len(filtered_labels) == 0:
        return graph, [], []

    # choose optimal solution
    opt_labels = [filtered_labels[0]]
    for i in range(1, len(filtered_labels)):
        label = filtered_labels[i]
        if label.cost < opt_labels[-1].cost:
            opt_labels = [label]
        elif label.cost == opt_labels[-1].cost:
            opt_labels.append(label)

    return graph, filtered_labels, opt_labels


def main():
    # 点中包含时间窗属性
    Nodes = {'s': (0, 0)
        , '1': (6, 14)
        , '2': (9, 12)
        , '3': (8, 12)
        , 't': (9, 15)
             }
    # 弧的属性包括duration, dist
    Arcs = {('s', '1'): (8, 3)
        , ('s', '2'): (5, 5)
        , ('s', '3'): (8, 1)
        , ('1', 't'): (4, 7)
        , ('2', 't'): (2, 6)
        , ('3', 't'): (4, 8)
            }

    # create the directed Graph
    graph = nx.DiGraph()
    cnt = 0
    # add nodes into the graph
    for name in Nodes.keys():
        cnt += 1
        graph.add_node(name, time_window=(Nodes[name][0], Nodes[name][1]))
    # add edges into the graph
    for key in Arcs.keys():
        graph.add_edge(key[0], key[1], duration=Arcs[key][0], dist=Arcs[key][1], cost=Arcs[key][1])

    org = 's'
    des = 't'
    begin_time = time()
    graph, filtered_labels, opt_labels = labeling_SPPRC(graph, org, des)
    end_time = time()
    print("计算时间： ", end_time - begin_time)

    for i in range(len(opt_labels)):
        opt_label = opt_labels[i]
        print(f'The {i + 1}-th optimal path : {opt_label.path}')
        print(f'The {i + 1}-th optimal path (dist): {opt_label.cost}')
        print(f'The {i + 1}-th optimal path (time): {opt_label.duration}')
    print()

if __name__ == '__main__':
    main()

