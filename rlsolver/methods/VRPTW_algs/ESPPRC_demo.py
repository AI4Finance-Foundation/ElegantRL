import numpy as np
import networkx as nx
from time import *
import copy
from typing import List, Tuple, Set, Dict

class Label:
    path = []
    travel_time = 0
    dist = 0

# dominance rule
def dominate(labels: List[Label], path_dict: Dict[int, Label]):
    labels_copy = copy.deepcopy(labels)
    path_dict_copy = copy.deepcopy(path_dict)

    # dominate Q
    for label in labels_copy:
        for another_label in labels:
            if (label.path[-1] == another_label.path[
                -1] and label.time < another_label.time and label.dis < another_label.dis):
                labels.remove(another_label)
            print("dominated path (Q) : ", another_label.path)

    # dominate Paths
    for key_1 in path_dict_copy.keys():
        for key_2 in path_dict_copy.keys():
            if (path_dict_copy[key_1].path[-1] == path_dict_copy[key_2].path[-1]
                    and path_dict_copy[key_1].travel_time < path_dict_copy[key_2].travel_time
                    and path_dict_copy[key_1].dist < path_dict_copy[key_2].dist
                    and (key_2 in path_dict.keys())):
                path_dict.pop(key_2)
                print("dominated path (P) : ", path_dict_copy[key_1].path)

    return labels, path_dict




# labeling algorithm
def labeling_SPPRC(graph, orig, dest):
    # initial Q
    labels: List[Label] = []
    path_dict: Dict = {}

    # creat initial label
    label = Label()
    label.path = [orig]
    label.travel_time = 0
    label.dist = 0
    labels.append(label)

    count = 0

    while (len(labels) > 0):
        count += 1
        cur_label = labels.pop()

        # extend the current label
        last_node = cur_label.path[-1]
        for child in graph.successors(last_node):
            extended_label = copy.deepcopy(cur_label)
            arc = (last_node, child)

            # check the feasibility
            arrive_time = cur_label.travel_time + graph.edges[arc]["travel_time"]
            time_window = graph.nodes[child]["time_window"]
            if (arrive_time >= time_window[0] and arrive_time <= time_window[1] and last_node != dest):
                extended_label.path.append(child)
                extended_label.travel_time += graph.edges[arc]["travel_time"]
                extended_label.dist += graph.edges[arc]["cost"]
                labels.append(extended_label)

    path_dict[count] = cur_label
    # 调用dominance rule
    labels, path_dict = dominate(labels, path_dict)

    # filtering Paths, only keep feasible solutions
    path_dict_copy = copy.deepcopy(path_dict)
    for key in path_dict_copy.keys():
        if (path_dict[key].path[-1] != dest):
            path_dict.pop(key)

    # choose optimal solution
    opt_path = {}
    min_dist = 1e6
    for key in path_dict.keys():
        if (path_dict[key].dist < min_dist):
            min_dist = path_dict[key].dist
            opt_path[1] = path_dict[key]

    return graph, labels, path_dict, opt_path


def main():
    # 点中包含时间窗属性
    Nodes = {'s': (0, 0)
        , '1': (6, 14)
        , '2': (9, 12)
        , '3': (8, 12)
        , 't': (9, 15)
             }
    # 弧的属性包括travel_time与dist
    Arcs = {('s', '1'): (8, 3)
        , ('s', '2'): (5, 5)
        , ('s', '3'): (12, 2)
        , ('1', 't'): (4, 7)
        , ('2', 't'): (2, 6)
        , ('3', 't'): (4, 3)
            }

    # create the directed Graph
    graph = nx.DiGraph()
    cnt = 0
    # add nodes into the graph
    for name in Nodes.keys():
        cnt += 1
        graph.add_node(name
                       , time_window=(Nodes[name][0], Nodes[name][1])
                       , min_dist=0
                       )
    # add edges into the graph
    for key in Arcs.keys():
        graph.add_edge(key[0], key[1]
                       , travel_time=Arcs[key][0]
                       , cost=Arcs[key][1]
                       )

    org = 's'
    des = 't'
    begin_time = time()
    graph, labels, path_dict, opt_path = labeling_SPPRC(graph, org, des)
    end_time = time()
    print("计算时间： ", end_time - begin_time)
    print('optimal path : ', opt_path[1].path)
    print('optimal path (dist): ', opt_path[1].dist)

if __name__ == '__main__':
    main()

