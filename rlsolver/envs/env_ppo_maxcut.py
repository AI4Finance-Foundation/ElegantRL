import os
from typing import List, Tuple

import networkx as nx
import torch as th

TEN = th.Tensor

'''graph'''
GraphList = List[Tuple[int, int, int]]  # 每条边两端点的索引以及边的权重 List[Tuple[Node0ID, Node1ID, WeightEdge]]
IndexList = List[List[int]]  # 按索引顺序记录每个点的所有邻居节点 IndexList[Node0ID] = [Node1ID, ...]
DataDir = './data/graph_max_cut'  # 保存图最大割的txt文件的目录，txt数据以稀疏的方式记录了GraphList，可以重建图的邻接矩阵

'''load graph'''


def load_graph_list_from_txt(txt_path: str = 'G14.txt') -> GraphList:
    with open(txt_path, 'r') as file:
        lines = file.readlines()
        lines = [[int(i1) for i1 in i0.split()] for i0 in lines]
    num_nodes, num_edges = lines[0]
    graph_list = [(n0 - 1, n1 - 1, dt) for n0, n1, dt in lines[1:]]  # 将node_id 由“从1开始”改为“从0开始”

    assert num_nodes == obtain_num_nodes(graph_list=graph_list)
    assert num_edges == len(graph_list)
    return graph_list


def generate_graph_list(graph_type: str, num_nodes: int) -> GraphList:
    graph_types = ['ErdosRenyi', 'BarabasiAlbert', 'PowerLaw']
    assert graph_type in graph_types

    if graph_type == 'ErdosRenyi':
        g = nx.erdos_renyi_graph(n=num_nodes, p=0.15)
    elif graph_type == 'BarabasiAlbert':
        g = nx.barabasi_albert_graph(n=num_nodes, m=4)
    elif graph_type == 'PowerLaw':
        g = nx.powerlaw_cluster_graph(n=num_nodes, m=4, p=0.05)
    elif graph_type == 'BarabasiAlbert':
        g = nx.barabasi_albert_graph(n=num_nodes, m=4)
    else:
        raise ValueError(f"g_type {graph_type} should in {graph_types}")

    distance = 1
    graph_list = [(node0, node1, distance) for node0, node1 in g.edges]
    return graph_list


def load_graph_list(graph_name: str):
    import random
    graph_types = ['ErdosRenyi', 'PowerLaw', 'BarabasiAlbert']
    graph_type = next((graph_type for graph_type in graph_types if graph_type in graph_name), None)  # 匹配 graph_type

    if os.path.exists(f"{DataDir}/{graph_name}.txt"):
        txt_path = f"{DataDir}/{graph_name}.txt"
        graph_list = load_graph_list_from_txt(txt_path=txt_path)
    elif os.path.isfile(graph_name) and os.path.splitext(graph_name)[-1] == '.txt':
        txt_path = graph_name
        graph_list = load_graph_list_from_txt(txt_path=txt_path)

    elif graph_type and graph_name.find('ID') == -1:
        num_nodes = int(graph_name.split('_')[-1])
        graph_list = generate_graph_list(num_nodes=num_nodes, graph_type=graph_type)
    elif graph_type and graph_name.find('ID') >= 0:
        num_nodes, valid_i = graph_name.split('_')[-2:]
        num_nodes = int(num_nodes)
        valid_i = int(valid_i[len('ID'):])
        random.seed(valid_i)
        graph_list = generate_graph_list(num_nodes=num_nodes, graph_type=graph_type)
        random.seed()

    else:
        raise ValueError(f"DataDir {DataDir} | graph_name {graph_name} txt_path {DataDir}/{graph_name}.txt")
    return graph_list


'''adjacency matrix'''


def build_adjacency_indies(graph_list: GraphList, if_bidirectional: bool = False) -> (IndexList, IndexList):
    """
    用二维列表list2d表示这个图：
    [
        [1, 2],
        [],
        [3],
        [],
    ]
    其中：
    - list2d[0] = [1, 2]
    - list2d[2] = [3]

    对于稀疏的矩阵，可以直接记录每条边两端节点的序号，用shape=(2,N)的二维列表 表示这个图：
    0, 1
    0, 2
    2, 3
    如果条边的长度为1，那么表示为shape=(2,N)的二维列表，并在第一行，写上 4个节点，3条边的信息，帮助重建这个图，然后保存在txt里：
    4, 3
    0, 1, 1
    0, 2, 1
    2, 3, 1
    """
    num_nodes = obtain_num_nodes(graph_list=graph_list)

    n0_to_n1s = [[] for _ in range(num_nodes)]  # 将 node0_id 映射到 node1_id
    n0_to_dts = [[] for _ in range(num_nodes)]  # 将 mode0_id 映射到 node1_id 与 node0_id 的距离
    for n0, n1, distance in graph_list:
        n0_to_n1s[n0].append(n1)
        n0_to_dts[n0].append(distance)
        if if_bidirectional:
            n0_to_n1s[n1].append(n0)
            n0_to_dts[n1].append(distance)
    n0_to_n1s = [th.tensor(node1s) for node1s in n0_to_n1s]
    n0_to_dts = [th.tensor(node1s) for node1s in n0_to_dts]
    assert num_nodes == len(n0_to_n1s)
    assert num_nodes == len(n0_to_dts)

    '''sort'''
    for i, node1s in enumerate(n0_to_n1s):
        sort_ids = th.argsort(node1s)
        n0_to_n1s[i] = n0_to_n1s[i][sort_ids]
        n0_to_dts[i] = n0_to_dts[i][sort_ids]
    return n0_to_n1s, n0_to_dts


def obtain_num_nodes(graph_list: GraphList) -> int:
    return max([max(n0, n1) for n0, n1, distance in graph_list]) + 1


class SimulatorGraphMaxCut:
    def __init__(self, args, graph_list: GraphList = (),
                 device=th.device('cpu'), if_bidirectional: bool = False):
        self.device = device
        self.int_type = int_type = th.long
        self.if_bidirectional = if_bidirectional
        self.num_nodes = args.num_nodes
        self.num_envs = args.num_envs
        self.xs = None
        self.action_count = 0
        self.last_reward = None
        self.num_steps = args.num_steps

        n0_to_n1s, n0_to_dts = build_adjacency_indies(graph_list=graph_list, if_bidirectional=if_bidirectional)
        n0_to_n1s = [t.to(int_type).to(device) for t in n0_to_n1s]
        self.num_edges = len(graph_list)
        n0_to_n0s = [(th.zeros_like(n1s) + i) for i, n1s in enumerate(n0_to_n1s)]
        self.n0_ids = th.hstack(n0_to_n0s)[None, :]
        self.n1_ids = th.hstack(n0_to_n1s)[None, :]
        len_sim_ids = self.num_edges * (2 if if_bidirectional else 1)
        self.sim_ids = th.zeros(len_sim_ids, dtype=int_type, device=device)[None, :]

    def reset(self):
        self.xs = self.generate_xs_randomly(num_sims=self.num_envs)
        self.xs = self.xs.to(th.float)
        self.last_reward = self.calculate_obj_values().to(th.float)

        return self.xs

    def step(self, action):
        self.action_count += 1
        for n in range(self.num_envs):
            self.xs[n, action[n]] = th.logical_not(self.xs[n, action[n]])
        cur_reward = self.calculate_obj_values().to(th.float)
        reward = cur_reward - self.last_reward
        self.last_reward = cur_reward

        if self.action_count == self.num_steps:
            self.action_count = 0
            next_done = th.ones([self.num_envs], dtype=th.float, device=self.device)
        else:
            next_done = th.zeros([self.num_envs], dtype=th.float, device=self.device)

        return self.xs, reward, next_done, cur_reward

    def calculate_obj_values(self, if_sum: bool = True) -> TEN:
        xs = self.xs > 0
        num_sims = xs.shape[0]  # 并行维度，环境数量。xs, vs第一个维度， dim0 , 就是环境数量
        if num_sims != self.sim_ids.shape[0]:
            self.n0_ids = self.n0_ids[0].repeat(num_sims, 1)
            self.n1_ids = self.n1_ids[0].repeat(num_sims, 1)
            self.sim_ids = self.sim_ids[0:1] + th.arange(num_sims, dtype=self.int_type, device=self.device)[:, None]

        values = xs[self.sim_ids, self.n0_ids] ^ xs[self.sim_ids, self.n1_ids]
        if if_sum:
            values = values.sum(1)
        if self.if_bidirectional:
            values = values // 2
        return values

    def generate_xs_randomly(self, num_sims):
        xs = th.randint(0, 2, size=(num_sims, self.num_nodes), dtype=th.bool, device=self.device)
        xs[:, 0] = 0
        return xs
