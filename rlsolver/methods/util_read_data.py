import sys
import os
cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../rlsolver')
sys.path.append(os.path.dirname(rlsolver_path))

from typing import List, Tuple, Union
from rlsolver.methods.config import GRAPH_TYPE
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import networkx as nx
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import torch as th
GraphList = List[Tuple[int, int, int]]  # 每条边两端点的索引以及边的权重 List[Tuple[Node0ID, Node1ID, WeightEdge]]
IndexList = List[List[int]]  # 按索引顺序记录每个点的所有邻居节点 IndexList[Node0ID] = [Node1ID, ...]

GraphTypes = ['BarabasiAlbert', 'ErdosRenyi', 'PowerLaw']
TEN = th.Tensor

from rlsolver.methods.util import calc_txt_files_with_prefixes

# read graph file, e.g., gset_14.txt, as networkx.Graph
# The nodes in file start from 1, but the nodes start from 0 in our codes.
def read_nxgraph(filename: str) -> nx.Graph():
    graph = nx.Graph()
    with open(filename, 'r') as file:
        # lines = []
        line = file.readline()
        is_first_line = True
        while line is not None and line != '':
            if '//' not in line:
                if is_first_line:
                    strings = line.split(" ")
                    num_nodes = int(strings[0])
                    num_edges = int(strings[1])
                    nodes = list(range(num_nodes))
                    graph.add_nodes_from(nodes)
                    is_first_line = False
                else:
                    node1, node2, weight = line.split(" ")
                    graph.add_edge(int(node1) - 1, int(node2) - 1, weight=weight)
            line = file.readline()
    return graph

def read_nxgraphs(directory: str, prefixes: List[str]) -> List[nx.Graph]:
    graphs = []
    files = calc_txt_files_with_prefixes(directory, prefixes)
    for i in range(len(files)):
        filename = files[i]
        graph = read_nxgraph(filename)
        graphs.append(graph)
    return graphs

def read_graphlist(filename: str) -> GraphList:
    with open(filename, 'r') as file:
        lines = file.readlines()
        lines = [[int(i1) for i1 in i0.split()] for i0 in lines]
    num_nodes, num_edges = lines[0]
    graph_list = [(n0 - 1, n1 - 1, dt) for n0, n1, dt in lines[1:]]  # 将node_id 由“从1开始”改为“从0开始”
    return graph_list

def generate_graph_list(graph_type: str, num_nodes: int) -> GraphList:
    graph_types = GraphTypes
    assert graph_type in graph_types

    if graph_type == 'BarabasiAlbert':
        g = nx.barabasi_albert_graph(n=num_nodes, m=4)
    elif graph_type == 'ErdosRenyi':
        g = nx.erdos_renyi_graph(n=num_nodes, p=0.15)
    elif graph_type == 'PowerLaw':
        g = nx.powerlaw_cluster_graph(n=num_nodes, m=4, p=0.05)
    else:
        raise ValueError(f"g_type {graph_type} should in {graph_types}")

    distance = 1
    graph_list = [(node0, node1, distance) for node0, node1 in g.edges]
    return graph_list


def load_graph_list(graph_name: str, if_force_exist: bool = False):
    import random
    graph_type = GRAPH_TYPE  # 匹配 graph_type
    DataDir = './data/syn_graph_list'  # 保存图最大割的txt文件的目录，txt数据以稀疏的方式记录了GraphList，可以重建图的邻接矩阵

    if if_force_exist:
        txt_path = f"{DataDir}/{graph_name}.txt"
        if_exist = os.path.exists(txt_path)
        print(f"| txt_path {txt_path} not exist") if not if_exist else None
        assert if_exist
    if os.path.exists(f"{DataDir}/{graph_name}.txt"):
        txt_path = f"{DataDir}/{graph_name}.txt"
        graph_list = read_graphlist(filename=txt_path)
    elif os.path.isfile(graph_name) and os.path.splitext(graph_name)[-1] == '.txt':
        txt_path = graph_name
        graph_list = read_graphlist(filename=txt_path)

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

def obtain_num_nodes(graph_list: GraphList) -> int:
    return max([max(n0, n1) for n0, n1, distance in graph_list]) + 1

def build_adjacency_matrix(graph_list: GraphList, if_bidirectional: bool = False) -> TEN:
    """例如，无向图里：
    - 节点0连接了节点1，边的权重为1
    - 节点0连接了节点2，边的权重为2
    - 节点2连接了节点3，边的权重为3

    用邻接阶矩阵Ary的上三角表示这个无向图：
      0 1 2 3
    0 F T T F
    1 _ F F F
    2 _ _ F T
    3 _ _ _ F

    其中：
    - Ary[0,1]=边的权重为1
    - Ary[0,2]=边的权重为2
    - Ary[2,3]=边的权重为3
    - 其余为-1，表示False 节点之间没有连接关系
    """
    not_connection = -1  # 选用-1去表示表示两个node之间没有edge相连，不选用0是为了避免两个节点的距离为0时出现冲突
    num_nodes = obtain_num_nodes(graph_list=graph_list)

    adjacency_matrix = th.zeros((num_nodes, num_nodes), dtype=th.float32)
    adjacency_matrix[:] = not_connection
    for n0, n1, distance in graph_list:
        adjacency_matrix[n0, n1] = distance
        if if_bidirectional:
            adjacency_matrix[n1, n0] = distance
    return adjacency_matrix


def build_adjacency_bool(graph_list: GraphList, num_nodes: int = 0, if_bidirectional: bool = False) -> TEN:
    """例如，无向图里：
    - 节点0连接了节点1
    - 节点0连接了节点2
    - 节点2连接了节点3

    用邻接阶矩阵Ary的上三角表示这个无向图：
      0 1 2 3
    0 F T T F
    1 _ F F F
    2 _ _ F T
    3 _ _ _ F

    其中：
    - Ary[0,1]=True
    - Ary[0,2]=True
    - Ary[2,3]=True
    - 其余为False
    """
    if num_nodes == 0:
        num_nodes = obtain_num_nodes(graph_list=graph_list)

    adjacency_bool = th.zeros((num_nodes, num_nodes), dtype=th.bool)
    node0s, node1s = list(zip(*graph_list))[:2]
    adjacency_bool[node0s, node1s] = True
    if if_bidirectional:
        adjacency_bool = th.logical_or(adjacency_bool, adjacency_bool.T)
    return adjacency_bool


def build_graph_list(adjacency_bool: TEN) -> GraphList:
    num_nodes = adjacency_bool.shape[0]

    graph_list = []
    for node_i in range(1, num_nodes):
        for node_j in range(node_i):
            edge_weight = int(adjacency_bool[node_i, node_j])
            if edge_weight > 0:
                graph_list.append((node_i, node_j, edge_weight))
    return graph_list


def check_convert_between_graph_list_and_adjacency_bool():
    num_nodes = 8
    adjacency_bool = th.tril(th.randint(0, 2, size=(num_nodes, num_nodes), dtype=th.bool))

    graph_list = build_graph_list(adjacency_bool)
    print("Original  graph list:", graph_list)
    adjacency_bool = build_adjacency_bool(graph_list, num_nodes, if_bidirectional=True)
    graph_list = build_graph_list(adjacency_bool)
    print("Converted graph list:", graph_list)


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



def update_xs_by_vs(xs0: TEN, vs0: TEN, xs1: TEN, vs1: TEN, if_maximize: bool) -> int:
    """
    并行的子模拟器数量为 num_sims, 解x 的节点数量为 num_nodes
    xs: 并行数量个解x,xs.shape == (num_sims, num_nodes)
    vs: 并行数量个解x对应的 objective value. vs.shape == (num_sims, )

    更新后，将xs1，vs1 中 objective value数值更高的解x 替换到xs0，vs0中
    如果被更新的解的数量大于0，将返回True
    """
    good_is = vs1.ge(vs0) if if_maximize else vs1.le(vs0)
    xs0[good_is] = xs1[good_is]
    vs0[good_is] = vs1[good_is]
    return good_is.shape[0]

def pick_xs_by_vs(xs: TEN, vs: TEN, num_repeats: int, if_maximize: bool) -> (TEN, TEN):
    # update good_xs: use .view() instead of .reshape() for saving GPU memory
    num_nodes = xs.shape[1]
    num_sims = xs.shape[0] // num_repeats

    xs_view = xs.view(num_repeats, num_sims, num_nodes)
    vs_view = vs.view(num_repeats, num_sims)
    ids = vs_view.argmax(dim=0) if if_maximize else vs_view.argmin(dim=0)

    sim_ids = th.arange(num_sims, device=xs.device)
    good_xs = xs_view[ids, sim_ids]
    good_vs = vs_view[ids, sim_ids]
    return good_xs, good_vs

# def read_set_cover(filename: str):
#     with open(filename, 'r') as file:
#         # lines = []
#         line = file.readline()
#         item_matrix = []
#         while line is not None and line != '':
#             if 'p set' in line:
#                 strings = line.split(" ")
#                 num_items = int(strings[-2])
#                 num_sets = int(strings[-1])
#             elif 's' in line:
#                 strings = line.split(" ")
#                 items = [int(s) for s in strings[1:]]
#                 item_matrix.append(items)
#             else:
#                 raise ValueError("error in read_set_cover")
#             line = file.readline()
#     return num_items, num_sets, item_matrix

def read_knapsack_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        N, W = map(int, lines[0].split())
        items = []
        for line in lines[1:]:
            weight, value = map(int, line.split())
            items.append((weight, value))
    return N, W, items


def read_set_cover_data(filename):
    with open(filename, 'r') as file:
        first_line = file.readline()
        total_elements, total_subsets = map(int, first_line.split())
        subsets = []
        for line in file:
            subset = list(map(int, line.strip().split()))
            subsets.append(subset)

    return total_elements, total_subsets, subsets




if __name__ == '__main__':

    read_txt = True
    if read_txt:
        graph1 = read_nxgraph('../data/gset/gset_14.txt')
        graph2 = read_nxgraph('../data/syn_BA/BA_100_ID0.txt')
