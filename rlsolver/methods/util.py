import sys
import os
cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../rlsolver')
sys.path.append(os.path.dirname(rlsolver_path))

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import copy
from torch.autograd import Variable
import functools
import time
import numpy as np
from typing import Union, Tuple, List
import networkx as nx
from torch import Tensor
import torch as th
from rlsolver.methods.config import *
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
GraphList = List[Tuple[int, int, int]]
IndexList = List[List[int]]

TEN = th.Tensor

def plot_nxgraph(g: nx.Graph(), fig_filename='.result/fig.png'):
    import matplotlib.pyplot as plt
    nx.draw_networkx(g)
    plt.savefig(fig_filename)
    plt.show()



def transfer_nxgraph_to_adjacencymatrix(graph: nx.Graph):
    return nx.to_numpy_array(graph)

# the returned weightmatrix has the following format： node1 node2 weight
# For example: 1 2 3 // the weight of node1 and node2 is 3
def transfer_nxgraph_to_weightmatrix(graph: nx.Graph):
    # edges = nx.edges(graph)
    res = np.array([])
    edges = graph.edges()
    for u, v in edges:
        u = int(u)
        v = int(v)
        # weight = graph[u][v]["weight"]
        weight = float(graph.get_edge_data(u, v)["weight"])
        vec = np.array([u, v, weight])
        if len(res) == 0:
            res = vec
        else:
            res = np.vstack((res, vec))
    return res

# weightmatrix: format of each vector: node1 node2 weight
# num_nodes: num of nodes
def transfer_weightmatrix_to_nxgraph(weightmatrix: List[List[int]], num_nodes: int) -> nx.Graph():
    graph = nx.Graph()
    nodes = list(range(num_nodes))
    graph.add_nodes_from(nodes)
    for i, j, weight in weightmatrix:
        graph.add_edge(i, j, weight=weight)
    return graph



def calc_file_name(front: str, id2: int, val: int, end: str):
    return front + "_" + str(id2) + "_" + str(val) + end + "pkl"

def gpu_info_str(device) -> str:
    if not th.cuda.is_available():
        return 'th.cuda.is_available() == False'

    total_memory = th.cuda.get_device_properties(device).total_memory / (1024 ** 3)  # GB
    max_allocated = th.cuda.max_memory_allocated(device) / (1024 ** 3)  # GB
    memory_allocated = th.cuda.memory_allocated(device) / (1024 ** 3)  # GB

    return (f"RAM(GB) {memory_allocated:.2f} < {max_allocated:.2f} < {total_memory:.2f}  "
            f"Rate {(max_allocated / total_memory):5.2f}")

def evolutionary_replacement(xs: TEN, vs: TEN, low_k: int, if_maximize: bool):
    num_sims = xs.shape[0]

    ids = vs.argsort()
    top_ids, low_ids = (ids[:-low_k], ids[-low_k:]) if if_maximize else (ids[:low_k], ids[low_k:])
    replace_ids = top_ids[th.randperm(num_sims - low_k, device=xs.device)[:low_k]]
    xs[replace_ids] = xs[low_ids]
    vs[replace_ids] = vs[low_ids]

def get_hot_image_of_graph(adj_bool, hot_type):
    if hot_type == 'avg':
        adj_matrix = adj_bool.float() / adj_bool.shape[0]
    elif hot_type == 'sum':
        adj_matrix = adj_bool.float()
        adj_matrix /= adj_matrix.sum(dim=1, keepdim=True).clip(1, None)
        adj_matrix /= adj_matrix.sum(dim=0, keepdim=True).clip(1, None)
        adj_matrix = adj_matrix * 0.25
    else:
        raise ValueError(f"| get_hot_image_of_graph() hot_type {hot_type} should in ['avg', 'sum']")

    num_nodes = adj_matrix.size(0)
    num_iters = int(th.tensor(num_nodes).log().item() + 1) * 2
    device = adj_matrix.device
    break_thresh = 2 ** -10

    log_matrix = None

    hot_matrix = adj_matrix.clone()
    adjust_eye = th.eye(num_nodes, device=device)
    prev_diff = 0
    for i in range(num_iters):
        hot_matrix = th.matmul(hot_matrix + adjust_eye, adj_matrix)
        hot_matrix = hot_matrix + hot_matrix.t()

        log_matrix = th.log(hot_matrix.clip(1e-12, 1e+12))

        curr_diff = log_matrix.std()
        if abs(prev_diff - curr_diff) < (prev_diff + curr_diff) * break_thresh:
            break
        # print(f'{num_iters:6} {i:6}  {curr_diff:9.3e}')
        prev_diff = curr_diff

    returns = (log_matrix - log_matrix.mean()) / (log_matrix.std() * 3)
    return returns

def get_adjacency_distance_matrix(adj_bool_ary):
    graph = nx.from_numpy_array(adj_bool_ary)
    # '''graph_list -> graph'''
    # graph = nx.Graph()
    # for n0, n1, distance in graph_list:
    #     graph.add_edge(n0, n1, weight=distance)

    dist_matrix = nx.floyd_warshall_numpy(graph)
    dist_matrix[dist_matrix == 0] = 0.5
    return 1.0 / dist_matrix

def detach_var(v, device):
    var = Variable(v.data, requires_grad=True).to(device)
    var.retain_grad()
    return var


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def plot_fig(scores: List[int], label: str):
    import matplotlib.pyplot as plt
    plt.figure()
    x = list(range(len(scores)))
    dic = {'0': 'ro-', '1': 'gs', '2': 'b^', '3': 'c>', '4': 'm<', '5': 'yp'}
    plt.plot(x, scores, dic['0'])
    plt.legend([label], loc=0)
    plt.savefig('../result/' + label + '.png')
    plt.show()

def plot_fig_over_durations(objs: List[int], durations: List[int], label: str):
    import matplotlib.pyplot as plt
    plt.figure()
    x = durations
    dic = {'0': 'ro-', '1': 'gs', '2': 'b^', '3': 'c>', '4': 'm<', '5': 'yp'}
    # plt.ylim(0, max(objs))
    plt.plot(x, objs, dic['0'])
    plt.legend([label], loc=0)
    plt.savefig('./result/' + label + '.png')
    plt.show()

def calc_txt_files_with_prefixes(directory: str, prefixes: List[str]):
    res = []
    files = os.listdir(directory)
    for prefix in prefixes:
        for file in files:
            if file.startswith(prefix):
                res.append(directory + '/' + file)
    return res

def calc_files_with_prefix_suffix(directory: str, prefix: str, suffix: str, extension: str = '.txt'):
    res = []
    files = os.listdir(directory)
    new_suffix = '_' + suffix + extension
    for file in files:
        if file.startswith(prefix) and file.endswith(new_suffix):
            res.append(directory + '/' + file)
    return res

# if the file name is '../data/BA_100.txt', the return is '../result/BA_100.txt'
# if the file name is '../result/BA_100.txt', the return is '../result/BA_100.txt'
# if the file name is '../data/syn_BA/BA_100.txt', the return is '../result/BA_100.txt'
def calc_result_file_name(file: str, add_tail: str= ''):
    new_file = copy.deepcopy(file)
    if 'data' in new_file:
        new_file = new_file.replace('data', 'result')
    new_file = new_file.split('result')[0] + 'result/' + new_file.split('/')[-1]
    if add_tail is not None:
        new_file = new_file.replace('.txt', '') + add_tail + '.txt'
    return new_file

# For example, syn_10_21_3601.txt, the prefix is 'syn_10_', time_limit is 3600 (seconds).
# The gap and running_duration are also be calculated.
def calc_avg_std_of_obj(directory: str, prefix: str, time_limit: int):
    init_time_limit = copy.deepcopy(time_limit)
    objs = []
    gaps = []
    obj_bounds = []
    running_durations = []
    suffix = str(time_limit)
    files = calc_files_with_prefix_suffix(directory, prefix, suffix)
    for i in range(len(files)):
        with open(files[i], 'r') as file:
            line = file.readline()
            assert 'obj' in line
            obj = float(line.split('obj:')[1].split('\n')[0])
            objs.append(obj)

            line2 = file.readline()
            running_duration_ = line2.split('running_duration:')
            running_duration = float(running_duration_[1]) if len(running_duration_) >= 2 else None
            running_durations.append(running_duration)

            line3 = file.readline()
            gap_ = line3.split('gap:')
            gap = float(gap_[1]) if len(gap_) >= 2 else None
            gaps.append(gap)

            line4 = file.readline()
            obj_bound_ = line4.split('obj_bound:')
            obj_bound = float(obj_bound_[1]) if len(obj_bound_) >= 2 else None
            obj_bounds.append(obj_bound)
    if len(objs) == 0:
        return
    avg_obj = np.average(objs)
    std_obj = np.std(objs)
    avg_running_duration = np.average(running_durations)
    avg_gap = np.average(gaps) if None not in gaps else None
    avg_obj_bound = np.average(obj_bounds) if None not in obj_bounds else None
    print(f'{directory} prefix {prefix}, suffix {suffix}: avg_obj {avg_obj}, std_obj {std_obj}, avg_running_duration {avg_running_duration}, avg_gap {avg_gap}, avg_obj_bound {avg_obj_bound}')
    if time_limit != init_time_limit:
        print()
    return {(prefix, time_limit): (avg_obj, std_obj, avg_running_duration, avg_gap, avg_obj_bound)}

def calc_avg_std_of_objs(directory: str, prefixes: List[str], time_limits: List[int]):
    res = []
    for i in range(len(prefixes)):
        for k in range(len(time_limits)):
            avg_std = calc_avg_std_of_obj(directory, prefixes[i], int(time_limits[k]))
            res.append(avg_std)
    return res

# transfer flot to binary. For example, 1e-7 -> 0, 1 + 1e-8 -> 1
def transfer_float_to_binary(value: float) -> int:
    if abs(value) < 1e-4:
        value = 0
    elif abs(value - 1) < 1e-4:
        value = 1
    else:
        raise ValueError('wrong value')
    return value

def fetch_node(line: str):
    if 'x[' in line:
        node = int(line.split('x[')[1].split(']')[0])
    else:
        node = None
    return node

# e.g., s = "// time_limit: ('TIME_LIMIT', <class 'float'>, 36.0, 0.0, inf, inf)",
# then returns 36
def obtain_first_number(s: str):
    res = ''
    pass_first_digit = False
    for i in range(len(s)):
        if s[i].isdigit() or s[i] == '.':
            res += s[i]
            pass_first_digit = True
        elif pass_first_digit:
            break
    value = int(float(res))
    return value


def load_graph_from_txt(txt_path: str = './data/gset_14.txt'):
    with open(txt_path, 'r') as file:
        lines = file.readlines()
        lines = [[int(i1) for i1 in i0.split()] for i0 in lines]
    num_nodes, num_edges = lines[0]
    graph = [(n0 - 1, n1 - 1, dt) for n0, n1, dt in lines[1:]]  # node_id “从1开始”改为“从0开始”
    return graph, num_nodes, num_edges

def get_adjacency_matrix(graph, num_nodes):
    adjacency_matrix = np.empty((num_nodes, num_nodes))
    adjacency_matrix[:] = -1  # 选用-1而非0表示表示两个node之间没有edge相连，避免两个节点的距离为0时出现冲突
    for n0, n1, dt in graph:
        adjacency_matrix[n0, n1] = dt
    return adjacency_matrix

# def load_graph(graph_name: str):
#     data_dir = DATA_DIR
#     graph_types = GRAPH_DISTRI_TYPES
#     if os.path.exists(f"{data_dir}/{graph_name}.txt"):
#         txt_path = f"{data_dir}/{graph_name}.txt"
#         graph, num_nodes, num_edges = load_graph_from_txt(txt_path=txt_path)
#     elif graph_name.split('_')[0] in graph_types:
#         g_type, num_nodes = graph_name.split('_')
#         num_nodes = int(num_nodes)
#         graph, num_nodes, num_edges = generate_graph(num_nodes=num_nodes, g_type=g_type)
#     else:
#         raise ValueError(f"graph_name {graph_name}")
#     return graph, num_nodes, num_edges
#
# def load_graph_auto(graph_name: str):
#     import random
#     graph_types = GRAPH_DISTRI_TYPES
#     if os.path.exists(f"{DataDir}/{graph_name}.txt"):
#         txt_path = f"{DataDir}/{graph_name}.txt"
#         graph = load_graph_from_txt(txt_path=txt_path)
#     elif graph_name.split('_')[0] in graph_types and len(graph_name.split('_')) == 3:
#         graph_type, num_nodes, valid_i = graph_name.split('_')
#         num_nodes = int(num_nodes)
#         valid_i = int(valid_i[len('ID'):])
#         random.seed(valid_i)
#         graph = generate_graph(num_nodes=num_nodes, graph_type=graph_type)
#         random.seed()
#     elif graph_name.split('_')[0] in graph_types and len(graph_name.split('_')) == 2:
#         graph_type, num_nodes = graph_name.split('_')
#         num_nodes = int(num_nodes)
#         graph = generate_graph(num_nodes=num_nodes, graph_type=graph_type)
#     else:
#         raise ValueError(f"DataDir {DataDir} | graph_name {graph_name}")
#     return graph

def save_graph_info_to_txt(txt_path, graph, num_nodes, num_edges):
    formatted_content = f"{num_nodes} {num_edges}\n"
    for node0, node1, distance in graph:
        row = [node0 + 1, node1 + 1, distance]  # node+1 is a bad design
        formatted_content += " ".join(str(item) for item in row) + "\n"
    with open(txt_path, "w") as file:
        file.write(formatted_content)


def build_adjacency_matrix(graph, num_nodes):
    adjacency_matrix = np.empty((num_nodes, num_nodes))
    adjacency_matrix[:] = -1  # 选用-1而非0表示表示两个node之间没有edge相连，避免两个节点的距离为0时出现冲突
    for n0, n1, dt in graph:
        adjacency_matrix[n0, n1] = dt
    return adjacency_matrix

def build_adjacency_matrix_auto(graph: GraphList, if_bidirectional: bool = False):
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
    not_connection = -1  # 选用-1去表示表示两个node之间没有edge相连，不选用0是为了避免两个节点的距离为0时出现冲突
    print(f"graph before enter: {graph}")
    num_nodes = obtain_num_nodes_auto(graph=graph)

    adjacency_matrix = th.zeros((num_nodes, num_nodes), dtype=th.float32)
    adjacency_matrix[:] = not_connection
    for n0, n1, distance in graph:
        adjacency_matrix[n0, n1] = distance
        if if_bidirectional:
            adjacency_matrix[n1, n0] = distance
    return adjacency_matrix

def build_adjacency_indies_auto(graph, if_bidirectional: bool = False) -> (IndexList, IndexList):
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
    num_nodes = obtain_num_nodes_auto(graph=graph)

    n0_to_n1s = [[] for _ in range(num_nodes)]  # 将 node0_id 映射到 node1_id
    n0_to_dts = [[] for _ in range(num_nodes)]  # 将 mode0_id 映射到 node1_id 与 node0_id 的距离
    for n0, n1, distance in graph:
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

def obtain_num_nodes_auto(graph: GraphList) -> int:
    return max([max(n0, n1) for n0, n1, distance in graph]) + 1


def convert_matrix_to_vector(matrix):
    vector = [row[i + 1:] for i, row in enumerate(matrix)]
    return th.hstack(vector)





def read_solution(filename: str):
    with open(filename, 'r') as file:
        # lines = []
        line = file.readline()
        while True:
            if '// num_nodes:' in line:
                strings = line.split("// num_nodes:")
                num_nodes = int(strings[1])
                solution = [0] * num_nodes
            if '//' not in line and len(line) >= 1:
                strings = line.split(" ")
                node = int(strings[0]) - 1
                label = int(strings[1]) - 1
                solution[node] = label
            if len(line) == 0:
                break
            line = file.readline()
    return solution



if __name__ == '__main__':
    s = "// time_limit: ('TIME_LIMIT', <class 'float'>, 36.0, 0.0, inf, inf)"
    val = obtain_first_number(s)

    # directory = 'result'
    # prefix = 'syn_10_'
    # time_limit = 3600
    # avg_std = calc_avg_std_of_obj(directory, prefix, time_limit)

    if_calc_avg_std = False
    if if_calc_avg_std:
        directory_result = 'result'
        prefixes = ['BA_100_', 'BA_200_']
        time_limits = GUROBI_TIME_LIMITS
        avgs_stds = calc_avg_std_of_objs(directory_result, prefixes, time_limits)

    # filename = 'result/syn_10_21_1800.sta'
    # new_filename = 'result/syn_10_21_1800.txt'
    # transfer_write_solver_result(filename, new_filename)

    # from_extension = '.sov'
    # to_extension = '.txt'
    # transfer_write_solver_results(directory_result, prefixes, time_limits, from_extension, to_extension)









    print()
