import os
import sys
import torch as th
import networkx as nx
from typing import List, Tuple

'''graph'''

TEN = th.Tensor

GraphList = List[Tuple[int, int, int]]  # 每条边两端点的索引以及边的权重 List[Tuple[Node0ID, Node1ID, WeightEdge]]
IndexList = List[List[int]]  # 按索引顺序记录每个点的所有邻居节点 IndexList[Node0ID] = [Node1ID, ...]
DataDir = '../../data/gset'  # 保存图最大割的txt文件的目录，txt数据以稀疏的方式记录了GraphList，可以重建图的邻接矩阵
GraphTypes = ['BarabasiAlbert', 'ErdosRenyi', 'PowerLaw']

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


def save_graph_list_to_txt(graph_list: GraphList, txt_path: str):
    num_nodes = obtain_num_nodes(graph_list=graph_list)
    num_edges = len(graph_list)

    lines = [f"{num_nodes} {num_edges}", ]
    lines.extend([f"{n0 + 1} {n1 + 1} {distance}" for n0, n1, distance in graph_list])
    lines = [line + '\n' for line in lines]
    with open(txt_path, 'w') as file:
        file.writelines(lines)


def check_load_graph_list_from_txt_and_save_graph_list_to_txt():
    graph_list1 = load_graph_list_from_txt(txt_path=f"./data/graph_max_cut/gset_14.txt")

    save_graph_list_to_txt(graph_list=graph_list1, txt_path='temp.txt')
    graph_list2 = load_graph_list(graph_name=f"temp.txt")

    for i in range(len(graph_list1)):
        assert graph_list1[i] == graph_list2[i]
    print("finish")


def check_save_graph_list_to_txt():
    graph_types = ['ErdosRenyi', 'BarabasiAlbert', 'PowerLaw']

    for graph_type in graph_types:
        data_dir = f"./data/syn_{graph_type}"
        os.makedirs(data_dir, exist_ok=True)
        for num_nodes in range(100, 2000 + 1, 100):
            for graph_id in range(30):
                graph_name = f"{graph_type}_{num_nodes}_ID{graph_id}"
                graph_list = load_graph_list(graph_name=graph_name)

                txt_path = f"{data_dir}/{graph_name}.txt"
                save_graph_list_to_txt(graph_list=graph_list, txt_path=txt_path)
                print(txt_path)


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
    graph_types = GraphTypes
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
            edge_weight = adjacency_bool[node_i, node_j]
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


'''get_hot_tensor_of_graph'''


def show_array2d(ary, title='array2d', if_save=False):
    import matplotlib.pyplot as plt

    if isinstance(ary, th.Tensor):
        ary = ary.cpu().data.numpy()
    # assert isinstance(show_array, np.ndarray)

    plt.cla()
    plt.imshow(ary, cmap='hot', interpolation='nearest')
    plt.colorbar(label='hot map')
    plt.title(title)
    plt.tight_layout()

    if if_save:
        plt.savefig(f"hot_image_{title}.jpg", dpi=400)
        plt.close('all')
    else:
        plt.show()


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
        # print(f';;; {num_iters:6} {i:6}  {curr_diff:9.3e}')
        prev_diff = curr_diff

    return (log_matrix - log_matrix.mean()) / (log_matrix.std() * 3)


def get_adjacency_distance_matrix(adj_bool_ary):
    graph = nx.from_numpy_array(adj_bool_ary)
    # '''graph_list -> graph'''
    # graph = nx.Graph()
    # for n0, n1, distance in graph_list:
    #     graph.add_edge(n0, n1, weight=distance)

    dist_matrix = nx.floyd_warshall_numpy(graph)
    return 1.0 / dist_matrix


def check_get_hot_tenor_of_graph():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    if_save = True

    graph_names = []
    for graph_type in ['ErdosRenyi', 'PowerLaw', 'BarabasiAlbert']:
        for num_nodes in (128, 1024):
            for seed_id in range(2):
                graph_names.append(f'{graph_type}_{num_nodes}_ID{seed_id}')
    for gset_id in (14, 15, 49, 50, 22, 55, 70):  # todo
        graph_names.append(f"gset_{gset_id}")

    for graph_name in graph_names:
        graph_list: GraphList = load_graph_list(graph_name=graph_name)

        graph = nx.Graph()
        for n0, n1, distance in graph_list:
            graph.add_edge(n0, n1, weight=distance)

        for hot_type in ('avg', 'sum'):
            adj_bool = build_adjacency_bool(graph_list=graph_list, if_bidirectional=True).to(device)
            hot_array = get_hot_image_of_graph(adj_bool=adj_bool, hot_type=hot_type).cpu().data.numpy()
            title = f"{hot_type}_{graph_name}_N{graph.number_of_nodes()}_E{graph.number_of_edges()}"
            show_array2d(ary=hot_array, title=title, if_save=if_save)
            print(f"title {title}")

    print()


def check_plot_the_distances_between_each_pair_of_nodes():
    graph_type, num_nodes, graph_id = 'PowerLaw', 512, 0
    graph_list = load_graph_list(f"{graph_type}_{num_nodes}_ID{graph_id}")

    graph = nx.Graph()
    for n0, n1, distance in graph_list:
        graph.add_edge(n0, n1, weight=distance)

    adj_matrix = build_adjacency_matrix(graph_list=graph_list, if_bidirectional=True)
    show_array2d(adj_matrix)

    graph = nx.from_numpy_array(A=adj_matrix.eq(1).cpu().data.numpy())
    dist_matrix = nx.floyd_warshall_numpy(graph)
    show_array2d(1.0 / dist_matrix, title='1/(the shortest distance)')

    # adj_matrix = np.zeros((num_nodes, num_nodes), dtype=bool)
    # for num_hop_radius in range(1, 5):
    #     for node_i in range(num_nodes):
    #         hop_neighbors = nx.ego_graph(graph, node_i, radius=num_hop_radius).nodes
    #         adj_matrix[np.full_like(hop_neighbors, fill_value=node_i), hop_neighbors] = True
    #     show_array2d(adj_matrix, title=f'num_hop_radius {num_hop_radius}')


'''local search'''


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


def evolutionary_replacement(xs: TEN, vs: TEN, low_k: int, if_maximize: bool):
    num_sims = xs.shape[0]

    ids = vs.argsort()
    top_ids, low_ids = (ids[:-low_k], ids[-low_k:]) if if_maximize else (ids[:low_k], ids[low_k:])
    replace_ids = top_ids[th.randperm(num_sims - low_k, device=xs.device)[:low_k]]
    xs[replace_ids] = xs[low_ids]
    vs[replace_ids] = vs[low_ids]


'''utils'''


def obtain_num_nodes(graph_list: GraphList) -> int:
    return max([max(n0, n1) for n0, n1, distance in graph_list]) + 1


def gpu_info_str(device) -> str:
    if not th.cuda.is_available():
        return 'th.cuda.is_available() == False'

    total_memory = th.cuda.get_device_properties(device).total_memory / (1024 ** 3)  # GB
    max_allocated = th.cuda.max_memory_allocated(device) / (1024 ** 3)  # GB
    memory_allocated = th.cuda.memory_allocated(device) / (1024 ** 3)  # GB

    return (f"RAM(GB) {memory_allocated:.2f} < {max_allocated:.2f} < {total_memory:.2f}  "
            f"Rate {(max_allocated / total_memory):5.2f}")


if __name__ == '__main__':
    check_convert_between_graph_list_and_adjacency_bool()
    check_get_hot_tenor_of_graph()
