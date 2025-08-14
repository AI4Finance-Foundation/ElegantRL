import os
import sys

cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../rlsolver')
sys.path.append(os.path.dirname(rlsolver_path))

import sys
sys.path.append('..')

"""
pip install torch_geometric
"""
from config import (GPU_ID,
                    calc_device)
import os, random
import torch
import sys
from torch_geometric.data import Data
# from evaluator import EncoderBase64
# from graph_max_cut_simulator import SimulatorGraphMaxCut, load_graph_list
import time
import torch as th
from typing import List, Tuple, Union
# from graph_max_cut_simulator import SimulatorGraphMaxCut
# from graph_max_cut_local_search import SolverLocalSearch
import networkx as nx
import numpy as np

fix_seed = True
if fix_seed:
    seed = 74
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# class Config:
test_sampling_speed = False

GPU_ID = 0
device = calc_device(GPU_ID)
total_mcmc_num = 512

max_epoch_num = 30
sample_epoch_num = 8
repeat_times = 1

show_gap = 2 ** 4

num_ls = 6
reset_epoch_num = 192

total_running_duration = 2000


DataDir = '../data/gset'  # 保存图最大割的txt文件的目录，txt数据以稀疏的方式记录了GraphList，可以重建图的邻接矩阵
path = '../data/gset/gset_14.txt'
# path = '../data/syn_BA/BA_100_ID0.txt'
# num_ls = 6
# reset_epoch_num = 192
# total_mcmc_num = 224
# path = 'data/gset_22.txt'

# num_ls = 8
# reset_epoch_num = 128
# total_mcmc_num = 256
# path = 'data/gset_55.txt'

# num_ls = 8
# reset_epoch_num = 256
# total_mcmc_num = 192
# path = 'data/gset_70.txt'

# num_ls = 8
# reset_epoch_num = 256
# repeat_times = 512
# total_mcmc_num = 2048
# path = 'data/gset_22.txt'  # GPU RAM 40GB

# num_ls = 8
# reset_epoch_num = 192
# repeat_times = 448
# total_mcmc_num = 1024
# path = 'data/gset_55.txt'  # GPU RAM 40GB

# num_ls = 8
# reset_epoch_num = 320
# repeat_times = 288
# total_mcmc_num = 768
# path = 'data/gset_70.txt'  # GPU RAM 40GB
    

TEN = th.Tensor
ARY = np.ndarray

GraphList = List[Tuple[int, int, int]]  # 每条边两端点的索引以及边的权重 List[Tuple[Node0ID, Node1ID, WeightEdge]]
IndexList = List[List[int]]  # 按索引顺序记录每个点的所有邻居节点 IndexList[Node0ID] = [Node1ID, ...]



class EncoderBase64:
    def __init__(self, encode_len: int):
        num_power = 6
        self.encode_len = encode_len
        self.string_len = -int(-(encode_len / num_power) // 1)  # ceil(num_nodes / num_power)

        self.base_digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_$"
        self.base_num = len(self.base_digits)
        assert self.base_num == 2 ** num_power

    def bool_to_str(self, x_bool: Union[TEN, ARY]) -> str:
        x_int = int(''.join([('1' if i else '0') for i in x_bool.tolist()]), 2)

        '''bin_int_to_str'''
        base_num = len(self.base_digits)
        x_str = ""
        while True:
            remainder = x_int % base_num
            x_str = self.base_digits[remainder] + x_str
            x_int //= base_num
            if x_int == 0:
                break

        if len(x_str) > 120:
            x_str = '\n'.join([x_str[i:i + 120] for i in range(0, len(x_str), 120)])
        if len(x_str) > 64:
            x_str = f"\n{x_str}"
        return x_str.zfill(self.string_len)

    def str_to_bool(self, x_str: str) -> TEN:
        x_b64 = x_str.replace('\n', '').replace(' ', '')

        '''b64_str_to_int'''
        x_int = 0
        base_len = len(x_b64)
        for i in range(base_len):
            digit = self.base_digits.index(x_b64[i])
            power = base_len - 1 - i
            x_int += digit * (self.base_num ** power)

        x_bin: str = bin(x_int)[2:]
        x_bool = th.zeros(self.encode_len, dtype=th.bool)
        x_bool[-len(x_bin):] = th.tensor([int(i) for i in x_bin], dtype=th.bool)
        return x_bool


class SolverLocalSearch:
    def __init__(self, simulator, num_nodes: int):
        self.simulator = simulator
        self.num_nodes = num_nodes

        self.num_sims = 0
        self.good_xs = th.tensor([])  # solution x
        self.good_vs = th.tensor([])  # objective value

    def reset(self, xs: TEN):
        vs = self.simulator.calculate_obj_values(xs=xs)

        self.good_xs = xs
        self.good_vs = vs
        self.num_sims = xs.shape[0]
        return vs

    def reset_search(self, num_sims):
        xs = th.empty((num_sims, self.num_nodes), dtype=th.bool, device=device)
        for sim_id in range(num_sims):
            _xs = self.simulator.generate_xs_randomly(num_sims=num_sims)
            _vs = self.simulator.calculate_obj_values(_xs)
            xs[sim_id] = _xs[_vs.argmax()]
        return xs

    # self.simulator是并行
    def random_search(self, num_iters: int = 8, num_spin: int = 8, noise_std: float = 0.3):
        sim = self.simulator
        kth = self.num_nodes - num_spin

        prev_xs = self.good_xs.clone()
        prev_vs_raw = sim.calculate_obj_values_for_loop(prev_xs, if_sum=False)
        prev_vs = prev_vs_raw.sum(dim=1)

        thresh = None
        for _ in range(num_iters):
            '''flip randomly with ws(weights)'''
            ws = sim.n0_num_n1 - (4 if sim.if_bidirectional else 2) * prev_vs_raw
            ws_std = ws.max(dim=0, keepdim=True)[0] - ws.min(dim=0, keepdim=True)[0]

            spin_rand = ws + th.randn_like(ws, dtype=th.float32) * (ws_std.float() * noise_std)
            thresh = th.kthvalue(spin_rand, k=kth, dim=1)[0][:, None] if thresh is None else thresh
            spin_mask = spin_rand.gt(thresh)

            xs = prev_xs.clone()
            xs[spin_mask] = th.logical_not(xs[spin_mask])
            vs = sim.calculate_obj_values(xs)

            update_xs_by_vs(prev_xs, prev_vs, xs, vs)

        '''addition'''
        for i in range(sim.num_nodes):
            xs1 = prev_xs.clone()
            xs1[:, i] = th.logical_not(xs1[:, i])
            vs1 = sim.calculate_obj_values(xs1)

            update_xs_by_vs(prev_xs, prev_vs, xs1, vs1)

        num_update = update_xs_by_vs(self.good_xs, self.good_vs, prev_xs, prev_vs)
        return self.good_xs, self.good_vs, num_update


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


def update_xs_by_vs(xs0, vs0, xs1, vs1, if_maximize: bool = True):
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


class SimulatorGraphMaxCut:
    def __init__(self, sim_name: str = 'max_cut', graph_list: GraphList = (),
                 device=device, if_bidirectional: bool = False):
        self.device = device
        self.sim_name = sim_name
        self.int_type = int_type = th.long
        self.if_maximize = True
        self.if_bidirectional = if_bidirectional

        '''load graph'''
        graph_list: GraphList = graph_list if graph_list else load_graph_list(graph_name=sim_name)

        '''建立邻接矩阵'''
        # self.adjacency_matrix = build_adjacency_matrix(graph_list=graph_list, if_bidirectional=True).to(device)
        self.adjacency_bool = build_adjacency_bool(graph_list=graph_list, if_bidirectional=True).to(device)

        '''建立邻接索引'''
        n0_to_n1s, n0_to_dts = build_adjacency_indies(graph_list=graph_list, if_bidirectional=if_bidirectional)
        n0_to_n1s = [t.to(int_type).to(device) for t in n0_to_n1s]
        self.num_nodes = obtain_num_nodes(graph_list)
        self.num_edges = len(graph_list)
        self.adjacency_indies = n0_to_n1s

        '''基于邻接索引，建立基于边edge的索引张量：(n0_ids, n1_ids)是所有边(第0个, 第1个)端点的索引'''
        n0_to_n0s = [(th.zeros_like(n1s) + i) for i, n1s in enumerate(n0_to_n1s)]
        self.n0_ids = th.hstack(n0_to_n0s)[None, :]
        self.n1_ids = th.hstack(n0_to_n1s)[None, :]
        len_sim_ids = self.num_edges * (2 if if_bidirectional else 1)
        self.sim_ids = th.zeros(len_sim_ids, dtype=int_type, device=device)[None, :]
        self.n0_num_n1 = th.tensor([n1s.shape[0] for n1s in n0_to_n1s], device=device)[None, :]
        print()

    def calculate_obj_values(self, xs: TEN, if_sum: bool = True) -> TEN:
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

    def calculate_obj_values_for_loop(self, xs: TEN, if_sum: bool = True) -> TEN:  # 代码简洁，但是计算效率低
        num_sims, num_nodes = xs.shape
        values = th.zeros((num_sims, num_nodes), dtype=self.int_type, device=self.device)
        for node0 in range(num_nodes):
            node1s = self.adjacency_indies[node0]
            if node1s.shape[0] > 0:
                values[:, node0] = (xs[:, node0, None] ^ xs[:, node1s]).sum(dim=1)

        if if_sum:
            values = values.sum(dim=1)
        if self.if_bidirectional:
            values = values.float() / 2
        return values

    def generate_xs_randomly(self, num_sims):
        xs = th.randint(0, 2, size=(num_sims, self.num_nodes), dtype=th.bool, device=self.device)
        xs[:, 0] = 0
        return xs

    def local_search_inplace(self, good_xs: TEN, good_vs: TEN,
                             num_iters: int = 8, num_spin: int = 8, noise_std: float = 0.3):

        vs_raw = self.calculate_obj_values_for_loop(good_xs, if_sum=False)
        good_vs = vs_raw.sum(dim=1).long() if good_vs.shape == () else good_vs.long()
        ws = self.n0_num_n1 - (2 if self.if_bidirectional else 1) * vs_raw
        ws_std = ws.max(dim=0, keepdim=True)[0] - ws.min(dim=0, keepdim=True)[0]
        rd_std = ws_std.float() * noise_std
        spin_rand = ws + th.randn_like(ws, dtype=th.float32) * rd_std
        thresh = th.kthvalue(spin_rand, k=self.num_nodes - num_spin, dim=1)[0][:, None]

        for _ in range(num_iters):
            '''flip randomly with ws(weights)'''
            spin_rand = ws + th.randn_like(ws, dtype=th.float32) * rd_std
            spin_mask = spin_rand.gt(thresh)

            xs = good_xs.clone()
            xs[spin_mask] = th.logical_not(xs[spin_mask])
            vs = self.calculate_obj_values(xs)

            update_xs_by_vs(good_xs, good_vs, xs, vs, if_maximize=self.if_maximize)

        '''addition'''
        for i in range(self.num_nodes):
            xs1 = good_xs.clone()
            xs1[:, i] = th.logical_not(xs1[:, i])
            vs1 = self.calculate_obj_values(xs1)

            update_xs_by_vs(good_xs, good_vs, xs1, vs1, if_maximize=self.if_maximize)
        return good_xs, good_vs


def obtain_num_nodes(graph_list: GraphList) -> int:
    return max([max(n0, n1) for n0, n1, distance in graph_list]) + 1


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


def metro_sampling(probs, start_status, max_transfer_time, device=None):
    # Metropolis-Hastings sampling
    torch.set_grad_enabled(False)
    if device is None:
        device = calc_device(GPU_ID)

    num_node = len(probs)
    num_chain = start_status.shape[1]
    index_col = torch.tensor(list(range(num_chain)), device=device)

    samples = start_status.bool().to(device)
    probs = probs.detach().to(device)

    count = 0
    for t in range(max_transfer_time * 5):
        if count >= num_chain * max_transfer_time:
            break

        index_row = torch.randint(low=0, high=num_node, size=[num_chain], device=device)
        chosen_probs_base = probs[index_row]
        chosen_value = samples[index_row, index_col]
        chosen_probs = torch.where(chosen_value, chosen_probs_base, 1 - chosen_probs_base)
        accept_rate = (1 - chosen_probs) / chosen_probs

        is_accept = torch.rand(num_chain, device=device).lt(accept_rate)
        samples[index_row, index_col] = torch.where(is_accept, ~chosen_value, chosen_value)

        count += is_accept.sum()
    torch.set_grad_enabled(True)
    return samples.float().to(device)


def sampler_func(data, xs_sample,
                 num_ls, total_mcmc_num, repeat_times,
                 device=device):
    torch.set_grad_enabled(False)
    k = 1 / 4

    # num_nodes = data.num_nodes
    num_edges = data.num_edges
    n0s_tensor = data.edge_index[0]
    n1s_tensor = data.edge_index[1]

    xs_loc_sample = xs_sample.clone()
    xs_loc_sample *= 2  # map (0, 1) to (-0.5, 1.5)
    xs_loc_sample -= 0.5  # map (0, 1) to (-0.5, 1.5)

    # local search
    for cnt in range(num_ls):
        for node0_id in data.sorted_degree_nodes:
            node1_ids = data.neighbors[node0_id]

            node_rand_v = (xs_loc_sample[node1_ids].sum(dim=0) +
                           torch.rand(total_mcmc_num * repeat_times, device=device) * k)
            xs_loc_sample[node0_id] = node_rand_v.lt((data.weighted_degree[node0_id] + k) / 2).long()
    # pass
    # vs1 = simulator.calculate_obj_values(xs_sample.t().bool())
    # vs2 = simulator.calculate_obj_values(xs_loc_sample.t().bool())
    # pass
    expected_cut = torch.empty(total_mcmc_num * repeat_times, dtype=torch.float32, device=device)
    for j in range(repeat_times):
        j0 = total_mcmc_num * j
        j1 = total_mcmc_num * (j + 1)

        nlr_probs = 2 * xs_loc_sample[n0s_tensor.type(torch.long), j0:j1] - 1
        nlc_probs = 2 * xs_loc_sample[n1s_tensor.type(torch.long), j0:j1] - 1
        expected_cut[j0:j1] = (nlr_probs * nlc_probs).sum(dim=0)

    expected_cut_reshape = expected_cut.reshape((-1, total_mcmc_num))
    index = torch.argmin(expected_cut_reshape, dim=0)
    index = torch.arange(total_mcmc_num, device=device) + index * total_mcmc_num
    max_cut = expected_cut[index]
    vs_good = (num_edges - max_cut) / 2

    xs_good = xs_loc_sample[:, index]
    value = expected_cut.float()
    value -= value.mean()
    torch.set_grad_enabled(True)
    return vs_good, xs_good, value


class Simpler(torch.nn.Module):
    def __init__(self, output_num):
        super().__init__()
        self.lin = torch.nn.Linear(1, output_num)
        self.sigmoid = torch.nn.Sigmoid()

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, device=device):
        x = torch.ones(1).to(device)
        x = self.lin(x)
        x = self.sigmoid(x)

        x = (x - 0.5) * 0.6 + 0.5
        return x


def maxcut_dataloader(path,
                      device=device):
    with open(path) as f:
        fline = f.readline()
        fline = fline.split()
        num_nodes, num_edges = int(fline[0]), int(fline[1])
        edge_index = torch.LongTensor(2, num_edges)
        cnt = 0
        while True:
            lines = f.readlines(num_edges * 2)
            if not lines:
                break
            for line in lines:
                line = line.rstrip('\n').split()
                edge_index[0][cnt] = int(line[0]) - 1
                edge_index[1][cnt] = int(line[1]) - 1
                cnt += 1

        data = Data(num_nodes=num_nodes, edge_index=edge_index.to(device))
        data = append_neighbors(data)

        data.single_degree = []
        data.weighted_degree = []
        tensor_abs_weighted_degree = []
        for i0 in range(data.num_nodes):
            data.single_degree.append(len(data.neighbors[i0]))
            data.weighted_degree.append(
                float(torch.sum(data.neighbor_edges[i0])))
            tensor_abs_weighted_degree.append(
                float(torch.sum(torch.abs(data.neighbor_edges[i0]))))
        tensor_abs_weighted_degree = torch.tensor(tensor_abs_weighted_degree)
        data.sorted_degree_nodes = torch.argsort(
            tensor_abs_weighted_degree, descending=True)

        edge_degree = []
        add = torch.zeros(3, num_edges).to(device)
        for i0 in range(num_edges):
            edge_degree.append(
                tensor_abs_weighted_degree[edge_index[0][i0]] + tensor_abs_weighted_degree[edge_index[1][i0]])
            node_r = edge_index[0][i0]
            node_c = edge_index[1][i0]
            add[0][i0] = 1 - data.weighted_degree[node_r] / 2 - 0.05
            add[1][i0] = 1 - data.weighted_degree[node_c] / 2 - 0.05
            add[2][i0] = 1 + 0.05

        for i0 in range(num_nodes):
            data.neighbor_edges[i0] = data.neighbor_edges[i0].unsqueeze(0)
        data.add_items = add
        edge_degree = torch.tensor(edge_degree)
        data.sorted_degree_edges = torch.argsort(
            edge_degree, descending=True)
        return data, num_nodes


def append_neighbors(data,
                     device=device):
    data.neighbors = []
    data.neighbor_edges = []
    # num_nodes = data.encode_len
    num_nodes = data.num_nodes
    for i in range(num_nodes):
        data.neighbors.append([])
        data.neighbor_edges.append([])
    edge_number = data.edge_index.shape[1]

    edge_weight = 1
    for index in range(0, edge_number):
        row = data.edge_index[0][index]
        col = data.edge_index[1][index]

        data.neighbors[row].append(col.item())
        data.neighbor_edges[row].append(edge_weight)
        data.neighbors[col].append(row.item())
        data.neighbor_edges[col].append(edge_weight)

    data.n0 = []
    data.n1 = []
    data.n0_edges = []
    data.n1_edges = []
    for index in range(0, edge_number):
        row = data.edge_index[0][index]
        col = data.edge_index[1][index]
        data.n0.append(data.neighbors[row].copy())
        data.n1.append(data.neighbors[col].copy())
        data.n0_edges.append(data.neighbor_edges[row].copy())
        data.n1_edges.append(data.neighbor_edges[col].copy())
        i = 0
        for i in range(len(data.n0[index])):
            if data.n0[index][i] == col:
                break
        data.n0[index].pop(i)
        data.n0_edges[index].pop(i)
        for i in range(len(data.n1[index])):
            if data.n1[index][i] == row:
                break
        data.n1[index].pop(i)
        data.n1_edges[index].pop(i)

        data.n0[index] = torch.LongTensor(data.n0[index]).to(device)
        data.n1[index] = torch.LongTensor(data.n1[index]).to(device)
        data.n0_edges[index] = torch.tensor(
            data.n0_edges[index]).unsqueeze(0).to(device)
        data.n1_edges[index] = torch.tensor(
            data.n1_edges[index]).unsqueeze(0).to(device)

    for i in range(num_nodes):
        data.neighbors[i] = torch.LongTensor(data.neighbors[i]).to(device)
        data.neighbor_edges[i] = torch.tensor(
            data.neighbor_edges[i]).to(device)

    return data


def get_return(probs, samples, value, total_mcmc_num, repeat_times):
    log_prob_sum = torch.empty_like(value)
    for j in range(repeat_times):
        j0 = total_mcmc_num * j
        j1 = total_mcmc_num * (j + 1)

        _samples = samples[j0:j1]
        log_prob = (_samples * probs + (1 - _samples) * (1 - probs)).log()
        log_prob_sum[j0:j1] = log_prob.sum(dim=1)
    objective = (log_prob_sum * value.detach()).mean()
    return objective


def save_graph_list_to_txt(graph_list, txt_path: str):
    num_nodes = max([max(n0, n1) for n0, n1, distance in graph_list]) + 1
    num_edges = len(graph_list)

    lines = [f"{num_nodes} {num_edges}", ]
    lines.extend([f"{n0 + 1} {n1 + 1} {distance}" for n0, n1, distance in graph_list])
    lines = [l + '\n' for l in lines]
    # with open(txt_path, 'w') as file:
    #     file.writelines(lines)


def print_gpu_memory(device):
    if not torch.cuda.is_available():
        return

    total_memory = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)  # GB
    max_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)  # GB
    memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)  # GB

    print(f"AllRAM {total_memory:.2f} GB, "
          f"MaxRAM {max_allocated:.2f} GB, "
          f"NowRAM {memory_allocated:.2f} GB, "
          f"Rate {(max_allocated / total_memory) * 100:.2f}%")


def run():

    #path = '../data/gset/BA_100_ID0.txt'
    # path = 'data/gset_15.txt'
    # path = 'data/gset_49.txt'
    # path = 'data/gset_50.txt'
    graph_type = ['ErdosRenyi', 'barabasi_albert', 'PowerLaw'][1]
    num_nodes = 1000
    graph_id = 0
    # graph_name = f"{graph_type}_{num_nodes}_ID{graph_id}"
    # path = f'temp_{graph_name}.txt'
    # graph_name = "barabasi_albert_1000_ID0"
    # graph_name = "gset_55"
    # save_graph_list_to_txt(graph_list=load_graph_list(graph_name=graph_name), txt_path=path)

    # num_ls = 6
    # reset_epoch_num = 192
    # total_mcmc_num = 224
    # path = 'data/gset_22.txt'

    # num_ls = 8
    # reset_epoch_num = 128
    # total_mcmc_num = 256
    # path = 'data/gset_55.txt'

    # num_ls = 8
    # reset_epoch_num = 256
    # total_mcmc_num = 192
    # path = 'data/gset_70.txt'

    # num_ls = 8
    # reset_epoch_num = 256
    # repeat_times = 512
    # total_mcmc_num = 2048
    # path = 'data/gset_22.txt'  # GPU RAM 40GB

    # num_ls = 8
    # reset_epoch_num = 192
    # repeat_times = 448
    # total_mcmc_num = 1024
    # path = 'data/gset_55.txt'  # GPU RAM 40GB

    # num_ls = 8
    # reset_epoch_num = 320
    # repeat_times = 288
    # total_mcmc_num = 768
    # path = 'data/gset_70.txt'  # GPU RAM 40GB

    '''init'''
    sim_name = path  # os.path.splitext(os.path.basename(path))[0]
    data, num_nodes = maxcut_dataloader(sim_name)
    device = calc_device(GPU_ID)
    print("GPU_ID:", GPU_ID, "cuda available:", torch.cuda.is_available())
    print("device: ", device)
    change_times = int(num_nodes / 10)  # transition times for metropolis sampling

    net = Simpler(num_nodes)
    net.to(device).reset_parameters()
    optimizer = torch.optim.Adam(net.parameters(), lr=8e-2)

    '''addition'''

    sim = SimulatorGraphMaxCut(sim_name=sim_name, device=device)
    solver = SolverLocalSearch(simulator=sim, num_nodes=num_nodes)

    xs = sim.generate_xs_randomly(num_sims=total_mcmc_num)
    solver.reset(xs.bool())
    for _ in range(16):
        solver.random_search(num_iters=repeat_times // 16)
    now_max_info = solver.good_xs.t()
    now_max_res = solver.good_vs
    del sim
    del solver

    '''loop'''
    net.train()
    xs_prob = (torch.zeros(num_nodes) + 0.5).to(device)
    xs_bool = now_max_info.repeat(1, repeat_times)

    print('start loop')
    rewardss = []

    sys.stdout.flush()  # add for slurm stdout
    objs_of_epochs = []
    sum_samples_per_second = []
    duration_obj_dict = {}
    start_time = time.time()
    start_time_of_dict = time.time()
    for epoch in range(1, max_epoch_num + 1):
        #start_time = time.time()
        y_dict = {}
        rewards = []
        net.to(device).reset_parameters()
        for j1 in range(reset_epoch_num // sample_epoch_num):
            if test_sampling_speed == True:
                start_time = time.time()
            xs_sample = metro_sampling(xs_prob, xs_bool.clone(), change_times)

            temp_max, temp_max_info, value = sampler_func(
                data, xs_sample, num_ls, total_mcmc_num, repeat_times, device)
            if not test_sampling_speed:
                # update now_max
                for i0 in range(total_mcmc_num):
                    if temp_max[i0] > now_max_res[i0]:
                        now_max_res[i0] = temp_max[i0]
                        now_max_info[:, i0] = temp_max_info[:, i0]
                res = torch.Tensor.cpu(now_max_res)
                res = max(res.tolist())
                rewards.append(res)
                # update if min is too small
                now_max = max(now_max_res).item()
                now_max_index = torch.argmax(now_max_res)
                now_min_index = torch.argmin(now_max_res)
                now_max_res[now_min_index] = now_max
                now_max_info[:, now_min_index] = now_max_info[:, now_max_index]
                temp_max_info[:, now_min_index] = now_max_info[:, now_max_index]

                # select best samples
                xs_bool = temp_max_info.clone()
                xs_bool = xs_bool.repeat(1, repeat_times)
                # construct the start point for next iteration
                start_samples = xs_sample.t()

                probs = xs_prob[None, :]
                _probs = 1 - probs
                entropy = -(probs * probs.log2() + _probs * _probs.log2()).mean(dim=1)
                obj_entropy = entropy.mean()

                duration = time.time() - start_time_of_dict
                duration_obj_dict[duration] = max(now_max_res).item()

                print(f"value {max(now_max_res).item():9.2f}  entropy {obj_entropy:9.3f}")
                run_time = time.time() - start_time
                max_value = max(now_max_res).item()
                y_dict[round(run_time, 2)] = max_value
                sys.stdout.flush()  # add for slurm stdout

                if run_time > total_running_duration:
                    break

            if test_sampling_speed:
                running_duration = time.time() - start_time
                # num_samples = xs_sample.shape[1]
                num_samples = temp_max.shape[0]
                num_samples_per_second = num_samples / running_duration
                print("num_samples_per_second: ", num_samples_per_second)
                sum_samples_per_second.append(num_samples_per_second)

            if not test_sampling_speed:
                for _ in range(sample_epoch_num):
                    xs_prob = net()
                    ret_loss_ls = get_return(xs_prob, start_samples, value, total_mcmc_num, repeat_times)

                    optimizer.zero_grad()
                    ret_loss_ls.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
                    optimizer.step()

            if j1 % show_gap == 0:
                total_max = now_max_res
                best_sort = torch.argsort(now_max_res, descending=True)
                total_best_info = torch.squeeze(now_max_info[:, best_sort[0]])

                objective_value = max(total_max)
                solution = total_best_info

                encoder = EncoderBase64(encode_len=num_nodes)
                x_str = encoder.bool_to_str(x_bool=solution)

                print(f"epoch {epoch:6}  value {objective_value.item():8.2f}  {x_str}")
                # print_gpu_memory(device)
            if os.path.exists('./stop'):
                break
        print("objective_value.item():", objective_value.item())
        objs_of_epochs.append(objective_value.item())
        # filename = 'mcpg_speed.txt'
        # num_samples_per_second = sum(sum_samples_per_second)/len(sum_samples_per_second)
        filename = os.path.basename(path)  # 提取文件名 gset_22.txt
        graph_name = os.path.splitext(filename)[0]
        filename = f'mcpg_{graph_name}_seed{seed}_{total_mcmc_num}.txt'
        # 将 a 和 b 写入到 txt 文件
        if test_sampling_speed:
            filename = f'mcpg_speed_{graph_name}.txt'
            num_samples_per_second = sum(sum_samples_per_second) / len(sum_samples_per_second)
            with open(filename, 'a') as file:
                file.write(f"seed:{seed}\ny2_{total_mcmc_num} = {num_samples_per_second}\n")
        if not test_sampling_speed:
            running_duration = time.time() - start_time
            with open(filename, 'a') as file:  # 使用 'a' 模式打开文件以附加内容
                file.write(f"seed:{seed}\ny_{total_mcmc_num}_{epoch} = {y_dict}\nx_{total_mcmc_num} = {running_duration}\n")

        # print("rewards", rewards)
        #rewards_numpy = rewards.numpy()
        print("rewards:", rewards)
        rewardss.append(rewards)
        #rewardss.append(rewards)
        if os.path.exists('./stop'):
            break

        print()

    print("total_mcmc_num", total_mcmc_num)
    print(f"objs_of_epochs: {objs_of_epochs}")
    print("rewardss: ", rewardss)
    if os.path.exists('./stop'):
        print(f"break: os.path.exists('./stop') {os.path.exists('./stop')}")
        sys.stdout.flush()  # add for slurm stdout


if __name__ == '__main__':
    run()
