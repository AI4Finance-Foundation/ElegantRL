import os
import sys
import time
import torch as th

from graph_utils import load_graph_list, GraphList
from graph_utils import build_adjacency_bool, build_adjacency_indies, obtain_num_nodes
from graph_utils import update_xs_by_vs, gpu_info_str, evolutionary_replacement

TEN = th.Tensor


class SimulatorMaxcut:
    def __init__(self, sim_name: str = 'max_cut', graph_list: GraphList = (),
                 device=th.device('cpu'), if_bidirectional: bool = False):
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


'''check'''


def find_best_num_sims():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    calculate_obj_func = 'calculate_obj_values'
    graph_name = 'gset_14'
    num_sims = 2 ** 16
    num_iter = 2 ** 6
    # calculate_obj_func = 'calculate_obj_values_for_loop'
    # graph_name = 'gset_14'
    # num_sims = 2 ** 13
    # num_iter = 2 ** 9

    if os.name == 'nt':
        graph_name = 'powerlaw_64'
        num_sims = 2 ** 4
        num_iter = 2 ** 3

    graph = load_graph_list(graph_name=graph_name)
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')
    simulator = SimulatorMaxcut(sim_name=graph_name, graph_list=graph, device=device, if_bidirectional=False)

    print('find the best num_sims')
    from math import ceil
    for j in (1, 1, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32):
        _num_sims = int(num_sims * j)
        _num_iter = ceil(num_iter * num_sims / _num_sims)

        timer = time.time()
        for i in range(_num_iter):
            xs = simulator.generate_xs_randomly(num_sims=_num_sims)
            vs = getattr(simulator, calculate_obj_func)(xs=xs)
            assert isinstance(vs, TEN)
            # print(f"| {i}  max_obj_value {vs.max().item()}")
        print(f"_num_iter {_num_iter:8}  "
              f"_num_sims {_num_sims:8}  "
              f"UsedTime {time.time() - timer:9.3f}  "
              f"GPU {gpu_info_str(device)}")


def check_simulator():
    gpu_id = -1
    num_sims = 16
    num_nodes = 24
    graph_name = f'powerlaw_{num_nodes}'

    graph = load_graph_list(graph_name=graph_name)
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')
    simulator = SimulatorMaxcut(sim_name=graph_name, graph_list=graph, device=device)

    for i in range(8):
        xs = simulator.generate_xs_randomly(num_sims=num_sims)
        obj = simulator.obj(xs=xs)
        print(f"| {i}  max_obj_value {obj.max().item()}")
    pass


def check_local_search():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    graph_type = 'gset_14'
    graph_list = load_graph_list(graph_name=graph_type)
    num_nodes = obtain_num_nodes(graph_list)

    show_gap = 4

    num_sims = 2 ** 8
    num_iters = 2 ** 8
    reset_gap = 2 ** 6
    save_dir = f"./{graph_type}_{num_nodes}"

    if os.name == 'nt':
        num_sims = 2 ** 2
        num_iters = 2 ** 5

    '''simulator'''
    sim = SimulatorMaxcut(graph_list=graph_list, device=device, if_bidirectional=True)
    if_maximize = sim.if_maximize

    '''evaluator'''
    good_xs = sim.generate_xs_randomly(num_sims=num_sims)
    good_vs = sim.obj(xs=good_xs)
    from evaluator import Evaluator
    evaluator = Evaluator(save_dir=save_dir, num_bits=num_nodes, if_maximize=if_maximize,
                          x=good_xs[0], v=good_vs[0].item(), )

    for i in range(num_iters):
        evolutionary_replacement(good_xs, good_vs, low_k=2, if_maximize=if_maximize)

        for _ in range(4):
            sim.local_search_inplace(good_xs, good_vs)

        if_show_x = evaluator.record2(i=i, vs=good_vs, xs=good_xs)
        if (i + 1) % show_gap == 0 or if_show_x:
            show_str = f"| cut_value {good_vs.float().mean():8.2f} < {good_vs.max():6}"
            evaluator.logging_print(show_str=show_str, if_show_x=if_show_x)
            sys.stdout.flush()

        if (i + 1) % reset_gap == 0:
            print(f"| reset {gpu_info_str(device=device)} "
                  f"| up_rate {evaluator.best_v / evaluator.first_v - 1.:8.5f}")
            sys.stdout.flush()

            good_xs = sim.generate_xs_randomly(num_sims=num_sims)
            good_vs = sim.obj(xs=good_xs)

    print(f"\nbest_x.shape {evaluator.best_x.shape}"
          f"\nbest_v {evaluator.best_v}"
          f"\nbest_x_str {evaluator.best_x_str}")


if __name__ == '__main__':
    check_simulator()
    # check_local_search()
