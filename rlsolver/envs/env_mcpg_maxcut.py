import sys
import os
cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../rlsolver')
sys.path.append(os.path.dirname(rlsolver_path))

import os
import sys
import time
import torch as th

from rlsolver.methods.util_read_data import (load_graph_list, GraphList,
                                             build_adjacency_bool,
                                             build_adjacency_indies,
                                             obtain_num_nodes,
                                             update_xs_by_vs,)
from rlsolver.methods.util import gpu_info_str, evolutionary_replacement
from rlsolver.methods.config import *
TEN = th.Tensor

class SimulatorMaxcut:
    def __init__(self, sim_name: str = 'max_cut', graph_list: GraphList = [],
                 device=calc_device(GPU_ID), if_bidirectional: bool = False):
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
        obj = simulator.calculate_obj_values(xs=xs)
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
    good_vs = sim.calculate_obj_values(xs=good_xs)

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
            good_vs = sim.calculate_obj_values(xs=good_xs)

    print(f"\nbest_x.shape {evaluator.best_x.shape}"
          f"\nbest_v {evaluator.best_v}"
          f"\nbest_x_str {evaluator.best_x_str}")

import os
import sys
import time
import torch as th
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from rlsolver.methods.util_evaluator import X_G14, X_G15, X_G49, X_G50, X_G22, X_G55, X_G70
from rlsolver.methods.util_evaluator import Evaluator, EncoderBase64

# TODO plan to remove

TEN = th.Tensor

'''local search'''





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


'''network'''


# FIXME plan to remove PolicyMLP from here, because PolicyMLP now in network.py
class PolicyMLP(nn.Module):
    def __init__(self, inp_dim, mid_dim, out_dim):
        super().__init__()
        self.net1 = nn.Sequential(nn.Linear(inp_dim, mid_dim), nn.GELU(), nn.LayerNorm(mid_dim),
                                  nn.Linear(mid_dim, mid_dim), nn.GELU(), nn.LayerNorm(mid_dim),
                                  nn.Linear(mid_dim, out_dim), nn.Tanh(), )
        self.net2 = nn.Sequential(nn.Linear(1 + out_dim // inp_dim, 4), nn.Tanh(),
                                  nn.Linear(4, 1), nn.Sigmoid(), )

    def forward(self, xs0):
        num_sims, num_nodes = xs0.shape
        xs1 = self.net1(xs0).reshape((num_sims, num_nodes, -1))
        xs2 = th.cat((xs0.unsqueeze(2), xs1), dim=2)
        xs3 = self.net2(xs2).squeeze(2)
        return xs3


def train_loop(num_train, device, seq_len, best_x, num_sims1, sim, net, optimizer, show_gap, noise_std):
    num_nodes = best_x.shape[0]
    sim_ids = th.arange(num_sims1, device=sim.device)
    start_time = time.time()
    assert seq_len <= num_nodes

    for j in range(num_train):
        mask = th.zeros(num_nodes, dtype=th.bool, device=device)
        n_std = (num_nodes - seq_len - 1) // 4
        n_avg = seq_len + 1 + n_std * 2
        rand_n = int(th.randn(size=(1,)).clip(-2, +2).item() * n_std + n_avg)
        mask[:rand_n] = True
        mask = mask[th.randperm(num_nodes)]
        rand_x = best_x.clone()
        rand_x[mask] = th.logical_not(rand_x[mask])
        rand_v = sim.calculate_obj_values(rand_x[None, :])[0]
        good_xs = rand_x.repeat(num_sims1, 1)
        good_vs = rand_v.repeat(num_sims1, )

        xs = good_xs.clone()
        num_not_equal = xs[0].ne(best_x).sum().item()
        # assert num_not_equal == rand_n
        # assert num_not_equal >= seq_len

        out_list = th.empty((num_sims1, seq_len), dtype=th.float32, device=device)
        for i in range(seq_len):
            net.train()
            inp = xs.float()
            out = net(inp) + xs.ne(best_x).float().detach()

            noise = th.randn_like(out) * noise_std
            sample = (out + noise).argmax(dim=1)
            xs[sim_ids, sample] = th.logical_not(xs[sim_ids, sample])
            vs = sim.calculate_obj_values(xs)

            out_list[:, i] = out[sim_ids, sample]

            update_xs_by_vs(good_xs, good_vs, xs, vs)

        good_vs = good_vs.float()
        advantage = (good_vs - good_vs.mean()) / (good_vs.std() + 1e-6)

        objective = (out_list.mean(dim=1) * advantage.detach()).mean()
        optimizer.zero_grad()
        objective.backward()
        clip_grad_norm_(net.parameters(), 2)
        optimizer.step()

        if (j + 1) % show_gap == 0:
            vs_avg = good_vs.mean().item()
            print(f'{j:8}  {time.time() - start_time:9.0f} '
                  f'| {vs_avg:9.3f}  {vs_avg - rand_v.item():9.3f} |  {num_not_equal}')
    pass


def check_net(net, sim, num_sims):
    num_nodes = sim.encode_len
    good_xs = sim.generate_xs_randomly(num_sims=num_sims)
    good_vs = sim.calculate_obj_values(good_xs)

    xs = good_xs.clone()
    sim_ids = th.arange(num_sims, device=sim.device)
    for i in range(num_nodes):
        inp = xs.float()
        out = net(inp)

        sample = out.argmax(dim=1)
        xs[sim_ids, sample] = th.logical_not(xs[sim_ids, sample])
        vs = sim.calculate_obj_values(xs)

        update_xs_by_vs(good_xs, good_vs, xs, vs)
    return good_xs, good_vs


def check_generate_best_x():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    # sim_name = 'gset_14'
    # x_str = X_G14
    sim_name = 'gset_70'
    x_str = X_G70
    lr = 1e-3
    noise_std = 0.1

    num_train = 2 ** 9
    mid_dim = 2 ** 8
    seq_len = 2 ** 6
    show_gap = 2 ** 5

    num_sims = 2 ** 8
    if os.name == 'nt':  # windows new type
        num_sims = 2 ** 4

    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    '''simulator'''
    sim = SimulatorMaxcut(sim_name=sim_name, device=device)
    enc = EncoderBase64(encode_len=sim.num_nodes)
    num_nodes = sim.num_nodes

    '''network'''
    net = PolicyMLP(inp_dim=num_nodes, mid_dim=mid_dim, out_dim=num_nodes * 3).to(device)
    optimizer = th.optim.Adam(net.parameters(), lr=lr, maximize=True)

    best_x = enc.str_to_bool(x_str).to(device)
    best_v = sim.calculate_obj_values(best_x[None, :])[0]
    print(f"{sim_name:32}  num_nodes {sim.num_nodes:4}  obj_value {best_v.item()}  ")

    train_loop(num_train, device, seq_len, best_x, num_sims, sim, net, optimizer, show_gap, noise_std)


'''utils'''


def show_gpu_memory(device):
    if not th.cuda.is_available():
        return 'not th.cuda.is_available()'

    all_memory = th.cuda.get_device_properties(device).total_memory / (1024 ** 3)  # GB
    max_memory = th.cuda.max_memory_allocated(device) / (1024 ** 3)  # GB
    now_memory = th.cuda.memory_allocated(device) / (1024 ** 3)  # GB

    show_str = (
        f"AllRAM {all_memory:.2f} GB, "
        f"MaxRAM {max_memory:.2f} GB, "
        f"NowRAM {now_memory:.2f} GB, "
        f"Rate {(max_memory / all_memory) * 100:.2f}%"
    )
    return show_str


'''run'''


def find_smallest_nth_power_of_2(target):
    n = 0
    while 2 ** n < target:
        n += 1
    return 2 ** n


def search_and_evaluate_local_search():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    num_reset = 2 ** 0
    num_iter1 = 2 ** 6
    num_iter1_wait = 2 ** 3
    num_iter0 = 2 ** 4
    num_iter0_wait = 2 ** 0
    num_sims = 2 ** 12

    num_skip = 2 ** 0
    gap_print = 2 ** 0

    sim_name = 'gset_14'

    if os.name == 'nt':  # windows new type
        num_sims = 2 ** 4
        num_reset = 2 ** 1
        num_iter0 = 2 ** 2

    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    simulator_class = SimulatorMaxcut
    local_search_class = LocalSearch

    '''simulator'''
    sim = simulator_class(sim_name=sim_name, device=device)
    num_nodes = sim.num_nodes

    '''evaluator'''
    temp_xs = sim.generate_xs_randomly(num_sims=1)
    temp_vs = sim.calculate_obj_values(xs=temp_xs)
    evaluator = Evaluator(save_dir=f"{sim_name}_{gpu_id}", num_bits=num_nodes, x=temp_xs[0], v=temp_vs[0].item())

    '''solver'''
    solver = local_search_class(simulator=sim, num_nodes=sim.num_nodes)

    """loop"""
    th.set_grad_enabled(True)
    print(f"start searching, {sim_name}  num_nodes={num_nodes}")
    sim_ids = th.arange(num_sims, device=device)
    for j2 in range(num_reset):
        print(f"|\n| reset {j2}")
        best_xs = sim.generate_xs_randomly(num_sims)
        best_vs = sim.calculate_obj_values(best_xs)

        update_j1 = 0
        for j1 in range(num_iter1):
            best_i = best_vs.argmax()
            best_xs[:] = best_xs[best_i]
            best_vs[:] = best_vs[best_i]

            '''update xs via probability'''
            xs = best_xs.clone()
            for _ in range(num_iter0):
                sample = th.randint(num_nodes, size=(num_sims,), device=device)
                xs[sim_ids, sample] = th.logical_not(xs[sim_ids, sample])

            '''update xs via local search'''
            solver.reset(xs)

            update_j0 = 0
            for j0 in range(num_iter0):
                solver.random_search(num_iters=2 ** 6, num_spin=4)
                if_update0 = update_xs_by_vs(best_xs, best_vs, solver.good_xs, solver.good_vs)
                if if_update0:
                    update_j0 = j0
                elif j0 - update_j0 > num_iter0_wait:
                    break

            if j1 > num_skip and (j1 + 1) % gap_print == 0:
                i = j2 * num_iter1 + j1

                good_i = solver.good_vs.argmax()
                good_x = solver.good_xs[good_i]
                good_v = solver.good_vs[good_i].item()

                if_update1 = evaluator.record2(i=i, vs=good_v, xs=good_x)
                evaluator.logging_print(x=good_x, v=good_v, show_str=f"{good_v:6}", if_show_x=if_update1)
                if if_update1:
                    update_j1 = j1
                elif j1 - update_j1 > num_iter1_wait:
                    break
        evaluator.save_record_draw_plot()


# if __name__ == '__main__':
#     search_and_evaluate_local_search()

from rlsolver.envs.env_mcpg_maxcut import SimulatorMaxcut
from rlsolver.methods.util_read_data import read_graphlist
def check_solution_x():
    filename = '../data/syn_BA/BA_100_ID0.txt'
    graph = read_graphlist(filename)
    simulator = SimulatorMaxcut(sim_name=filename, graph_list=graph)

    x_str = X_G14
    num_nodes = simulator.num_nodes
    encoder = EncoderBase64(encode_len=num_nodes)

    x = encoder.str_to_bool(x_str)
    vs = simulator.calculate_obj_values(xs=x[None, :])
    print(f"objective value  {vs[0].item():8.2f}  solution {x_str}", flush=True)


if __name__ == '__main__':
    check_simulator()
    # check_local_search()

    search_and_evaluate_local_search()
