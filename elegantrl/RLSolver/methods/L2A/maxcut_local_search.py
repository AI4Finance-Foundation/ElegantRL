import os
import sys
import time
import torch as th
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from methods.L2A.maxcut_simulator import SimulatorMaxcut
from methods.L2A.evaluator import X_G14, X_G15, X_G49, X_G50, X_G22, X_G55, X_G70
from methods.L2A.evaluator import Evaluator, EncoderBase64

# TODO plan to remove

TEN = th.Tensor

'''local search'''


class SolverLocalSearch:
    def __init__(self, simulator: SimulatorMaxcut, num_nodes: int):
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
        xs = th.empty((num_sims, self.num_nodes), dtype=th.bool, device=self.simulator.device)
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


# FIXME plan to remove PolicyMLP from here, because PolicyMLP now in l2a_network.py
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
        rand_v = sim.obj(rand_x[None, :])[0]
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
            vs = sim.obj(xs)

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
    good_vs = sim.obj(good_xs)

    xs = good_xs.clone()
    sim_ids = th.arange(num_sims, device=sim.device)
    for i in range(num_nodes):
        inp = xs.float()
        out = net(inp)

        sample = out.argmax(dim=1)
        xs[sim_ids, sample] = th.logical_not(xs[sim_ids, sample])
        vs = sim.obj(xs)

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
    best_v = sim.obj(best_x[None, :])[0]
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
    solver_class = SolverLocalSearch

    '''simulator'''
    sim = simulator_class(sim_name=sim_name, device=device)
    num_nodes = sim.num_nodes

    '''evaluator'''
    temp_xs = sim.generate_xs_randomly(num_sims=1)
    temp_vs = sim.obj(xs=temp_xs)
    evaluator = Evaluator(save_dir=f"{sim_name}_{gpu_id}", num_bits=num_nodes, x=temp_xs[0], v=temp_vs[0].item())

    '''solver'''
    solver = solver_class(simulator=sim, num_nodes=sim.num_nodes)

    """loop"""
    th.set_grad_enabled(True)
    print(f"start searching, {sim_name}  num_nodes={num_nodes}")
    sim_ids = th.arange(num_sims, device=device)
    for j2 in range(num_reset):
        print(f"|\n| reset {j2}")
        best_xs = sim.generate_xs_randomly(num_sims)
        best_vs = sim.obj(best_xs)

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


if __name__ == '__main__':
    search_and_evaluate_local_search()
