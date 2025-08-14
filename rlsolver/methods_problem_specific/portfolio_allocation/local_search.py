import os
import sys
import time
import torch as th
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from simulator import SimulatorGraphMaxCut
from simulator import X_G14, X_G15, X_G49, X_G50, X_G22, X_G55, X_G70
from evaluator import Evaluator, EncoderBase64

TEN = th.Tensor

'''local search'''


class SolverLocalSearch:
    def __init__(self, simulator: SimulatorGraphMaxCut, num_nodes: int):
        self.simulator = simulator
        self.num_nodes = num_nodes

        self.num_sims = 0
        self.good_xs = th.tensor([])  # solution x
        self.good_vs = th.tensor([])  # objective value

    def reset(self, xs: TEN):
        self.good_xs = xs
        self.good_vs = self.simulator.calculate_obj_values(xs=xs)
        self.num_sims = xs.shape[0]

    def reset_search(self, num_sims):
        xs = th.empty((num_sims, self.num_nodes), dtype=th.bool, device=self.simulator.device)
        for sim_id in range(num_sims):
            _xs = self.simulator.generate_xs_randomly(num_sims=num_sims)
            _vs = self.simulator.calculate_obj_values(_xs)
            xs[sim_id] = _xs[_vs.argmax()]
        return xs

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

        update_xs_by_vs(self.good_xs, self.good_vs, prev_xs, prev_vs)
        return self.good_xs, self.good_vs


def update_xs_by_vs(xs0, vs0, xs1, vs1):
    good_is = vs1.gt(vs0)
    xs0[good_is] = xs1[good_is]
    vs0[good_is] = vs1[good_is]


'''network'''


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
    num_nodes = sim.num_nodes
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
    sim = SimulatorGraphMaxCut(sim_name=sim_name, device=device)
    enc = EncoderBase64(num_nodes=sim.num_nodes)
    num_nodes = sim.num_nodes

    '''network'''
    net = PolicyMLP(inp_dim=num_nodes, mid_dim=mid_dim, out_dim=num_nodes * 3).to(device)
    optimizer = th.optim.Adam(net.parameters(), lr=lr, maximize=True)

    best_x = enc.str_to_bool(x_str).to(device)
    best_v = sim.calculate_obj_values(best_x[None, :])[0]
    print(f"{sim_name:32}  num_nodes {sim.num_nodes:4}  obj_value {best_v.item()}  ")

    train_loop(num_train, device, seq_len, best_x, num_sims, sim, net, optimizer, show_gap, noise_std)


'''run'''


def find_smallest_nth_power_of_2(target):
    n = 0
    while 2 ** n < target:
        n += 1
    return 2 ** n


def search_and_evaluate_local_search():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    # if_reinforce = False
    # num_reset = 2 ** 1
    # num_iter1 = 2 ** 6
    # num_iter0 = 2 ** 4
    # num_sims = 2 ** 13

    if_reinforce = True
    num_reset = 2 ** 8
    num_iter1 = 2 ** 5
    num_iter0 = 2 ** 6
    num_sims = 2 ** 11
    num_sims1 = 2 ** 10

    seq_len = 2 ** 7
    show_gap = 2 ** 6
    num_train = 2 ** 9

    noise_std = 0.1
    mid_dim = 2 ** 7
    lr = 1e-5

    num_skip = 2 ** 0
    gap_print = 2 ** 0

    x_str = None
    sim_name = 'gset_14'
    if gpu_id == 0:
        sim_name = 'gset_14'  # num_nodes==800
        x_str = """yNpHTLH7e2OIdP6rCrMPIFDIONjekuOTSIcsZHJ4oVznK_DN98AUJKV9cN3W3PSVLS$h4eoCIzHrCBcGhMSuL4JD3JTg89BkvDZXVY07h6z9NPO5QWjRxCyC
FUAYMjofiS5er"""  # 3022
    if gpu_id == 1:
        sim_name = 'gset_15'  # num_nodes==800
        x_str = """PoaFXUkt2uOnZNChgBeg8ljjVkK_2VvBmhul_GbbYmI8GQ9h6wPDKxowYppuj9MzV_pg8oQ69gXqaFOJWCaMRnaDvqUnmtTe9ua9xVe2NS5bKcazkHsW6kO7
hUH4vj0nAzi24"""
    # if gpu_id == 2:
    #     sim_name = 'gset_49'  # num_nodes==3000
    # if gpu_id == 3:
    #     sim_name = 'gset_50'  # num_nodes==3000
    if gpu_id in {2, }:
        sim_name = 'gset_22'  # num_nodes==2000
        x_str = X_G22
        seq_len = 2 ** 6
        num_sims1 = 2 ** 9
        num_iter1 = 2 ** 5
        num_iter0 = 2 ** 7
    if gpu_id in {3, }:
        sim_name = 'gset_22'  # num_nodes==2000
        x_str = X_G22
        seq_len = 2 ** 6
        num_sims1 = 2 ** 9
        num_iter1 = 2 ** 6
        num_iter0 = 2 ** 6
    if gpu_id in {4, 5}:
        sim_name = 'gset_55'  # num_nodes==5000
        x_str = X_G55
        num_sims1 = 2 ** 9
        seq_len = 2 ** 6
        num_iter1 = 2 ** 6
        num_iter0 = 2 ** 7
    if gpu_id in {6, 7}:
        sim_name = 'gset_70'  # num_nodes==10000
        x_str = X_G70
        num_sims1 = 2 ** 9
        seq_len = 2 ** 5
        mid_dim = 2 ** 6
        num_iter1 = 2 ** 6
        num_iter0 = 2 ** 8

    if os.name == 'nt':  # windows new type
        num_sims = 2 ** 4
        num_reset = 2 ** 1
        num_iter0 = 2 ** 2

    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    simulator_class = SimulatorGraphMaxCut
    solver_class = SolverLocalSearch

    '''simulator'''
    sim = simulator_class(sim_name=sim_name, device=device)
    num_nodes = sim.num_nodes

    '''evaluator'''
    temp_xs = sim.generate_xs_randomly(num_sims=1)
    temp_vs = sim.calculate_obj_values(xs=temp_xs)
    evaluator = Evaluator(save_dir=f"{sim_name}_{gpu_id}", num_nodes=num_nodes, x=temp_xs[0], v=temp_vs[0].item())

    '''solver'''
    solver = solver_class(simulator=sim, num_nodes=sim.num_nodes)

    '''network'''
    mid_dim = mid_dim if mid_dim else find_smallest_nth_power_of_2(num_nodes)
    net = PolicyMLP(inp_dim=num_nodes, mid_dim=mid_dim, out_dim=num_nodes).to(device)
    optimizer = th.optim.Adam(net.parameters(), lr=lr, maximize=False)

    """loop"""
    th.set_grad_enabled(True)
    print(f"start searching, {sim_name}  num_nodes={num_nodes}")
    sim_ids = th.arange(num_sims, device=device)
    for j2 in range(num_reset):
        print(f"|\n| reset {j2}")
        best_xs = sim.generate_xs_randomly(num_sims)
        best_vs = sim.calculate_obj_values(best_xs)

        if (j2 == 0) and (x_str is not None):
            _num_iter1 = 0  # skip

            evaluator.best_x = evaluator.encoder_base64.str_to_bool(x_str).to(device)
            evaluator.best_v = sim.calculate_obj_values(evaluator.best_x[None, :])[0]
        else:
            _num_iter1 = num_iter1
        for j1 in range(_num_iter1):
            best_i = best_vs.argmax()
            best_xs[:] = best_xs[best_i]
            best_vs[:] = best_vs[best_i]

            '''update xs via probability'''
            xs = best_xs.clone()
            _num_iter0 = th.randint(int(num_iter0 * 0.75), int(num_iter0 * 1.25), size=(1,)).item()
            for _ in range(num_iter0):
                if if_reinforce and (j2 != 0):
                    best_x = evaluator.best_x
                    out = net(xs.float()) + xs.ne(best_x[None, :]).float()
                    sample = (out + th.rand_like(out) * noise_std).argmax(dim=1)
                else:
                    sample = th.randint(num_nodes, size=(num_sims,), device=device)
                xs[sim_ids, sample] = th.logical_not(xs[sim_ids, sample])

            '''update xs via local search'''
            solver.reset(xs)
            solver.random_search(num_iters=2 ** 6, num_spin=4)

            update_xs_by_vs(best_xs, best_vs, solver.good_xs, solver.good_vs)

            if j1 > num_skip and (j1 + 1) % gap_print == 0:
                i = j2 * num_iter1 + j1

                good_i = solver.good_vs.argmax()
                good_x = solver.good_xs[good_i]
                good_v = solver.good_vs[good_i].item()

                if_show_x = evaluator.record2(i=i, v=good_v, x=good_x)
                evaluator.logging_print(x=good_x, v=good_v, show_str=f"{good_v:6}", if_show_x=if_show_x)

        if if_reinforce:
            best_x = evaluator.best_x
            best_v = evaluator.best_v
            evaluator.logging_print(x=best_x, v=best_v, show_str=f"{best_v:9.0f}", if_show_x=True)

            train_loop(num_train, device, seq_len, best_x, num_sims1, sim, net, optimizer, show_gap, noise_std)

        evaluator.plot_record()


if __name__ == '__main__':
    search_and_evaluate_local_search()
