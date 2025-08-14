import os
import sys
import torch as th

import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from subset_sum_simulator import SimulatorSubsetSum, SimulatorSubsetSumWithTag
from evaluator import Evaluator, EncoderBase64

TEN = th.Tensor

'''local search'''


class SolverLocalSearch:
    def __init__(self, simulator: SimulatorSubsetSum, num_nodes: int):
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
            xs1 = self.simulator.generate_xs_randomly(num_sims=num_sims)
            vs1 = self.simulator.calculate_obj_values(xs1)
            xs[sim_id] = xs1[vs1.argmax()]
        return xs

    def random_search(self, num_iters: int = 8, num_spin: int = 8):
        sim = self.simulator
        kth = self.num_nodes - num_spin

        prev_xs = self.good_xs.clone()
        prev_vs = self.good_vs.clone()

        num_sims = prev_xs.shape[0]
        sim_ids = th.arange(num_sims, device=prev_xs.device)
        thresh = None
        for _ in range(num_iters):
            '''flip randomly'''
            xs0 = prev_xs.clone()
            spin_rand = th.rand_like(prev_xs, dtype=th.float32)
            thresh = th.kthvalue(spin_rand, k=kth, dim=1)[0][:, None] if thresh is None else thresh
            spin_mask = spin_rand.gt(thresh)
            xs0[spin_mask] = th.logical_not(xs0[spin_mask])

            '''addition'''
            for j1 in range(self.num_nodes):
                xs1 = xs0.clone()
                xs1[:, j1] = th.logical_not(xs1[:, j1])

                xs2 = xs1.long()
                xs2[th.logical_not(xs1)] = -1
                xs2_amount_sum = (sim.amount[None, :] * xs1).sum(dim=1, keepdim=True)
                flip_id = th.abs(xs2 + xs2_amount_sum).argmin(dim=1)
                xs1[sim_ids, flip_id] = th.logical_not(xs1[sim_ids, flip_id])

                vs1 = sim.calculate_obj_values(xs1)
                update_xs_by_vs(prev_xs, prev_vs, xs1, vs1)

        update_xs_by_vs(self.good_xs, self.good_vs, prev_xs, prev_vs)
        return self.good_xs, self.good_vs


class SolverLocalSearchWithTag(SolverLocalSearch):
    def __init__(self, simulator: SimulatorSubsetSumWithTag, num_nodes: int):
        super().__init__(simulator, num_nodes)
        self.simulator = simulator
        self.num_nodes = num_nodes

        self.num_sims = 0
        self.good_xs = th.tensor([])  # solution x
        self.good_vs = th.tensor([])  # objective value

    def random_search(self, num_iters: int = 8, num_spin: int = 8):
        sim = self.simulator
        kth = self.num_nodes - num_spin

        prev_xs = self.good_xs.clone()
        prev_vs = self.good_vs.clone()

        num_sims = prev_xs.shape[0]
        sim_ids = th.arange(num_sims, device=prev_xs.device)
        thresh = None
        for _ in range(num_iters):
            '''flip randomly'''
            xs0 = prev_xs.clone()
            spin_rand = th.rand_like(prev_xs, dtype=th.float32)
            thresh = th.kthvalue(spin_rand, k=kth, dim=1)[0][:, None] if thresh is None else thresh
            spin_mask = spin_rand.gt(thresh)
            xs0[spin_mask] = th.logical_not(xs0[spin_mask])

            '''addition for constraint 2'''
            tag_jf_num = (sim.tag_jf * xs0).sum(dim=1)  # 被选中的JF订单数量
            tag_jw_num = (sim.tag_jw * xs0).sum(dim=1)  # 被选中的JW订单数量
            tag_num = th.stack((tag_jf_num, tag_jw_num), dim=1)
            tag_min_is = tag_num.argmin(dim=1)

            tag_mask = th.stack((sim.tag_jf, sim.tag_jw), dim=0)
            ban_mask = tag_mask[tag_min_is, :]
            tmp_amount = sim.amount[None, :] * th.logical_not(ban_mask)

            '''addition for constraint 1'''
            for j1 in range(self.num_nodes):
                xs1 = xs0.clone()
                xs1[:, j1] = th.logical_not(xs1[:, j1])

                xs2 = xs1.long()
                xs2[th.logical_not(xs1)] = -1
                xs2_amount_sum = (tmp_amount * xs1).sum(dim=1, keepdim=True)
                flip_id = th.abs(xs2 + xs2_amount_sum).argmin(dim=1)
                xs1[sim_ids, flip_id] = th.logical_not(xs1[sim_ids, flip_id])

                xs1[ban_mask] = False  # ban
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
        self.net = nn.Sequential(nn.Linear(inp_dim, mid_dim), nn.ReLU(), nn.LayerNorm(mid_dim),
                                 nn.Linear(mid_dim, mid_dim), nn.ReLU(), nn.LayerNorm(mid_dim),
                                 nn.Linear(mid_dim, out_dim), nn.Sigmoid(), )

    def forward(self, xs):
        return self.net(xs)


def train_loop(num_sims, num_nodes, best_x, sim, net, optimizer):
    sim_ids = th.arange(num_sims, device=sim.device)
    best_xs = best_x[None, :]
    show_gap = 4
    max_iter = 2 ** 6

    if_success = False
    objective = 0
    for j in range(max_iter):
        xs = sim.generate_xs_randomly(num_sims=num_sims)

        count = 0
        for i in range(num_nodes):
            inp = xs.float()
            out = net(inp)
            lab = xs.ne(best_xs).float().detach()

            objective = th.pow(out - lab, 2).sum(dim=1).mean()

            optimizer.zero_grad()
            objective.backward()
            clip_grad_norm_(net.parameters(), 3)
            optimizer.step()

            sample = out.argmax(dim=1)
            xs[sim_ids, sample] = th.logical_not(xs[sim_ids, sample])

            mask = lab.sum(dim=1).eq(0)
            num_mask = mask.sum().item()
            if num_mask >= 1:
                count += num_mask
                xs[mask] = sim.generate_xs_randomly(num_sims=num_mask)

        if_success = count >= num_sims
        if if_success or ((j % show_gap) == (show_gap - 1)):
            print(f"{j:6}  {count:9} of {num_sims:9} | {objective.item():9.3e}")
        if if_success:
            break
    return if_success


def check_generate_best_x():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0


    sim_name = ''  # 551
    x_str = "V$$xxxyztmzdzz_zthzi_o_lvLVVx$zJVg$FxBVn9VVzFu$UoRNUn$xxR$nu$7lPTyrBwTspy$x$Ecnmpz$BtFxot$d$"  # 402, 0

    # sim_name = 'Phase2_EUR-big.npy'
    # x_str = "1$$$x$$ltixTI_$$fd7t$tU$tffVFT$gb8tV_k_rzy$Y$ztN$8Tokz$J$t_NFkk_z_9tylE$zT$$$x$grrwzti"  # 385, 0

    num_sims = 2 ** 13
    mid_dim = 2 ** 9
    if os.name == 'nt':  # windows new type
        num_sims = 2 ** 4
        mid_dim = 2 ** 6

    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    '''simulator'''
    sim = SimulatorSubsetSum(sim_name=sim_name, device=device)
    enc = EncoderBase64(num_nodes=sim.num_nodes)
    num_nodes = sim.num_nodes

    '''evaluator'''
    # temp_xs = sim.generate_xs_randomly(num_sims=1)
    # temp_vs = sim.calculate_obj_values(xs=temp_xs)
    # evaluator = Evaluator(save_dir=f"{sim_name}_{gpu_id}", num_nodes=num_nodes, x=temp_xs[0], v=temp_vs[0].item())

    '''network'''
    net = PolicyMLP(inp_dim=num_nodes, mid_dim=mid_dim, out_dim=num_nodes).to(device)
    net_params = list(net.parameters())
    optimizer = th.optim.Adam(net_params, lr=1e-4, maximize=False)

    best_x = enc.str_to_bool(x_str).to(device)
    best_v = sim.calculate_obj_values(best_x[None, :])[0]
    print(f"{sim_name:32}  "
          f"abs_min_amount {th.abs(sim.amount).min().item():4}  num_nodes {sim.num_nodes:4}  obj_value {best_v.item()}  "
          f"H_b {best_x.long().sum().item():4}  H_a {th.abs((sim.amount * best_x).sum()).item():4}")

    '''loop'''
    train_loop(num_sims=num_sims, num_nodes=num_nodes, best_x=best_x, sim=sim, net=net, optimizer=optimizer)

    net.eval()
    best_xs = sim.generate_xs_randomly(num_sims=num_sims)
    best_vs = sim.calculate_obj_values(best_xs)

    xs = best_xs.clone()
    sim_ids = th.arange(num_sims, device=sim.device)
    for i in range(num_nodes * 2):
        inp = xs.float()
        out = net(inp)

        sample = out.argmax(dim=1)
        xs[sim_ids, sample] = th.logical_not(xs[sim_ids, sample])
        vs = sim.calculate_obj_values(xs)

        update_xs_by_vs(best_xs, best_vs, xs, vs)

    print(best_vs)
    print(best_vs.max())
    print(best_vs.eq(best_v.item()).sum(), num_sims)


'''run'''


def search_and_evaluate_local_search():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    x_str = None
    sim_name = 'xxx.csv'

    # if_reinforce = False
    # num_reset = 2 ** 1
    # num_iter1 = 2 ** 6
    # num_iter0 = 2 ** 4
    # num_sims = 2 ** 13

    if_reinforce = True
    num_reset = 2 ** 5
    num_iter1 = 2 ** 5
    num_iter0 = 2 ** 4
    num_sims = 2 ** 13

    mid_dim = 2 ** 9
    lr = 1e-4

    num_skip = 2 ** 0
    gap_print = 2 ** 0

    if os.name == 'nt':  # windows new type
        num_sims = 2 ** 4
        num_reset = 2 ** 1
        num_iter1 = 2 ** 4
        num_iter0 = 2 ** 2
        mid_dim = 2 ** 6

    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    if sim_name.find('tag') == -1:
        simulator_class = SimulatorSubsetSum
        solver_class = SolverLocalSearch
    else:
        simulator_class = SimulatorSubsetSumWithTag
        solver_class = SolverLocalSearchWithTag

    '''simulator'''
    sim = simulator_class(sim_name=sim_name, device=device)
    num_nodes = sim.num_nodes

    '''evaluator'''
    temp_xs = sim.generate_xs_randomly(num_sims=1)
    temp_vs = sim.calculate_obj_values(xs=temp_xs)
    evaluator = Evaluator(save_dir=f"{sim_name}_{gpu_id}", num_nodes=num_nodes, x=temp_xs[0], v=temp_vs[0].item())
    value_and_const = sim.calculate_obj_values(xs=temp_xs[0:1], if_sum=False)[0].cpu().data.numpy()
    best_const_value = value_and_const[1:].sum()

    '''solver'''
    solver = solver_class(simulator=sim, num_nodes=sim.num_nodes)

    '''network'''
    net = PolicyMLP(inp_dim=num_nodes, mid_dim=mid_dim, out_dim=num_nodes).to(device)
    optimizer = th.optim.Adam(net.parameters(), lr=lr, maximize=False)

    """loop"""
    th.set_grad_enabled(True)
    print(f"start searching, {sim_name}  num_nodes={num_nodes}")
    sim_ids = th.arange(num_sims, device=device)
    for j2 in range(num_reset):
        print(f"|\n| reset {j2}  lamb {sim.lamb.cpu().data.numpy()}")
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
            for _ in range(num_iter0):
                if if_reinforce and (j2 != 0):
                    sample = net(xs.float()).argmax(dim=1)
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

                value_and_const = sim.calculate_obj_values(xs=good_x[None, :], if_sum=False)[0].cpu().data.numpy()
                value = value_and_const[0]
                const = value_and_const[1:].sum()

                if_show_x = evaluator.record2(i=i, v=good_v, x=good_x)
                if (const == 0) and (value > best_const_value):
                    best_const_value = value
                    if_show_x = True
                evaluator.logging_print(x=good_x, v=good_v, show_str=str(value_and_const), if_show_x=if_show_x)

        if if_reinforce:
            best_x = evaluator.best_x
            if_success = train_loop(num_sims=num_sims, num_nodes=num_nodes, best_x=best_x,
                                    sim=sim, net=net, optimizer=optimizer)
            if not if_success:
                net = PolicyMLP(inp_dim=num_nodes, mid_dim=mid_dim, out_dim=num_nodes).to(device)
                optimizer = th.optim.Adam(net.parameters(), lr=lr, maximize=False)
                train_loop(num_sims=num_sims, num_nodes=num_nodes, best_x=best_x,
                           sim=sim, net=net, optimizer=optimizer)

        evaluator.plot_record()


if __name__ == '__main__':
    search_and_evaluate_local_search()
