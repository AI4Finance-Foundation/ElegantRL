import os
import time

import torch as th
from TNCO_simulator import SimulatorTensorNetContract
from evaluator import Evaluator

TEN = th.Tensor


class SolverLocalSearch:
    def __init__(self, simulator: SimulatorTensorNetContract, num_bits: int):
        # the num_nodes of SolverLocalSearch is not the num_nodes of TensorNetworkEnv
        self.simulator = simulator
        self.num_bits = num_bits
        self.if_maximize = False

        self.num_sims = 0
        self.good_xs = th.tensor([])  # solution x
        self.good_vs = th.tensor([])  # objective value

        '''flip randomly with ws(weights)'''
        num_edges = simulator.num_edges
        num_bases = simulator.num_bases
        device = simulator.device
        w = (th.arange(num_bases, device=device).float() + 1) / num_bases
        self.ws = w.repeat(num_edges)[None, :]
        assert self.ws.shape[1] == (num_edges * num_bases) == num_bits

    def reset(self, xs: TEN):
        vs = self.simulator.calculate_obj_values(xs=xs)

        self.good_xs = xs
        self.good_vs = vs
        self.num_sims = xs.shape[0]
        return vs

    def reset_search(self, num_sims):
        xs = th.empty((num_sims, self.num_bits), dtype=th.bool, device=self.simulator.device)
        for sim_id in range(num_sims):
            _xs = self.simulator.generate_xs_randomly(num_sims=num_sims)
            _vs = self.simulator.calculate_obj_values(_xs)
            xs[sim_id] = _xs[_vs.argmax()]
        return xs

    def random_search(self, num_iters: int = 8, num_spin: int = 8, noise_std: float = 0.3):
        if_acc = False
        if_maximize = self.if_maximize

        sim = self.simulator
        num_sims = self.num_sims
        num_edges = sim.num_edges
        device = self.good_xs.device

        prev_es = sim.convert_binary_xs_to_edge_sorts(self.good_xs)  # edge_sorts
        prev_fs = sim.matching_sorts(prev_es).float() / sim.num_edges
        prev_vs = self.good_vs.clone()
        sim_ids = th.arange(num_sims, device=device)[:, None]
        for _ in range(num_iters):
            '''change randomly'''
            change_mask = th.randint(num_edges, size=(num_sims, num_spin), device=device)
            change_rand = th.randn(size=(num_sims, num_spin), device=device) * noise_std

            fs = prev_fs.clone()
            fs[[sim_ids, change_mask]] = fs[[sim_ids, change_mask]] + change_rand
            vs = sim.get_log10_multiple_times(edge_sorts=fs.argsort(dim=1), if_acc=if_acc)

            update_xs_by_vs(prev_fs, prev_vs, fs, vs, if_maximize=if_maximize)

        prev_xs = sim.convert_edge_sorts_to_binary_xs(prev_fs.argsort(dim=1))
        num_update = update_xs_by_vs(self.good_xs, self.good_vs, prev_xs, prev_vs, if_maximize=if_maximize)
        return self.good_xs, self.good_vs, num_update

    def explore_xs(self, xs, num_spin: int = 8, noise_std: float = 0.3):
        sim = self.simulator
        num_edges = sim.num_edges
        num_sims = xs.shape[0]
        device = xs.device

        sim_ids = th.arange(num_sims, device=device)[:, None]

        prev_es = sim.convert_binary_xs_to_edge_sorts(xs)  # edge_sorts
        prev_fs = sim.matching_sorts(prev_es).float() / num_edges

        change_mask = th.randint(num_edges, size=(num_sims, num_spin), device=device)
        change_rand = th.randn(size=(num_sims, num_spin), device=device) * noise_std

        fs = prev_fs.clone()
        fs[[sim_ids, change_mask]] = fs[[sim_ids, change_mask]] + change_rand
        edge_sorts = fs.argsort(dim=1)

        xs = sim.convert_edge_sorts_to_binary_xs(edge_sorts=edge_sorts)
        return xs


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


def check_searcher():
    import sys
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    from TNCO_simulator import EdgeSortStrH2OSycamoreN53M20, NodesSycamoreN53M20
    graph_name = 'SycamoreN53M20'
    nodes_list, ban_edges = NodesSycamoreN53M20, 0
    edge_sort_str = EdgeSortStrH2OSycamoreN53M20  # 21.1282464668655585, otherSOTA 18.544

    from TNCO_simulator import convert_str_ary_to_list_as_edge_sort
    edge_sort = convert_str_ary_to_list_as_edge_sort(edge_sort_str)
    edge_sort = th.tensor(edge_sort, dtype=th.long).to(device)

    '''auto choose NodesSycamore'''
    # env = TensorNetworkEnv(nodes_list=nodes_list, ban_edges=ban_edges, device=device, num_bases=-1)
    env = SimulatorTensorNetContract(nodes_list=nodes_list, ban_edges=ban_edges, device=device)
    print(f"\nnum_nodes      {env.num_nodes:9}"
          f"\nnum_edges      {env.num_edges:9}"
          f"\nban_edges      {env.ban_edges:9}")

    '''get multiple_times'''
    edge_sorts = edge_sort.unsqueeze(0)
    multiple_times = env.get_log10_multiple_times(edge_sorts=edge_sorts.to(device), if_acc=False)
    multiple_times = multiple_times.cpu().numpy()[0]
    print(f"multiple_times(log10) {multiple_times:9.6f} historical")
    print()

    '''edge_sort to x'''
    # edge_sort = th.arange(env.num_edges, device=device)
    from TNCO_simulator import StrSycamoreN53M20
    x_str = StrSycamoreN53M20
    from evaluator import EncoderBase64
    encoder_base64 = EncoderBase64(encode_len=env.num_bits)
    x = encoder_base64.str_to_bool(x_str=x_str).to(device)
    edge_sorts = env.convert_binary_xs_to_edge_sorts(xs=x[None, :])
    xs = env.convert_edge_sorts_to_binary_xs(edge_sorts=edge_sorts)
    vs = env.calculate_obj_values(xs=xs, if_acc=True)
    multiple_times = vs[0].item()
    print(f"multiple_times(log10) {multiple_times:9.6f} historical")
    print()

    searcher = SolverLocalSearch(simulator=env, num_bits=env.num_bits)
    if_maximize = False

    num_sims = 2 ** 5

    k = num_sims // 16
    if os.name == 'nt':
        num_sims = 2 ** 2
        k = num_sims // 4

    # temp_xs = env.generate_xs_randomly(num_sims=num_sims)
    temp_xs = searcher.explore_xs(xs=xs.repeat(num_sims, 1), num_spin=32, noise_std=0.3)
    temp_vs = searcher.reset(xs=temp_xs)
    print(f"|{0:6}  {temp_vs.min().item():9.6f}  {temp_vs.mean().item():9.6f}")

    th.set_grad_enabled(False)
    evaluator = Evaluator(save_dir=f"{graph_name}_{gpu_id}", num_bits=searcher.num_bits, if_maximize=if_maximize,
                          x=temp_xs[0], v=temp_vs[0].item())
    for j in range(2 ** 5):
        ids = temp_vs.argsort()
        ids[-k:] = ids[th.randperm(num_sims - k, device=device)[:k]]
        searcher.good_xs = temp_xs[ids]
        searcher.good_vs = temp_vs[ids]

        # temp_vs = searcher.reset(xs=searcher.explore_xs(temp_xs, num_spin=64, noise_std=0.01))
        for k in range(2 ** 5):
            # searcher.random_search(num_iters=2 ** 4, num_spin=8, noise_std=0.5)  # GPU 4
            searcher.random_search(num_iters=2 ** 4, num_spin=4, noise_std=0.5)  # GPU 5
            good_xs = searcher.good_xs
            good_vs = searcher.good_vs

            update_xs_by_vs(temp_xs, temp_vs, good_xs, good_vs, if_maximize=if_maximize)

            '''update good_xs'''
            good_i = good_vs.argmax() if if_maximize else good_vs.argmin()
            good_x = good_xs[good_i]
            good_v = good_vs[good_i]
            if_show_x = evaluator.record2(i=j * (2 ** 5) + k, vs=good_v.item(), xs=good_x)
            show_str = f"{good_vs.min():9.6f} < {good_vs.float().mean():9.6f}"
            evaluator.logging_print(x=good_x, v=good_v, show_str=show_str, if_show_x=if_show_x)
        evaluator.save_record_draw_plot(fig_dpi=300)
    evaluator.save_record_draw_plot(fig_dpi=300)


if __name__ == '__main__':
    check_searcher()
