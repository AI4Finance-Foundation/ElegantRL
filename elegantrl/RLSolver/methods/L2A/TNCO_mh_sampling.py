from TNCO_simulator import *
from TNCO_local_search import *
from maxcut_end2end import show_gpu_memory, reset_parameters_of_model
from config import ConfigPolicy
from torch.nn.utils import clip_grad_norm_


def metropolis_hastings_sampling(probs: TEN, start_xs: TEN, num_repeats: int, num_iters: int = -1,
                                 accept_rate: float = 0.25) -> TEN:
    """随机平稳采样 是 metropolis-hastings sampling is:
    - 在状态转移链上 using transition kernel in Markov Chain
    - 使用随机采样估计 Monte Carlo sampling
    - 依赖接受概率去达成细致平稳条件 with accept ratio to satisfy detailed balance condition
    的采样方法。

    工程实现上:
    - 让它从多个随机的起始位置 start_xs 开始
    - 每个起始位置建立多个副本 repeat
    - 循环多次 for _ in range
    - 直到接受的样本数量超过阈值 count >= 再停止

    这种采样方法允许不同的区域能够以平稳的概率采集到符合采样概率的样本，确保样本数量比例符合期望。
    具体而言，每个区域内采集到的样本数与该区域所占的平稳分布概率成正比。
    """
    # metropolis-hastings sampling: Monte Carlo sampling using transition kernel in Markov Chain with accept ratio
    xs = start_xs.repeat(num_repeats, 1)
    ps = probs.repeat(num_repeats, 1)

    num, dim = xs.shape
    device = xs.device
    num_iters = int(dim * accept_rate) if num_iters == -1 else num_iters  # 希望至少有accept_rate的节点的采样结果被接受

    count = 0
    for _ in range(4):  # 迭代4轮后，即便被拒绝的节点很多，也不再迭代了。
        ids = th.randperm(dim, device=device)  # 按随机的顺序，均匀地选择节点进行采样。避免不同节点被采样的次数不同。
        for i in range(dim):
            idx = ids[i]
            chosen_p0 = ps[:, idx]
            chosen_xs = xs[:, idx]
            chosen_ps = th.where(chosen_xs, chosen_p0, 1 - chosen_p0)

            accept_rates = (1 - chosen_ps) / chosen_ps
            accept_masks = th.rand(num, device=device).lt(accept_rates)
            xs[:, idx] = th.where(accept_masks, th.logical_not(chosen_xs), chosen_xs)

            count += accept_masks.sum()
            if count >= num * num_iters:
                break
        if count >= num * num_iters:
            break
    return xs


class MCMC_TNCO:
    def __init__(self, num_sims: int, num_repeats: int, num_searches: int,
                 graph_type: str = 'graph', nodes_list: list = (), device=th.device('cpu')):
        self.num_sims = num_sims
        self.num_repeats = num_repeats
        self.num_searches = num_searches
        self.device = device
        self.sim_ids = th.arange(num_sims, device=device)

        # build in reset
        self.simulator = SimulatorTensorNetContract(nodes_list=nodes_list, ban_edges=0, device=self.device)
        self.num_bits = self.simulator.num_bits
        self.searcher = SolverLocalSearch(simulator=self.simulator, num_bits=self.num_bits)
        self.if_maximize = self.searcher.if_maximize

    def reset(self):
        xs = self.simulator.generate_xs_randomly(num_sims=self.num_sims)
        self.searcher.reset(xs)
        for _ in range(self.num_searches * 2):
            self.searcher.random_search(num_iters=8)

        good_xs = self.searcher.good_xs
        good_vs = self.searcher.good_vs
        return good_xs, good_vs

    def step(self, start_xs: TEN, probs: TEN) -> (TEN, TEN):
        xs = metropolis_hastings_sampling(probs=probs, start_xs=start_xs, num_repeats=self.num_repeats, num_iters=-1)
        vs = self.searcher.reset(xs)
        for _ in range(self.num_searches):
            xs, vs, num_update = self.searcher.random_search(num_iters=2 ** 3, num_spin=8, noise_std=0.5)
        return xs, vs

    def good(self, full_xs, full_vs) -> (TEN, TEN):
        # update good_xs: use .view() instead of .reshape() for saving GPU memory
        xs_view = full_xs.view(self.num_repeats, self.num_sims, self.num_bits)
        vs_view = full_vs.view(self.num_repeats, self.num_sims)
        ids = vs_view.argmax(dim=0) if self.if_maximize else vs_view.argmin(dim=0)

        good_xs = xs_view[ids, self.sim_ids]
        good_vs = vs_view[ids, self.sim_ids]
        return good_xs, good_vs


def valid_in_single_graph(
        args0: ConfigPolicy = None,
        nodes_list: list = None,
):
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    '''dummy value'''
    # args0 = args0 if args0 else ConfigPolicy(graph_type='SycamoreN12M14', num_nodes=1)  # todo plan remove num_nodes
    # nodes_list = NodesSycamoreN12M14 if nodes_list is None else nodes_list
    args0 = args0 if args0 else ConfigPolicy(graph_type='SycamoreN53M20', num_nodes=1)  # todo plan remove num_nodes
    nodes_list = NodesSycamoreN53M20 if nodes_list is None else nodes_list

    '''custom'''
    args0.num_iters = 2 ** 6 * 7
    args0.reset_gap = 2 ** 6
    args0.num_sims = 2 ** 4  # LocalSearch 的初始解数量
    args0.num_repeats = 2 ** 4  # LocalSearch 对于每以个初始解进行复制的数量

    if os.name == 'nt':
        args0.num_sims = 2 ** 2
        args0.num_repeats = 2 ** 3

    '''config: graph'''
    graph_type = args0.graph_type
    num_nodes = args0.num_nodes

    '''config: train'''
    num_sims = args0.num_sims
    num_repeats = args0.num_repeats
    num_searches = args0.num_searches
    reset_gap = args0.reset_gap
    num_iters = args0.num_iters
    num_sgd_steps = args0.num_sgd_steps
    entropy_weight = args0.entropy_weight

    weight_decay = args0.weight_decay
    learning_rate = args0.learning_rate

    show_gap = args0.show_gap

    '''iterator'''
    iterator = MCMC_TNCO(num_sims=num_sims, num_repeats=num_repeats, num_searches=num_searches,
                         graph_type=graph_type, nodes_list=nodes_list, device=device)
    num_bits = iterator.num_bits  # todo add num_bits
    if_maximize = iterator.if_maximize

    '''model'''
    # from network import PolicyORG
    # policy_net = PolicyORG(num_bits=num_bits).to(device)
    from network import PolicyMLP
    policy_net = PolicyMLP(num_bits=num_bits).to(device)
    policy_net = th.compile(policy_net) if th.__version__ < '2.0' else policy_net

    net_params = list(policy_net.parameters())
    optimizer = th.optim.Adam(net_params, lr=learning_rate, maximize=False) if weight_decay \
        else th.optim.AdamW(net_params, lr=learning_rate, maximize=False, weight_decay=weight_decay)

    '''evaluator'''
    save_dir = f"./ORG_{graph_type}_{num_bits}"
    os.makedirs(save_dir, exist_ok=True)
    good_xs, good_vs = iterator.reset()
    evaluator = Evaluator(save_dir=save_dir, num_bits=num_nodes, if_maximize=if_maximize,
                          x=good_xs[0], v=good_vs[0].item())

    '''loop'''
    th.set_grad_enabled(False)
    lamb_entropy = (th.cos(th.arange(reset_gap, device=device) / (reset_gap - 1) * th.pi) + 1) / 2 * entropy_weight
    for i in range(num_iters):
        good_i = good_vs.argmax() if if_maximize else good_vs.argmin()
        probs = policy_net.auto_regressive(xs_flt=good_xs[good_i, None, :].float())
        probs = probs.repeat(num_sims, 1)

        full_xs, full_vs = iterator.step(start_xs=good_xs, probs=probs)
        good_xs, good_vs = iterator.good(full_xs=full_xs, full_vs=full_vs)

        advantages = full_vs.float()
        if if_maximize:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = (advantages.mean() - advantages) / (advantages.std() + 1e-8)
        del full_vs

        th.set_grad_enabled(True)  # ↓↓↓↓↓↓↓↓↓↓ gradient
        for j in range(num_sgd_steps):
            good_i = good_vs.argmax() if if_maximize else good_vs.argmin()
            probs = policy_net.auto_regressive(xs_flt=good_xs[good_i, None, :].float())
            probs = probs.repeat(num_sims, 1)

            full_ps = probs.repeat(num_repeats, 1)
            logprobs = th.log(th.where(full_xs, full_ps, 1 - full_ps)).sum(dim=1)

            _probs = 1 - probs
            entropy = (probs * probs.log2() + _probs * _probs.log2()).mean(dim=1)
            obj_entropy = entropy.mean()
            obj_values = (th.softmax(logprobs, dim=0) * advantages).sum()

            objective = obj_values + obj_entropy * lamb_entropy[i % reset_gap]
            optimizer.zero_grad()
            objective.backward()
            clip_grad_norm_(net_params, 3)
            optimizer.step()
        th.set_grad_enabled(False)  # ↑↑↑↑↑↑↑↑↑ gradient

        '''update good_xs'''
        good_i = good_vs.argmax() if if_maximize else good_vs.argmin()
        good_x = good_xs[good_i]
        good_v = good_vs[good_i]
        if_show_x = evaluator.record2(i=i, vs=good_v.item(), xs=good_x)

        if (i + 1) % show_gap == 0 or if_show_x:
            _probs = 1 - probs
            entropy = (probs * probs.log2() + _probs * _probs.log2()).mean(dim=1)
            obj_entropy = -entropy.mean().item()

            show_str = f"| entropy {obj_entropy:9.4f} obj_value {good_vs.min():9.6f} < {good_vs.mean():9.6f}"
            evaluator.logging_print(x=good_x, v=good_v, show_str=show_str, if_show_x=if_show_x)
            sys.stdout.flush()

        if (i + 1) % reset_gap == 0:
            print(f"| reset {show_gpu_memory(device=device)} "
                  f"| up_rate {evaluator.best_v / evaluator.first_v - 1.:8.5f}")
            sys.stdout.flush()

            '''method1: reset (keep old graph)'''
            reset_parameters_of_model(model=policy_net)
            good_xs, good_vs = iterator.reset()

    evaluator.save_record_draw_plot(fig_dpi=300)


if __name__ == '__main__':
    valid_in_single_graph()
    # check_searcher()
