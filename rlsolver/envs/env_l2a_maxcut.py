import torch as th
import sys
sys.path.append('..')
from rlsolver.methods.L2A.graph_utils import  GraphList
from rlsolver.methods.L2A.maxcut_simulator import SimulatorMaxcut
from rlsolver.methods.L2A.maxcut_local_search import SolverLocalSearch

TEN = th.Tensor


def metropolis_hastings_sampling(probs: TEN, start_xs: TEN, num_repeats: int, num_iters: int = -1) -> TEN:
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
    xs = start_xs.repeat(num_repeats, 1)  # 并行，，
    ps = probs.repeat(num_repeats, 1)

    num, dim = xs.shape
    device = xs.device
    num_iters = int(dim // 4) if num_iters == -1 else num_iters  # 希望至少有1/4的节点的采样结果被接受

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


class MCMC_Maxcut:
    def __init__(self, num_nodes: int, num_sims: int, num_repeats: int, num_searches: int,
                 graph_list: GraphList = (), device=th.device('cpu')):
        self.num_nodes = num_nodes
        self.num_sims = num_sims
        self.num_repeats = num_repeats
        self.num_searches = num_searches
        self.device = device
        self.sim_ids = th.arange(num_sims, device=device)

        # build in reset
        self.simulator = SimulatorMaxcut(graph_list=graph_list, device=self.device)  # 初始值
        self.searcher = SolverLocalSearch(simulator=self.simulator, num_nodes=self.num_nodes)

    # 如果end to end, graph_list为空元组。如果distribution, 抽样赋值
    def reset(self, graph_list: GraphList = ()):
        self.simulator = SimulatorMaxcut(graph_list=graph_list, device=self.device)
        self.searcher = SolverLocalSearch(simulator=self.simulator, num_nodes=self.num_nodes)
        self.searcher.reset(xs=self.simulator.generate_xs_randomly(num_sims=self.num_sims))

        good_xs = self.searcher.good_xs
        good_vs = self.searcher.good_vs
        return good_xs, good_vs

    # probs: 策略网络输出值
    # start_xs： 上一轮step的输出的解中，并行环境输出的最好的解
    def step(self, start_xs: TEN, probs: TEN) -> (TEN, TEN):
        xs = metropolis_hastings_sampling(probs=probs, start_xs=start_xs, num_repeats=self.num_repeats, num_iters=-1)
        vs = self.searcher.reset(xs)
        for _ in range(self.num_searches):
            # 再用local search, searcher 是local search
            xs, vs, num_update = self.searcher.random_search(num_iters=8)
        return xs, vs

    # 好的解的数量是sim数量，一个环境出一个好的解
    def pick_good_xs(self, full_xs, full_vs) -> (TEN, TEN):
        # update good_xs: use .view() instead of .reshape() for saving GPU memory
        xs_view = full_xs.view(self.num_repeats, self.num_sims, self.num_nodes)
        vs_view = full_vs.view(self.num_repeats, self.num_sims)
        ids = vs_view.argmax(dim=0)

        good_xs = xs_view[ids, self.sim_ids]
        good_vs = vs_view[ids, self.sim_ids]
        return good_xs, good_vs
