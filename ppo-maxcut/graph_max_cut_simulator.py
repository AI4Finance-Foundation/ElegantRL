import torch as th
from graph_utils import GraphList, build_adjacency_indies
TEN = th.Tensor


class SimulatorGraphMaxCut:
    def __init__(self, args, graph_list: GraphList = (),
                 device=th.device('cpu'), if_bidirectional: bool = False):
        self.device = device
        self.int_type = int_type = th.long
        self.if_bidirectional = if_bidirectional
        self.num_nodes = args.num_nodes
        self.num_envs = args.num_envs
        self.xs = None
        self.action_count = 0
        self.last_reward = None
        self.num_steps = args.num_steps

        n0_to_n1s, n0_to_dts = build_adjacency_indies(graph_list=graph_list, if_bidirectional=if_bidirectional)
        n0_to_n1s = [t.to(int_type).to(device) for t in n0_to_n1s]
        self.num_edges = len(graph_list)
        n0_to_n0s = [(th.zeros_like(n1s) + i) for i, n1s in enumerate(n0_to_n1s)]
        self.n0_ids = th.hstack(n0_to_n0s)[None, :]
        self.n1_ids = th.hstack(n0_to_n1s)[None, :]
        len_sim_ids = self.num_edges * (2 if if_bidirectional else 1)
        self.sim_ids = th.zeros(len_sim_ids, dtype=int_type, device=device)[None, :]

    def reset(self):
        self.xs = self.generate_xs_randomly(num_sims=self.num_envs)
        self.xs = self.xs.to(th.float)
        self.last_reward = self.calculate_obj_values().to(th.float)

        return self.xs

    def step(self, action):
        self.action_count += 1
        for n in range(self.num_envs):
            self.xs[n, action[n]] = th.logical_not(self.xs[n, action[n]])
        cur_reward = self.calculate_obj_values().to(th.float)
        reward = cur_reward - self.last_reward
        self.last_reward = cur_reward

        if self.action_count == self.num_steps:
            self.action_count = 0
            next_done = th.ones([self.num_envs], dtype=th.float, device=self.device)
        else:
            next_done = th.zeros([self.num_envs], dtype=th.float, device=self.device)

        return self.xs, reward, next_done, cur_reward

    def calculate_obj_values(self, if_sum: bool = True) -> TEN:
        xs = self.xs > 0
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

    def generate_xs_randomly(self, num_sims):
        xs = th.randint(0, 2, size=(num_sims, self.num_nodes), dtype=th.bool, device=self.device)
        xs[:, 0] = 0
        return xs
