import os
import sys
from typing import Tuple, Optional

import torch as th
import torch.nn as nn

from config import ConfigGraph, ConfigPolicy
from graph_embedding_pretrain import sort_adj_bools
from network import GraphTRS, create_mask
from rlsolver.methods.util_evaluator import Evaluator
from rlsolver.methods.util_read_data import load_graph_list, update_xs_by_vs, pick_xs_by_vs, GraphList, build_adjacency_bool

TEN = th.Tensor
'''network'''


def group_concat(t0: TEN, t1: TEN, num_heads: int) -> TEN:
    d0, d1 = t0.shape[:2]
    t2 = th.concat(tensors=(t0.view(d0, d1, num_heads, -1),
                            t1.view(d0, d1, num_heads, -1)), dim=3)
    t2 = t2.view(d0, d1, -1)
    return t2


def group_split(t2: TEN, t0_dim: int, t1_dim: int, num_heads: int) -> Tuple[TEN, TEN]:
    d0, d1 = t2.shape[:2]
    t2 = t2.view(d0, d1, num_heads, -1)
    # assert (t0_dim + t1_dim) == (t2.shape[3] + self.num_heads)
    t0 = t2[:, :, :, :+(t0_dim // num_heads)].reshape(d0, d1, t0_dim)
    t1 = t2[:, :, :, -(t1_dim // num_heads):].reshape(d0, d1, t1_dim)
    return t0, t1


def convert_solution_to_prob(solution_xs: TEN) -> TEN:
    """
    solution_xs.shape == (num_sims, num_nodes); th.bool
    seq_prob.shape == (num_nodes, num_sims, 2); th.float32
    """
    solution_xs_float = th.where(solution_xs.T, 1, -1)[:, :, None]
    seq_prob = th.concat((solution_xs_float, -solution_xs_float), dim=2)
    return seq_prob


class TrsDecoderLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, out_dim: int):
        super().__init__()
        prob_num = 2

        self.num_heads = num_heads
        self.f_dim = embed_dim
        self.p_dim = prob_num * num_heads

        full_embed_dim = self.f_dim + self.p_dim
        self.self_attn = nn.MultiheadAttention(full_embed_dim, num_heads, dropout=0)
        self.multi_head_attn = nn.MultiheadAttention(full_embed_dim, num_heads, dropout=0)

        self.mlp_y_f = nn.Sequential(nn.Tanh(), nn.Linear(self.f_dim, self.f_dim))
        self.mlp_y_p = nn.Sequential(nn.Tanh(), nn.Linear(self.p_dim, out_dim))

    def forward(self, seq_prob: TEN, seq_graph: TEN, seq_memory: Optional[TEN] = None,
                if_memory: bool = True) -> Tuple[TEN, TEN]:
        """
        seq_prob.shape   == (num_nodes, num_sims, 2); th.float32            # prob of each node in a solution
        seq_graph.shape  == (num_nodes, num_sims, feature_dim); th.float32  # fixed graph_embedding features
        seq_memory.shape == (num_nodes, num_sims, feature_dim); th.float32  # memory of this transformer layer
        """
        num_sims = seq_prob.shape[1]
        x_e = seq_graph[:, 0:1, :].repeat(1, num_sims, 1) if seq_graph.shape[1] == 1 else seq_graph
        x_p = seq_prob.repeat((1, 1, self.num_heads))
        x_f = seq_memory

        x_query = group_concat(t0=x_e, t1=x_p, num_heads=self.num_heads)
        x_memo = group_concat(t0=x_f, t1=x_p, num_heads=self.num_heads) if seq_memory is not None \
            else x_query

        x = x_query + self.self_attn(x_query, x_query, x_query, attn_mask=None, need_weights=False)[0]
        y = x + self.multi_head_attn(x, x_memo, x_memo, attn_mask=None, need_weights=False)[0]

        y_f, y_p = group_split(t2=y, t0_dim=self.f_dim, t1_dim=self.p_dim, num_heads=self.num_heads)

        if if_memory:
            seq_memory = self.mlp_y_f(y_f)
        else:
            seq_memory = None

        seq_prob = self.mlp_y_p(y_p)
        return seq_prob, seq_memory

    def get_logprob_entropy(self, curr_xs: TEN, next_xs: TEN, seq_graph: TEN) -> Tuple[TEN, TEN]:
        """
        curr_xs.shape == (num_sims, num_nodes)
        next_xs.shape == (num_sims, num_nodes)
        """
        seq_prob = convert_solution_to_prob(solution_xs=curr_xs)
        seq_prob, _seq_memory = self.forward(seq_prob=seq_prob, seq_graph=seq_graph, seq_memory=None)
        seq_prob = seq_prob.permute(1, 0, 2)  # [num_sims, num_nodes, 2]

        # logprob = th.log(th.where(next_xs, prob, _prob) + 1e-4).sum(dim=-1)
        logprob = th.log_softmax(seq_prob, dim=2)  # important for gradient
        logprob = th.where(next_xs, logprob[:, :, 0], logprob[:, :, 1]).sum(dim=-1)

        _seq_prob = th.softmax(seq_prob, dim=2)  # [num_sims, num_nodes, 2]
        prob = _seq_prob[:, :, 0]
        _prob = _seq_prob[:, :, 1]
        entropy = (prob * prob.log2() + _prob * _prob.log2()).mean(dim=-1)
        return logprob, entropy


class TrsCell(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_layers: int):
        super().__init__()
        '''TransformerDecoderLayer'''
        self.trs_decoder_layers = []
        for layer_id in range(num_layers):
            trs_decoder_layer = TrsDecoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                out_dim=2,
            )
            self.trs_decoder_layers.append(trs_decoder_layer)
            setattr(self, f'trs_decoder_layer{layer_id:02}', trs_decoder_layer)

        self.mlp_value = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Tanh(),
                                       nn.Linear(embed_dim, 1), )

    def forward(self, seq_prob: TEN, seq_graph: TEN, seq_memory: Optional[TEN], layer_i: int):
        """
        seq_prob.shape   == (num_nodes, num_sims, 2); th.float32
        seq_memory.shape == (num_nodes, num_sims, feature_dim); th.float32
        seq_graph.shape  == (num_nodes, 1, feature_dim); th.float32
        """
        trs_decoder_layer = self.trs_decoder_layers[layer_i]
        seq_prob, seq_memory = trs_decoder_layer(
            seq_prob=seq_prob,
            seq_graph=seq_graph,
            seq_memory=seq_memory,
        )
        return th.softmax(seq_prob, dim=-1), seq_memory

    def get_value(self, seq_memory: TEN):
        seq_value = self.mlp_value(seq_memory)  # shape = (seq_len, num_sims, embed_dim)
        return seq_value.sum(dim=0)  # shape = (num_sims, embed_dim)

    def get_logprob_entropy(self, curr_xs: TEN, next_xs: TEN, seq_graph: TEN, layer_i: int) -> Tuple[TEN, TEN]:
        trs_decoder_layer = self.trs_decoder_layers[layer_i]
        return trs_decoder_layer.get_logprob_entropy(curr_xs=curr_xs, next_xs=next_xs, seq_graph=seq_graph)


def check_policy_trs_layer():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    """single_graph"""
    '''config'''
    graph_type, num_nodes, graph_id = 'BarabasiAlbert', 500, 0
    graph_name = f'{graph_type}_{num_nodes}_ID{graph_id}'
    # graph_type, num_nodes, graph_id = 'gset_14', 800, 0
    # graph_name = graph_type
    graph_list = load_graph_list(graph_name=graph_name)

    '''simulator'''
    from graph_max_cut_simulator import SimulatorGraphMaxCut
    sim = SimulatorGraphMaxCut(graph_list=graph_list, device=device, if_bidirectional=True)
    # if_max = sim.if_maximize

    '''seq_adj_float'''
    adj_bool = build_adjacency_bool(graph_list=graph_list, num_nodes=num_nodes, if_bidirectional=True).to(device)
    seq_adj_float = adj_bool[:, None, :].float()  # input tensor, sequence of adjacency_bool
    seq_mask_bool = create_mask(seq_len=num_nodes, mask_type='eye').to(device).detach()

    """get seq_graph"""
    args_graph = ConfigGraph(graph_list=graph_list, graph_type=graph_type)
    '''build graph_embedding_net'''
    graph_embed_net = GraphTRS(
        inp_dim=args_graph.inp_dim,
        mid_dim=args_graph.mid_dim,
        out_dim=args_graph.out_dim,
        embed_dim=args_graph.embed_dim,
        num_heads=args_graph.num_heads,
        num_layers=args_graph.num_layers
    ).to(device)

    graph_embedding_net_path = f'./model/graph_trs_{graph_type}_{num_nodes}.pth'
    print(f"graph_embedding_net_path {graph_embedding_net_path}  exists? {os.path.exists(graph_embedding_net_path)}")
    if not os.path.exists(graph_embedding_net_path):
        raise FileNotFoundError(f"graph_embedding_net_path {graph_embedding_net_path}")
    graph_embed_net.load_state_dict(th.load(graph_embedding_net_path, map_location=device, weights_only=True))

    '''output sequence_of_graph_embedding_features'''
    seq_graph = graph_embed_net.get_seq_graph(seq_adj_float=seq_adj_float, mask=seq_mask_bool).detach()
    seq_graph = seq_graph / seq_graph.std(dim=-1, keepdim=True)  # layer_norm
    del graph_embed_net, graph_embedding_net_path
    print(f"seq_graph.shape {seq_graph.shape}")

    """solve single_graph"""
    '''config'''
    # from config import ConfigPolicy
    # args_policy = ConfigPolicy
    # weight_decay = 0
    # learning_rate = 1e-4
    # show_gap = 4
    num_sims = 3
    # num_iters = 2 ** 8
    # reset_gap = 2 ** 6
    # save_dir = f"./{graph_type}_{num_nodes}"
    # num_searchers = 1
    # lambda_entropy = 1

    num_layers = 4
    # num_layer_repeats = 8
    # top_k = 2 ** 4
    # num_repeats = 2 ** 4

    '''build policy_trs_net'''
    policy_net = TrsCell(
        embed_dim=args_graph.embed_dim,
        num_heads=args_graph.num_heads,
        num_layers=num_layers,
    ).to(device)

    # net_params = list(policy_net.parameters())
    # optimizer = th.optim.Adam(net_params, lr=learning_rate, maximize=False) if weight_decay \
    #     else th.optim.AdamW(net_params, lr=learning_rate, maximize=False, weight_decay=weight_decay)

    best_xs = sim.generate_xs_randomly(num_sims=num_sims).detach()
    layer_i = 0

    seq_memory = seq_graph[:, 0:1, :].repeat(1, num_sims, 1)  # first seq_memory = seq_graph
    seq_prob = convert_solution_to_prob(solution_xs=best_xs)
    seq_prob, seq_memory = policy_net.forward(seq_prob=seq_prob, seq_graph=seq_graph, seq_memory=seq_memory,
                                              layer_i=layer_i)
    print(f"| seq_prob   {seq_prob.shape}")
    print(f"| seq_graph  {seq_graph.shape}")
    print(f"| seq_memory {seq_memory.shape}")


'''utils for run'''


class Buffer:
    def __init__(self, max_size: int, seq_len: int, num_nodes: int, device=th.device('cpu')):
        self.states = th.empty((seq_len + 1, max_size, num_nodes), dtype=th.bool)
        self.rewards = th.empty((seq_len, max_size), dtype=th.float32)
        self.logprobs = th.empty((seq_len, max_size), dtype=th.float32)

        self.obj_values = th.empty(max_size, dtype=th.float32)

        self.if_full = False
        self.p = 0
        self.add_size = 0
        self.max_size = max_size
        self.device = device

    def update(self, states: TEN, rewards: TEN, logprobs: TEN, obj_values: TEN):
        # assert states.shape     == (seq_len, num_sims, num_nodes)
        # assert rewards.shape    == (seq_len, num_sims)
        # assert logprobs.shape   == (seq_len, num_sims)
        # assert obj_values.shape == (seq_len, num_sims)

        self.add_size = rewards.shape[1]

        p = self.p + self.add_size  # pointer
        if (not self.if_full) and (p <= self.max_size):
            self.states[:, self.p:p] = states.to(self.device)
            self.rewards[:, self.p:p] = rewards.to(self.device)
            self.logprobs[:, self.p:p] = logprobs.to(self.device)
            self.obj_values[self.p:p] = obj_values.to(self.device)

            self.p = p
        else:
            _, ids = th.topk(self.obj_values[:self.p], k=self.add_size, largest=False)
            self.states[:, ids] = states.to(self.device)
            self.rewards[:, ids] = rewards.to(self.device)
            self.logprobs[:, ids] = logprobs.to(self.device)
            self.obj_values[ids] = obj_values.float().to(self.device)

    def sample(self, batch_size: int, device: th.device) -> Tuple[TEN, TEN, TEN]:
        ids = th.randint(self.p, size=(batch_size,), requires_grad=False)
        return (self.states[:, ids].to(device),
                self.rewards[:, ids].to(device),
                self.logprobs[:, ids].to(device),)


def get_advantages(rewards: TEN, values: TEN, lambda_gae_adv: float = 0.98) -> TEN:
    advantages = th.empty_like(values)  # advantage value

    horizon_len = rewards.shape[0]

    next_value = th.zeros_like(rewards[0])
    advantage = th.zeros_like(rewards[0])
    for t in range(horizon_len - 1, -1, -1):
        delta = rewards[t] + next_value - values[t]
        advantages[t] = advantage = delta + lambda_gae_adv * advantage
        next_value = values[t]
    return advantages


def get_seq_graph(graph_list: GraphList, args_graph: ConfigGraph, args_policy: ConfigPolicy, device: th.device,
                  graph_embed_net):
    from graph_max_cut_simulator import SimulatorGraphMaxCut

    '''config'''
    graph_type = args_graph.graph_type
    num_nodes = args_graph.num_nodes
    num_sims = args_policy.num_sims
    save_dir = f"./{graph_type}_{num_nodes}"  # TODO wait for adding to ConfigPolicy

    sim = SimulatorGraphMaxCut(graph_list=graph_list, device=device, if_bidirectional=True)
    if_max = sim.if_maximize

    '''seq_adj_float'''
    adj_bool = build_adjacency_bool(graph_list=graph_list, num_nodes=num_nodes, if_bidirectional=True).to(device)
    adj_bool = sort_adj_bools(adj_bools=adj_bool.unsqueeze(0)).squeeze(0)  # TODO sort
    seq_adj_float = adj_bool[:, None, :].float()  # input tensor, sequence of adjacency_bool

    '''output sequence_of_graph_embedding_features'''
    seq_mask_bool = create_mask(seq_len=num_nodes, mask_type='eye').to(device).detach()
    seq_graph = graph_embed_net.get_seq_graph(seq_adj_float=seq_adj_float, mask=seq_mask_bool).detach()
    seq_graph = seq_graph / seq_graph.std(dim=-1, keepdim=True)  # layer_norm

    '''evaluator'''
    best_xs = sim.generate_xs_randomly(num_sims=num_sims).detach()
    best_vs = sim.calculate_obj_values(xs=best_xs).detach()
    evaluator = Evaluator(save_dir=save_dir, num_bits=num_nodes, if_maximize=if_max, x=best_xs[0], v=best_vs[0].item())
    if best_xs.shape[0] == 256:
        aaa = 1
    return seq_graph, sim, evaluator, best_xs, best_vs


def sub_set_sampling(probs: TEN, start_xs: TEN, num_repeats: int, top_k: int) -> (TEN, TEN):
    determinism = th.abs(probs - 0.5)
    max_k = probs.shape[1]
    # 概率越远离0.5，则确定性越高
    top_values, top_ids = th.topk(determinism, k=max(max_k - top_k, 0), largest=True, dim=1)
    # 找出确定性高的（1-top_k）个比特位的概率，根据是否大于0.5的概率赋值为(1.0, 0.0)
    probs = probs.scatter(dim=1, index=top_ids, src=probs.gather(dim=1, index=top_ids).lt(0.5).float())

    xs = start_xs.repeat(num_repeats, 1)

    sim_ids = th.arange(xs.shape[0], device=xs.device)
    top_values, top_ids = th.topk(determinism, k=min(top_k, max_k), largest=False, dim=1)

    for top_i in range(top_values.shape[1]):
        _prob = top_values[:, top_i].repeat(num_repeats)
        _ids = top_ids[:, top_i].repeat(num_repeats)

        xs[sim_ids, _ids] = th.rand_like(_prob).lt(_prob)
    return xs, probs


def check_sub_set_sampling():
    num_sims = 8
    num_nodes = 6
    top_k = 3
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    start_xs = th.randint(0, 2, size=(num_sims, num_nodes), dtype=th.bool, device=device)
    probs = th.rand(size=(num_sims, num_nodes), dtype=th.float32, device=device)

    sub_set_sampling(probs=probs, start_xs=start_xs, num_repeats=4, top_k=top_k)


def valid_net(sim, net, evaluator, seq_graph: TEN, iter_i: int, graph_id: int,
              num_sims: int, seq_len: int, num_repeats: int, top_k: int, num_searchers: int):
    if_max = sim.if_maximize
    # seq_len = num_layers * num_layer_repeats

    best_xs = sim.generate_xs_randomly(num_sims=num_sims)
    best_vs = sim.calculate_obj_values(xs=best_xs)

    th.set_grad_enabled(False)
    for t in range(seq_len):
        seq_prob = convert_solution_to_prob(solution_xs=best_xs)
        seq_prob, _seq_memory = net.forward(seq_prob=seq_prob, seq_graph=seq_graph, seq_memory=None, layer_i=0)
        seq_prob = th.softmax(seq_prob.permute(1, 0, 2), dim=2)  # [num_sims, num_nodes, 2]
        seq_prob = seq_prob + th.randn_like(seq_prob) * 0.05  # TODO

        probs = seq_prob[:, :, 0]  # [num_sims, num_nodes]
        full_xs, _probs = sub_set_sampling(probs=probs, start_xs=best_xs, num_repeats=num_repeats, top_k=top_k)
        full_vs = None

        for _ in range(num_searchers):
            full_xs, full_vs = sim.local_search_inplace(good_xs=full_xs, good_vs=th.empty((), ))

        good_xs, good_vs = pick_xs_by_vs(xs=full_xs, vs=full_vs, num_repeats=num_repeats, if_maximize=if_max)
        del full_vs

        update_xs_by_vs(xs0=best_xs, vs0=best_vs, xs1=good_xs, vs1=good_vs, if_maximize=if_max)

        if_show_x = evaluator.record2(i=iter_i, vs=best_vs, xs=best_xs)
        if (t + 1) % 2 == 0:
            show_str = (f"  {t:4} value {best_vs.float().mean().long():6} < {best_vs.max():6} < {evaluator.best_v:6}"
                        f"  graph_id {graph_id}")
            evaluator.logging_print(show_str=show_str, if_show_x=if_show_x)
