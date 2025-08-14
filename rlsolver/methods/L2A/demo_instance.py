import os
import sys

cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../../rlsolver')
sys.path.append(os.path.dirname(rlsolver_path))

import numpy as np
import torch as th
from torch.nn.utils import clip_grad_norm_

from config import ConfigGraph, ConfigPolicy
from rlsolver.methods.util_evaluator import Evaluator
from transformer import TrsCell, Buffer, convert_solution_to_prob, sub_set_sampling, get_advantages
from network import GraphTRS, create_mask
from rlsolver.methods.util_read_data import load_graph_list, GraphTypes, update_xs_by_vs, pick_xs_by_vs, GraphList, build_adjacency_bool
from rlsolver.methods.util import gpu_info_str
from graph_embedding_pretrain import train_graph_net_in_a_single_graph, train_graph_net_in_graph_distribution


def solve_single_graph_problem_using_trs(
        graph_list: GraphList,
        args_graph: ConfigGraph,
        args_policy: ConfigPolicy,
        gpu_id: int = 0
):
    # graph_type, num_nodes, graph_id = 'PowerLaw', 100, 0
    # graph_name = f'{graph_type}_{num_nodes}_ID{graph_id}'
    # graph_type, num_nodes, graph_id = 'gset_14', 800, 0
    # args_graph = ConfigGraph(graph_list=graph_list, graph_type=graph_type)
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    '''config'''
    graph_type = args_graph.graph_type
    num_nodes = args_graph.num_nodes

    '''simulator'''
    from graph_max_cut_simulator import SimulatorGraphMaxCut
    sim = SimulatorGraphMaxCut(graph_list=graph_list, device=device, if_bidirectional=True)
    if_max = sim.if_maximize

    '''seq_adj_float'''
    adj_bool = build_adjacency_bool(graph_list=graph_list, num_nodes=num_nodes, if_bidirectional=True).to(device)
    seq_adj_float = adj_bool[:, None, :].float()  # input tensor, sequence of adjacency_bool
    seq_mask_bool = create_mask(seq_len=num_nodes, mask_type='eye').to(device).detach()

    """get seq_graph"""
    '''build graph_embedding_net'''
    graph_embed_net = GraphTRS(
        inp_dim=args_graph.inp_dim,
        mid_dim=args_graph.mid_dim,
        out_dim=args_graph.out_dim,
        embed_dim=args_graph.embed_dim,
        num_heads=args_graph.num_heads,
        num_layers=args_graph.num_layers
    ).to(device)
    net_path = f'./model/graph_trs_{graph_type}_{num_nodes}.pth'
    print(f"graph_embedding_net_path {net_path}  exists? {os.path.exists(net_path)}", flush=True)
    if not os.path.exists(net_path):
        if graph_type in GraphTypes:
            train_graph_net_in_graph_distribution(args=args_graph, net_path=net_path)
        else:
            train_graph_net_in_a_single_graph(args=args_graph, graph_list=graph_list, net_path=net_path, gpu_id=gpu_id)

    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, message=".*FlashAttention.*")
    graph_embed_net.load_state_dict(th.load(net_path, map_location=device, weights_only=True))
    # TODO torch_cuda_12.4代码 加载 旧torch模型 引发 UserWarning 与 Flash Attention (new feature in CUDA 12.4) 相关

    '''output sequence_of_graph_embedding_features'''
    seq_graph = graph_embed_net.get_seq_graph(seq_adj_float=seq_adj_float, mask=seq_mask_bool).detach()
    seq_graph = seq_graph / seq_graph.std(dim=-1, keepdim=True)  # layer_norm
    del graph_embed_net, net_path
    print(f"seq_graph.shape {seq_graph.shape}", flush=True)

    """solve single_graph"""
    '''config'''
    assert isinstance(args_policy, ConfigPolicy)
    weight_decay = args_policy.weight_decay
    learning_rate = args_policy.learning_rate
    show_gap = args_policy.show_gap
    num_sims = args_policy.num_sims
    num_iters = args_policy.num_iters
    reset_gap = args_policy.reset_gap

    save_dir = f"./{graph_type}_{num_nodes}"  # TODO wait for adding to ConfigPolicy
    num_searchers = 8
    num_layers = 1
    num_layer_repeats = 2 ** 5
    top_k = num_nodes // 4
    num_repeats = 2 ** 6
    max_buffer_size = 2 ** 12

    repeat_times = 8
    ratio_clip = 0.25
    lambda_entropy = 4

    criterion = th.nn.SmoothL1Loss()

    if os.name == 'nt':
        print("| Warning: checking mode.", flush=True)
        num_sims = 2 ** 6
        show_gap = 1
        num_layer_repeats = 2 ** 2

    '''build policy_trs_net'''
    embed_dim = args_graph.embed_dim
    num_heads = args_graph.num_heads
    net = TrsCell(embed_dim=embed_dim, num_heads=num_heads, num_layers=1).to(device)
    net_param = net.parameters()
    net_optim = th.optim.Adam(net_param, lr=learning_rate) if weight_decay \
        else th.optim.AdamW(net_param, lr=learning_rate, weight_decay=weight_decay)
    # net_optim = th.optim.SGD(net_param, lr=learning_rate, momentum=0.5)  # TODO

    '''buffer'''
    seq_len = num_layers * num_layer_repeats
    buffer = Buffer(max_size=max_buffer_size, seq_len=seq_len, num_nodes=num_nodes, device=th.device('cpu'))

    '''evaluator'''
    best_xs = sim.generate_xs_randomly(num_sims=num_sims).detach()
    best_vs = sim.calculate_obj_values(xs=best_xs).detach()
    evaluator = Evaluator(save_dir=save_dir, num_bits=num_nodes, if_maximize=if_max,
                          x=best_xs[0], v=best_vs[0].item())

    '''loop'''
    # seq_memory = seq_graph[:, 0:1, :].repeat(1, num_sims, 1)  # first seq_memory = seq_graph

    th.set_grad_enabled(False)
    for iter_i in range(num_iters):
        # evolutionary_replacement(best_xs, best_vs, low_k=2, if_max=if_max)
        # best_xs = sim.generate_xs_randomly(num_sims=num_sims).detach()
        # best_vs = sim.calculate_obj_values(xs=best_xs).detach()

        states = th.empty((seq_len + 1, num_sims, num_nodes), dtype=th.bool, device=device)
        rewards = th.empty((seq_len, num_sims), dtype=th.float32, device=device)
        logprobs = th.empty((seq_len, num_sims), dtype=th.float32, device=device)
        curr_vs = None

        for t in range(seq_len):
            seq_prob = convert_solution_to_prob(solution_xs=best_xs)
            seq_prob, _seq_memory = net.forward(seq_prob=seq_prob, seq_graph=seq_graph, seq_memory=None, layer_i=0)
            seq_prob = th.softmax(seq_prob.permute(1, 0, 2), dim=2)  # [num_sims, num_nodes, 2]
            seq_prob = seq_prob + th.randn_like(seq_prob) * 0.02  # TODO

            probs = seq_prob[:, :, 0]  # [num_sims, num_nodes]
            full_xs, _probs = sub_set_sampling(probs=probs, start_xs=best_xs, num_repeats=num_repeats, top_k=top_k)
            full_vs = None

            for _ in range(num_searchers):
                full_xs, full_vs = sim.local_search_inplace(good_xs=full_xs, good_vs=th.empty((), ))

            good_xs, good_vs = pick_xs_by_vs(xs=full_xs, vs=full_vs, num_repeats=num_repeats, if_maximize=if_max)
            del full_vs

            curr_xs = best_xs.clone()
            curr_vs = best_vs.clone()
            update_xs_by_vs(xs0=best_xs, vs0=best_vs, xs1=good_xs, vs1=good_vs, if_maximize=if_max)

            states[t] = curr_xs
            rewards[t] = best_vs - curr_vs
            logprobs[t] = th.log(th.where(best_xs, probs, 1 - probs).clip(0.005, 0.995)).sum(dim=1)

            if_show_x = evaluator.record2(i=iter_i, vs=best_vs, xs=best_xs)
            if (t + 1) % 2 == 0:
                show_str = f"  {t:4} value {best_vs.float().mean().long():6} < {best_vs.max():6} < {evaluator.best_v:6}"
                evaluator.logging_print(show_str=show_str, if_show_x=if_show_x)

        states[seq_len] = best_xs

        '''buffer'''
        buffer.update(states=states, rewards=rewards, logprobs=logprobs, obj_values=curr_vs)

        # print(f"|;;;| {iter_i}  {gpu_info_str(device=device)}", flush=True)

        del states, rewards, logprobs
        th.cuda.empty_cache()
        if buffer.p * num_sims < num_sims * 4:
            continue
        states, rewards, logprobs = buffer.sample(batch_size=num_sims, device=device)

        """update network"""
        th.set_grad_enabled(True)
        with th.no_grad():
            '''get advantages and reward_sums'''
            values = th.empty_like(rewards)  # values.shape == (buffer_size, buffer_num)
            for seq_t in range(seq_len):
                best_xs = states[seq_t]
                seq_prob = convert_solution_to_prob(solution_xs=best_xs)

                # seq_value, _seq_memory = critic_net.forward(seq_prob, seq_memory=None)
                # value = seq_value.sum(dim=0).squeeze(-1)
                seq_value, _seq_memory = net.forward(seq_prob=seq_prob, seq_graph=seq_graph, seq_memory=None, layer_i=0)
                value = net.get_value(seq_memory=_seq_memory).squeeze(-1)

                values[seq_t] = value

            '''get advantages reward_sums'''
            advantages = get_advantages(rewards, values, lambda_gae_adv=0.98)  # advantages.shape == (buffer_size, )
            reward_sums = advantages + values  # reward_sums.shape == (buffer_size, )
            del rewards, values

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        # assert logprobs.shape == advantages.shape == reward_sums.shape == (buffer_size, buffer_num)

        obj_critic_avg = []
        obj_policy_avg = []
        obj_entropy_avg = []

        update_times = int((seq_len * num_sims) * repeat_times // num_sims)
        assert update_times >= 1
        for t in range(update_times):
            ids = th.randint(seq_len * num_sims, size=(num_sims,), requires_grad=False)
            ids0 = th.fmod(ids, seq_len)  # ids % sample_len
            ids1 = th.div(ids, seq_len, rounding_mode='floor')  # ids // sample_len

            curr_xs = states[ids0, ids1]
            next_xs = states[ids0 + 1, ids1]
            logprob = logprobs[ids0, ids1]
            advantage = advantages[ids0, ids1]
            reward_sum = reward_sums[ids0, ids1]

            seq_prob = convert_solution_to_prob(solution_xs=curr_xs)
            seq_value, _seq_memory = net.forward(seq_prob=seq_prob, seq_graph=seq_graph, seq_memory=None, layer_i=0)
            value = net.get_value(seq_memory=_seq_memory).squeeze(-1)

            obj_critic = criterion(value, reward_sum)

            new_logprob, entropy = net.get_logprob_entropy(
                curr_xs=curr_xs, next_xs=next_xs, seq_graph=seq_graph, layer_i=0)
            obj_entropy = entropy.mean()

            ratio = (new_logprob - logprob.detach()).clip(-12, +12).exp()
            surrogate1 = advantage * ratio
            surrogate2 = advantage * ratio.clamp(1 - ratio_clip, 1 + ratio_clip)
            obj_surrogate = th.min(surrogate1, surrogate2).mean()

            obj_policy = obj_surrogate + obj_entropy * lambda_entropy

            objective = obj_critic + obj_policy
            net_optim.zero_grad()
            objective.backward()
            clip_grad_norm_(net_param, 1)
            net_optim.step()

            obj_critic_avg.append(obj_critic.item())
            obj_policy_avg.append(obj_policy.item())
            obj_entropy_avg.append(obj_entropy.item())

            # print(f';;;;;;;;; {t:4}  {obj_critic:9.3f} {obj_policy:9.3f} {obj_entropy:9.3f}', flush=True)
        th.set_grad_enabled(False)

        obj_critic_avg = np.nanmean(obj_critic_avg) if not np.all(np.isnan(obj_critic_avg)) else np.nan
        obj_policy_avg = np.nanmean(obj_policy_avg) if not np.all(np.isnan(obj_policy_avg)) else np.nan
        obj_entropy_avg = np.nanmean(obj_entropy_avg) if not np.all(np.isnan(obj_entropy_avg)) else np.nan

        '''record and show'''
        if_show_x = evaluator.record2(i=iter_i, vs=best_vs, xs=best_xs)
        if (iter_i + 1) % show_gap == 0 or if_show_x:
            show_str = (
                f"| value {best_vs.float().mean().long():6} < {best_vs.max():6} < {evaluator.best_v:6}"
                f"\n||critic {obj_critic_avg:8.4f}  policy {obj_policy_avg:8.4f}  entropy {obj_entropy_avg:8.4f}"
            )
            evaluator.logging_print(show_str=show_str, if_show_x=if_show_x)
        if (iter_i + 1) % reset_gap == 0:
            evaluator.save_record_draw_plot()
            up_rate = evaluator.best_v / evaluator.first_v - 1
            print(f"\n| reset {gpu_info_str(device=device)} | up_rate {up_rate :8.5f}", flush=True)

            best_xs = sim.generate_xs_randomly(num_sims=num_sims)
            best_vs = sim.calculate_obj_values(xs=best_xs)

            # TODO
            # net = TrsCell(embed_dim=embed_dim, num_heads=num_heads, num_layers=1, seq_graph=seq_graph).to(device)
            # net_param = net.parameters()
            # net_optim = th.optim.Adam(net_param, lr=learning_rate) if weight_decay \
            #     else th.optim.AdamW(net_param, lr=learning_rate, weight_decay=weight_decay)


def run_graph_set_14_15(graph_type: str = 'G14', gpu_id: int = 0):
    # graph_list = load_graph_list(graph_name=graph_type)
    graph_list = load_graph_list(graph_name=graph_type)

    args_graph = ConfigGraph(graph_list=graph_list, graph_type=graph_type)
    args_graph.batch_size = 2 ** 5
    args_graph.train_times = 2 ** 10
    args_graph.show_gap = 2 ** 2

    args_policy = ConfigPolicy(graph_list=graph_list, graph_type=graph_type)

    '''run'''
    # net_path = f'./model/graph_trs_{graph_type}_{args_graph.num_nodes}.pth'
    # train_graph_trs_net_in_a_single_graph(args=args_graph, graph_list=graph_list, net_path=net_path, gpu_id=gpu_id)
    solve_single_graph_problem_using_trs(graph_list, args_graph, args_policy, gpu_id=gpu_id)


def run_graph_set_22_23(graph_type: str = 'G22', gpu_id: int = 0):
    graph_list = load_graph_list(graph_name=graph_type)

    args_graph = ConfigGraph(graph_list=graph_list, graph_type=graph_type)
    args_graph.batch_size = 2 ** 3
    args_graph.train_times = 2 ** 10

    args_policy = ConfigPolicy(graph_list=graph_list, graph_type=graph_type)

    '''run'''
    # net_path = f'./model/graph_trs_{graph_type}_{args_graph.num_nodes}.pth'
    # train_graph_trs_net_in_a_single_graph(args=args_graph, graph_list=graph_list, net_path=net_path, gpu_id=gpu_id)
    solve_single_graph_problem_using_trs(graph_list, args_graph, args_policy, gpu_id=gpu_id)


def run_graph_set_55_58(graph_type: str = 'G55', gpu_id: int = 0):
    graph_list = load_graph_list(graph_name=graph_type)

    args_graph = ConfigGraph(graph_list=graph_list, graph_type=graph_type)
    args_graph.batch_size = 2 ** 3
    args_graph.train_times = 2 ** 9

    args_policy = ConfigPolicy(graph_list=graph_list, graph_type=graph_type)

    '''run'''
    # net_path = f'./model/graph_trs_{graph_type}_{args_graph.num_nodes}.pth'
    # train_graph_trs_net_in_a_single_graph(args=args_graph, graph_list=graph_list, net_path=net_path, gpu_id=gpu_id)
    solve_single_graph_problem_using_trs(graph_list, args_graph, args_policy, gpu_id=gpu_id)


def run_graph_set_63(graph_type: str = 'G63', gpu_id: int = 0):
    graph_list = load_graph_list(graph_name=graph_type)

    args_graph = ConfigGraph(graph_list=graph_list, graph_type=graph_type)
    args_graph.batch_size = 2 ** 2
    args_graph.train_times = 2 ** 8

    args_policy = ConfigPolicy(graph_list=graph_list, graph_type=graph_type)

    '''run'''
    # net_path = f'./model/graph_trs_{graph_type}_{args_graph.num_nodes}.pth'
    # train_graph_trs_net_in_a_single_graph(args=args_graph, graph_list=graph_list, net_path=net_path, gpu_id=gpu_id)
    solve_single_graph_problem_using_trs(graph_list, args_graph, args_policy, gpu_id=gpu_id)


def run_graph_set_70(graph_type: str = 'G70', gpu_id: int = 0):
    graph_list = load_graph_list(graph_name=graph_type)

    args_graph = ConfigGraph(graph_list=graph_list, graph_type=graph_type)
    args_graph.batch_size = 2 ** 1
    args_graph.train_times = 2 ** 8

    args_policy = ConfigPolicy(graph_list=graph_list, graph_type=graph_type)

    '''run'''
    # net_path = f'./model/graph_trs_{graph_type}_{args_graph.num_nodes}.pth'
    # train_graph_trs_net_in_a_single_graph(args=args_graph, graph_list=graph_list, net_path=net_path, gpu_id=gpu_id)
    solve_single_graph_problem_using_trs(graph_list, args_graph, args_policy, gpu_id=gpu_id)


if __name__ == '__main__':
    run_graph_set_14_15()
    # run_graph_set_22_23()
    # run_graph_set_55_58()
    # run_graph_set_63()
    # run_graph_set_70()
