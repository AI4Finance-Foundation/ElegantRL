import os
import sys

cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../../rlsolver')
sys.path.append(os.path.dirname(rlsolver_path))

import numpy as np
import torch as th
import sys
import os

from config import ConfigGraph, ConfigPolicy
from transformer import get_seq_graph, valid_net
from transformer import TrsCell, convert_solution_to_prob, sub_set_sampling, get_advantages
from network import GraphTRS
from rlsolver.methods.util_read_data import GraphList, load_graph_list, update_xs_by_vs, pick_xs_by_vs
from rlsolver.methods.util import gpu_info_str
from graph_embedding_pretrain import train_graph_net_in_graph_distribution


def solve_graph_dist_problem_using_trs(
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

    graph_lists = [load_graph_list(graph_name=f"{graph_type}_{num_nodes}_ID{graph_id}", if_force_exist=True)
                   for graph_id in range(30)]
    # 加载测试集的30个实例，用于训练中打印出策略模型学习曲线

    '''simulator'''
    from graph_max_cut_simulator import SimulatorGraphMaxCut
    sim = SimulatorGraphMaxCut(graph_list=graph_list, device=device, if_bidirectional=True)
    if_max = sim.if_maximize

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
    # 创建第一阶段的自动编码器，用于将邻接矩阵编码为特征嵌入序列
    graph_net_path = f'./model/graph_trs_{graph_type}_{num_nodes}.pth'
    policy_net_path = f'./model/policy_trs_{graph_type}_{num_nodes}.pth'
    print(f"graph_embedding_net_path {graph_net_path}  exists? {os.path.exists(graph_net_path)}", flush=True)
    if not os.path.exists(graph_net_path):
        train_graph_net_in_graph_distribution(args=args_graph, net_path=graph_net_path)
    # 使用监督学习标签（输入邻接矩阵，拟合邻接矩阵）训练自动编码器
    # 警告：目前已经开发了不需要第一阶段的方法：在不读取邻接矩阵的情况下，直接用随机产生的嵌入特征表达节点序列。
    #     嵌入特征只在优化 critic objective 的阶段被更新。嵌入特征在 actor objective 的阶段不被更新。

    graph_embed_net.load_state_dict(th.load(graph_net_path, map_location=device, weights_only=True))
    del graph_net_path

    """solve"""
    '''config'''
    assert isinstance(args_policy, ConfigPolicy)
    weight_decay = args_policy.weight_decay
    learning_rate = args_policy.learning_rate
    show_gap = args_policy.show_gap
    num_sims = args_policy.num_sims  # 尽量占满显存，越大越好，甚至可以用累计梯度的方式牺牲训练速度、代码可读性，换来更大并行，得到更好得分
    num_iters = args_policy.num_iters
    reset_gap = args_policy.reset_gap

    num_searchers = args_policy.num_searches
    num_layers = args_policy.num_layers
    seq_len = args_policy.seq_len  # 搜索的问题越难，需要的轨迹长度越长。训练后期的轨迹长度可以更短，可惜调节代码会降低可读性
    top_k = args_policy.top_k  # 并行数量越大，能用来探索的节点数量就越多。优先探索不确定性排名靠前的top_k个节点
    num_repeats = args_policy.num_repeats  # 探索的节点越多，需要的每个并行对应的副本数量也越多。

    # 以下参数同PPO算法
    repeat_times = args_policy.repeat_times
    clip_ratio = args_policy.clip_ratio
    lambda_entropy = args_policy.lambda_entropy

    num_instances = 30  # GraphDist in validation set

    criterion = th.nn.MSELoss()  # 想要长期训练不崩溃，还是得用 MSE，不能用SmoothL1 或者 L1

    # if os.name == 'nt':
    #     print("| Warning: checking mode.", flush=True)
    #     num_sims = 2 ** 6
    #     show_gap = 1

    '''build policy_trs_net'''
    embed_dim = args_graph.embed_dim
    num_heads = args_graph.num_heads
    net = TrsCell(embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers).to(device)
    net_param = net.parameters()
    net_optim = th.optim.Adam(net_param, lr=learning_rate) if weight_decay \
        else th.optim.AdamW(net_param, lr=learning_rate, weight_decay=weight_decay)

    seq_graphs = []
    evaluators = []
    for graph_id, graph_list in enumerate(graph_lists):
        seq_graph, sim, evaluator, best_xs, best_vs = get_seq_graph(
            graph_list=graph_list,
            args_graph=args_graph,
            args_policy=args_policy,
            graph_embed_net=graph_embed_net,
            device=device
        )

        seq_graphs.append(seq_graph.cpu())
        evaluators.append(evaluator)
    # 提前计算好测试集的30个实例的嵌入特征，避免在训练过程中重复计算

    seq_graph, sim, evaluator, best_xs, best_vs = get_seq_graph(
        graph_list=load_graph_list(graph_name=f"{graph_type}_{num_nodes}"),
        args_graph=args_graph,
        args_policy=args_policy,
        graph_embed_net=graph_embed_net,
        device=device
    )
    print(f"seq_graph.shape {seq_graph.shape}", flush=True)
    # graph_name=f"{graph_type}_{num_nodes}" 这里没有指定graph_id，就是随机从图的分布中采样得到新的图用于训练。

    '''loop'''
    buf_device = th.device('cpu')
    # seq_memory = seq_graph[:, 0:1, :].repeat(1, num_sims, 1)  # first seq_memory = seq_graph
    th.set_grad_enabled(False)
    for iter_i in range(num_iters):
        # evolutionary_replacement(best_xs, best_vs, low_k=2, if_max=if_max)
        # best_xs = sim.generate_xs_randomly(num_sims=num_sims).detach()
        # best_vs = sim.calculate_obj_values(xs=best_xs).detach()

        states = th.empty((seq_len + 1, num_sims, num_nodes), dtype=th.bool, device=buf_device)
        rewards = th.empty((seq_len, num_sims), dtype=th.float32, device=buf_device)
        logprobs = th.empty((seq_len, num_sims), dtype=th.float32, device=buf_device)
        # 创建PPO风格的经验回放缓存，只需要保存这三个，

        '''explore'''
        for t in range(seq_len):
            seq_prob = convert_solution_to_prob(solution_xs=best_xs)
            # 从当前最好的解开始迭代，将离散的二进制序列转化为连续的独热编码序列
            seq_prob, _seq_memory = net.forward(seq_prob=seq_prob, seq_graph=seq_graph, seq_memory=None, layer_i=0)
            # 让策略网络根据当前的解输出要采样的概率序列
            seq_prob = th.softmax(seq_prob.permute(1, 0, 2), dim=2)  # [num_sims, num_nodes, 2]
            seq_prob = seq_prob + th.randn_like(seq_prob) * 0.1
            # 主动为离散动作的概率添加探索噪声

            probs = seq_prob[:, :, 0]  # [num_sims, num_nodes]
            full_xs, _probs = sub_set_sampling(probs=probs, start_xs=best_xs, num_repeats=num_repeats, top_k=top_k)
            full_vs = None
            # 搜索不确定性最高的top_k位，得到修改后的概率序列

            for _ in range(num_searchers):
                full_xs, full_vs = sim.local_search_inplace(good_xs=full_xs, good_vs=th.empty((), ))
            # 在这些解的附近区域搜索num_searchers次

            good_xs, good_vs = pick_xs_by_vs(xs=full_xs, vs=full_vs, num_repeats=num_repeats, if_maximize=if_max)
            del full_vs
            # 将这些解附近区域num_searchers次得到的最好的解作为这一步的结果

            curr_xs = best_xs.clone()
            curr_vs = best_vs.clone()
            update_xs_by_vs(xs0=best_xs, vs0=best_vs, xs1=good_xs, vs1=good_vs, if_maximize=if_max)
            # 将搜索出来更优解的结果更新到最优解记录里

            reward = best_vs - curr_vs
            # 将初始解和最终解的差值作为奖励信号，这是稀疏奖励，只有最后一步有奖励。其余步数的奖励是0。探索轨迹的长度为 seq_len
            # 以上代码为了节省显存，没有打开梯度记录，下面的代码重新计算梯度，用于更新网络参数
            logprob = th.log(th.where(best_xs, probs, 1 - probs).clip(0.005, 0.995)).sum(dim=1)
            if buf_device == device:
                states[t] = curr_xs
                rewards[t] = reward
                logprobs[t] = logprob
            else:
                states[t] = curr_xs.to(buf_device)
                rewards[t] = reward.to(buf_device)
                logprobs[t] = logprob.to(buf_device)

            # 训练过程中，记录当前探索的进度。注意，训练过程使用从图分布中随机采样的图进行训练，不使用训练集的图。
            if_show_x = evaluator.record2(i=iter_i, vs=best_vs, xs=best_xs)
            if (t + 1) % 2 == 0:
                show_str = f"  {t:4} value {best_vs.float().mean().long():6} < {best_vs.max():6} < {evaluator.best_v:6}"
                evaluator.logging_print(show_str=show_str, if_show_x=if_show_x)

        if buf_device == device:
            states[seq_len] = best_xs
        else:
            states[seq_len] = best_xs.to(buf_device)

        """update network"""
        # PPO风格的策略函数以及优势值函数的更新方法
        th.set_grad_enabled(True)
        with th.no_grad():
            '''get advantages and reward_sums'''
            values = th.empty_like(rewards, device=buf_device)  # values.shape == (buffer_size, buffer_num)
            for seq_t in range(seq_len):
                if buf_device == device:
                    best_xs = states[seq_t]
                else:
                    best_xs = states[seq_t].to(device)

                seq_prob = convert_solution_to_prob(solution_xs=best_xs)

                seq_value, _seq_memory = net.forward(seq_prob=seq_prob, seq_graph=seq_graph, seq_memory=None, layer_i=0)
                value = net.get_value(seq_memory=_seq_memory).squeeze(-1)

                if buf_device == device:
                    values[seq_t] = value
                else:
                    values[seq_t] = value.to(buf_device)

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
            # 矢量并行风格的随机梯度下降样本抽取方法
            ids = th.randint(seq_len * num_sims, size=(num_sims,), requires_grad=False, device=buf_device)
            ids0 = th.fmod(ids, seq_len)  # ids % sample_len
            ids1 = th.div(ids, seq_len, rounding_mode='floor')  # ids // sample_len

            if buf_device == device:
                curr_xs = states[ids0, ids1]
                next_xs = states[ids0 + 1, ids1]
                logprob = logprobs[ids0, ids1]
                advantage = advantages[ids0, ids1]
                reward_sum = reward_sums[ids0, ids1]
            else:
                curr_xs = states[ids0, ids1].to(device)
                next_xs = states[ids0 + 1, ids1].to(device)
                logprob = logprobs[ids0, ids1].to(device)
                advantage = advantages[ids0, ids1].to(device)
                reward_sum = reward_sums[ids0, ids1].to(device)

            seq_prob = convert_solution_to_prob(solution_xs=curr_xs)
            seq_value, _seq_memory = net.forward(seq_prob=seq_prob, seq_graph=seq_graph, seq_memory=None, layer_i=0)
            value = net.get_value(seq_memory=_seq_memory).squeeze(-1)

            obj_critic = criterion(value, reward_sum)

            new_logprob, entropy = net.get_logprob_entropy(
                curr_xs=curr_xs, next_xs=next_xs, seq_graph=seq_graph, layer_i=0)
            obj_entropy = entropy.mean()

            ratio = (new_logprob - logprob.detach()).clip(-12, +12).exp()
            surrogate1 = advantage * ratio
            surrogate2 = advantage * ratio.clamp(1 - clip_ratio, 1 + clip_ratio)
            obj_surrogate = th.min(surrogate1, surrogate2).mean()

            obj_policy = obj_surrogate + obj_entropy * lambda_entropy
            # 这里obj_entropy使用不同符号，效果会有差别。无论PPO算法的 obj_entropy 是什么符号，按这里代码设置的方式更好。

            objective = obj_critic + obj_policy
            net_optim.zero_grad()
            objective.backward()
            # clip_grad_norm_(net_param, 1)
            net_optim.step()

            obj_critic_avg.append(obj_critic.item())
            obj_policy_avg.append(obj_policy.item())
            obj_entropy_avg.append(obj_entropy.item())
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
            print(f"\n| reset {gpu_info_str(device=device)} | up_rate {up_rate:8.5f}", flush=True)

            graph_id = th.randint(num_instances, 2 ** 24, size=(1,)).item()
            # 重新从图分布里采样新的图，用于训练。避免抽取到训练集的图。
            seq_graph, sim, evaluator, best_xs, best_vs = get_seq_graph(
                graph_list=load_graph_list(graph_name=f"{graph_type}_{num_nodes}_ID{graph_id}"),
                args_graph=args_graph,
                args_policy=args_policy,
                graph_embed_net=graph_embed_net,
                device=device
            )

            for _graph_id in range(num_instances):
                _evaluator = evaluators[_graph_id]
                _seq_graph = seq_graphs[_graph_id].to(device)

                _graph_list = graph_lists[_graph_id]
                _sim = SimulatorGraphMaxCut(graph_list=_graph_list, device=device, if_bidirectional=True)
                valid_net(sim=_sim, net=net, evaluator=_evaluator, seq_graph=_seq_graph, iter_i=iter_i,
                          num_sims=num_sims, graph_id=_graph_id,
                          seq_len=seq_len, num_repeats=num_repeats, top_k=top_k, num_searchers=num_searchers)

            valid_graph_lists_write_into_log(evaluators=evaluators, log_path=f"{policy_net_path}.txt")
            th.save(net.state_dict(), policy_net_path)

    for _graph_id in range(num_instances):
        _evaluator = evaluators[_graph_id]
        _seq_graph = seq_graphs[_graph_id].to(device)

        _graph_list = graph_lists[_graph_id]
        _sim = SimulatorGraphMaxCut(graph_list=graph_list, device=device, if_bidirectional=True)
        valid_net(sim=_sim, net=net, evaluator=_evaluator, seq_graph=_seq_graph, iter_i=-1,
                  num_sims=num_sims, graph_id=_graph_id,
                  seq_len=seq_len, num_repeats=num_repeats, top_k=top_k, num_searchers=num_searchers)

    valid_graph_lists_write_into_log(evaluators=evaluators, log_path=f"{policy_net_path}.txt")
    th.save(net.state_dict(), policy_net_path)


def valid_graph_lists_write_into_log(evaluators, log_path: str):
    print("|||")
    with open(file=log_path, mode='a') as log_file:
        for _graph_id, _evaluator in enumerate(evaluators):
            log_str = _evaluator.logging_print(show_str=f'GraphID {_graph_id}', if_show_x=True)
            log_file.write(log_str + '\n')
    print("|||")


def convert_log_txt_in_dir_to_df(model_dir='./model'):
    import re
    import pandas as pd

    df_cols = ['graph_dist', 'num_nodes', 'graph_id', 'cut_value', 'x_str']
    csv_path = "./GraphMaxCut_TRS1010.csv"

    rows = []

    log_txt_files = [file for file in os.listdir(model_dir) if file[-4:] == '.txt']
    for log_txt_file in log_txt_files:
        infos = log_txt_file[:log_txt_file.index('.')].split('_')
        graph_dist = infos[2]
        num_nodes = infos[3]

        log_txt_path = f"{model_dir}/{log_txt_file}"
        with open(log_txt_path, 'r') as f:
            texts = [line.strip() for line in f]

        for text in texts:
            # text = "|    11 26797 sec  best   10970.0000 GraphID 29  x_str: 2XNmabYjtSLNDvm3SaCsGunhBUh"
            pattern = r'\|?\s*(-?\d+)\s+(\d+)\s+(\w+)\s+(\w+)\s+(\d+\.\d+)\s+(\w+)\s+(\d+)\s+x_str:\s+(\S+)'
            match = re.match(pattern, text)
            if not match:
                continue

            rows.append({
                'graph_dist': str(graph_dist),
                'num_nodes': int(num_nodes),
                'round_id': int(match.group(1)),
                'train_sec': int(match.group(2)),
                'cut_value': int(float(match.group(5))),
                'graph_id': int(match.group(7)),
                'x_str': str(match.group(8))
            })

    new_df = pd.DataFrame(rows)
    new_df = new_df[df_cols]
    if os.path.exists(csv_path):
        old_df = pd.read_csv(csv_path)

        new_df = pd.concat((old_df, new_df)).reset_index()
        new_df['graph_name'] = (new_df['graph_dist'] + '_' + new_df['num_nodes'].astype(str)
                                + '_ID' + new_df['graph_id'].astype(str))

        group_df = new_df.loc[new_df.groupby('graph_name')['cut_value'].idxmax()]
        sort_df = group_df.sort_values(by=['graph_dist', 'num_nodes', 'graph_id'])
        new_df = sort_df[df_cols]
    new_df.to_csv(csv_path, index=False)
    print(new_df)

    new_df = new_df
    pd.set_option('display.max_rows', None)  # 显示所有行
    pd.set_option('display.max_columns', None)  # 显示所有列
    new_df = new_df.copy()
    new_df.loc[:, 'graph_name'] = new_df['graph_dist'] + '_' + new_df['num_nodes'].astype(str).str.rjust(4)
    result = new_df.groupby('graph_name')['cut_value'].mean().reset_index()
    print(result)


def run_graph_dist_num_nodes():
    sys_argv_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    gpu_id = sys_argv_id
    cwd = os.getcwd()  # "/home/shixi/yonv/L2A_Algorithm"

    # graph_name.replace('BA', 'BarabasiAlbert').replace('ER', 'ErdosRenyi').replace('PL', 'PowerLaw')
    graph_type, num_nodes = 'ER', 100
    # graph_type, num_nodes = 'ErdosRenyi', 1000
    # graph_type, num_nodes = 'BarabasiAlbert', 2000

    if num_nodes <= 500:
        graph_num_sims = 2 ** 5
        buffer_repeats = 32
        num_buffers = 10

        policy_num_sims = 2 ** 8
        num_repeats = 2 ** 5
        seq_len = 16
    elif num_nodes <= 800:
        graph_num_sims = 2 ** 4
        buffer_repeats = 8
        num_buffers = 8

        policy_num_sims = 2 ** 7
        num_repeats = 2 ** 4
        seq_len = 16
    elif num_nodes <= 1000:
        graph_num_sims = 2 ** 3
        buffer_repeats = 4
        num_buffers = 6

        policy_num_sims = 2 ** 6
        num_repeats = 2 ** 6
        seq_len = 32
        if graph_type == 'ER':
            num_repeats //= 2
    elif num_nodes <= 1200:
        graph_num_sims = 2 ** 3
        buffer_repeats = 4
        num_buffers = 4

        policy_num_sims = 2 ** 7
        num_repeats = 2 ** 7
        seq_len = 32
        if graph_type == 'ER':
            policy_num_sims //= 4
            num_repeats //= 2
    elif num_nodes <= 2000:
        graph_num_sims = 2 ** 4
        buffer_repeats = 2
        num_buffers = 2

        policy_num_sims = 2 ** 6
        num_repeats = 2 ** 6
        seq_len = 48
        if graph_type == 'ER':
            policy_num_sims //= 4
            num_repeats //= 8
    # elif num_nodes <= 3000:
    #     graph_num_sims = 2 ** 1
    #     buffer_repeats = 1
    #     num_buffers = 2
    #
    #     policy_num_sims = 2 ** 4
    #     num_repeats = 2 ** 4
    #     seq_len = 32
    else:
        return None

    args_graph = ConfigGraph(graph_type=graph_type, num_nodes=num_nodes)
    args_graph.buffer_dir = f"{cwd}/buffer"
    args_graph.batch_size = graph_num_sims
    args_graph.buffer_repeats = buffer_repeats
    args_graph.num_buffers = num_buffers

    graph_list = load_graph_list(graph_name=f"{graph_type}_{num_nodes}")
    args_policy = ConfigPolicy(graph_list=graph_list)
    args_policy.num_sims = policy_num_sims
    args_policy.num_repeats = num_repeats
    args_policy.reset_gap = 2 ** 2
    args_policy.seq_len = seq_len

    assert args_graph.embed_dim == args_policy.embed_dim
    assert args_graph.mid_dim == args_policy.mid_dim
    # solve_single_graph_problem_using_trs(graph_list, args_graph=args_graph, args_policy=args_policy, gpu_id=gpu_id)
    solve_graph_dist_problem_using_trs(graph_list, args_graph=args_graph, args_policy=args_policy, gpu_id=gpu_id)


def check_x_str():
    csv_path = "./GraphMaxCut_TRS1010.csv"
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    import pandas as pd
    from graph_max_cut_simulator import SimulatorGraphMaxCut
    from rlsolver.methods.util_evaluator import EncoderBase64
    df = pd.read_csv(csv_path)

    for row_id, row in df.iterrows():
        graph_dist = str(row['graph_dist'])
        num_nodes = int(row['num_nodes'])
        graph_id = int(row['graph_id'])

        x_str = str(row['x_str'])
        cut_value = int(row['cut_value'])

        graph_list = load_graph_list(graph_name=f'{graph_dist}_{num_nodes}_ID{graph_id}', if_force_exist=True)
        sim = SimulatorGraphMaxCut(graph_list=graph_list)
        encoder = EncoderBase64(encode_len=num_nodes)
        x = encoder.str_to_bool(x_str=x_str)
        obj_value = sim.calculate_obj_values(xs=x[None, :].to(device))[0].item()

        if cut_value != obj_value:
            print(f"| {graph_dist}_{num_nodes}_ID{graph_id:<2}  cut_value-check {cut_value:6}-{obj_value:6}")
    print(f"| Finish checking")


if __name__ == '__main__':
    run_graph_dist_num_nodes()
    # convert_log_txt_in_dir_to_df()
    # check_x_str()
