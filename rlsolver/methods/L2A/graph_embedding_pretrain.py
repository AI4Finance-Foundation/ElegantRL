import os
import sys
import tqdm
import torch as th
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from config import ConfigGraph, GraphList
from network import GraphTRS, create_mask
from rlsolver.methods.util_evaluator import Recorder
from rlsolver.methods.util_read_data import build_adjacency_bool, load_graph_list, generate_graph_list
from rlsolver.methods.util import get_hot_image_of_graph, get_adjacency_distance_matrix

TEN = th.Tensor


def generate_adjacency_seq(num_sims, graph_type, num_nodes, if_tqdm=False):
    adjacency_seq = th.empty((num_nodes, num_sims, num_nodes), dtype=th.bool)

    i_iteration = tqdm.trange(num_sims, ascii=True) if if_tqdm else range(num_sims)
    for i in i_iteration:
        graph_list = generate_graph_list(graph_type=graph_type, num_nodes=num_nodes)
        adjacency_seq[:, i, :] = build_adjacency_bool(graph_list=graph_list, if_bidirectional=True)
    return adjacency_seq


def get_adj_bool_ary(train_inp_path, buf_size, graph_type, num_nodes):
    train_inp = th.empty((num_nodes, buf_size, num_nodes), dtype=th.bool)
    if os.path.exists(train_inp_path):
        inp = th.load(train_inp_path, map_location=th.device('cpu'), weights_only=True)
        load_size = min(inp.shape[1], buf_size)
        train_inp[:, :load_size, :] = inp[:, :load_size, :]
    else:
        print(f"| get_adj_bool_ary() FileNotExist {train_inp_path}", flush=True)
        load_size = 0
    generate_size = buf_size - load_size
    if generate_size > 0:
        inp = generate_adjacency_seq(num_sims=generate_size, if_tqdm=True,
                                     graph_type=graph_type, num_nodes=num_nodes)
        train_inp[:, load_size:, :] = inp

    if buf_size > load_size:
        th.save(train_inp, train_inp_path)
        print(f"| get_adj_bool_ary() save in {train_inp_path}", flush=True)
    return train_inp


def get_objective(adj_bools, net, mask, criterion, if_train):
    if if_train:
        net.train()
        _mask = mask
    else:
        net.eval()
        _mask = None

    '''get lab:lab1 lab2 lab3'''
    th.set_grad_enabled(False)

    adj_bools = sort_adj_bools(adj_bools=adj_bools)  # TODO
    inp0 = adj_bools.float()
    lab1 = th.empty_like(inp0)
    lab2 = th.empty_like(inp0)
    lab3 = th.empty_like(inp0)

    batch_size = adj_bools.shape[1]
    for i in range(batch_size):
        adj_bool = adj_bools[:, i, :]
        adj_ary = get_adjacency_distance_matrix(adj_bool_ary=adj_bool.eq(1).cpu().data.numpy())
        lab1[:, i, :] = get_hot_image_of_graph(adj_bool=adj_bool, hot_type='avg')  # average hot image
        lab2[:, i, :] = get_hot_image_of_graph(adj_bool=adj_bool, hot_type='sum')  # sum hot image
        lab3[:, i, :] = th.tensor(adj_ary, dtype=th.float32, device=inp0.device)
    seq_lab = th.concat((lab1, lab2, lab3), dim=2)  # [num_nodes, batch_size, num_nodes*3]

    '''get lab: eye_lab'''
    num_nodes, batch_size = adj_bools.shape[0:2]
    eye_lab = th.eye(num_nodes, device=adj_bools.device)[:, None, :].repeat((1, batch_size, 1))

    '''get objective'''
    th.set_grad_enabled(True)
    seq_out, seq_memory = net.forward(inp0, mask)
    eye_out = net.get_node_classify(tgt=seq_memory)
    objective = criterion(seq_out, seq_lab) + criterion(eye_out, eye_lab)
    return objective


def sort_adj_bools(adj_bools: TEN) -> TEN:
    adj_bools = adj_bools.transpose(1, 2)

    num_sims, num_nodes = adj_bools.shape[0:2]
    neighbor_counts = adj_bools.sum(dim=2)
    sorted_ids = th.argsort(neighbor_counts, dim=1)

    sim_ids = th.arange(num_sims, device=adj_bools.device)[:, None].repeat(1, num_nodes)
    sorted_adj_bools = adj_bools[sim_ids, sorted_ids]

    sorted_adj_bools = sorted_adj_bools.transpose(1, 2)
    return sorted_adj_bools


'''run'''


def train_graph_net_in_a_single_graph(graph_list: GraphList, args: ConfigGraph, net_path: str, gpu_id: int = 0):
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    '''config'''
    num_nodes = args.num_nodes
    inp_dim = args.inp_dim
    out_dim = args.out_dim
    mid_dim = args.mid_dim
    embed_dim = args.embed_dim
    num_heads = args.num_heads
    num_layers = args.num_layers

    show_gap = args.show_gap
    batch_size = args.batch_size
    train_times = args.train_times
    weight_decay = args.weight_decay
    learning_rate = args.learning_rate

    '''graph'''
    adj_bool = build_adjacency_bool(graph_list=graph_list, num_nodes=num_nodes, if_bidirectional=True).to(device)

    '''model'''
    net = GraphTRS(inp_dim=inp_dim, mid_dim=mid_dim, out_dim=out_dim,
                   embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers).to(device)

    net_params = list(net.parameters())
    optimizer = th.optim.Adam(net_params, lr=learning_rate, maximize=False) if weight_decay \
        else th.optim.AdamW(net_params, lr=learning_rate, maximize=False, weight_decay=weight_decay)

    criterion = nn.MSELoss()

    '''loop'''
    print(f"| {args.graph_type}  Nodes {num_nodes}  Edges {len(graph_list)}", flush=True)
    mask = create_mask(seq_len=num_nodes, mask_type='eye').to(device)

    recorder = Recorder()
    for repeat_id in range(train_times):
        '''序号为0的固定为原始图graph_list，在训练中对原始图graph_list随机添加边edge，生成相近的图graph 作为其他序号的值'''
        adj_bools = th.zeros((num_nodes, batch_size, num_nodes), dtype=th.float32, device=device)

        _adj_bool = adj_bool.clone()
        adj_bools[:, 0, :] = _adj_bool

        rd_size = 8
        rd_ij_list = th.randint(num_nodes, size=(rd_size, 2)).data.numpy()
        for k in range(th.randint(rd_size, size=(1,)).item()):
            rd_i, rd_j = rd_ij_list[k]
            _adj_bool[rd_i, rd_j] = _adj_bool[rd_j, rd_i] = not _adj_bool[rd_i, rd_j]

        rd_ij_list = th.randint(num_nodes, size=(batch_size, 2)).data.numpy()
        for k in range(1, batch_size):
            rd_i, rd_j = rd_ij_list[k]
            _adj_bool[rd_i, rd_j] = _adj_bool[rd_j, rd_i] = not _adj_bool[rd_i, rd_j]
            adj_bools[:, k, :] = _adj_bool

        objective = get_objective(adj_bools, net, mask, criterion, if_train=True)

        optimizer.zero_grad()
        objective.backward()
        clip_grad_norm_(net_params, 3)
        optimizer.step()

        if repeat_id % show_gap == 0:
            recorder.add_and_print(repeat_id=repeat_id, buffer_id=0, objective_item=objective.item())

    os.makedirs(os.path.dirname(net_path), exist_ok=True)
    csv_path = f"{net_path}.csv"
    recorder.save_as_csv(csv_path=csv_path)
    recorder.plot_training_recorder(csv_path=csv_path, ignore_n=4)
    th.save(net.state_dict(), net_path)
    print(f"| save net in {net_path}", flush=True)


def check_train_graph_trs_net_in_a_single_graph():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    graph_type, num_nodes, graph_id = 'PowerLaw', 100, 0
    graph_list = load_graph_list(f"{graph_type}_{num_nodes}_ID{graph_id}")
    net_path = f'./model/graph_trs_{graph_type}_{num_nodes}_ID{graph_id}.pth'
    args = ConfigGraph(graph_list=graph_list, graph_type=graph_type, num_nodes=num_nodes)

    args.train_times = 2 ** 6
    args.show_gap = 2 ** 5

    train_graph_net_in_a_single_graph(graph_list=graph_list, args=args, net_path=net_path, gpu_id=gpu_id)


def train_graph_net_in_graph_distribution(args: ConfigGraph, net_path: str, gpu_id: int = 0):
    print(f"| train_graph_trs_net_in_graph_distribution {args.graph_type}  {args.num_nodes}", flush=True)
    # gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    os.makedirs(os.path.dirname(net_path), exist_ok=True)

    args = args if args else ConfigGraph(graph_type='PowerLaw', num_nodes=100)

    '''config: graph'''
    graph_type = args.graph_type
    num_nodes = args.num_nodes

    '''config: model'''
    inp_dim = args.inp_dim
    out_dim = args.out_dim
    mid_dim = args.mid_dim
    embed_dim = args.embed_dim
    num_heads = args.num_heads
    num_layers = args.num_layers

    '''config: train'''
    buffer_size = args.buffer_size
    buffer_dir = args.buffer_dir
    batch_size = args.batch_size
    num_repeats = args.buffer_repeats
    num_buffers = args.num_buffers
    weight_decay = args.weight_decay
    learning_rate = args.learning_rate

    show_gap = args.show_gap

    # if os.name == 'nt':  # debug model (nt: NewType ~= WindowsOS)
    #     buffer_size = 2 ** 4
    #     batch_size = 2 ** 3
    #     num_buffers = 1

    '''model'''
    net = GraphTRS(inp_dim=inp_dim, mid_dim=mid_dim, out_dim=out_dim,
                   embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers).to(device)
    th.save(net.state_dict(), net_path)

    net_params = list(net.parameters())
    optimizer = th.optim.Adam(net_params, lr=learning_rate, maximize=False) if weight_decay \
        else th.optim.AdamW(net_params, lr=learning_rate, maximize=False, weight_decay=weight_decay)

    criterion = nn.MSELoss()

    '''loop'''
    print(f"| train_graph_net_in_graph_distribution num_repeats {num_repeats}  num_buffers {num_buffers}", flush=True)

    mask = create_mask(seq_len=num_nodes, mask_type='eye').to(device)
    num_epochs = buffer_size // batch_size

    recorder = Recorder()
    for repeat_id in range(num_repeats):
        for buffer_id in range(num_buffers):
            with th.no_grad():
                dir_path = f"{buffer_dir}/buffer_{graph_type}_Node{num_nodes}"
                os.makedirs(dir_path, exist_ok=True)
                adj_bool_ary_path = f"{dir_path}/buffer_{buffer_id:02}.pth"
                adj_bool_ary = get_adj_bool_ary(adj_bool_ary_path, buffer_size, graph_type, num_nodes)

            rand_ids = th.randperm(buffer_size)
            for j in range(num_epochs):
                j0 = j * batch_size
                j1 = j0 + batch_size
                adj_bools = adj_bool_ary[:, rand_ids[j0:j1], :].to(device)

                '''valid'''
                if j % show_gap == 0:
                    objective = get_objective(adj_bools, net, mask, criterion, if_train=False)
                    recorder.add_and_print(repeat_id=repeat_id, buffer_id=buffer_id, objective_item=objective.item())

                '''train'''
                objective = get_objective(adj_bools, net, mask, criterion, if_train=True)

                optimizer.zero_grad()
                objective.backward()
                clip_grad_norm_(net_params, 3)
                optimizer.step()

            th.save(net.state_dict(), net_path)
            print(f"| save net in {net_path}", flush=True)

    recorder.save_as_csv(csv_path=f"{buffer_dir}/{net_path}.csv")


def check_train_graph_trs_net_in_graph_distribution():
    for task_id in range(0, 8):
        print(task_id, flush=True)
        num_nodes, graph_type, num_buffer, num_sims = [
            [100, 'ErdosRenyi', 5, 2 ** 6], [100, 'BarabasiAlbert', 6, 2 ** 6], [100, 'PowerLaw', 7, 2 ** 6],
            [200, 'ErdosRenyi', 5, 2 ** 6], [200, 'BarabasiAlbert', 6, 2 ** 6], [200, 'PowerLaw', 7, 2 ** 6],
            [300, 'ErdosRenyi', 4, 2 ** 5], [300, 'BarabasiAlbert', 5, 2 ** 5], [300, 'PowerLaw', 6, 2 ** 5],
            [400, 'ErdosRenyi', 3, 2 ** 5], [400, 'BarabasiAlbert', 4, 2 ** 5], [400, 'PowerLaw', 5, 2 ** 5],
            [500, 'ErdosRenyi', 3, 2 ** 5], [500, 'BarabasiAlbert', 4, 2 ** 5], [500, 'PowerLaw', 5, 2 ** 5],
            [600, 'ErdosRenyi', 3, 2 ** 5], [600, 'BarabasiAlbert', 4, 2 ** 5], [600, 'PowerLaw', 5, 2 ** 5],
            [1000, 'ErdosRenyi', 6, 2 ** 3], [1000, 'BarabasiAlbert', 10, 2 ** 3], [1000, 'PowerLaw', 12, 2 ** 3],
            [2000, 'ErdosRenyi', 9, 2 ** 2], [2000, 'BarabasiAlbert', 15, 2 ** 2], [2000, 'PowerLaw', 18, 2 ** 2],
        ][task_id]

        args = ConfigGraph(graph_type=graph_type, num_nodes=num_nodes)
        setattr(args, 'num_buffers', num_buffer)
        setattr(args, 'num_sims', num_sims)
        net_path = f'./model/graph_trs_{graph_type}_{num_nodes}.pth'
        train_graph_net_in_graph_distribution(args=args, net_path=net_path)

    recorder = Recorder()
    recorder.plot_training_recorders(data_dir='./recorder', graph_type_id=1)


if __name__ == '__main__':
    check_train_graph_trs_net_in_a_single_graph()
    # check_train_graph_trs_net_in_graph_distribution()
