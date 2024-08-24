import os
import sys
import tqdm
import torch as th
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from config import ConfigGraph, GraphList
from network import GraphTRS, create_mask
from evaluator import TrainingLogger
from graph_utils import build_adjacency_bool, load_graph_list
from graph_utils import generate_graph_list, get_hot_image_of_graph, get_adjacency_distance_matrix


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
        inp = th.load(train_inp_path, map_location=th.device('cpu'))
        load_size = min(inp.shape[1], buf_size)
        train_inp[:, :load_size, :] = inp[:, :load_size, :]
    else:
        print(f"| get_adj_bool_ary() FileNotExist {train_inp_path}")
        load_size = 0
    generate_size = buf_size - load_size
    if generate_size > 0:
        inp = generate_adjacency_seq(num_sims=generate_size, if_tqdm=True,
                                     graph_type=graph_type, num_nodes=num_nodes)
        train_inp[:, load_size:, :] = inp
    if buf_size > load_size:
        th.save(train_inp, train_inp_path)
    return train_inp


def get_objective(adj_bools, net, mask, criterion, if_train):
    if if_train:
        net.train()
        _mask = mask
    else:
        net.eval()
        _mask = None

    '''get lab1 lab2 lab_eyes'''
    th.set_grad_enabled(False)
    # num_nodes, batch_size = adj_bools.shape[0:2]
    # device = adj_bools.device
    # lab_eyes = th.eye(num_nodes, device=device)[:, None, :].repeat((1, batch_size, 1))

    inp = adj_bools.float()
    lab1 = th.empty_like(inp)
    lab2 = th.empty_like(inp)
    lab3 = th.empty_like(inp)
    for i in range(lab1.shape[1]):
        adj_bool = adj_bools[:, i, :]
        lab1[:, i, :] = get_hot_image_of_graph(adj_bool=adj_bool, hot_type='avg')  # average hot image
        lab2[:, i, :] = get_hot_image_of_graph(adj_bool=adj_bool, hot_type='sum')  # sum hot image
        lab3[:, i, :] = get_adjacency_distance_matrix(adj_bool_ary=adj_bool.eq(1).cpu().data.numpy())

    '''get objective'''
    th.set_grad_enabled(True)
    # out = net(inp.float(), mask=_mask)
    # out1, out2 = th.chunk(out, chunks=2, dim=2)
    # objective = criterion(out1, lab1.detach()) + criterion(out2, lab2.detach())

    enc1, dec_matrix, dec_node = net.encoder_trs(inp, mask)
    out_hot_image = net.decoder_trs(enc1, dec_matrix, dec_node)
    out1, out2 = th.chunk(out_hot_image, chunks=2, dim=2)
    out3 = net.get_node_classify(dec_node)

    objective = criterion(out1, lab1) + criterion(out2, lab2) + criterion(out3, lab3)
    return objective


'''run'''


def train_graph_trs_net_in_a_single_graph(graph_list: GraphList, args: ConfigGraph, net_path: str, if_adj_matrix=True,
                                          train_times=2 ** 11, batch_size=2 ** 4, show_gap=2 ** 5):
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    '''config'''
    num_nodes = args.num_nodes
    inp_dim = args.inp_dim
    out_dim = args.out_dim
    mid_dim = args.mid_dim
    embed_dim = args.embed_dim
    num_heads = args.num_heads
    num_layers = args.num_layers
    weight_decay = args.weight_decay
    learning_rate = args.learning_rate

    '''graph'''
    adj_bool = build_adjacency_bool(graph_list=graph_list, num_nodes=num_nodes, if_bidirectional=True).to(device)
    label_hot_image_avg = get_hot_image_of_graph(adj_bool=adj_bool, hot_type='avg')  # average hot image
    label_hot_image_sum = get_hot_image_of_graph(adj_bool=adj_bool, hot_type='sum')  # sum hot image

    '''build (inp, lab1, lab2)'''
    inp = adj_bool[:, None, :].repeat(1, batch_size, 1).float()
    lab1 = th.zeros((num_nodes, batch_size, num_nodes), dtype=th.float32, device=device).detach()
    lab2 = th.zeros((num_nodes, batch_size, num_nodes), dtype=th.float32, device=device).detach()
    lab_eyes = th.eye(num_nodes, device=device)[:, None, :].repeat(1, batch_size, 1)

    '''序号为0的固定为原始图graph_list，在训练中对原始图graph_list随机添加边edge，生成相近的图graph 作为其他序号的值'''
    inp[:, 0, :] = adj_bool
    lab1[:, 0, :] = label_hot_image_avg
    lab2[:, 0, :] = label_hot_image_sum

    '''model'''
    net = GraphTRS(inp_dim=inp_dim, mid_dim=mid_dim, out_dim=out_dim,
                   embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers).to(device)

    net_params = list(net.parameters())
    optimizer = th.optim.Adam(net_params, lr=learning_rate, maximize=False) if weight_decay \
        else th.optim.AdamW(net_params, lr=learning_rate, maximize=False, weight_decay=weight_decay)

    criterion = nn.MSELoss()

    '''loop'''
    mask = create_mask(seq_len=num_nodes, mask_type='eye').to(device)

    recorder = TrainingLogger()
    for repeat_id in range(train_times):
        '''序号为0的固定为原始图graph_list，在训练中对原始图graph_list随机添加边edge，生成相近的图graph 作为其他序号的值'''
        rd_ij_list = th.randint(num_nodes, size=(batch_size, 2)).data.numpy()
        _adj_bool = adj_bool.clone()
        for k in range(1, batch_size):
            rd_i, rd_j = rd_ij_list[k]
            _adj_bool[rd_i, rd_j] = _adj_bool[rd_j, rd_i] = not _adj_bool[rd_i, rd_j]

            if if_adj_matrix:
                label_hot_image_avg = _adj_bool
                label_hot_image_sum = _adj_bool
            else:
                label_hot_image_avg = get_hot_image_of_graph(adj_bool=_adj_bool, hot_type='avg')  # average hot image
                label_hot_image_sum = get_hot_image_of_graph(adj_bool=_adj_bool, hot_type='sum')  # sum hot image

            inp[:, k, :] = _adj_bool
            lab1[:, k, :] = label_hot_image_avg
            lab2[:, k, :] = label_hot_image_sum

        output_tensor, tgt = net.forward(inp0=inp.detach(), mask=mask)
        out1, out2 = th.chunk(output_tensor, chunks=2, dim=2)
        out3 = net.get_node_classify(tgt=tgt)

        objective = (criterion(out1, lab1.detach()) +
                     criterion(out2, lab2.detach()) +
                     criterion(out3, lab_eyes.detach()))

        optimizer.zero_grad()
        objective.backward()
        clip_grad_norm_(net_params, 3)
        optimizer.step()

        if repeat_id % show_gap == 0:
            recorder.add_and_print(repeat_id=repeat_id, buffer_id=0, objective_item=objective.item())

    os.makedirs(os.path.dirname(net_path), exist_ok=True)
    csv_path = f"{net_path}.csv"
    recorder.save_as_csv(csv_path=csv_path)
    recorder.plot_training_recorder(csv_path=csv_path, ignore_n=16)
    th.save(net.state_dict(), net_path)
    print(f"| save net in {net_path}")


def check_train_graph_trs_net_in_a_single_graph():
    graph_type, num_nodes, graph_id = 'PowerLaw', 100, 0
    graph_list = load_graph_list(f"{graph_type}_{num_nodes}_ID{graph_id}")
    net_path = f'./model/graph_trs_{graph_type}_{num_nodes}_ID{graph_id}.pth'
    args = ConfigGraph(graph_list=graph_list, graph_type=graph_type, num_nodes=num_nodes)

    train_graph_trs_net_in_a_single_graph(graph_list=graph_list, args=args, net_path=net_path,
                                          train_times=2 ** 6, batch_size=2 ** 4, show_gap=2 ** 5)


def train_graph_trs_net_in_graph_distribution(args: ConfigGraph, net_path: str):
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

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
    num_sims = args.num_sims
    num_repeats = args.buffer_repeats
    num_buffers = args.num_buffers
    weight_decay = args.weight_decay
    learning_rate = args.learning_rate

    show_gap = args.show_gap

    if num_nodes >= 1000:
        num_sims = int(2 ** 3 * 1.5)
    if num_nodes >= 2000:
        num_sims = int(2 ** 1 * 1.5)

    if os.name == 'nt':  # debug model (nt: NewType ~= WindowsOS)
        buffer_size = 2 ** 4
        num_sims = 2 ** 3
        num_buffers = 1

    '''model'''
    net = GraphTRS(inp_dim=inp_dim, mid_dim=mid_dim, out_dim=out_dim,
                   embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers).to(device)
    th.save(net.state_dict(), net_path)

    net_params = list(net.parameters())
    optimizer = th.optim.Adam(net_params, lr=learning_rate, maximize=False) if weight_decay \
        else th.optim.AdamW(net_params, lr=learning_rate, maximize=False, weight_decay=weight_decay)

    criterion = nn.MSELoss()

    '''loop'''
    mask = create_mask(seq_len=num_nodes, mask_type='eye').to(device)
    num_epochs = buffer_size // num_sims

    recorder = TrainingLogger()
    for repeat_id in range(num_repeats):
        for buffer_id in range(num_buffers):
            with th.no_grad():
                dir_path = f"{buffer_dir}/buffer_{graph_type}_Node{num_nodes}"
                os.makedirs(dir_path, exist_ok=True)
                adj_bool_ary_path = f"{dir_path}/buffer_{buffer_id:02}.pth"
                adj_bool_ary = get_adj_bool_ary(adj_bool_ary_path, buffer_size, graph_type, num_nodes)

            rand_ids = th.randperm(buffer_size)
            for j in range(num_epochs):
                j0 = j * num_sims
                j1 = j0 + num_sims
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
        print(f"| save net in {net_path}")

    recorder.save_as_csv(csv_path=f"{buffer_dir}/{net_path}.csv")


def check_train_graph_trs_net_in_graph_distribution():
    for task_id in range(0, 8):
        print(task_id)
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
        train_graph_trs_net_in_graph_distribution(args=args, net_path=net_path)

    recorder = TrainingLogger()
    recorder.plot_training_recorders(data_dir='./recorder', graph_type_id=1)


if __name__ == '__main__':
    check_train_graph_trs_net_in_a_single_graph()
    check_train_graph_trs_net_in_graph_distribution()
