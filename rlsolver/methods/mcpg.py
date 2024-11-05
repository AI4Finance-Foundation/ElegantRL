import sys
import os
cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../rlsolver')
sys.path.append(os.path.dirname(rlsolver_path))

from typing import List, Tuple
import os
import torch
import sys
sys.path.append('..')
from torch_geometric.data import Data
from rlsolver.methods.L2A.evaluator import EncoderBase64
from rlsolver.methods.L2A.maxcut_simulator import SimulatorMaxcut, load_graph_list
from rlsolver.methods.util import calc_txt_files_with_prefixes
import time
from rlsolver.methods.L2A.maxcut_local_search import SolverLocalSearch
from rlsolver.methods.util_read_data import (read_nxgraph, read_graphlist
                            )
from rlsolver.methods.util_result import write_graph_result
"""
pip install torch_geometric
"""
from config import (GPU_ID, 
                    calc_device,
                    DATA_FILENAME,
                    DIRECTORY_DATA,
                    PREFIXES)

class Config:
    show_gap = 2 ** 4

    max_epoch_num = 30
    sample_epoch_num = 8
    repeat_times = 128

    num_ls = 8
    reset_epoch_num = 128
    total_mcmc_num = 512

    # num_ls = 6
    # reset_epoch_num = 192
    # total_mcmc_num = 224
    # path = 'data/gset_22.txt'

    # num_ls = 8
    # reset_epoch_num = 128
    # total_mcmc_num = 256
    # path = 'data/gset_55.txt'

    # num_ls = 8
    # reset_epoch_num = 256
    # total_mcmc_num = 192
    # path = 'data/gset_70.txt'

    # num_ls = 8
    # reset_epoch_num = 256
    # repeat_times = 512
    # total_mcmc_num = 2048
    # path = 'data/gset_22.txt'  # GPU RAM 40GB

    # num_ls = 8
    # reset_epoch_num = 192
    # repeat_times = 448
    # total_mcmc_num = 1024
    # path = 'data/gset_55.txt'  # GPU RAM 40GB

    # num_ls = 8
    # reset_epoch_num = 320
    # repeat_times = 288
    # total_mcmc_num = 768
    # path = 'data/gset_70.txt'  # GPU RAM 40GB




def metro_sampling(probs, start_status, max_transfer_time, device=None):
    # Metropolis-Hastings sampling
    torch.set_grad_enabled(False)
    if device is None:
        device = calc_device(GPU_ID)

    num_node = len(probs)
    num_chain = start_status.shape[1]
    index_col = torch.tensor(list(range(num_chain)), device=device)

    samples = start_status.bool().to(device)
    probs = probs.detach().to(device)

    count = 0
    for t in range(max_transfer_time * 5):
        if count >= num_chain * max_transfer_time:
            break

        index_row = torch.randint(low=0, high=num_node, size=[num_chain], device=device)
        chosen_probs_base = probs[index_row]
        chosen_value = samples[index_row, index_col]
        chosen_probs = torch.where(chosen_value, chosen_probs_base, 1 - chosen_probs_base)
        accept_rate = (1 - chosen_probs) / chosen_probs

        is_accept = torch.rand(num_chain, device=device).lt(accept_rate)
        samples[index_row, index_col] = torch.where(is_accept, ~chosen_value, chosen_value)

        count += is_accept.sum()
    torch.set_grad_enabled(True)
    return samples.float().to(device)


def sampler_func(data, xs_sample,
                 num_ls, total_mcmc_num, repeat_times,
                 device=calc_device(GPU_ID)):
    torch.set_grad_enabled(False)
    k = 1 / 4

    # num_nodes = data.num_nodes
    num_edges = data.num_edges
    n0s_tensor = data.edge_index[0]
    n1s_tensor = data.edge_index[1]

    xs_loc_sample = xs_sample.clone()
    xs_loc_sample *= 2  # map (0, 1) to (-0.5, 1.5)
    xs_loc_sample -= 0.5  # map (0, 1) to (-0.5, 1.5)

    # local search
    for cnt in range(num_ls):
        for node0_id in data.sorted_degree_nodes:
            node1_ids = data.neighbors[node0_id]

            node_rand_v = (xs_loc_sample[node1_ids].sum(dim=0) +
                           torch.rand(total_mcmc_num * repeat_times, device=device) * k)
            xs_loc_sample[node0_id] = node_rand_v.lt((data.weighted_degree[node0_id] + k) / 2).long()
    # pass
    # vs1 = simulator.calculate_obj_values(xs_sample.t().bool())
    # vs2 = simulator.calculate_obj_values(xs_loc_sample.t().bool())
    # pass
    expected_cut = torch.empty(total_mcmc_num * repeat_times, dtype=torch.float32, device=device)
    for j in range(repeat_times):
        j0 = total_mcmc_num * j
        j1 = total_mcmc_num * (j + 1)

        nlr_probs = 2 * xs_loc_sample[n0s_tensor.type(torch.long), j0:j1] - 1
        nlc_probs = 2 * xs_loc_sample[n1s_tensor.type(torch.long), j0:j1] - 1
        expected_cut[j0:j1] = (nlr_probs * nlc_probs).sum(dim=0)

    expected_cut_reshape = expected_cut.reshape((-1, total_mcmc_num))
    index = torch.argmin(expected_cut_reshape, dim=0)
    index = torch.arange(total_mcmc_num, device=device) + index * total_mcmc_num
    max_cut = expected_cut[index]
    vs_good = (num_edges - max_cut) / 2

    xs_good = xs_loc_sample[:, index]
    value = expected_cut.float()
    value -= value.mean()
    torch.set_grad_enabled(True)
    return vs_good, xs_good, value


class Simpler(torch.nn.Module):
    def __init__(self, output_num):
        super().__init__()
        self.lin = torch.nn.Linear(1, output_num)
        self.sigmoid = torch.nn.Sigmoid()

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, device=calc_device(GPU_ID)):
        x = torch.ones(1).to(device)
        x = self.lin(x)
        x = self.sigmoid(x)

        x = (x - 0.5) * 0.6 + 0.5
        return x


def maxcut_dataloader(path, device=calc_device(GPU_ID)):
    with open(path) as f:
        fline = f.readline()
        fline = fline.split()
        num_nodes, num_edges = int(fline[0]), int(fline[1])
        edge_index = torch.LongTensor(2, num_edges)
        cnt = 0
        while True:
            lines = f.readlines(num_edges * 2)
            if not lines:
                break
            for line in lines:
                line = line.rstrip('\n').split()
                edge_index[0][cnt] = int(line[0]) - 1
                edge_index[1][cnt] = int(line[1]) - 1
                cnt += 1

        data = Data(num_nodes=num_nodes, edge_index=edge_index.to(device))
        data = append_neighbors(data)

        data.single_degree = []
        data.weighted_degree = []
        tensor_abs_weighted_degree = []
        for i0 in range(data.num_nodes):
            data.single_degree.append(len(data.neighbors[i0]))
            data.weighted_degree.append(float(torch.sum(data.neighbor_edges[i0])))
            tensor_abs_weighted_degree.append(float(torch.sum(torch.abs(data.neighbor_edges[i0]))))
        tensor_abs_weighted_degree = torch.tensor(tensor_abs_weighted_degree)
        data.sorted_degree_nodes = torch.argsort(tensor_abs_weighted_degree, descending=True)

        edge_degree = []
        add = torch.zeros(3, num_edges).to(device)
        for i0 in range(num_edges):
            edge_degree.append(tensor_abs_weighted_degree[edge_index[0][i0]] + tensor_abs_weighted_degree[edge_index[1][i0]])
            node_r = edge_index[0][i0]
            node_c = edge_index[1][i0]
            add[0][i0] = 1 - data.weighted_degree[node_r] / 2 - 0.05
            add[1][i0] = 1 - data.weighted_degree[node_c] / 2 - 0.05
            add[2][i0] = 1 + 0.05

        for i0 in range(num_nodes):
            data.neighbor_edges[i0] = data.neighbor_edges[i0].unsqueeze(0)
        data.add_items = add
        edge_degree = torch.tensor(edge_degree)
        data.sorted_degree_edges = torch.argsort(edge_degree, descending=True)
        return data, num_nodes


def append_neighbors(data, device=calc_device(GPU_ID)):
    data.neighbors = []
    data.neighbor_edges = []
    # num_nodes = data.encode_len
    num_nodes = data.num_nodes
    for i in range(num_nodes):
        data.neighbors.append([])
        data.neighbor_edges.append([])
    edge_number = data.edge_index.shape[1]

    edge_weight = 1
    for index in range(0, edge_number):
        row = data.edge_index[0][index]
        col = data.edge_index[1][index]

        data.neighbors[row].append(col.item())
        data.neighbor_edges[row].append(edge_weight)
        data.neighbors[col].append(row.item())
        data.neighbor_edges[col].append(edge_weight)

    data.n0 = []
    data.n1 = []
    data.n0_edges = []
    data.n1_edges = []
    for index in range(0, edge_number):
        row = data.edge_index[0][index]
        col = data.edge_index[1][index]
        data.n0.append(data.neighbors[row].copy())
        data.n1.append(data.neighbors[col].copy())
        data.n0_edges.append(data.neighbor_edges[row].copy())
        data.n1_edges.append(data.neighbor_edges[col].copy())
        i = 0
        for i in range(len(data.n0[index])):
            if data.n0[index][i] == col:
                break
        data.n0[index].pop(i)
        data.n0_edges[index].pop(i)
        for i in range(len(data.n1[index])):
            if data.n1[index][i] == row:
                break
        data.n1[index].pop(i)
        data.n1_edges[index].pop(i)

        data.n0[index] = torch.LongTensor(data.n0[index]).to(device)
        data.n1[index] = torch.LongTensor(data.n1[index]).to(device)
        data.n0_edges[index] = torch.tensor(
            data.n0_edges[index]).unsqueeze(0).to(device)
        data.n1_edges[index] = torch.tensor(
            data.n1_edges[index]).unsqueeze(0).to(device)

    for i in range(num_nodes):
        data.neighbors[i] = torch.LongTensor(data.neighbors[i]).to(device)
        data.neighbor_edges[i] = torch.tensor(data.neighbor_edges[i]).to(device)

    return data


def get_return(probs, samples, value, total_mcmc_num, repeat_times):
    log_prob_sum = torch.empty_like(value)
    for j in range(repeat_times):
        j0 = total_mcmc_num * j
        j1 = total_mcmc_num * (j + 1)

        _samples = samples[j0:j1]
        log_prob = (_samples * probs + (1 - _samples) * (1 - probs)).log()
        log_prob_sum[j0:j1] = log_prob.sum(dim=1)
    objective = (log_prob_sum * value.detach()).mean()
    return objective


def save_graph_list_to_txt(graph_list, txt_path: str):
    num_nodes = max([max(n0, n1) for n0, n1, distance in graph_list]) + 1
    num_edges = len(graph_list)

    lines = [f"{num_nodes} {num_edges}", ]
    lines.extend([f"{n0 + 1} {n1 + 1} {distance}" for n0, n1, distance in graph_list])
    lines = [l + '\n' for l in lines]
    with open(txt_path, 'w') as file:
        file.writelines(lines)


def print_gpu_memory(device):
    if not torch.cuda.is_available():
        return

    total_memory = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)  # GB
    max_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)  # GB
    memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)  # GB

    print(f"AllRAM {total_memory:.2f} GB, "
          f"MaxRAM {max_allocated:.2f} GB, "
          f"NowRAM {memory_allocated:.2f} GB, "
          f"Rate {(max_allocated / total_memory) * 100:.2f}%")


def mcpg(filename: str):
    print(f"filename: {filename}")

    '''init'''
    sim_name = filename  # os.path.splitext(os.path.basename(path))[0]
    data, num_nodes = maxcut_dataloader(filename)
    device = calc_device(GPU_ID)

    change_times = int(num_nodes / 10)  # transition times for metropolis sampling

    net = Simpler(num_nodes)
    net.to(device).reset_parameters()
    optimizer = torch.optim.Adam(net.parameters(), lr=8e-2)

    '''addition'''
    from rlsolver.envs.env_mcpg_maxcut import SimulatorMaxcut
    from rlsolver.envs.LocalSearch import LocalSearch
    # from graph_max_cut_simulator import SimulatorGraphMaxCut
    # from graph_max_cut_local_search import SolverLocalSearch
    sim = SimulatorMaxcut(sim_name=sim_name, device=device)
    local_search = LocalSearch(simulator=sim, num_nodes=num_nodes)

    xs = sim.generate_xs_randomly(num_sims=Config.total_mcmc_num)
    local_search.reset(xs.bool())
    for _ in range(16):
        local_search.random_search(num_iters=Config.repeat_times // 16)
    now_max_info = local_search.good_xs.t()
    now_max_res = local_search.good_vs
    del sim
    del local_search

    '''loop'''
    net.train()
    xs_prob = (torch.zeros(num_nodes) + 0.5).to(device)
    xs_bool = now_max_info.repeat(1, Config.repeat_times)

    print('start loop')
    sys.stdout.flush()  # add for slurm stdout
    objs_epochs = []
    xs_epochs = []
    for epoch in range(1, Config.max_epoch_num + 1):
        net.to(device).reset_parameters()
        objs_each_epoch = []
        xs_each_epoch = []
        for j1 in range(Config.reset_epoch_num // Config.sample_epoch_num):
            start_time = time.time()
            xs_sample = metro_sampling(xs_prob, xs_bool.clone(), change_times)

            temp_max, temp_max_info, value = sampler_func(
                data, xs_sample, Config.num_ls, Config.total_mcmc_num, Config.repeat_times, device)
            # update now_max
            for i0 in range(Config.total_mcmc_num):
                if temp_max[i0] > now_max_res[i0]:
                    now_max_res[i0] = temp_max[i0]
                    now_max_info[:, i0] = temp_max_info[:, i0]

            # update if min is too small
            now_max = max(now_max_res).item()
            now_max_index = torch.argmax(now_max_res)
            now_min_index = torch.argmin(now_max_res)
            objs_each_epoch.append(now_max)

            x = now_max_info[:, now_max_index]
            xs_each_epoch.append(x)
            now_max_res[now_min_index] = now_max
            now_max_info[:, now_min_index] = now_max_info[:, now_max_index]
            temp_max_info[:, now_min_index] = now_max_info[:, now_max_index]

            # select best samples
            xs_bool = temp_max_info.clone()
            xs_bool = xs_bool.repeat(1, Config.repeat_times)
            # construct the start point for next iteration
            start_samples = xs_sample.t()

            probs = xs_prob[None, :]
            _probs = 1 - probs
            entropy = -(probs * probs.log2() + _probs * _probs.log2()).mean(dim=1)
            obj_entropy = entropy.mean()
            print(f"value {now_max: 9.2f}  entropy {obj_entropy: 9.3f}")
            # sys.stdout.flush()  # add for slurm stdout

            running_duration = time.time() - start_time
            num_samples = temp_max.shape[0]
            num_samples_per_second = num_samples / running_duration
            print("num_samples_per_second: ", num_samples_per_second)

            for _ in range(Config.sample_epoch_num):
                xs_prob = net()
                ret_loss_ls = get_return(xs_prob, start_samples, value, Config.total_mcmc_num, Config.repeat_times)

                optimizer.zero_grad()
                ret_loss_ls.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
                optimizer.step()
            torch.cuda.empty_cache()

            if j1 % Config.show_gap == 0:
                total_max = now_max_res
                best_sort = torch.argsort(now_max_res, descending=True)
                total_best_info = torch.squeeze(now_max_info[:, best_sort[0]])

                objective_value = max(total_max)
                solution = total_best_info

                encoder = EncoderBase64(encode_len=num_nodes)
                x_str = encoder.bool_to_str(x_bool=solution)

                print(f"epoch {epoch:6}  value {objective_value.item():8.2f}  {x_str}")
                print_gpu_memory(device)

            if os.path.exists('./stop'):
                break
        if os.path.exists('./stop'):
            break

        objs_epochs.extend(objs_each_epoch)
        xs_epochs.extend(xs_each_epoch)
        print()
    if os.path.exists('./stop'):
        print(f"break: os.path.exists('./stop') {os.path.exists('./stop')}")
        sys.stdout.flush()  # add for slurm stdout
    best_obj = max(objs_epochs)
    best_index = objs_epochs.index(best_obj)
    best_x = xs_epochs[best_index]
    return best_obj, best_x

def mcpg_multifiles(directory_data: str, prefixes: List[str]):
    files = calc_txt_files_with_prefixes(directory_data, prefixes)
    files.sort()
    for i in range(len(files)):
        start_time = time.time()
        filename = files[i]
        print(f'Start the {i}-th file: {filename}')
        best_obj, best_x = mcpg(filename)
        running_duration = time.time() - start_time
        alg_name = "mcpg"
        solution = best_x.tolist()
        num_nodes = len(solution)
        write_graph_result(best_obj, running_duration, num_nodes, alg_name, solution, filename)



if __name__ == '__main__':
    run_one_file = False
    if run_one_file:
        filename = DATA_FILENAME
        mcpg(filename)

    run_multifiles = True
    if run_multifiles:
        directory_data = DIRECTORY_DATA
        prefixes = PREFIXES
        mcpg_multifiles(directory_data, prefixes)





