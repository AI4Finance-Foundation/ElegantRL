import os
import torch as th
import sys


"""
pip install th_geometric
"""

GPU_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0


def metro_sampling(probs, start_status, max_transfer_time, device=None):
    # Metropolis-Hastings sampling
    th.set_grad_enabled(False)
    if device is None:
        device = th.device(f'cuda:{GPU_ID}' if th.cuda.is_available() else 'cpu')

    num_node = len(probs)
    num_chain = start_status.shape[1]
    index_col = th.tensor(list(range(num_chain)), device=device)

    samples = start_status.bool().to(device)
    probs = probs.detach().to(device)

    count = 0
    for t in range(max_transfer_time * 5):
        if count >= num_chain * max_transfer_time:
            break

        index_row = th.randint(low=0, high=num_node, size=[num_chain], device=device)
        chosen_probs_base = probs[index_row]
        chosen_value = samples[index_row, index_col]
        chosen_probs = th.where(chosen_value, chosen_probs_base, 1 - chosen_probs_base)
        accept_rate = (1 - chosen_probs) / chosen_probs

        is_accept = th.rand(num_chain, device=device).lt(accept_rate)
        samples[index_row, index_col] = th.where(is_accept, ~chosen_value, chosen_value)

        count += is_accept.sum()
    th.set_grad_enabled(True)
    return samples.float().to(device)

def obj(xs_sample, total_mcmc_num, repeat_times, data, device):
    n0s_tensor = data.edge_index[0]
    n1s_tensor = data.edge_index[1]
    xs_loc_sample = xs_sample.clone()
    expected_cut = th.empty(total_mcmc_num * repeat_times, dtype=th.float32, device=device)
    for j in range(repeat_times):
        j0 = total_mcmc_num * j
        j1 = total_mcmc_num * (j + 1)

        nlr_probs = 2 * xs_loc_sample[n0s_tensor.type(th.long), j0:j1] - 1
        nlc_probs = 2 * xs_loc_sample[n1s_tensor.type(th.long), j0:j1] - 1
        expected_cut[j0:j1] = (nlr_probs * nlc_probs).sum(dim=0)
    return expected_cut

def pick_good_xs(data, xs_sample,
                 num_ls, total_mcmc_num, repeat_times,
                 device=th.device(f'cuda:{GPU_ID}' if th.cuda.is_available() else 'cpu')):
    th.set_grad_enabled(False)
    k = 1 / 4

    # num_nodes = data.num_nodes
    num_edges = data.num_edges

    xs_loc_sample = xs_sample.clone()
    xs_loc_sample *= 2  # map (0, 1) to (-0.5, 1.5)
    xs_loc_sample -= 0.5  # map (0, 1) to (-0.5, 1.5)

    # local search
    for cnt in range(num_ls):
        for node0_id in data.sorted_degree_nodes:
            node1_ids = data.neighbors[node0_id]

            node_rand_v = (xs_loc_sample[node1_ids].sum(dim=0) +
                           th.rand(total_mcmc_num * repeat_times, device=device) * k)
            xs_loc_sample[node0_id] = node_rand_v.lt((data.weighted_degree[node0_id] + k) / 2).long()

    # expected_cut = th.empty(total_mcmc_num * repeat_times, dtype=th.float32, device=device)
    # for j in range(repeat_times):
    #     j0 = total_mcmc_num * j
    #     j1 = total_mcmc_num * (j + 1)
    #
    #     nlr_probs = 2 * xs_loc_sample[n0s_tensor.type(th.long), j0:j1] - 1
    #     nlc_probs = 2 * xs_loc_sample[n1s_tensor.type(th.long), j0:j1] - 1
    #     expected_cut[j0:j1] = (nlr_probs * nlc_probs).sum(dim=0)
    expected_cut = obj(xs_sample, total_mcmc_num, repeat_times, data, device)

    expected_cut_reshape = expected_cut.reshape((-1, total_mcmc_num))
    index = th.argmin(expected_cut_reshape, dim=0)
    index = th.arange(total_mcmc_num, device=device) + index * total_mcmc_num
    max_cut = expected_cut[index]
    vs_good = (num_edges - max_cut) / 2

    xs_good = xs_loc_sample[:, index]
    value = expected_cut.float()
    value -= value.mean()
    th.set_grad_enabled(True)
    return vs_good, xs_good, value

def get_return(probs, samples, value, total_mcmc_num, repeat_times):
    log_prob_sum = th.empty_like(value)
    for j in range(repeat_times):
        j0 = total_mcmc_num * j
        j1 = total_mcmc_num * (j + 1)

        _samples = samples[j0:j1]
        log_prob = (_samples * probs + (1 - _samples) * (1 - probs)).log()
        log_prob_sum[j0:j1] = log_prob.sum(dim=1)
    objective = (log_prob_sum * value.detach()).mean()
    return objective

class Sampler(th.nn.Module):
    def __init__(self, output_num):
        super().__init__()
        self.lin = th.nn.Linear(1, output_num)
        self.sigmoid = th.nn.Sigmoid()

    def reset(self):
        self.lin.reset_parameters()

    def forward(self, device=th.device(f'cuda:{GPU_ID}' if th.cuda.is_available() else 'cpu')):
        x = th.ones(1).to(device)
        x = self.lin(x)
        x = self.sigmoid(x)

        x = (x - 0.5) * 0.6 + 0.5
        return x
