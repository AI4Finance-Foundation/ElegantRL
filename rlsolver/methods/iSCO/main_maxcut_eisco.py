import sys
import os

cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../../rlsolver')
sys.path.append(os.path.dirname(rlsolver_path))

from absl import app
from rlsolver.envs.env_eisco_maxcut import iSCO
from rlsolver.methods.iSCO.config.config_maxcut import *
from rlsolver.methods.iSCO.util import maxcut_util
import torch
import time
import tqdm
from rlsolver.methods.util_result import write_graph_result
import torch.nn.functional as F


# The results are written in this directory: 'rlsolver/result/maxcut_iSCO'
def main(_):
    params_dict = maxcut_util.load_data(DATAPATH)
    sampler = iSCO(params_dict)
    sample = sampler.random_gen_init_sample()
    pad_rows = ((sampler.max_num_nodes + 7) // 8 * 8) - sampler.max_num_nodes
    sample = F.pad(sample, (0, pad_rows), mode='constant', value=0)
    mu = torch.ones(BATCH_SIZE, device=DEVICE, dtype=torch.float) * 10
    start_time = time.time()
    best_energy = torch.tensor(0, device=DEVICE, dtype=torch.float)
    best_sample = torch.zeros(sampler.max_num_nodes, device=DEVICE, dtype=torch.float16)
    for step in tqdm.tqdm(range(0, sampler.chain_length)):
        path_length = torch.clamp(torch.poisson(mu), min=1, max=sampler.max_num_nodes).long()
        temperature = sampler.init_temperature - step / sampler.chain_length * (
                sampler.init_temperature - sampler.final_temperature)
        sample, new_energy, acc = sampler.step(sample, path_length, temperature)
        mu = torch.clamp((mu + 0.01 * (acc - 0.574)), min=1.0, max=float(sampler.max_num_nodes))
        best_energy, best_sample = maxcut_util.record(best_energy, best_sample, new_energy, sample)
    obj = 0.5 * sampler.edge_from.shape[0] + best_energy
    end_time = time.time()
    obj = obj.item()

    result = best_sample.to(torch.int)
    running_duration = end_time - start_time
    print(running_duration)
    alg_name = "iSCO"
    write_graph_result(obj, running_duration, params_dict["num_nodes"], alg_name, result, DATAPATH)


if __name__ == '__main__':
    app.run(main)
