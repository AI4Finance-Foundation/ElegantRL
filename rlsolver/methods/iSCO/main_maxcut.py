import sys
import os
cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../../rlsolver')
sys.path.append(os.path.dirname(rlsolver_path))

from absl import app
from rlsolver.envs.env_isco_maxcut import iSCO
from rlsolver.methods.iSCO.config.maxcut_config import *
from rlsolver.methods.iSCO.util import maxcut_util
import torch
import time
import tqdm
from rlsolver.methods.util_result import write_graph_result

# The results are written in this directory: 'rlsolver/result/maxcut_iSCO'
def main(_):
    params_dict = maxcut_util.load_data(DATAPATH)
    sampler = iSCO(params_dict)
    sample = sampler.random_gen_init_sample(params_dict)
    mu = torch.ones(BATCH_SIZE,device=DEVICE,dtype=torch.float)*10
    start_time = time.time()
    energy = torch.tensor(0,device=DEVICE,dtype=torch.float)
    for step in tqdm.tqdm(range(0, sampler.chain_length)):
        path_length = 10*torch.ones(BATCH_SIZE,device=DEVICE,dtype=torch.long)
        temperature = sampler.init_temperature - step / sampler.chain_length * (
                    sampler.init_temperature - sampler.final_temperature)
        sample, new_energy, acc = sampler.step(sample, path_length, temperature)
        mu = torch.clamp((mu + 0.01 * (acc - 0.574)),min = 1.0,max = float(sampler.max_num_nodes))
        energy = torch.max(energy,torch.max(new_energy, dim=0)[0])
    obj, obj_index = torch.max(new_energy, dim=0)
    print(energy)


    obj = obj.item()
    result = sample[obj_index]

    end_time = time.time()
    running_duration = end_time - start_time
    alg_name = "iSCO"
    write_graph_result(obj, running_duration, params_dict["num_nodes"], alg_name, result, DATAPATH)

if __name__ == '__main__':
    app.run(main)
