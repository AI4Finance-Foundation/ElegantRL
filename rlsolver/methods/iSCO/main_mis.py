from absl import app
from rlsolver.envs.env_isco_mis import iSCO_local_search
from rlsolver.methods.iSCO.config.mis_config import *
from rlsolver.methods.iSCO.util import mis_util
from rlsolver.methods.iSCO.util.mis_util import load_data
import torch
import time
import tqdm
from torch.func import vmap

# The results are written in this directory: 'rlsolver/result/mis_iSCO'
def main(_):
    data = load_data(DATAPATH)
    sampler = iSCO_local_search(data)
    sample = sampler.random_gen_init_sample()
    mu = 10.0
    average_acc = 0
    average_path_length = 0
    sampler.x2y = vmap(sampler.x2y,in_dims=(0,0,0,None,None,None), randomness='different')
    sampler.y2x = vmap(sampler.y2x,in_dims=(0,0,0,0,0,0,None,None,None), randomness='different')
    start_time = time.time()
    for step in tqdm.tqdm(range(0,sampler.chain_length)):
        poisson_sample = torch.poisson(torch.tensor([mu]))
        path_length = max(1, int(poisson_sample.item()))
        average_path_length += path_length
        temperature = sampler.init_temperature  - step / sampler.chain_length * (sampler.init_temperature -sampler.final_temperature)
        sample,new_energy,acc = sampler.step(path_length,temperature,sample)
        acc = acc.item()
        mu = min(sampler.max_num_nodes,max(1.0,(mu + 0.01*(acc - 0.574))))
        average_acc+=acc

    obj, obj_index = torch.min(new_energy, dim=0)
    obj = obj.item()
    result = sample[obj_index].squeeze()
    end_time = time.time()
    running_duration = end_time - start_time
    mis_util.write_result(DATAPATH, result, obj, running_duration, data['num_nodes'])

if __name__ == '__main__':
    app.run(main)
