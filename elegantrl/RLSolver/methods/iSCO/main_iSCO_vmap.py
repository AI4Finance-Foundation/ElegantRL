from until import get_data
from absl import app
from envs.env_isco_maxcut import iSCO_fast_vmap
from config import *
import until
import torch
import time
import tqdm
from scipy.stats import poisson

def main(_):
    data = get_data(DATA_ROOT)
    sampler = iSCO_fast_vmap(data)
    sample = sampler.get_init_sample()
    mu = 10.0
    average_acc = 0
    average_path_length = 0
    sampler.x2y,sampler.y2x = until.parallelization(sampler.x2y,sampler.y2x)
    start_time = time.time()
    for step in tqdm.tqdm(range(0,sampler.chain_length)):
        poisson_sample = torch.poisson(torch.tensor([mu]))
        path_length = max(1, int(poisson_sample.item()))
        average_path_length += path_length
        temperature = sampler.init_temperature  - step / sampler.chain_length * (sampler.init_temperature -sampler.final_temperature)
        sample,new_energy,acc = sampler.step(path_length,temperature,sample)
        acc = acc.item()
        mu = min(sampler.max_num_nodes,max(1,(mu + 0.01*(acc - 0.574))))
        average_acc+=acc

    obj, obj_index = torch.min(new_energy, dim=0)
    obj = obj.item()
    result = sample[obj_index].squeeze()
    average_acc,average_path_length = average_acc/sampler.chain_length,average_path_length/sampler.chain_length
    end_time = time.time()
    running_duration = end_time - start_time
    until.write_result(sampler.data_directory,result,obj,running_duration,sampler.max_num_nodes)

if __name__ == '__main__':
    app.run(main)

    
