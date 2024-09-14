from absl import app
from rlsolver.envs.env_isco_maxcut import iSCO
from rlsolver.methods.iSCO.config.maxcut_config import *
from rlsolver.methods.iSCO.util import maxcut_util
import torch
import time
import tqdm
from rlsolver.methods.util_result import write_result3

# The results are written in this directory: 'rlsolver/result/maxcut_iSCO'
def main(_):
    params_dict = maxcut_util.load_data(DATAPATH)
    sampler = iSCO(params_dict)
    sample = sampler.random_gen_init_sample(params_dict)
    mu = 10.0
    start_time = time.time()
    for step in tqdm.tqdm(range(0, sampler.chain_length)):
        poisson_sample = torch.poisson(torch.tensor([mu]))
        path_length = max(1, int(poisson_sample.item()))
        temperature = sampler.init_temperature - step / sampler.chain_length * (
                    sampler.init_temperature - sampler.final_temperature)
        sample, new_energy, acc = sampler.step(sample, path_length, temperature)
        acc = acc.item()
        mu = min(float(sampler.max_num_nodes), max(1.0, (mu + 0.01 * (acc - 0.574))))

    obj, obj_index = torch.max(new_energy, dim=0)
    obj = obj.item()
    result = sample[obj_index]

    end_time = time.time()
    running_duration = end_time - start_time
    # maxcut_util.write_result(DATAPATH,result,obj,running_duration,params_dict['num_nodes'])
    alg_name = "iSCO"
    write_result3(obj, running_duration, params_dict["num_nodes"], alg_name, result, DATAPATH)

if __name__ == '__main__':
    app.run(main)
