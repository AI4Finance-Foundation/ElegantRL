from absl import app
from rlsolver.envs.env_isco_tsp import iSCO
from rlsolver.methods.iSCO.config.tsp_config import *
from rlsolver.methods.iSCO.util import tsp_util
import torch
import time
import tqdm

#The results are written in this directory: 'rlsolver/result/tsp_iSCO'
def main(_):
    params_dict = tsp_util.load_data(DATA_PATH,SOL_PATH )
    sampler = iSCO(params_dict)
    sample = sampler.random_gen_init_sample(params_dict)
    mu = 10.0
    start_time = time.time()
    for step in tqdm.tqdm(range(0,sampler.chain_length)):
        poisson_sample = torch.poisson(torch.tensor([mu]))
        path_length = int(min(float(99),max(1.0, int(poisson_sample.item()))))
        temperature = sampler.init_temperature  - step / sampler.chain_length * (sampler.init_temperature -sampler.final_temperature)
        sample,acc = sampler.step(sample,path_length,temperature)
        acc = acc.item()
        mu = min(float(params_dict['num_nodes']),max(1.0,(mu + 0.01*(acc - 0.574))))

    distance = 0
    for k in range(sample.shape[0]):
        distance += params_dict['distance'][sample[k],sample[k+1-sample.shape[0]]]

    zero_index = torch.nonzero(sample == 0, as_tuple=True)[0].item()
    result = torch.cat((sample[zero_index:zero_index+1], torch.cat((sample[zero_index+1:], sample[:zero_index]))))
    end_time = time.time()
    running_duration = end_time - start_time
    tsp_util.write_result(DATA_PATH,result,distance,running_duration,params_dict['num_nodes'])



if __name__ == '__main__':
    app.run(main)