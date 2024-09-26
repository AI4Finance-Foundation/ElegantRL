import torch as th

INIT_TEMPERATURE = 1.0
FINAL_TEMPERATURE = 0.1
CHAIN_LENGTH = 10000
BATCH_SIZE = 1
GPU_ID = 0
K = 20
DATAPATH = '../../../rlsolver/data/tsplib/berlin52.tsp'
GPU_ID = 0


def calc_device(gpu_id: int):
    return th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')


DEVICE = calc_device(GPU_ID)
