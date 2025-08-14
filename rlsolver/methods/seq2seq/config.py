import torch

NUM_NODES = 10000
DATA_PATH = 'data/test_data'
GPU_ID = 7
BATCH_SIZE = 2**5
def calc_device(gpu_id: int):
    return torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() and gpu_id >= 0 else 'cpu')
DEVICE = calc_device(GPU_ID)

