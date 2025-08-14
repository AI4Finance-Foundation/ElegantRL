# launch.py
import torch.multiprocessing as mp
import os
import torch
from train_ddp import main_worker

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

WORLD_SIZE = torch.cuda.device_count()

if __name__ == "__main__":
    if WORLD_SIZE < 1:
        raise RuntimeError("No CUDA devices available")
    
    print(f"Starting DDP training with {WORLD_SIZE} GPUs")
    mp.spawn(main_worker, nprocs=WORLD_SIZE, args=(WORLD_SIZE,))