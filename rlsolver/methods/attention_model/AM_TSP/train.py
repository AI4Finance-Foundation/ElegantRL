# train.py

import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from models import TSPSolver
from dataset import create_distributed_data_loaders
from trainer import DistributedTSPTrainer
import config as args

# Global performance settings
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass
torch.backends.cudnn.benchmark = True


def setup_ddp(rank, world_size):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """Clean up distributed training."""
    dist.destroy_process_group()


def main_worker(rank, world_size):
    """Main worker function for each GPU."""
    # Setup distributed training
    setup_ddp(rank, world_size)
    
    # Set random seeds (different seed per rank for diversity)
    torch.manual_seed(args.SEED + rank)
    torch.cuda.manual_seed(args.SEED + rank)
    
    # Print configuration from rank 0
    if rank == 0:
        print("Configuration:")
        for key, value in vars(args).items():
            print(f"  {key}: {value}")
        print(f"  world_size: {world_size}")
        print()
    
    # Create data loaders with distributed sampling
    train_loader, test_loader, eval_loader, test_dataset = create_distributed_data_loaders(args, rank, world_size)
    
    # Create model
    model = TSPSolver(
        embedding_size=args.EMBEDDING_SIZE,
        hidden_size=args.HIDDEN_SIZE,
        seq_len=args.SEQ_LEN,
        n_head=args.N_HEAD,
        C=args.C
    ).cuda(rank)
    
    # ============ BEST PRACTICE TORCH.COMPILE ============
    # Apply torch.compile AFTER moving to CUDA, BEFORE DDP wrapping
    # Use reduce-overhead mode for training with fixed batch sizes
    model = torch.compile(
        model, 
        mode="reduce-overhead",
        fullgraph=False  # Allow graph breaks for better stability with pointer network
    )
    # =====================================================
    
    # Wrap model with DDP - all parameters are used, so find_unused_parameters=False
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    
    # Print model information from rank 0
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        print(f"Compilation: torch.compile enabled with reduce-overhead mode")
        print()
    
    # Create trainer
    trainer = DistributedTSPTrainer(model, args, rank, world_size)
    
    # Train model
    trainer.train(train_loader, eval_loader, test_dataset)
    
    if rank == 0:
        print("\nTraining completed!")
    
    cleanup()


def main():
    """Main entry point."""
    world_size = torch.cuda.device_count()
    
    if world_size == 0:
        raise RuntimeError("No GPUs available for training")
    
    print(f"Starting distributed training on {world_size} GPUs")
    mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)


if __name__ == '__main__':
    main()