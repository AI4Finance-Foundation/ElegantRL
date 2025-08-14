# dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


class VRPDataset(Dataset):
    """Dataset for VRP instances."""
    
    def __init__(self, num_nodes, num_samples, random_seed=111):
        """
        Args:
            num_nodes: Number of nodes in each VRP instance
            num_samples: Number of VRP instances to generate
            random_seed: Random seed for reproducibility
        """
        super().__init__()
        
        # Set random seed for reproducibility
        generator = torch.Generator()
        generator.manual_seed(random_seed)
        
        # Generate random VRP instances
        self.data = []
        for _ in range(num_samples):
            # Generate random 2D coordinates in [0, 1] x [0, 1]
            nodes = torch.rand(num_nodes, 2, generator=generator)
            self.data.append(nodes)
            
        self.size = len(self.data)
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return idx, self.data[idx]


def create_distributed_data_loaders(args, rank, world_size):
    """Create distributed train and test data loaders.
    
    Args:
        args: Arguments containing dataset parameters
        rank: Current process rank
        world_size: Total number of processes
        
    Returns:
        train_loader: Distributed DataLoader for training data
        test_loader: DataLoader for test data
        eval_loader: DataLoader for evaluation (full test set)
        test_dataset: Test dataset for heuristic comparison
    """
    # Create datasets
    train_dataset = VRPDataset(args.SEQ_LEN, args.NUM_TR_DATASET)
    test_dataset = VRPDataset(args.SEQ_LEN, args.NUM_TE_DATASET)
    
    # Create distributed sampler for training
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    # Common DataLoader kwargs for performance
    common_kwargs = {
        "pin_memory": True,
        "num_workers": args.NUM_WORKERS,
        "persistent_workers": args.NUM_WORKERS > 0,
    }
    
    # Add prefetch_factor only when num_workers > 0
    if args.NUM_WORKERS > 0:
        common_kwargs["prefetch_factor"] = 4
    
    # Train loader with drop_last=True to avoid DDP issues with uneven batches
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.BATCH_SIZE,
        sampler=train_sampler,
        drop_last=True,
        **common_kwargs
    )
    
    # Test loader without distributed sampling (for evaluation)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.BATCH_SIZE,
        shuffle=False,
        **common_kwargs
    )
    
    # Evaluation loader uses full test set
    eval_loader = DataLoader(
        test_dataset,
        batch_size=args.NUM_TE_DATASET,
        shuffle=False,
        **common_kwargs
    )
    
    return train_loader, test_loader, eval_loader, test_dataset