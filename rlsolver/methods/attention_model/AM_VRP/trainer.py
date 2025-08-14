# trainer.py

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from utils import moving_average, clip_grad_norm, AverageMeter, get_heuristic_solution


class DistributedVRPTrainer:
    """Distributed trainer for TSP solver using REINFORCE algorithm."""
    
    def __init__(self, model, args, rank, world_size):
        """
        Args:
            model: DDP-wrapped TSPSolver model
            args: Training arguments
            rank: GPU rank
            world_size: Number of GPUs
        """
        self.model = model
        self.args = args
        self.rank = rank
        self.world_size = world_size
        
        # Fixed learning rate (no scaling with world size to avoid instability)
        self.optimizer = optim.Adam(model.parameters(), lr=args.LR)
        
        # ============ COMPILE OPTIMIZER FOR ADDITIONAL SPEEDUP ============
        # Compile optimizer to reduce overhead during parameter updates
        # This is especially beneficial with DDP gradient synchronization
        try:
            # Only compile if PyTorch version supports it (2.2+)
            self.optimizer = torch.compile(self.optimizer, mode="reduce-overhead")
            if rank == 0:
                print("Optimizer compilation enabled")
        except Exception as e:
            if rank == 0:
                print(f"Optimizer compilation not available: {e}")
        # ====================================================================
        
        # Mixed precision training (critical parts use FP32 internally)
        self.scaler = GradScaler()
        
        # Moving average baseline (local to each GPU)
        self.moving_avg = None
        self.beta = args.BETA
        self.baseline_sync_freq = 500
        self.step_count = 0
    
    def sync_baseline(self):
        """Synchronize baselines across GPUs."""
        if self.moving_avg is not None:
            # Average baselines across all GPUs
            baseline_sum = self.moving_avg.clone()
            dist.all_reduce(baseline_sum, op=dist.ReduceOp.SUM)
            self.moving_avg = baseline_sum / self.world_size
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch.
        
        Args:
            train_loader: Distributed DataLoader for training data
            epoch: Current epoch number
            
        Returns:
            avg_loss: Average loss for the epoch
            avg_reward: Average reward for the epoch
        """
        self.model.train()
        train_loader.sampler.set_epoch(epoch)
        
        loss_meter = AverageMeter()
        reward_meter = AverageMeter()
        
        # Progress bar only on rank 0
        iterator = tqdm(train_loader, desc=f"Epoch {epoch}") if self.rank == 0 else train_loader
        
        for batch_idx, (indices, batch) in enumerate(iterator):
            batch = batch.cuda(self.rank)
            batch_size = batch.size(0)
            
            with autocast('cuda'):
                # Forward pass (critical FP32 operations are handled inside model)
                rewards, log_probs, actions = self.model(batch)
            
            # Initialize moving average if needed
            if self.moving_avg is None:
                self.moving_avg = torch.zeros(len(train_loader.dataset), device=f'cuda:{self.rank}')
            
            # Update baseline for current batch
            self.moving_avg[indices] = moving_average(
                self.moving_avg[indices],
                rewards.detach(),
                self.beta
            )
            
            # Calculate advantage
            advantage = rewards - self.moving_avg[indices]
            
            # REINFORCE loss
            log_probs_sum = log_probs.sum(dim=1)
            log_probs_sum = log_probs_sum.clamp(min=-100)  # Prevent extreme values
            loss = (advantage * log_probs_sum).mean()
            
            # Backward pass with automatic gradient synchronization
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # Gradient clipping (essential for stability)
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm(self.model.parameters(), self.args.GRAD_CLIP)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update meters
            loss_meter.update(loss.item(), batch_size)
            reward_meter.update(rewards.mean().item(), batch_size)
            
            # Periodic baseline synchronization
            self.step_count += 1
            if self.step_count % self.baseline_sync_freq == 0:
                self.sync_baseline()
        
        return loss_meter.avg, reward_meter.avg
    
    def evaluate(self, eval_loader, heuristic_distances=None):
        """Evaluate model performance (only on rank 0).
        
        Args:
            eval_loader: DataLoader for evaluation
            heuristic_distances: Pre-computed heuristic solutions
            
        Returns:
            avg_reward: Average tour length
            gap: Gap compared to heuristic (if available)
        """
        if self.rank == 0:
            self.model.eval()
            all_rewards = []
            
            with torch.no_grad():
                for indices, batch in eval_loader:
                    batch = batch.cuda(self.rank)
                    
                    with autocast('cuda'):
                        rewards, _, _ = self.model(batch)
                    
                    all_rewards.append(rewards.cpu())
            
            all_rewards = torch.cat(all_rewards)
            avg_reward = all_rewards.mean().item()
            
            # Calculate gap if heuristic solutions available
            gap = None
            if heuristic_distances is not None:
                ratio = all_rewards / heuristic_distances
                gap = ratio.mean().item()
            
            return avg_reward, gap
        else:
            return None, None
    
    def initialize_baseline(self, train_loader):
        """Initialize moving average baseline."""
        if self.rank == 0:
            print("Initializing baseline...")
        
        self.model.eval()
        
        if self.moving_avg is None:
            self.moving_avg = torch.zeros(len(train_loader.dataset), device=f'cuda:{self.rank}')
        
        with torch.no_grad():
            for indices, batch in train_loader:
                batch = batch.cuda(self.rank)
                
                with autocast('cuda'):
                    rewards, _, _ = self.model(batch)
                
                self.moving_avg[indices] = rewards
        
        # Sync baseline after initialization
        self.sync_baseline()
    
    def train(self, train_loader, eval_loader, test_dataset=None):
        """Full training loop.
        
        Args:
            train_loader: Distributed DataLoader for training
            eval_loader: DataLoader for evaluation
            test_dataset: Test dataset for heuristic comparison
        """
        # Compute heuristic solutions on rank 0
        heuristic_distances = None
        if self.rank == 0 and test_dataset is not None:
            print("Computing heuristic solutions...")
            heuristic_distances = []
            
            for i, (_, pointset) in enumerate(tqdm(test_dataset)):
                dist_val = get_heuristic_solution(pointset)
                if dist_val is not None:
                    heuristic_distances.append(dist_val)
                else:
                    heuristic_distances = None
                    break
            
            if heuristic_distances is not None:
                heuristic_distances = torch.tensor(heuristic_distances)
        
        # Initialize baseline
        self.initialize_baseline(train_loader)
        
        # Training loop
        for epoch in range(self.args.NUM_EPOCHS):
            # Train
            avg_loss, avg_reward = self.train_epoch(train_loader, epoch)
            
            # Evaluate only on rank 0
            eval_reward, gap = self.evaluate(eval_loader, heuristic_distances)
            
            # Print results from rank 0
            if self.rank == 0:
                print(f"\n[Epoch {epoch}]")
                print(f"  Train Loss: {avg_loss:.4f}")
                print(f"  Train Reward: {avg_reward:.4f}")
                print(f"  Eval Reward: {eval_reward:.4f}")
                if gap is not None:
                    print(f"  Gap vs Heuristic: {gap:.4f}x")
        
        # Save final model only from rank 0
        if self.rank == 0:
            print("\nSaving final model to model.pth...")
            torch.save(self.model.module.state_dict(), "model.pth")
            print("Model saved successfully!")