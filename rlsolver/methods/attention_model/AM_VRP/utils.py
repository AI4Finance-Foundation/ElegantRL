"""Utility functions for VRP solver."""

import numpy as np
import torch

try:
    import elkai
    ELKAI_AVAILABLE = True
except ImportError:
    ELKAI_AVAILABLE = False


def get_heuristic_solution(pointset, scale=100000.0):
    """Get heuristic solution using elkai (LKH algorithm).
    
    Args:
        pointset: Tensor or numpy array of shape [num_nodes, 2]
        scale: Scaling factor for coordinates
        
    Returns:
        tour_length: Length of the heuristic tour
    """
    if not ELKAI_AVAILABLE:
        return None
        
    # Convert to numpy if needed
    if isinstance(pointset, (torch.Tensor, torch.cuda.FloatTensor)):
        pointset = pointset.detach().cpu().numpy()
    
    num_points = len(pointset)
    
    # Create distance matrix (elkai requires integers)
    dist_matrix = np.zeros((num_points, num_points), dtype=np.int32)
    for i in range(num_points):
        for j in range(i + 1, num_points):
            dist = int(np.linalg.norm(pointset[i] - pointset[j]) * scale)
            dist_matrix[i, j] = dist_matrix[j, i] = dist
    
    # Solve using elkai
    tour = elkai.solve_int_matrix(dist_matrix)
    
    # Calculate tour length
    tour_length = 0.0
    for i in range(num_points):
        tour_length += dist_matrix[tour[i], tour[(i + 1) % num_points]]
    
    return tour_length / scale


def compute_tour_length(nodes, tour):
    """Compute length of a tour.
    
    Args:
        nodes: Tensor of shape [batch_size, num_nodes, 2] or [num_nodes, 2]
        tour: Tensor of shape [batch_size, num_nodes] or [num_nodes]
        
    Returns:
        lengths: Tensor of shape [batch_size] or scalar
    """
    if nodes.dim() == 2:
        # Single instance
        nodes = nodes.unsqueeze(0)
        tour = tour.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    # Gather nodes in tour order
    batch_size = nodes.size(0)
    ordered_nodes = nodes.gather(1, tour.unsqueeze(2).expand(-1, -1, 2))
    
    # Calculate distances between consecutive nodes
    distances = torch.norm(ordered_nodes[:, 1:] - ordered_nodes[:, :-1], dim=2)
    
    # Add distance from last node back to first
    distances_to_first = torch.norm(ordered_nodes[:, -1] - ordered_nodes[:, 0], dim=1)
    
    # Total tour length
    lengths = distances.sum(dim=1) + distances_to_first
    
    if squeeze_output:
        lengths = lengths.squeeze(0)
        
    return lengths


def moving_average(old_avg, new_value, beta=0.9):
    """Update moving average.
    
    Args:
        old_avg: Previous average value
        new_value: New value to incorporate
        beta: Smoothing factor (0 < beta < 1)
        
    Returns:
        new_avg: Updated average
    """
    return old_avg * beta + new_value * (1.0 - beta)


def clip_grad_norm(parameters, max_norm):
    """Clip gradients by norm.
    
    Args:
        parameters: Model parameters
        max_norm: Maximum gradient norm
        
    Returns:
        total_norm: Total norm of gradients
    """
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm)


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count