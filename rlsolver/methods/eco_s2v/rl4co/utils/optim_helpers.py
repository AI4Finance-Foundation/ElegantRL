import inspect

import torch
from torch.optim import Optimizer


def get_pytorch_lr_schedulers():
    """Get all learning rate schedulers from `torch.optim.lr_scheduler`"""
    return torch.optim.lr_scheduler.__all__


def get_pytorch_optimizers():
    """Get all optimizers from `torch.optim`"""
    optimizers = []
    for name, obj in inspect.getmembers(torch.optim):
        if inspect.isclass(obj) and issubclass(obj, Optimizer):
            optimizers.append(name)
    return optimizers


def create_optimizer(parameters, optimizer_name: str, **optimizer_kwargs) -> Optimizer:
    """Create optimizer for model. If `optimizer_name` is not found, raise ValueError."""
    if optimizer_name in get_pytorch_optimizers():
        optimizer_cls = getattr(torch.optim, optimizer_name)
        return optimizer_cls(parameters, **optimizer_kwargs)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not found.")


def create_scheduler(
        optimizer: Optimizer, scheduler_name: str, **scheduler_kwargs
) -> torch.optim.lr_scheduler.LRScheduler:
    """Create scheduler for optimizer. If `scheduler_name` is not found, raise ValueError."""
    if scheduler_name in get_pytorch_lr_schedulers():
        scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_name)
        return scheduler_cls(optimizer, **scheduler_kwargs)
    else:
        raise ValueError(f"Scheduler {scheduler_name} not found.")
