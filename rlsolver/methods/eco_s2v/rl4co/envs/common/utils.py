import abc
from typing import Callable

import torch
from tensordict.tensordict import TensorDict
from torch.distributions import Exponential, Normal, Poisson, Uniform

from rlsolver.methods.eco_s2v.rl4co.envs.common.distribution_utils import (
    Cluster,
    Gaussian_Mixture,
    Mix_Distribution,
    Mix_Multi_Distributions,
    Mixed,
)


class Generator(metaclass=abc.ABCMeta):
    """Base data generator class, to be called with `env.generator(batch_size)`"""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, batch_size) -> TensorDict:
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        return self._generate(batch_size)

    @abc.abstractmethod
    def _generate(self, batch_size, **kwargs) -> TensorDict:
        raise NotImplementedError


def get_sampler(
        val_name: str,
        distribution: int | float | str | type | Callable,
        low: float = 0,
        high: float = 1.0,
        **kwargs,
):
    """Get the sampler for the variable with the given distribution.
    If kwargs are passed, they will be parsed e.g. with `val_name` + `_dist_arg` (e.g. `loc_std` for Normal distribution).

    Args:
        val_name: Name of the variable
        distribution: int/float value (as constant distribution), or string with the distribution name (supporting
            uniform, normal, exponential, and poisson) or PyTorch Distribution type or a callable function that
            returns a PyTorch Distribution
        low: Minimum value for the variable, used for Uniform distribution
        high: Maximum value for the variable, used for Uniform distribution
        kwargs: Additional arguments for the distribution

    Example:
        ```python
        sampler_uniform = get_sampler("loc", "uniform", 0, 1)
        sampler_normal = get_sampler("loc", "normal", loc_mean=0.5, loc_std=.2)
        ```
    """
    if isinstance(distribution, (int, float)):
        return Uniform(low=distribution, high=distribution)
    elif distribution == Uniform or distribution == "uniform":
        return Uniform(low=low, high=high)
    elif distribution == Normal or distribution == "normal" or distribution == "gaussian":
        assert (
                kwargs.get(val_name + "_mean", None) is not None
        ), "mean is required for Normal distribution"
        assert (
                kwargs.get(val_name + "_std", None) is not None
        ), "std is required for Normal distribution"
        return Normal(loc=kwargs[val_name + "_mean"], scale=kwargs[val_name + "_std"])
    elif distribution == Exponential or distribution == "exponential":
        assert (
                kwargs.get(val_name + "_rate", None) is not None
        ), "rate is required for Exponential/Poisson distribution"
        return Exponential(rate=kwargs[val_name + "_rate"])
    elif distribution == Poisson or distribution == "poisson":
        assert (
                kwargs.get(val_name + "_rate", None) is not None
        ), "rate is required for Exponential/Poisson distribution"
        return Poisson(rate=kwargs[val_name + "_rate"])
    elif distribution == "center":
        return Uniform(low=(high - low) / 2, high=(high - low) / 2)
    elif distribution == "corner":
        return Uniform(
            low=low, high=low
        )  # todo: should be also `low, high` and any other corner
    elif isinstance(distribution, Callable):
        return distribution(**kwargs)
    elif distribution == "gaussian_mixture":
        return Gaussian_Mixture(num_modes=kwargs["num_modes"], cdist=kwargs["cdist"])
    elif distribution == "cluster":
        return Cluster(kwargs["n_cluster"])
    elif distribution == "mixed":
        return Mixed(kwargs["n_cluster_mix"])
    elif distribution == "mix_distribution":
        return Mix_Distribution(kwargs["n_cluster"], kwargs["n_cluster_mix"])
    elif distribution == "mix_multi_distributions":
        return Mix_Multi_Distributions()
    else:
        raise ValueError(f"Invalid distribution type of {distribution}")


def batch_to_scalar(param):
    """Return first element if in batch. Used for batched parameters that are the same for all elements in the batch."""
    if len(param.shape) > 0:
        return param[0].item()
    if isinstance(param, torch.Tensor):
        return param.item()
    return param
