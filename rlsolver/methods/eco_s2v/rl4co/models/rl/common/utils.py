import torch


class RewardScaler:
    """This class calculates the running mean and variance of a stepwise observed
    quantity, like the RL reward / advantage using the Welford online algorithm.
    The mean and variance are either used to standardize the input (scale='norm') or
    to scale it (scale='scale').

    Args:
        scale: None | 'scale' | 'mean': specifies how to transform the input; defaults to None
    """

    def __init__(self, scale: str = None):
        self.scale = scale
        self.count = 0
        self.mean = 0
        self.M2 = 0

    def __call__(self, scores: torch.Tensor):
        if self.scale is None:
            return scores
        elif isinstance(self.scale, int):
            return scores / self.scale
        # Score scaling
        self.update(scores)
        tensor_to_kwargs = dict(dtype=scores.dtype, device=scores.device)
        std = (self.M2 / (self.count - 1)).float().sqrt()
        score_scaling_factor = std.to(**tensor_to_kwargs) + torch.finfo(scores.dtype).eps
        if self.scale == "norm":
            scores = (scores - self.mean.to(**tensor_to_kwargs)) / score_scaling_factor
        elif self.scale == "scale":
            scores /= score_scaling_factor
        else:
            raise ValueError("unknown scaling operation requested: %s" % self.scale)
        return scores

    @torch.no_grad()
    def update(self, batch: torch.Tensor):
        batch = batch.reshape(-1)
        self.count += len(batch)

        # newvalues - oldMean
        delta = batch - self.mean
        self.mean += (delta / self.count).sum()
        # newvalues - newMeant
        delta2 = batch - self.mean
        self.M2 += (delta * delta2).sum()
