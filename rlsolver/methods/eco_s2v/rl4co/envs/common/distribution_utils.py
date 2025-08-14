import random

import torch


class Cluster:
    """
    Multiple gaussian distributed clusters, as in the Solomon benchmark dataset
    Following the setting in Bi et al. 2022 (https://arxiv.org/abs/2210.07686)

    Args:
        n_cluster: Number of the gaussian distributed clusters
    """

    def __init__(self, n_cluster: int = 3):
        super().__init__()
        self.lower, self.upper = 0.2, 0.8
        self.std = 0.07
        self.n_cluster = n_cluster

    def sample(self, size):

        batch_size, num_loc, _ = size

        # Generate the centers of the clusters
        center = self.lower + (self.upper - self.lower) * torch.rand(
            batch_size, self.n_cluster * 2
        )

        # Pre-define the coordinates
        coords = torch.zeros(batch_size, num_loc, 2)

        # Calculate the size of each cluster
        cluster_sizes = [num_loc // self.n_cluster] * self.n_cluster
        for i in range(num_loc % self.n_cluster):
            cluster_sizes[i] += 1

        # Generate the coordinates
        current_index = 0
        for i in range(self.n_cluster):
            means = center[:, i * 2: (i + 1) * 2]
            stds = torch.full((batch_size, 2), self.std)
            points = torch.normal(
                means.unsqueeze(1).expand(-1, cluster_sizes[i], -1),
                stds.unsqueeze(1).expand(-1, cluster_sizes[i], -1),
            )
            coords[:, current_index: current_index + cluster_sizes[i], :] = points
            current_index += cluster_sizes[i]

        # Confine the coordinates to range [0, 1]
        coords.clamp_(0, 1)

        return coords


class Mixed:
    """
    50% nodes sampled from uniform distribution, 50% nodes sampled from gaussian distribution, as in the Solomon benchmark dataset
    Following the setting in Bi et al. 2022 (https://arxiv.org/abs/2210.07686)

    Args:
        n_cluster_mix: Number of the gaussian distributed clusters
    """

    def __init__(self, n_cluster_mix=1):
        super().__init__()
        self.lower, self.upper = 0.2, 0.8
        self.std = 0.07
        self.n_cluster_mix = n_cluster_mix

    def sample(self, size):
        batch_size, num_loc, _ = size

        # Generate the centers of the clusters
        center = self.lower + (self.upper - self.lower) * torch.rand(
            batch_size, self.n_cluster_mix * 2
        )

        # Pre-define the coordinates sampled under uniform distribution
        coords = torch.FloatTensor(batch_size, num_loc, 2).uniform_(0, 1)

        # Sample mutated index (default setting: 50% mutation)
        mutate_idx = torch.stack(
            [torch.randperm(num_loc)[: num_loc // 2] for _ in range(batch_size)]
        )

        # Generate the coordinates
        segment_size = num_loc // (2 * self.n_cluster_mix)
        remaining_indices = num_loc // 2 - segment_size * (self.n_cluster_mix - 1)
        sizes = [segment_size] * (self.n_cluster_mix - 1) + [remaining_indices]
        for i in range(self.n_cluster_mix):
            indices = mutate_idx[:, sum(sizes[:i]): sum(sizes[: i + 1])]
            means_x = center[:, 2 * i].unsqueeze(1).expand(-1, sizes[i])
            means_y = center[:, 2 * i + 1].unsqueeze(1).expand(-1, sizes[i])
            coords.scatter_(
                1,
                indices.unsqueeze(-1).expand(-1, -1, 2),
                torch.stack(
                    [
                        torch.normal(means_x.expand(-1, sizes[i]), self.std),
                        torch.normal(means_y.expand(-1, sizes[i]), self.std),
                    ],
                    dim=2,
                ),
            )

        # Confine the coordinates to range [0, 1]
        coords.clamp_(0, 1)

        return coords


class Gaussian_Mixture:
    """
    Following Zhou et al. (2023): https://arxiv.org/abs/2305.19587

    Args:
        num_modes: the number of clusters/modes in the Gaussian Mixture.
        cdist: scale of the uniform distribution for center generation.
    """

    def __init__(self, num_modes: int = 0, cdist: int = 0):
        super().__init__()
        self.num_modes = num_modes
        self.cdist = cdist

    def sample(self, size):

        batch_size, num_loc, _ = size

        if self.num_modes == 0:  # (0, 0) - uniform
            return torch.rand((batch_size, num_loc, 2))
        elif self.num_modes == 1 and self.cdist == 1:  # (1, 1) - gaussian
            return self.generate_gaussian(batch_size, num_loc)
        else:
            res = [self.generate_gaussian_mixture(num_loc) for _ in range(batch_size)]
            return torch.stack(res)

    def generate_gaussian_mixture(self, num_loc):
        """Following the setting in Zhang et al. 2022 (https://arxiv.org/abs/2204.03236)"""

        # Randomly decide how many points each mode gets
        nums = torch.multinomial(
            input=torch.ones(self.num_modes) / self.num_modes,
            num_samples=num_loc,
            replacement=True,
        )

        # Prepare to collect points
        coords = torch.empty((0, 2))

        # Generate points for each mode
        for i in range(self.num_modes):
            num = (nums == i).sum()  # Number of points in this mode
            if num > 0:
                center = torch.rand((1, 2)) * self.cdist
                cov = torch.eye(2)  # Covariance matrix
                nxy = torch.distributions.MultivariateNormal(
                    center.squeeze(), covariance_matrix=cov
                ).sample((num,))
                coords = torch.cat((coords, nxy), dim=0)

        return self._global_min_max_scaling(coords)

    def generate_gaussian(self, batch_size, num_loc):
        """Following the setting in Xin et al. 2022 (https://openreview.net/pdf?id=nJuzV-izmPJ)"""

        # Mean and random covariances
        mean = torch.full((batch_size, num_loc, 2), 0.5)
        covs = torch.rand(batch_size)  # Random covariances between 0 and 1

        # Generate the coordinates
        coords = torch.zeros((batch_size, num_loc, 2))
        for i in range(batch_size):
            # Construct covariance matrix for each sample
            cov_matrix = torch.tensor([[1.0, covs[i]], [covs[i], 1.0]])
            m = torch.distributions.MultivariateNormal(
                mean[i], covariance_matrix=cov_matrix
            )
            coords[i] = m.sample()

        # Shuffle the coordinates
        indices = torch.randperm(coords.size(0))
        coords = coords[indices]

        return self._batch_normalize_and_center(coords)

    def _global_min_max_scaling(self, coords):

        # Scale the points to [0, 1] using min-max scaling
        coords_min = coords.min(0, keepdim=True).values
        coords_max = coords.max(0, keepdim=True).values
        coords = (coords - coords_min) / (coords_max - coords_min)

        return coords

    def _batch_normalize_and_center(self, coords):
        # Step 1: Compute min and max along each batch
        coords_min = coords.min(dim=1, keepdim=True).values
        coords_max = coords.max(dim=1, keepdim=True).values

        # Step 2: Normalize coordinates to range [0, 1]
        coords = (
                coords - coords_min
        )  # Broadcasting subtracts min value on each coordinate
        range_max = (
            (coords_max - coords_min).max(dim=-1, keepdim=True).values
        )  # The maximum range among both coordinates
        coords = coords / range_max  # Divide by the max range to normalize

        # Step 3: Center the batch in the middle of the [0, 1] range
        coords = (
                coords + (1 - coords.max(dim=1, keepdim=True).values) / 2
        )  # Centering the batch

        return coords


class Mix_Distribution:
    """
    Mixture of three exemplar distributions in batch-level, i.e. Uniform, Cluster, Mixed
    Following the setting in Bi et al. 2022 (https://arxiv.org/abs/2210.07686)

    Args:
        n_cluster: Number of the gaussian distributed clusters in Cluster distribution
        n_cluster_mix: Number of the gaussian distributed clusters in Mixed distribution
    """

    def __init__(self, n_cluster=3, n_cluster_mix=1):
        super().__init__()
        self.lower, self.upper = 0.2, 0.8
        self.std = 0.07
        self.Mixed = Mixed(n_cluster_mix=n_cluster_mix)
        self.Cluster = Cluster(n_cluster=n_cluster)

    def sample(self, size):

        batch_size, num_loc, _ = size

        # Pre-define the coordinates sampled under uniform distribution
        coords = torch.FloatTensor(batch_size, num_loc, 2).uniform_(0, 1)

        # Random sample probability for the distribution of each sample
        p = torch.rand(batch_size)

        # Mixed
        mask = p <= 0.33
        n_mixed = mask.sum().item()
        if n_mixed > 0:
            coords[mask] = self.Mixed.sample((n_mixed, num_loc, 2))

        # Cluster
        mask = (p > 0.33) & (p <= 0.66)
        n_cluster = mask.sum().item()
        if n_cluster > 0:
            coords[mask] = self.Cluster.sample((n_cluster, num_loc, 2))

        # The remaining ones are uniformly distributed
        return coords


class Mix_Multi_Distributions:
    """
    Mixture of 11 Gaussian-like distributions in batch-level
    Following the setting in Zhou et al. (2023): https://arxiv.org/abs/2305.19587
    """

    def __init__(self):
        super().__init__()
        self.dist_set = [(0, 0), (1, 1)] + [
            (m, c) for m in [3, 5, 7] for c in [10, 30, 50]
        ]

    def sample(self, size):
        batch_size, num_loc, _ = size
        coords = torch.zeros(batch_size, num_loc, 2)

        # Pre-select distributions for the entire batch
        dists = [random.choice(self.dist_set) for _ in range(batch_size)]
        unique_dists = list(
            set(dists)
        )  # Unique distributions to minimize re-instantiation

        # Instantiate Gaussian_Mixture only once per unique distribution
        gm_instances = {dist: Gaussian_Mixture(*dist) for dist in unique_dists}

        # Batch process where possible
        for i, dist in enumerate(dists):
            coords[i] = gm_instances[dist].sample((1, num_loc, 2)).squeeze(0)

        return coords
