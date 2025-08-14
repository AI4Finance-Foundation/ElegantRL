import numpy as np
import torch
import torch.nn as nn


def pos_init_embedding(pos_name: str, config: dict) -> nn.Module:
    """Get positional embedding. The positional embedding is used for improvement methods to encode current solutions.

    Args:
        pos_name: Positional embeding method name.
        config: A dictionary of configuration options for the initlization.
    """
    embedding_registry = {
        "APE": AbsolutePositionalEmbedding,
        "CPE": CyclicPositionalEmbedding,
    }

    if pos_name not in embedding_registry:
        raise ValueError(
            f"Unknown positional embedding name '{pos_name}'. Available positional embeddings: {embedding_registry.keys()}"
        )

    return embedding_registry[pos_name](**config)


class AbsolutePositionalEmbedding(nn.Module):
    """Absolute Positional Embedding in the original Transformer."""

    def __init__(self, embed_dim):
        super(AbsolutePositionalEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.pattern = None

    def _init(self, n_position, emb_dim):
        pattern = torch.tensor(
            [
                [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
                for pos in range(1, n_position + 1)
            ],
            dtype=torch.float32,
        )

        pattern[1:, 0::2] = torch.sin(pattern[1:, 0::2])  # dim 2i
        pattern[1:, 1::2] = torch.cos(pattern[1:, 1::2])  # dim 2i+1

        return pattern

    def forward(self, td):
        batch_size, seq_length = td["rec_current"].size()
        visited_time = td["visited_time"]
        embedding_dim = self.embed_dim

        # expand for every batch
        if self.pattern is None or self.pattern.size(0) != seq_length:
            self.pattern = self._init(seq_length, self.embed_dim)

        batch_vector = (
            self.pattern.expand(batch_size, seq_length, embedding_dim)
            .clone()
            .to(visited_time.device)
        )
        index = (
            (visited_time % seq_length)
            .long()
            .unsqueeze(-1)
            .expand(batch_size, seq_length, embedding_dim)
        )

        return torch.gather(batch_vector, 1, index)


class CyclicPositionalEmbedding(nn.Module):
    """Cyclic Positional Embedding presented in Ma et al.(2021)
    See https://arxiv.org/abs/2110.02544
    """

    def __init__(self, embed_dim, mean_pooling=True):
        super(CyclicPositionalEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.mean_pooling = mean_pooling
        self.pattern = None

    def _basesin(self, x, T, fai=0):
        return np.sin(2 * np.pi / T * np.abs(np.mod(x, 2 * T) - T) + fai)

    def _basecos(self, x, T, fai=0):
        return np.cos(2 * np.pi / T * np.abs(np.mod(x, 2 * T) - T) + fai)

    def _init(self, n_position, emb_dim, mean_pooling):
        Td_set = np.linspace(
            np.power(n_position, 1 / (emb_dim // 2)),
            n_position,
            emb_dim // 2,
            dtype="int",
        )
        x = np.zeros((n_position, emb_dim))

        for i in range(emb_dim):
            Td = (
                Td_set[i // 3 * 3 + 1]
                if (i // 3 * 3 + 1) < (emb_dim // 2)
                else Td_set[-1]
            )
            fai = (
                0
                if i <= (emb_dim // 2)
                else 2 * np.pi * ((-i + (emb_dim // 2)) / (emb_dim // 2))
            )
            longer_pattern = np.arange(0, np.ceil((n_position) / Td) * Td, 0.01)
            if i % 2 == 1:
                x[:, i] = self._basecos(longer_pattern, Td, fai)[
                    np.linspace(
                        0, len(longer_pattern), n_position, dtype="int", endpoint=False
                    )
                ]
            else:
                x[:, i] = self._basesin(longer_pattern, Td, fai)[
                    np.linspace(
                        0, len(longer_pattern), n_position, dtype="int", endpoint=False
                    )
                ]

        pattern = torch.from_numpy(x).type(torch.FloatTensor)
        pattern_sum = torch.zeros_like(pattern)

        # averaging the adjacient embeddings if needed (optional, almost the same performance)
        arange = torch.arange(n_position)
        pooling = [0] if not mean_pooling else [-2, -1, 0, 1, 2]
        time = 0
        for i in pooling:
            time += 1
            index = (arange + i + n_position) % n_position
            pattern_sum += pattern.gather(0, index.view(-1, 1).expand_as(pattern))
        pattern = 1.0 / time * pattern_sum - pattern.mean(0)

        return pattern

    def forward(self, td):
        batch_size, seq_length = td["rec_current"].size()
        visited_time = td["visited_time"]
        embedding_dim = self.embed_dim

        # expand for every batch
        if self.pattern is None or self.pattern.size(0) != seq_length:
            self.pattern = self._init(seq_length, self.embed_dim, self.mean_pooling)

        batch_vector = (
            self.pattern.expand(batch_size, seq_length, embedding_dim)
            .clone()
            .to(visited_time.device)
        )
        index = (
            (visited_time % seq_length)
            .long()
            .unsqueeze(-1)
            .expand(batch_size, seq_length, embedding_dim)
        )

        return torch.gather(batch_vector, 1, index)
