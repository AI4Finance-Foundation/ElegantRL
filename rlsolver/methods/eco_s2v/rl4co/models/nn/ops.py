import math
from typing import Tuple

import torch
import torch.nn as nn

from rlsolver.methods.eco_s2v.rl4co.utils.ops import gather_by_index


class SkipConnection(nn.Module):
    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


class AdaptiveSequential(nn.Sequential):
    def forward(
            self, *inputs: Tuple[torch.Tensor] | torch.Tensor
    ) -> Tuple[torch.Tensor] | torch.Tensor:
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class Normalization(nn.Module):
    def __init__(self, embed_dim, normalization="batch"):
        super(Normalization, self).__init__()
        if normalization != "layer":
            normalizer_class = {
                "batch": nn.BatchNorm1d,
                "instance": nn.InstanceNorm1d,
            }.get(normalization, None)

            self.normalizer = normalizer_class(embed_dim, affine=True)
        else:
            self.normalizer = "layer"

    def forward(self, x):
        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(x.view(-1, x.size(-1))).view(*x.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(x.permute(0, 2, 1)).permute(0, 2, 1)
        elif self.normalizer == "layer":
            return (x - x.mean((1, 2)).view(-1, 1, 1)) / torch.sqrt(
                x.var((1, 2)).view(-1, 1, 1) + 1e-05
            )
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return x


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = embed_dim
        max_len = max_len
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(max_len, 1, self.d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0, 1)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)
        breakpoint()

    def forward(self, hidden: torch.Tensor, seq_pos) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
            seq_pos: Tensor, shape ``[batch_size, seq_len]``
        """
        pes = self.pe.expand(hidden.size(0), -1, -1).gather(
            1, seq_pos.unsqueeze(-1).expand(-1, -1, self.d_model)
        )
        hidden = hidden + pes
        return self.dropout(hidden)


class TransformerFFN(nn.Module):
    def __init__(self, embed_dim, feed_forward_hidden, normalization="batch") -> None:
        super().__init__()

        self.ops = nn.ModuleDict(
            {
                "norm1": Normalization(embed_dim, normalization),
                "ffn": nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim),
                ),
                "norm2": Normalization(embed_dim, normalization),
            }
        )

    def forward(self, x, x_old):
        x = self.ops["norm1"](x_old + x)
        x = self.ops["norm2"](x + self.ops["ffn"](x))

        return x


class RandomEncoding(nn.Module):
    """This is like torch.nn.Embedding but with rows of embeddings are randomly
    permuted in each forward pass before lookup operation. This might be useful
    in cases where classes have no fixed meaning but rather indicate a connection
    between different elements in a sequence. Reference is the MatNet model.
    """

    def __init__(self, embed_dim: int, max_classes: int = 100):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_classes = max_classes
        rand_emb = torch.rand(max_classes, self.embed_dim)
        self.register_buffer("emb", rand_emb)
        breakpoint()

    def forward(self, hidden: torch.Tensor, classes=None) -> torch.Tensor:
        b, s, _ = hidden.shape
        if classes is None:
            classes = torch.eye(s).unsqueeze(0).expand(b, s)
        assert (
                classes.max() < self.max_classes
        ), "number of classes larger than embedding table"
        classes = classes.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        rand_idx = torch.rand(b, self.max_classes).argsort(dim=1)
        embs_permuted = self.emb[rand_idx]
        rand_emb = gather_by_index(embs_permuted, classes, dim=1)
        hidden = hidden + rand_emb
        return hidden
