import itertools
import math
import warnings
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from rlsolver.methods.eco_s2v.rl4co.models.nn.moe import MoE
from rlsolver.methods.eco_s2v.rl4co.utils import get_pylogger

log = get_pylogger(__name__)


def scaled_dot_product_attention_simple(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
):
    """Simple (exact) Scaled Dot-Product Attention in RL4CO without customized kernels (i.e. no Flash Attention)."""

    # Check for causal and attn_mask conflict
    if is_causal and attn_mask is not None:
        raise ValueError("Cannot set both is_causal and attn_mask")

    # Calculate scaled dot product
    scores = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)

    # Apply the provided attention mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores.masked_fill_(~attn_mask, float("-inf"))
        else:
            scores += attn_mask

    # Apply causal mask
    if is_causal:
        s, l_ = scores.size(-2), scores.size(-1)
        mask = torch.triu(torch.ones((s, l_), device=scores.device), diagonal=1)
        scores.masked_fill_(mask.bool(), float("-inf"))

    # Softmax to get attention weights
    attn_weights = F.softmax(scores, dim=-1)

    # Apply dropout
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    # Compute the weighted sum of values
    return torch.matmul(attn_weights, v)


try:
    from torch.nn.functional import scaled_dot_product_attention
except ImportError:
    log.warning(
        "torch.nn.functional.scaled_dot_product_attention not found. Make sure you are using PyTorch >= 2.0.0."
        "Alternatively, install Flash Attention https://github.com/HazyResearch/flash-attention ."
        "Using custom implementation of scaled_dot_product_attention without Flash Attention. "
    )
    scaled_dot_product_attention = scaled_dot_product_attention_simple


class MultiHeadAttention(nn.Module):
    """PyTorch native implementation of Flash Multi-Head Attention with automatic mixed precision support.
    Uses PyTorch's native `scaled_dot_product_attention` implementation, available from 2.0

    Note:
        If `scaled_dot_product_attention` is not available, use custom implementation of `scaled_dot_product_attention` without Flash Attention.

    Args:
        embed_dim: total dimension of the model
        num_heads: number of heads
        bias: whether to use bias
        attention_dropout: dropout rate for attention weights
        causal: whether to apply causal mask to attention scores
        device: torch device
        dtype: torch dtype
        sdpa_fn: scaled dot product attention function (SDPA) implementation
    """

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            bias: bool = True,
            attention_dropout: float = 0.0,
            causal: bool = False,
            device: str = None,
            dtype: torch.dtype = None,
            sdpa_fn: Optional[Callable] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.attention_dropout = attention_dropout
        self.sdpa_fn = sdpa_fn if sdpa_fn is not None else scaled_dot_product_attention

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert (
                self.head_dim % 8 == 0 and self.head_dim <= 128
        ), "Only support head_dim <= 128 and divisible by 8"

        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

    def forward(self, x, attn_mask=None):
        """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        attn_mask: bool tensor of shape (batch, seqlen)
        """
        # Project query, key, value
        q, k, v = rearrange(
            self.Wqkv(x), "b s (three h d) -> three b h s d", three=3, h=self.num_heads
        ).unbind(dim=0)

        if attn_mask is not None:
            attn_mask = (
                attn_mask.unsqueeze(1)
                if attn_mask.ndim == 3
                else attn_mask.unsqueeze(1).unsqueeze(2)
            )

        # Scaled dot product attention
        out = self.sdpa_fn(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attention_dropout,
        )
        return self.out_proj(rearrange(out, "b h s d -> b s (h d)"))


def sdpa_fn_wrapper(q, k, v, attn_mask=None, dmat=None, dropout_p=0.0, is_causal=False):
    if dmat is not None:
        log.warning(
            "Edge weights passed to simple attention-fn, which is not supported. Weights will be ignored..."
        )
    return scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal
    )


class MultiHeadCrossAttention(nn.Module):
    """PyTorch native implementation of Flash Multi-Head Cross Attention with automatic mixed precision support.
    Uses PyTorch's native `scaled_dot_product_attention` implementation, available from 2.0

    Note:
        If `scaled_dot_product_attention` is not available, use custom implementation of `scaled_dot_product_attention` without Flash Attention.

    Args:
        embed_dim: total dimension of the model
        num_heads: number of heads
        bias: whether to use bias
        attention_dropout: dropout rate for attention weights
        device: torch device
        dtype: torch dtype
        sdpa_fn: scaled dot product attention function (SDPA)
    """

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            bias: bool = False,
            attention_dropout: float = 0.0,
            device: str = None,
            dtype: torch.dtype = None,
            sdpa_fn: Optional[Callable | nn.Module] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.attention_dropout = attention_dropout

        # Default to `scaled_dot_product_attention` if `sdpa_fn` is not provided
        if sdpa_fn is None:
            sdpa_fn = sdpa_fn_wrapper
        self.sdpa_fn = sdpa_fn

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert (
                self.head_dim % 8 == 0 and self.head_dim <= 128
        ), "Only support head_dim <= 128 and divisible by 8"

        self.Wq = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.Wkv = nn.Linear(embed_dim, 2 * embed_dim, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

    def forward(self, q_input, kv_input, cross_attn_mask=None, dmat=None):
        # Project query, key, value
        q = rearrange(
            self.Wq(q_input), "b m (h d) -> b h m d", h=self.num_heads
        )  # [b, h, m, d]
        k, v = rearrange(
            self.Wkv(kv_input), "b n (two h d) -> two b h n d", two=2, h=self.num_heads
        ).unbind(
            dim=0
        )  # [b, h, n, d]

        if cross_attn_mask is not None:
            # add head dim
            cross_attn_mask = cross_attn_mask.unsqueeze(1)

        # Scaled dot product attention
        out = self.sdpa_fn(
            q,
            k,
            v,
            attn_mask=cross_attn_mask,
            dmat=dmat,
            dropout_p=self.attention_dropout,
        )
        return self.out_proj(rearrange(out, "b h s d -> b s (h d)"))


class PointerAttention(nn.Module):
    """Calculate logits given query, key and value and logit key.
    This follows the pointer mechanism of Vinyals et al. (2015) (https://arxiv.org/abs/1506.03134).

    Note:
        With Flash Attention, masking is not supported

    Performs the following:
        1. Apply cross attention to get the heads
        2. Project heads to get glimpse
        3. Compute attention score between glimpse and logit key

    Args:
        embed_dim: total dimension of the model
        num_heads: number of heads
        mask_inner: whether to mask inner attention
        linear_bias: whether to use bias in linear projection
        check_nan: whether to check for NaNs in logits
        sdpa_fn: scaled dot product attention function (SDPA) implementation
    """

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            mask_inner: bool = True,
            out_bias: bool = False,
            check_nan: bool = True,
            sdpa_fn: Callable | str = "default",
            **kwargs,
    ):
        super(PointerAttention, self).__init__()
        self.num_heads = num_heads
        self.mask_inner = mask_inner

        # Projection - query, key, value already include projections
        self.project_out = nn.Linear(embed_dim, embed_dim, bias=out_bias)
        self.check_nan = check_nan

        # Defaults for sdpa_fn implementation
        # see https://github.com/ai4co/rl4co/issues/228
        if isinstance(sdpa_fn, str):
            if sdpa_fn == "default":
                sdpa_fn = scaled_dot_product_attention
            elif sdpa_fn == "simple":
                sdpa_fn = scaled_dot_product_attention_simple
            else:
                raise ValueError(
                    f"Unknown sdpa_fn: {sdpa_fn}. Available options: ['default', 'simple']"
                )
        else:
            if sdpa_fn is None:
                sdpa_fn = scaled_dot_product_attention
                log.info(
                    "Using default scaled_dot_product_attention for PointerAttention"
                )
        self.sdpa_fn = sdpa_fn

    def forward(self, query, key, value, logit_key, attn_mask=None):
        """Compute attention logits given query, key, value, logit key and attention mask.

        Args:
            query: query tensor of shape [B, ..., L, E]
            key: key tensor of shape [B, ..., S, E]
            value: value tensor of shape [B, ..., S, E]
            logit_key: logit key tensor of shape [B, ..., S, E]
            attn_mask: attention mask tensor of shape [B, ..., S]. Note that `True` means that the value _should_ take part in attention
                as described in the [PyTorch Documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
        """
        # Compute inner multi-head attention with no projections.
        heads = self._inner_mha(query, key, value, attn_mask)
        glimpse = self._project_out(heads, attn_mask)

        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # bmm is slightly faster than einsum and matmul
        logits = (torch.bmm(glimpse, logit_key.squeeze(-2).transpose(-2, -1))).squeeze(
            -2
        ) / math.sqrt(glimpse.size(-1))

        if self.check_nan:
            assert not torch.isnan(logits).any(), "Logits contain NaNs"

        return logits

    def _inner_mha(self, query, key, value, attn_mask):
        q = self._make_heads(query)
        k = self._make_heads(key)
        v = self._make_heads(value)
        if self.mask_inner:
            # make mask the same number of dimensions as q
            attn_mask = (
                attn_mask.unsqueeze(1)
                if attn_mask.ndim == 3
                else attn_mask.unsqueeze(1).unsqueeze(2)
            )
        else:
            attn_mask = None
        heads = self.sdpa_fn(q, k, v, attn_mask=attn_mask)
        return rearrange(heads, "... h n g -> ... n (h g)", h=self.num_heads)

    def _make_heads(self, v):
        return rearrange(v, "... g (h s) -> ... h g s", h=self.num_heads)

    def _project_out(self, out, *kwargs):
        return self.project_out(out)


class PointerAttnMoE(PointerAttention):
    """Calculate logits given query, key and value and logit key.
    This follows the pointer mechanism of Vinyals et al. (2015) <https://arxiv.org/abs/1506.03134>,
        and the MoE gating mechanism of Zhou et al. (2024) <https://arxiv.org/abs/2405.01029>.

    Note:
        With Flash Attention, masking is not supported

    Performs the following:
        1. Apply cross attention to get the heads
        2. Project heads to get glimpse
        3. Compute attention score between glimpse and logit key

    Args:
        embed_dim: total dimension of the model
        num_heads: number of heads
        mask_inner: whether to mask inner attention
        linear_bias: whether to use bias in linear projection
        check_nan: whether to check for NaNs in logits
        sdpa_fn: scaled dot product attention function (SDPA) implementation
        moe_kwargs: Keyword arguments for MoE
    """

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            mask_inner: bool = True,
            out_bias: bool = False,
            check_nan: bool = True,
            sdpa_fn: Optional[Callable] = None,
            moe_kwargs: Optional[dict] = None,
    ):
        super(PointerAttnMoE, self).__init__(
            embed_dim, num_heads, mask_inner, out_bias, check_nan, sdpa_fn
        )
        self.moe_kwargs = moe_kwargs

        self.project_out = None
        self.project_out_moe = MoE(
            embed_dim, embed_dim, num_neurons=[], out_bias=out_bias, **moe_kwargs
        )
        if self.moe_kwargs["light_version"]:
            self.dense_or_moe = nn.Linear(embed_dim, 2, bias=False)
            self.project_out = nn.Linear(embed_dim, embed_dim, bias=out_bias)

    def _project_out(self, out, attn_mask):
        """Implementation of Hierarchical Gating based on Zhou et al. (2024) <https://arxiv.org/abs/2405.01029>."""
        if self.moe_kwargs["light_version"]:
            num_nodes, num_available_nodes = attn_mask.size(-1), attn_mask.sum(-1)
            # only do this at the "second" step, which is depot -> pomo -> first select
            if (num_available_nodes >= num_nodes - 1).any():
                self.probs = F.softmax(
                    self.dense_or_moe(
                        out.view(-1, out.size(-1)).mean(dim=0, keepdim=True)
                    ),
                    dim=-1,
                )
            selected = self.probs.multinomial(1).squeeze(0)
            out = (
                self.project_out_moe(out)
                if selected.item() == 1
                else self.project_out(out)
            )
            glimpse = out * self.probs.squeeze(0)[selected]
        else:
            glimpse = self.project_out_moe(out)
        return glimpse


# Deprecated
class LogitAttention(PointerAttention):
    def __init__(self, *args, **kwargs):
        warnings.simplefilter("always", DeprecationWarning)
        warnings.warn(
            "LogitAttention is deprecated and will be removed in a future release. "
            "Please use PointerAttention instead."
            "Note that several components of the previous LogitAttention have moved to `rl4co.models.nn.dec_strategies`.",
            category=DeprecationWarning,
        )
        super(LogitAttention, self).__init__(*args, **kwargs)


# MultiHeadCompat
class MultiHeadCompat(nn.Module):
    def __init__(self, n_heads, input_dim, embed_dim=None, val_dim=None, key_dim=None):
        super(MultiHeadCompat, self).__init__()

        if val_dim is None:
            # assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))

        self.init_parameters()

    # used for init nn.Parameter
    def init_parameters(self):
        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """

        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """

        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)

        hflat = h.contiguous().view(-1, input_dim)  #################   reshape
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        K = torch.matmul(hflat, self.W_key).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility_s2n = torch.matmul(Q, K.transpose(2, 3))

        return compatibility_s2n


class PolyNetAttention(PointerAttention):
    """Calculate logits given query, key and value and logit key.
    This implements a modified version the pointer mechanism of Vinyals et al. (2015) (https://arxiv.org/abs/1506.03134)
    as described in Hottung et al. (2024) (https://arxiv.org/abs/2402.14048) PolyNetAttention conditions the attention logits on
    a set of k different binary vectors allowing to learn k different solution strategies.

    Note:
        With Flash Attention, masking is not supported

    Performs the following:
        1. Apply cross attention to get the heads
        2. Project heads to get glimpse
        3. Apply PolyNet layers
        4. Compute attention score between glimpse and logit key

    Args:
        k: Number unique bit vectors used to compute attention score
        embed_dim: total dimension of the model
        poly_layer_dim: Dimension of the PolyNet layers
        num_heads: number of heads
        mask_inner: whether to mask inner attention
        linear_bias: whether to use bias in linear projection
        check_nan: whether to check for NaNs in logits
        sdpa_fn: scaled dot product attention function (SDPA) implementation
    """

    def __init__(
            self, k: int, embed_dim: int, poly_layer_dim: int, num_heads: int, **kwargs
    ):
        super(PolyNetAttention, self).__init__(embed_dim, num_heads, **kwargs)

        self.k = k
        self.binary_vector_dim = math.ceil(math.log2(k))
        self.binary_vectors = torch.nn.Parameter(
            torch.Tensor(
                list(itertools.product([0, 1], repeat=self.binary_vector_dim))[:k]
            ),
            requires_grad=False,
        )

        self.poly_layer_1 = nn.Linear(embed_dim + self.binary_vector_dim, poly_layer_dim)
        self.poly_layer_2 = nn.Linear(poly_layer_dim, embed_dim)

    def forward(self, query, key, value, logit_key, attn_mask=None):
        """Compute attention logits given query, key, value, logit key and attention mask.

        Args:
            query: query tensor of shape [B, ..., L, E]
            key: key tensor of shape [B, ..., S, E]
            value: value tensor of shape [B, ..., S, E]
            logit_key: logit key tensor of shape [B, ..., S, E]
            attn_mask: attention mask tensor of shape [B, ..., S]. Note that `True` means that the value _should_ take part in attention
                as described in the [PyTorch Documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
        """
        # Compute inner multi-head attention with no projections.
        heads = self._inner_mha(query, key, value, attn_mask)
        glimpse = self.project_out(heads)

        num_solutions = glimpse.shape[1]
        z = self.binary_vectors.repeat(math.ceil(num_solutions / self.k), 1)[
            :num_solutions
            ]
        z = z[None].expand(glimpse.shape[0], num_solutions, self.binary_vector_dim)

        # PolyNet layers
        poly_out = self.poly_layer_1(torch.cat((glimpse, z), dim=2))
        poly_out = F.relu(poly_out)
        poly_out = self.poly_layer_2(poly_out)

        glimpse += poly_out

        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # bmm is slightly faster than einsum and matmul
        logits = (torch.bmm(glimpse, logit_key.squeeze(-2).transpose(-2, -1))).squeeze(
            -2
        ) / math.sqrt(glimpse.size(-1))

        if self.check_nan:
            assert not torch.isnan(logits).any(), "Logits contain NaNs"

        return logits
