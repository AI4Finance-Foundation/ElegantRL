from typing import Callable

import torch.nn as nn

from rlsolver.methods.eco_s2v.rl4co.models.common.constructive.autoregressive.policy import AutoregressivePolicy
from rlsolver.methods.eco_s2v.rl4co.models.zoo.s2v.decoder import S2VModelDecoder
from rlsolver.methods.eco_s2v.rl4co.models.zoo.s2v.encoder import S2VModelEncoder


class S2VModelPolicy(AutoregressivePolicy):

    def __init__(
            self,
            encoder: nn.Module = None,
            decoder: nn.Module = None,
            embed_dim: int = 128,
            num_encoder_layers: int = 3,
            num_heads: int = 8,
            normalization: str = "batch",
            feedforward_hidden: int = 512,
            env_name: str = "tsp",
            encoder_network: nn.Module = None,
            init_embedding: nn.Module = None,
            context_embedding: nn.Module = None,
            dynamic_embedding: nn.Module = None,
            use_graph_context: bool = True,
            linear_bias_decoder: bool = False,
            sdpa_fn: Callable = None,
            sdpa_fn_encoder: Callable = None,
            sdpa_fn_decoder: Callable = None,
            mask_inner: bool = True,
            out_bias_pointer_attn: bool = False,
            check_nan: bool = True,
            temperature: float = 1.0,
            tanh_clipping: float = 10.0,
            mask_logits: bool = True,
            train_decode_type: str = "sampling",
            val_decode_type: str = "greedy",
            test_decode_type: str = "greedy",
            moe_kwargs: dict = {"encoder": None, "decoder": None},
            **unused_kwargs,
    ):
        if encoder is None:
            encoder = S2VModelEncoder(
                env_name=env_name,
                embed_dim=embed_dim
            )

        if decoder is None:
            decoder = S2VModelDecoder(
                env_name=env_name,
                embed_dim=embed_dim
            )

        super(S2VModelPolicy, self).__init__(
            encoder=encoder,
            decoder=decoder,
            env_name=env_name,
            temperature=temperature,
            tanh_clipping=tanh_clipping,
            mask_logits=mask_logits,
            train_decode_type=train_decode_type,
            val_decode_type=val_decode_type,
            test_decode_type=test_decode_type,
            **unused_kwargs,
        )
