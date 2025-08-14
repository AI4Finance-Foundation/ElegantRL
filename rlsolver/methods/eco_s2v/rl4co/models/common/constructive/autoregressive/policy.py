from rlsolver.methods.eco_s2v.rl4co.models.common.constructive.base import ConstructivePolicy

from .decoder import AutoregressiveDecoder
from .encoder import AutoregressiveEncoder


class AutoregressivePolicy(ConstructivePolicy):
    """Template class for an autoregressive policy, simple wrapper around
    :class:`rl4co.models.common.constructive.base.ConstructivePolicy`.

    Note:
        While a decoder is required, an encoder is optional and will be initialized to
        :class:`rl4co.models.common.constructive.autoregressive.encoder.NoEncoder`.
        This can be used in decoder-only models in which at each step actions do not depend on
        previously encoded states.
    """

    def __init__(
            self,
            encoder: AutoregressiveEncoder,
            decoder: AutoregressiveDecoder,
            env_name: str = "tsp",
            temperature: float = 1.0,
            tanh_clipping: float = 0,
            mask_logits: bool = True,
            train_decode_type: str = "sampling",
            val_decode_type: str = "greedy",
            test_decode_type: str = "greedy",
            **unused_kw,
    ):
        # We raise an error for the user if no decoder was provided
        if decoder is None:
            raise ValueError("AutoregressivePolicy requires a decoder to be provided.")

        super(AutoregressivePolicy, self).__init__(
            encoder=encoder,
            decoder=decoder,
            env_name=env_name,
            temperature=temperature,
            tanh_clipping=tanh_clipping,
            mask_logits=mask_logits,
            train_decode_type=train_decode_type,
            val_decode_type=val_decode_type,
            test_decode_type=test_decode_type,
            **unused_kw,
        )
