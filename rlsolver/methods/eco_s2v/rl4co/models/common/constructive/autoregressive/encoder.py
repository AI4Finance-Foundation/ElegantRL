import abc

from rlsolver.methods.eco_s2v.rl4co.models.common.constructive.base import ConstructiveEncoder


class AutoregressiveEncoder(ConstructiveEncoder, metaclass=abc.ABCMeta):
    """Template class for an autoregressive encoder, simple wrapper around
    :class:`rl4co.models.common.constructive.base.ConstructiveEncoder`.

    Tip:
        This class will not work as it is and is just a template.
        An example for autoregressive encoder can be found as :class:`rl4co.models.zoo.am.encoder.AttentionModelEncoder`.
    """
