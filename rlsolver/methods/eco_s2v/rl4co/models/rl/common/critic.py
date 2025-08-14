import copy
from typing import Optional

from tensordict import TensorDict
from torch import Tensor, nn

from rlsolver.methods.eco_s2v.rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class CriticNetwork(nn.Module):
    """Create a critic network given an encoder (e.g. as the one in the policy network)
    with a value head to transform the embeddings to a scalar value.

    Args:
        encoder: Encoder module to encode the input
        value_head: Value head to transform the embeddings to a scalar value
        embed_dim: Dimension of the embeddings of the value head
        hidden_dim: Dimension of the hidden layer of the value head
    """

    def __init__(
            self,
            encoder: nn.Module,
            value_head: Optional[nn.Module] = None,
            embed_dim: int = 128,
            hidden_dim: int = 512,
            customized: bool = False,
    ):
        super(CriticNetwork, self).__init__()

        self.encoder = encoder
        if value_head is None:
            # check if embed dim of encoder is different, if so, use it
            if getattr(encoder, "embed_dim", embed_dim) != embed_dim:
                log.warning(
                    f"Found encoder with different embed_dim {encoder.embed_dim} than the value head {embed_dim}. Using encoder embed_dim for value head."
                )
                embed_dim = getattr(encoder, "embed_dim", embed_dim)
            value_head = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
            )
        self.value_head = value_head
        self.customized = customized

    def forward(self, x: Tensor | TensorDict, hidden=None) -> Tensor:
        """Forward pass of the critic network: encode the imput in embedding space and return the value

        Args:
            x: Input containing the environment state. Can be a Tensor or a TensorDict

        Returns:
            Value of the input state
        """
        if not self.customized:  # fir for most of costructive tasks
            h, _ = self.encoder(x)  # [batch_size, N, embed_dim] -> [batch_size, N]
            return self.value_head(h).mean(1)  # [batch_size, N] -> [batch_size]
        else:  # custimized encoder and value head with hidden input
            h = self.encoder(x)  # [batch_size, N, embed_dim] -> [batch_size, N]
            return self.value_head(h, hidden)


def create_critic_from_actor(
        policy: nn.Module, backbone: str = "encoder", **critic_kwargs
):
    # we reuse the network of the policy's backbone, such as an encoder
    encoder = getattr(policy, backbone, None)
    if encoder is None:
        raise ValueError(
            f"CriticBaseline requires a backbone in the policy network: {backbone}"
        )
    critic = CriticNetwork(copy.deepcopy(encoder), **critic_kwargs).to(
        next(policy.parameters()).device
    )
    return critic
