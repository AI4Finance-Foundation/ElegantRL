import torch
import torch.nn as nn

from rlsolver.methods.eco_s2v.rl4co.utils.ops import gather_by_index
from rlsolver.methods.eco_s2v.rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def env_dynamic_embedding(env_name: str, config: dict) -> nn.Module:
    """Get environment dynamic embedding. The dynamic embedding is used to modify query, key and value vectors of the attention mechanism
    based on the current state of the environment (which is changing during the rollout).
    Consists of a linear layer that projects the node features to the embedding space.

    Args:
        env: Environment or its name.
        config: A dictionary of configuration options for the environment.
    """
    embedding_registry = {
        "tsp": StaticEmbedding,
        "atsp": StaticEmbedding,
        "cvrp": StaticEmbedding,
        "cvrptw": StaticEmbedding,
        "ffsp": StaticEmbedding,
        "svrp": StaticEmbedding,
        "sdvrp": SDVRPDynamicEmbedding,
        "pctsp": StaticEmbedding,
        "spctsp": StaticEmbedding,
        "op": StaticEmbedding,
        "dpp": StaticEmbedding,
        "mdpp": StaticEmbedding,
        "pdp": StaticEmbedding,
        "mtsp": StaticEmbedding,
        "smtwtp": StaticEmbedding,
        "jssp": JSSPDynamicEmbedding,
        "fjsp": JSSPDynamicEmbedding,
        "mtvrp": StaticEmbedding,
    }

    if env_name not in embedding_registry:
        log.warning(
            f"Unknown environment name '{env_name}'. Available dynamic embeddings: {embedding_registry.keys()}. Defaulting to StaticEmbedding."
        )
    return embedding_registry.get(env_name, StaticEmbedding)(**config)


class StaticEmbedding(nn.Module):
    """Static embedding for general problems.
    This is used for problems that do not have any dynamic information, except for the
    information regarding the current action (e.g. the current node in TSP). See context embedding for more details.
    """

    def __init__(self, *args, **kwargs):
        super(StaticEmbedding, self).__init__()

    def forward(self, td):
        return 0, 0, 0


class SDVRPDynamicEmbedding(nn.Module):
    """Dynamic embedding for the Split Delivery Vehicle Routing Problem (SDVRP).
    Embed the following node features to the embedding space:
        - demand_with_depot: demand of the customers and the depot
    The demand with depot is used to modify the query, key and value vectors of the attention mechanism
    based on the current state of the environment (which is changing during the rollout).
    """

    def __init__(self, embed_dim, linear_bias=False):
        super(SDVRPDynamicEmbedding, self).__init__()
        self.projection = nn.Linear(1, 3 * embed_dim, bias=linear_bias)

    def forward(self, td):
        demands_with_depot = td["demand_with_depot"][..., None].clone()
        demands_with_depot[..., 0, :] = 0
        glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic = self.projection(
            demands_with_depot
        ).chunk(3, dim=-1)
        return glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic


class JSSPDynamicEmbedding(nn.Module):
    def __init__(self, embed_dim, linear_bias=False, scaling_factor: int = 1000) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.project_node_step = nn.Linear(2, 3 * embed_dim, bias=linear_bias)
        self.project_edge_step = nn.Linear(1, 3, bias=linear_bias)
        self.scaling_factor = scaling_factor

    def forward(self, td, cache):
        ma_emb = cache.node_embeddings["machine_embeddings"]
        bs, _, emb_dim = ma_emb.shape
        num_jobs = td["next_op"].size(1)
        # updates
        updates = ma_emb.new_zeros((bs, num_jobs, 3 * emb_dim))

        lbs = torch.clip(td["lbs"] - td["time"][:, None], 0) / self.scaling_factor
        update_feat = torch.stack((lbs, td["is_ready"]), dim=-1)
        job_update_feat = gather_by_index(update_feat, td["next_op"], dim=1)
        updates = updates + self.project_node_step(job_update_feat)

        ma_busy = td["busy_until"] > td["time"][:, None]
        # mask machines currently busy
        masked_proc_times = td["proc_times"].clone() / self.scaling_factor
        # bs, ma, ops
        masked_proc_times[ma_busy] = 0.0
        # bs, ops, ma, 3
        edge_feat = self.project_edge_step(masked_proc_times.unsqueeze(-1)).transpose(
            1, 2
        )
        job_edge_feat = gather_by_index(edge_feat, td["next_op"], dim=1)
        # bs, nodes, 3*emb
        edge_upd = torch.einsum("ijkl,ikm->ijlm", job_edge_feat, ma_emb).view(
            bs, num_jobs, 3 * emb_dim
        )
        updates = updates + edge_upd

        # (bs, nodes, emb)
        glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic = updates.chunk(
            3, dim=-1
        )
        return glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic
