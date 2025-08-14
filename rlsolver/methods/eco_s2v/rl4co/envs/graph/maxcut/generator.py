import torch
from tensordict.tensordict import TensorDict

from rlsolver.methods.eco_s2v.rl4co.envs.common.utils import Generator
from rlsolver.methods.eco_s2v.rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class MaxCutGenerator(Generator):
    """Data generator for the Maximum Cut(MAxcut).
    
    Args:
        n_spins: number of spins in the graph
        graph_type: Distribution form of the graph
        m_insertion_edges
        p_connection
    Returns:


    """

    def __init__(
            self,
            n_spins: int = 100,
            graph_type: str = "BA",
            m_insertion_edges: int = 4,
            p_connection: float = 0.15,
            **kwargs
    ):
        self.graph_type = graph_type
        self.n_spins = n_spins
        self.m_insertion_edges = m_insertion_edges
        self.p_connection = p_connection

    def _generate(self, batch_size) -> TensorDict:
        # adj = torch.zeros((*batch_size, self.n_spins, self.n_spins),device=TRAIN_DEVICE)
        adj = torch.zeros((*batch_size, self.n_spins, self.n_spins))

        if self.graph_type == 'BA':
            for i in range(self.m_insertion_edges + 1):
                adj[:, i, :i + 1] = 1
                adj[:, :i + 1, i] = 1

            for new_node in range(self.m_insertion_edges + 1, self.n_spins):
                degree = adj.sum(dim=-1)
                prob = degree / degree.sum(dim=-1, keepdim=True)

                chosen_edges = torch.multinomial(prob, num_samples=self.m_insertion_edges, replacement=False)
                batch_indices = torch.arange(*batch_size).repeat_interleave(self.m_insertion_edges)
                adj[batch_indices, new_node, chosen_edges.view(-1)] = 1
                adj[batch_indices, chosen_edges.view(-1), new_node] = 1
        return TensorDict(
            {
                "adj": adj,
                "to_choose": torch.ones(*batch_size, dtype=torch.long) * self.n_spins,
                "state": torch.ones((*batch_size, self.n_spins), dtype=torch.bool)
            },
            batch_size=batch_size
        )
