import networkx as nx
import torch
from tensordict.tensordict import TensorDict

from rlsolver.methods.eco_s2v.rl4co.envs.common.utils import Generator
from rlsolver.methods.eco_s2v.rl4co.utils.pylogger import get_pylogger
from rlsolver.methods.eco_s2v.util import load_graph_from_txt

log = get_pylogger(__name__)


class MaxCutGenerator(Generator):

    def __init__(
            self,
            file: str,
            device: str = "cpu",
    ):
        self.device = device
        g = load_graph_from_txt(file)
        g_array = nx.to_numpy_array(g)
        self.g_tensor = torch.tensor(g_array, dtype=torch.float, device=self.device)
        self.n_spins = g_array.shape[0]

    def _generate(self, batch_size) -> TensorDict:
        adj = self.g_tensor.unsqueeze(0).expand(*batch_size, -1, -1).to(self.device)
        return TensorDict(
            {
                "adj": adj,
                "to_choose": torch.ones(*batch_size, dtype=torch.long) * self.g_tensor.shape[0],
                "state": torch.ones((*batch_size, self.g_tensor.shape[0]), dtype=torch.bool)
            },
            batch_size=batch_size
        )
