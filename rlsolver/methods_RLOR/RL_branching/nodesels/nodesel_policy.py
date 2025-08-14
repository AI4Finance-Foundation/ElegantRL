# import extract
from ..extract import extract_MLP_state
import torch as th
import pyscipopt as scip


# The standard nodesel wrapper for all policies
class NodeselPolicy(scip.Nodesel):
    def __init__(self, policy=None, device=None, name=""):
        super().__init__()
        # self.model = model
        self.policy = policy
        self.device = device
        self.name = name

    def __str__(self):
        return self.name

    def nodeselect(self):
        # calculate minimal and maximal plunging depth
        min_plunge_depth = int(self.model.getMaxDepth() / 10)
        if self.model.getNStrongbranchLPIterations() > 2*self.model.getNNodeLPIterations():
            min_plunge_depth += 10

        max_plunge_depth = int(self.model.getMaxDepth() / 2)
        max_plunge_depth = max(max_plunge_depth, min_plunge_depth)
        max_plunge_quot = 0.25

        # check if we are within the maximal plunging depth
        plunge_depth = self.model.getPlungeDepth()
        selnode = self.model.getBestChild()
        # possibly choose sibling if child is None
        if plunge_depth <= max_plunge_depth and selnode is not None:
            # get global lower and cutoff bound
            lower_bound = self.model.getLowerbound()
            cutoff_bound = self.model.getCutoffbound()

            # if we didn't find a solution yet,
            # the cutoff bound is usually very bad:
            # use 20% of the gap as cutoff bound
            if self.model.getNSolsFound() == 0:
                max_plunge_quot *= 0.2

            # check, if plunging is forced at the current depth
            # else calculate maximal plunging bound
            max_bound = self.model.infinity()
            if plunge_depth >= min_plunge_depth:
                max_bound = lower_bound + max_plunge_quot * (cutoff_bound - lower_bound)

            if selnode.getEstimate() < max_bound:
                return {'selnode': selnode}

        return {'selnode': self.model.getBestboundNode()}

    def nodecomp(self, node1, node2):
        if node1.getParent() != node2.getParent(): return 0

        state1, state2 = extract_MLP_state(self.model, node1, node2)
        state = (th.tensor(state1, dtype=th.float32),
                 th.tensor(state2, dtype=th.float32))

        with th.inference_mode():
            return self.policy(*state)
