import math
import pyscipopt as scip


class NodeselDFS(scip.Nodesel):
    def __init__(self):
        super().__init__()

    def nodeselect(self):
        # Try getPrioChild
        selnode = self.model.getPrioChild()
        if selnode is not None:
            return {"selnode": selnode}

        # Try getPrioSibling
        selnode = self.model.getPrioSibling()
        if selnode is not None:
            return {"selnode": selnode}

        # Default to getBestLeaf
        return {"selnode": self.model.getBestLeaf()}

    def nodecomp(self, node1, node2):
        depth1 = node1.getDepth()
        depth2 = node2.getDepth()

        if depth1 == depth2:
            lowerbound1 = node1.getLowerbound()
            lowerbound2 = node2.getLowerbound()
            if lowerbound1 == lowerbound2: return 0
            return 1 if lowerbound1 > lowerbound2 else -1
        return 1 if depth1 < depth2 else -1


class NodeselRestartDFS(scip.Nodesel):
    def __init__(self):
        super().__init__()
        self.last_restart = 0
        self.n_processed_leaves = 0
        self.select_best_freq = 100
        self.count_only_leaves = True

    def nodeinitsol(self):
        self.last_restart = 0
        self.n_processed_leaves = 0

    def nodeselect(self):
        # decide if we want to select the node with the lowest bound or the deepest node;
        # finish the current dive in any case
        selnode = self.model.getPrioChild()
        if selnode is not None:
            # self.handle_pruning(selnode)
            return {"selnode": selnode}

        # increase the number of processed leafs since we are in a leaf
        self.n_processed_leaves += 1
        nnodes = self.model.getNNodes()
        leaves = int(self.count_only_leaves)

        # when "only leaves", check if the number of processed leaves exceeds the frequency or
        # when not "only leaves", check if the number of processed nodes exceeds the frequency
        if leaves * self.n_processed_leaves + (1 - leaves) * (nnodes - self.last_restart) >= self.select_best_freq:
            self.last_restart = nnodes
            self.n_processed_leaves = 0
            return {"selnode": self.model.getBestboundNode()}

        selnode = self.model.getPrioSibling()
        if selnode is not None:
            return {'selnode': selnode}
        return {"selnode": self.model.getBestLeaf()}

    def nodecomp(self, node1, node2):
        return int(node2.getNumber() - node1.getNumber())


class NodeselEstimate(scip.Nodesel):
    """
    Mimics the BestEstimate node selector of SCIP
    """
    def __init__(self):
        super().__init__()
        self.min_plunge_depth = -1
        self.max_plunge_depth = -1
        self.max_plunge_quot = 0.25
        self.best_node_freq = 10
        self.breadth_first_depth = -1
        self.plunge_offset = 0

    def nodeselect(self):
        # check if the breadth-first search should be applied
        if self.model.getDepth() <= self.breadth_first_depth:
            selnode = self.model.getPrioSibling()
            if selnode is not None:
                return {'selnode': selnode}

            selnode = self.model.getPrioChild()
            if selnode is not None:
                return {'selnode': selnode}

        best_node_freq = math.inf if self.best_node_freq == 0 else self.best_node_freq

        # check if we want to do plunging yet
        if self.model.getNNodes() < self.plunge_offset:
            # we don't want to plunge yet: select best node from the tree
            if self.model.getNNodes() % best_node_freq == 0:
                return {'selnode': self.model.getBestboundNode()}
            return {'selnode': self.model.getBestNode()}

        # calculate minimal and maximal plunging depth
        min_plunge_depth = self.min_plunge_depth
        max_plunge_depth = self.max_plunge_depth
        max_plunge_quot = self.max_plunge_quot
        if self.min_plunge_depth == -1:
            min_plunge_depth = int(self.model.getMaxDepth() / 10)
            if self.model.getNStrongbranchLPIterations() > 2*self.model.getNNodeLPIterations():
                min_plunge_depth += 10
            if self.max_plunge_depth >= 0:
                min_plunge_depth = min(min_plunge_depth, max_plunge_depth)

        if self.max_plunge_depth == -1:
            max_plunge_depth = int(self.model.getMaxDepth() / 2)
        max_plunge_depth = max(max_plunge_depth, min_plunge_depth)

        # check if we are within the maximal plunging depth
        plunge_depth = self.model.getPlungeDepth()
        if plunge_depth <= max_plunge_depth:
            # get global lower and cutoff bound
            lower_bound = self.model.getLowerbound()
            cutoff_bound = self.model.getCutoffbound()

            # if we didn't find a solution yet,
            # the cutoff bound is usually very bad:
            # use 20% of the gap as cutoff bound
            if self.model.getNSolsFound() == 0:
                # cutoff_bound = lower_bound + 0.2 * (cutoff_bound - lower_bound)
                max_plunge_quot *= 0.2

            # It turns out that using 20% of the gap as cutoff bound is equal to
            # using 20% of the max_plunge_quot which can be shown by substitution
            #  new_cutoff_bound = lower_bound + 0.2 * (cutoff_bound - lower_bound)
            #  max_bound = lower_bound + max_plunge_quot * (new_cutoff_bound - lower_bound)
            # substitute new_cutoff_bound:
            #  max_bound = lower_bound + max_plunge_quot * (lower_bound + 0.2 * (cutoff_bound - lower_bound) - lower_bound)
            #  max_bound = lower_bound + max_plunge_quot * (0.2 * (cutoff_bound - lower_bound))
            #  max_bound = lower_bound + (0.2 * max_plunge_quot) * (cutoff_bound - lower_bound)

            # check, if plunging is forced at the current depth
            # else calculate maximal plunging bound
            max_bound = self.model.infinity()
            if plunge_depth >= min_plunge_depth:
                max_bound = lower_bound + max_plunge_quot * (cutoff_bound - lower_bound)

            # we want to plunge again: prioritize children over siblings, and siblings over leaves,
            # but only select a child or sibling if its estimate is small enough;
            # prefer using nodes with higher node selection priority assigned by the branching rule
            selnode = self.model.getPrioChild()
            if selnode is not None and selnode.getEstimate() < max_bound:
                return {'selnode': selnode}
            selnode = self.model.getBestChild()
            if selnode is not None and selnode.getEstimate() < max_bound:
                return {'selnode': selnode}
            selnode = self.model.getPrioSibling()
            if selnode is not None and selnode.getEstimate() < max_bound:
                return {'selnode': selnode}
            selnode = self.model.getBestSibling()
            if selnode is not None and selnode.getEstimate() < max_bound:
                return {'selnode': selnode}

        if self.model.getNNodes() % best_node_freq == 0:
            return {'selnode': self.model.getBestboundNode()}
        return {'selnode': self.model.getBestNode()}

    def nodecomp(self, node1, node2):
        estimate1 = node1.getEstimate()
        estimate2 = node2.getEstimate()

        is_pos_inf = self.model.isInfinity(estimate1) and self.model.isInfinity(estimate2)
        is_neg_inf = self.model.isInfinity(-estimate1) and self.model.isInfinity(-estimate2)
        is_eq = self.model.isEQ(estimate1, estimate2)

        if is_pos_inf or is_neg_inf or is_eq:
            lowerbound1 = node1.getLowerbound()
            lowerbound2 = node2.getLowerbound()
            if self.model.isEQ(lowerbound1, lowerbound2):
                nodetype1 = node1.getType()
                nodetype2 = node2.getType()
                nodetype_child = 3
                nodetype_sibling = 2
                if nodetype1 == nodetype_child and nodetype2 != nodetype_child:
                    return -1
                if nodetype1 != nodetype_child and nodetype2 == nodetype_child:
                    return 1
                if nodetype1 == nodetype_sibling and nodetype2 != nodetype_sibling:
                    return -1
                if nodetype1 != nodetype_sibling and nodetype2 == nodetype_sibling:
                    return 1
                depth1 = node1.getDepth()
                depth2 = node2.getDepth()
                if depth1 == depth2: return 0
                return 1 if depth1 > depth2 else -1
            return 1 if self.model.isGT(lowerbound1, lowerbound2) else -1
        return 1 if self.model.isGT(estimate1, estimate2) else -1
