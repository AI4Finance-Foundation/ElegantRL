import queue
from ..extract import extract_MLP_state
import numpy as np
import torch as th

from nodesel_policy import NodeselPolicy


class NodeselAgent(NodeselPolicy):
    def __init__(self, instance, opt_sol, seed, greedy, metric, sample_rate, requests_queue):
        super().__init__()
        # self.model = model
        self.instance = instance
        self.opt_sol = opt_sol
        self.metric = metric
        self.greedy = greedy
        self.random = np.random.default_rng(seed)

        self.sample_rate = sample_rate
        self.tree_recorder = TreeRecorder() if sample_rate > 0 else None
        self.transitions = []

        self.receiver_queue = queue.Queue()
        self.requests_queue = requests_queue

        self.penalty = 0
        self.GUB = None
        self.gap = None

        self.iter_count = 0
        self.info = {
            'nnodes': 0,  # ecole.reward.NNodes().cumsum(),
            'lpiters': 0,  # ecole.reward.LpIterations().cumsum(),
            'time': 0,  # ecole.reward.SolvingTime().cumsum()
        }

    def nodeinit(self, *args, **kwargs):
        self.GUB = self.model.getUpperbound()
        self.gap = self.GUB - self.opt_sol
        if self.metric == "gub+" and self.sample_rate > 0:
            self.sample_rate = 1

    def nodecomp(self, node1, node2):
        if node1.getParent() != node2.getParent(): return 0

        state1, state2 = extract_MLP_state(self.model, node1, node2)
        state = (th.tensor(state1, dtype=th.float32),
                 th.tensor(state2, dtype=th.float32))

        # send out policy requests
        self.requests_queue.put({'state': state,
                                 'greedy': self.greedy,
                                 'receiver': self.receiver_queue})
        action = self.receiver_queue.get()  # LEFT:0, RIGHT:1
        reward = 0

        if self.metric == "nnodes":  # For global tree size
            self.penalty = self.model.getNNodes()
        elif self.metric == "lb-obj":  # For optimality-bound penalty
            lower_bound = self.model.getCurrentNode().getLowerbound()
            self.penalty += self.model.isGT(lower_bound, self.opt_sol)
            reward = -self.model.isGT(lower_bound, self.opt_sol)
        elif self.metric == "gub+":  # For primal bound improvement
            GUB = self.model.getUpperbound()
            reward = (self.GUB - GUB) / self.gap
            self.GUB = GUB

        # collect transition samples if requested
        if self.sample_rate > 0:
            focus_node = self.model.getCurrentNode()
            self.tree_recorder.record_decision(focus_node)
            if self.random.random() < self.sample_rate or reward != 0:
                node_number = focus_node.getNumber()
                self.transitions.append({'state': state,
                                         'action': action,
                                         'reward': reward,
                                         'penalty': self.penalty,
                                         'node_id': node_number,
                                         })

        self.info.update({
            'nnodes': self.model.getNNodes(),
            'lpiters': self.model.getNLPIterations(),
            'time': self.model.getSolvingTime()
        })

        if (self.model.isGT(node1.getLowerbound(), self.opt_sol) and
                self.model.isGT(node2.getLowerbound(), self.opt_sol)):
            self.penalty -= 1

        self.iter_count += 1
        # avoid too large trees for stability
        if self.iter_count > 25000:
            self.model.interruptSolve()

        return 1 if action > 0.5 else -1


class TreeRecorder:
    """
    Records the branch-and-bound tree from a custom brancher.

    Every node in SCIP has a unique node ID. We identify nodes and their corresponding
    attributes through the same ID system.
    Depth groups keep track of groups of nodes at the same depth. This data structure
    is used to speed up the computation of the subtree size.
    """

    def __init__(self):
        self.tree = {}
        self.depth_groups = []

    def record_decision(self, focus_node):
        parent_node = focus_node.getParent()
        node_number = focus_node.getNumber()
        parent_number = (0 if node_number == 1 else
                         parent_node.getNumber())
        self.tree[node_number] = parent_number
        # Add to corresponding depth group
        depth = focus_node.getDepth()
        if len(self.depth_groups) > depth:
            self.depth_groups[depth].append(node_number)
        else:
            self.depth_groups.append([node_number])

    def calculate_subtree_sizes(self):
        subtree_sizes = {node_number: 0 for node_number in self.tree.keys()}
        for group in self.depth_groups[::-1]:  # [::-1] reverses the list
            for node_number in group:
                subtree_sizes[node_number] += 1
                if node_number > 1:
                    parent_number = self.tree[node_number]
                    subtree_sizes[parent_number] += subtree_sizes[node_number]
        return subtree_sizes
