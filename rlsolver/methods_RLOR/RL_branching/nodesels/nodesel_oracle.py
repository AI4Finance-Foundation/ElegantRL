import gzip
import pickle
# import extract
from..extract import extract_MLP_state
import numpy as np

from nodesels.nodesel_baseline import NodeselEstimate


class BaseSampler:
    def __init__(self, episode, tmp_dir, out_queue):
        self.episode = episode
        self.tmp_dir = tmp_dir
        self.out_queue = out_queue
        self.action_count = [0, 0]
        self.sample_count = 0

    def write_sample(self, state1, state2, action):
        filename = self.tmp_dir + f'/sample_{self.episode}_{self.sample_count}.pkl'
        with gzip.open(filename, 'wb') as f:
            f.write(pickle.dumps({
                'state1': state1,
                'state2': state2,
                'action': action,
            }))

        self.out_queue.put({
            'type': "sample",
            'episode': self.episode,
            'filename': filename,
        })

        self.action_count[action] += 1
        self.sample_count += 1

    def create_sample(self, state1, state2, action):
        # If the statistics functionality is removed
        # from write_sample(), move it here instead
        self.write_sample(state1, state2, action)


class RandomSampler(BaseSampler):
    def __init__(self, episode, tmp_dir, out_queue, seed):
        super().__init__(episode, tmp_dir, out_queue)
        self.random = np.random.default_rng(seed)

    def create_sample(self, state1, state2, action):
        if self.random.random() < 0.5:
            self.write_sample(state1, state2, action)
        else:
            self.write_sample(state2, state1, 1 - action)


class DoubleSampler(BaseSampler):
    def __init__(self, episode, tmp_dir, out_queue):
        super().__init__(episode, tmp_dir, out_queue)

    def create_sample(self, state1, state2, action):
        self.write_sample(state1, state2, action)
        self.write_sample(state2, state1, 1 - action)


# This class contains the sampler.
# Valid states reached by the BestEstimate selector are saved.
class NodeselOracle(NodeselEstimate):
    def __init__(self, sampler, sampling, solutions):
        super().__init__()
        self.sampler = sampler
        self.sampling = sampling
        self.solutions = solutions

        # root node contains all solutions
        # split the solutions at each node
        self.k_sols = len(solutions)
        indices = list(range(self.k_sols))
        self.sol_indices = {1: indices}

    def nodeselect(self):
        # Stop sampling after 5000 nodes
        if self.model.getNNodes() > 5000:
            print("early stopping")
            self.model.interruptSolve()
        if self.sampling == "Children":
            self.model.getBestChild()
        elif self.sampling == "Nodes":
            self.model.getBestNode()
        return super().nodeselect()
        # selnode = super().nodeselect()
        # print(f"Chose: {selnode['selnode'].getNumber()}")
        # print("*** ====================== ***")
        # return selnode

    def nodecomp(self, node1, node2):
        siblings = node1.getParent() == node2.getParent()
        if self.sampling == "Children" and not siblings:
            return super().nodecomp(node1, node2)

        sol_rank = [self.k_sols, self.k_sols]
        for node_index, node in enumerate([node1, node2]):
            node_number = node.getNumber()
            if node_number in self.sol_indices:
                sol_index = self.sol_indices[node_number][0]
                sol_rank[node_index] = sol_index
                continue
            # If the parent node contained the optimal sol,
            # it is sufficient to only check the new bounds
            parent_number = node.getParent().getNumber()
            if parent_number not in self.sol_indices: continue
            branchings = node.getParentBranchings()
            for sol_index in self.sol_indices[parent_number]:
                sol = self.solutions[sol_index]
                for bvar, bbound, btype in zip(*branchings):
                    if btype == 0 and sol[bvar] < bbound: break  # EXCEEDS LOWER BOUND
                    if btype == 1 and sol[bvar] > bbound: break  # EXCEEDS UPPER BOUND
                else:                                            # SATISFIES ALL BOUNDS
                    if node_number not in self.sol_indices:
                        self.sol_indices[node_number] = []
                        sol_rank[node_index] = sol_index
                    self.sol_indices[node_number].append(sol_index)

        # if self.sampling == "Children":
        #     # My children have been processed
        #     # My work here is done; goodbye.
        #     node = self.model.getCurrentNode()
        #     node_number = node.getNumber()
        #     del self.sol_indices[node_number]

        if sol_rank[0] == sol_rank[1]:
            return super().nodecomp(node1, node2)
        action = int(sol_rank[1] < sol_rank[0])
        print(f"| Node {node1.getNumber()}: {sol_rank[0]} "
              f"| Node {node2.getNumber()}: {sol_rank[1]} "
              f"| Action: {['left', 'right'][action]}")

        state = extract_MLP_state(self.model, node1, node2)
        self.sampler.create_sample(*state, action)

        return super().nodecomp(node1, node2)
