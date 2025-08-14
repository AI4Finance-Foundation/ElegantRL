
import time
import csv
import random
import math
from pathlib import Path

import networkx as nx
from .operator import local_search_one_step, perturb_operator
from .perturbation import choose_perturbation
from .utils import BucketSort, TabuList, compute_cut_value, compute_gain


class BLSMaxCut:
    def __init__(self, G, params):
        self.G = G
        self.n = G.number_of_nodes()
        # Algorithm parameters
        self.L0 = max(1, int(params['L0_ratio'] * self.n))
        self.T = params['T']
        self.phi_min = params['phi_min']
        self.phi_max = max(1, int(params['phi_max_ratio'] * self.n))
        self.P0 = params['P0']
        self.Q = params['Q']
        self.max_iters = params.get('max_iters', 200000 * self.n)

        # Current and best solutions
        self.cut = {}
        self.best_cut = {}
        self.curr_val = 0
        self.best_val = float('-inf')
        # Counters
        self.omega = 0       # non-improving attractor count
        self.iteration = 0   # global iteration count

        # Auxiliary structures
        self.tabu = TabuList()
        self.bucket = None

    def initialize(self):
        # Random initial partition
        for v in self.G:
            self.cut[v] = random.choice([True, False])
        # Compute initial objective
        self.curr_val = compute_cut_value(self.G, self.cut)
        self.best_val = self.curr_val
        self.best_cut = self.cut.copy()
        # Compute gains and build bucket
        gains = {v: compute_gain(self.G, self.cut, v) for v in self.G}
        self.bucket = BucketSort(gains)
        # Reset tabu
        self.tabu = TabuList()

    def run(self, target=None, time_limit=None):

        self.initialize()
        t_start = time.time()
        print(f"Start BLS: |V|={self.n}, max_iters={self.max_iters}")

        while self.iteration < self.max_iters:
            # Early stop on time limit
            if time_limit is not None and time.time() - t_start > time_limit:
                break
            self.iteration += 1

            # (Optional) status print
            if self.iteration % 100 == 0:
                print(f"  iter={self.iteration}, curr={self.curr_val}, best={self.best_val}")

            # Local search step
            improved, self.curr_val = local_search_one_step(
                self.G, self.cut, self.bucket, self.tabu,
                self.curr_val, self.best_val, self.iteration)

            # Update best if improved
            if self.curr_val > self.best_val:
                self.best_val = self.curr_val
                self.best_cut = self.cut.copy()
                self.omega = 0
                # Early stop on target
                if target is not None and self.best_val >= target:
                    break
            else:
                self.omega += 1

            # Perturbation if no improvement
            if not improved:
                mode = choose_perturbation(self.omega, self.T, self.P0, self.Q)
                if mode == 'random':
                    M = list(self.G.nodes())
                else:
                    M = list(self.bucket.get_max_nodes())
                L = self.L0 + (1 if self.omega > self.T or self.cut == self.best_cut else 0)
                self.curr_val, self.iteration = perturb_operator(
                    self.G, self.cut, self.bucket, self.tabu,
                    L, M, self.curr_val, self.iteration,
                    self.phi_min, self.phi_max)

        return self.best_cut, self.best_val

