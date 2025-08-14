# utils.py
import random
from collections import defaultdict

class BucketSort:

    def __init__(self, gains=None):
        # gains: dict node->gain
        self.node_gain = {}
        self.buckets = defaultdict(set)
        self.maxgain = None
        if gains:
            self.reset(gains)

    def reset(self, gains):

        self.node_gain = gains.copy()
        self.buckets.clear()
        for v, g in gains.items():
            self.buckets[g].add(v)
        self.maxgain = max(self.buckets) if self.buckets else 0

    def update(self, v, new_gain):

        old = self.node_gain[v]
        self.buckets[old].remove(v)
        if not self.buckets[old]:
            del self.buckets[old]
            if old == self.maxgain:
                self.maxgain = max(self.buckets) if self.buckets else 0

        self.node_gain[v] = new_gain
        self.buckets[new_gain].add(v)
        if self.maxgain is None or new_gain > self.maxgain:
            self.maxgain = new_gain

    def get_max_nodes(self):

        if not self.buckets:
            return set()
        return self.buckets[self.maxgain]

class TabuList:

    def __init__(self):
        self.expire = {}  # node -> iteration index when tabu 解除

    def forbid(self, v, current_iter, tenure):
        self.expire[v] = current_iter + tenure

    def is_allowed(self, v, current_iter, aspiration=False, best_val=None, move_gain=None, curr_val=None):

        if current_iter >= self.expire.get(v, 0):
            return True
        if aspiration and best_val is not None and move_gain is not None and curr_val is not None:
            return (curr_val + move_gain) > best_val
        return False

def compute_cut_value(G, cut):

    val = 0
    for u, v, data in G.edges(data=True):
        if cut[u] != cut[v]:
            val += data.get("weight", 1)
    return val

def compute_gain(G, cut, v):
    delta = 0
    side = cut[v]
    for u in G[v]:
        w = G[v][u].get("weight", 1)
        if cut[u] == side:
            delta += w
        else:
            delta -= w
    return delta