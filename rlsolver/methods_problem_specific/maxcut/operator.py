# operator.py
import random
from .utils import BucketSort, TabuList, compute_cut_value

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

def local_search_one_step(G, cut, bucket: BucketSort, tabu: TabuList, curr_val, best_val, iteration):


    max_nodes = bucket.get_max_nodes()
    best_v = None
    for v in max_nodes:
        if tabu.is_allowed(v, iteration, aspiration=True,
                          best_val=best_val,
                          move_gain=bucket.node_gain[v],
                          curr_val=curr_val):
            best_v = v
            break
    if best_v is None:
        return False, curr_val


    gain = bucket.node_gain[best_v]
    cut[best_v] = not cut[best_v]
    curr_val += gain


    bucket.update(best_v, -gain)
    for u in G[best_v]:
        g_u = compute_gain(G, cut, u)
        bucket.update(u, g_u)

    return True, curr_val

def perturb_operator(G, cut, bucket: BucketSort, tabu: TabuList,
                     L, move_set, curr_val, iteration, phi_min, phi_max):

    for _ in range(L):
        v = random.choice(move_set)
        gain = compute_gain(G, cut, v)
        cut[v] = not cut[v]
        curr_val += gain


        tenure = random.randint(phi_min, phi_max)
        tabu.forbid(v, iteration, tenure)


        bucket.update(v, -gain)
        for u in G[v]:
            g_u = compute_gain(G, cut, u)
            bucket.update(u, g_u)

        iteration += 1

    return curr_val, iteration
