import os
import sys
import cPickle as cp
import random
import numpy as np
import networkx as nx
# from tqdm import tqdm

def gen_setcover_inst(opt):
    min_n = int(opt['min_n'])
    max_n = int(opt['max_n'])
    frac_primal = float(opt['frac_primal'])
    p = float(opt['edge_prob'])

    cur_n = np.random.randint(max_n - min_n + 1) + min_n
    num_primal = int(cur_n * frac_primal)
    num_dual = cur_n - num_primal

    a = range(num_primal)
    b = range(num_primal, num_dual + num_primal)

    g = nx.Graph()
    g.add_nodes_from(a, bipartite=0)
    g.add_nodes_from(b, bipartite=1)

    colHasOneBool = [0]*num_primal

    for i in range(num_dual):
        # guarantee that each element is in at least 2 sets, based on http://link.springer.com/chapter/10.1007%2FBFb0120886#page-1
        k1 = np.random.randint(num_primal)
        g.add_edge(k1, i + num_primal)
        k2 = np.random.randint(num_primal)
        g.add_edge(k2, i + num_primal)
        for j in range(num_primal):
            if j == k1 or j == k2:
                continue
            r = np.random.rand()
            if r < p:
                g.add_edge(j, i + num_primal)
                colHasOneBool[j] = 1

    # guarantee that each set has at least 1 element, based on http://link.springer.com/chapter/10.1007%2FBFb0120886#page-1
    for j in range(num_primal):
        if colHasOneBool[j] == 0:
            randrow = np.random.randint(num_dual)
            g.add_edge(j, randrow + num_primal)
    
    return g

if __name__ == '__main__':
    opt = {}
    for i in range(1, len(sys.argv), 2):
        opt[sys.argv[i][1:]] = sys.argv[i + 1]

    num_graph = int(opt['num_graph'])

    if 'seed' not in opt:
        seed = 1
    else:
        seed = int(opt['seed'])
    np.random.seed(seed=seed)

    with open('%s/nrange-%s-%s-n_graph-%s-p-%s-frac_primal-%s-seed-%s.pkl' % (opt['save_dir'], opt['min_n'], opt['max_n'], opt['num_graph'], opt['edge_prob'], opt['frac_primal'], seed), 'wb') as fout:
        for i in range(num_graph):
            g = gen_setcover_inst(opt)
            cp.dump(g, fout, cp.HIGHEST_PROTOCOL)