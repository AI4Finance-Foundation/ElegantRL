import numpy as np
import networkx as nx
import cPickle as cp
import random
import ctypes
import os
import sys
from tqdm import tqdm

sys.path.append( '%s/setcover_lib' % os.path.dirname(os.path.realpath(__file__)) )
from setcover_lib import SetCoverLib

n_valid = 100

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

def gen_new_graphs(opt):
    print 'generating new training graphs'
    sys.stdout.flush()
    api.ClearTrainGraphs()
    for i in tqdm(range(1000)):
        g = gen_setcover_inst(opt)
        api.InsertGraph(g, is_test=False)

def PrepareValidData(opt):
    print 'load validation graphs'
    sys.stdout.flush()
    for i in tqdm(range(n_valid)):
        g = gen_setcover_inst(opt)
        api.InsertGraph(g, is_test=True)

if __name__ == '__main__':
    api = SetCoverLib(sys.argv)
    print(sys.argv)
    opt = {}
    for i in range(1, len(sys.argv), 2):
        opt[sys.argv[i][1:]] = sys.argv[i + 1]
    
    PrepareValidData(opt)

    # startup
    gen_new_graphs(opt)
    for i in range(10):
        api.lib.PlayGame(100, ctypes.c_double(1.0))
    api.TakeSnapshot()

    eps_start = 1.0
    eps_end = 0.05
    eps_step = 10000.0
    for iter in range(int(opt['max_iter'])):
        if iter and iter % 5000 == 0:
            gen_new_graphs(opt)
        eps = eps_end + max(0., (eps_start - eps_end) * (eps_step - iter) / eps_step)
        if iter % 10 == 0:
            api.lib.PlayGame(10, ctypes.c_double(eps))

        if iter % 300 == 0:
            frac = 0.0
            for idx in range(n_valid):
                frac += api.lib.Test(idx)
            print 'iter', iter, 'eps', eps, 'average size of cover: ', frac / n_valid
            sys.stdout.flush()
            model_path = '%s/nrange_%d_%d_iter_%d.model' % (opt['save_dir'], int(opt['min_n']), int(opt['max_n']), iter)
            api.SaveModel(model_path)

        if iter % 500 == 0:
            api.TakeSnapshot()

        api.lib.Fit()
