import numpy as np
import networkx as nx
import cPickle as cp
import random
import ctypes
import os
import sys
from tqdm import tqdm

sys.path.append( '%s/tsp2d_lib' % os.path.dirname(os.path.realpath(__file__)) )
from tsp2d_lib import Tsp2dLib

def PrepareGraphs(fname, isTest):
    if not isTest:
        api.ClearTrainGraphs()
    norm = 1.0
    with open(fname, 'r') as f_tsp:
        coors = {}
        in_sec = False
        n_nodes = -1
        for l in f_tsp:
            if 'DIMENSION' in l:
                n_nodes = int(l.strip().split(' ')[-1].strip())
            if in_sec:
                idx, x, y = [w.strip() for w in l.strip().split(' ') if len(w.strip())]
                idx = int(idx)
		if np.fabs(float(x)) > norm:
		    norm = np.fabs(float(x))
		if np.fabs(float(y)) > norm:
		    norm = np.fabs(float(y))
                coors[idx - 1] = [float(x), float(y)]
                assert len(coors) == idx
                if len(coors) == n_nodes:
                    break
            elif 'NODE_COORD_SECTION' in l:
                in_sec = True

	for i in coors:
	    coors[i][0] /= norm
	    coors[i][1] /= norm
        assert len(coors) == n_nodes
        g = nx.Graph()
        g.add_nodes_from(range(n_nodes))
        nx.set_node_attributes(g, 'pos', coors)
        api.InsertGraph(g, is_test=isTest)

if __name__ == '__main__':
    api = Tsp2dLib(sys.argv)
    
    opt = {}
    for i in range(1, len(sys.argv), 2):
        opt[sys.argv[i][1:]] = sys.argv[i + 1]

    fname = '%s/%s/%s' % (opt['data_root'], opt['folder'], opt['sample_name'])
    PrepareGraphs(fname, isTest=True)
    PrepareGraphs(fname, isTest=False)

    # startup    
    for i in range(10):
        api.lib.PlayGame(100, ctypes.c_double(1.0))
    api.TakeSnapshot()

    eps_start = 1.0
    eps_end = 1.0

    eps_step = 10000.0
    api.lib.SetSign(1)

    lr = float(opt['learning_rate'])
    for iter in range(int(opt['max_iter'])):
        eps = eps_end + max(0., (eps_start - eps_end) * (eps_step - iter) / eps_step)
        if iter % 10 == 0:
            api.lib.PlayGame(10, ctypes.c_double(eps))

        if iter % 100 == 0:
            frac = api.lib.Test(0)
            print 'iter', iter, 'lr', lr, 'eps', eps, 'average tour length: ', frac
            sys.stdout.flush()
            model_path = '%s/%s_iter_%d.model' % (opt['save_dir'], opt['sample_name'], iter)
            api.SaveModel(model_path)

        if iter % 1000 == 0:
            api.TakeSnapshot()
            lr = lr * 0.95

        api.lib.Fit(ctypes.c_double(lr))
