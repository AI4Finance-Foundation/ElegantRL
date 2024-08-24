import sys
import numpy as np
import networkx as nx
import cPickle as cp
import random
import ctypes
import os
from tqdm import tqdm

sys.path.append( '%s/maxcut_lib' % os.path.dirname(os.path.realpath(__file__)) )
from maxcut_lib import MaxcutLib

n_valid = 10

file_list = ['sg3dl051000.mc', 
             'sg3dl052000.mc',
             'sg3dl053000.mc',
             'sg3dl054000.mc',
             'sg3dl055000.mc',
             'sg3dl056000.mc',
             'sg3dl057000.mc',
             'sg3dl058000.mc',
             'sg3dl059000.mc',
             'sg3dl0510000.mc']

def gen_graph(idx = -1, rand_weight=False):
    if idx < 0:
        idx = np.random.randint(len(file_list))
    fname = '%s/optsicom/%s' % (opt['data_root'], file_list[idx])
    g = nx.Graph()

    with open(fname, 'r') as f:
        line = f.readline()
        num_nodes, num_edges = [int(w) for w in line.strip().split(' ')]
        for row in f:
            x, y, w = [int(t) for t in row.strip().split(' ')]
            x -= 1
            y -= 1
            w = float(w)
            if rand_weight:
                t = np.random.randn() * 0.1
                w += t
            g.add_edge(x, y, weight=w)
    
    assert len(g) == num_nodes
    assert len(g.edges()) == num_edges
    return g

def gen_new_graphs(opt):
    print 'generating new training graphs'
    api.ClearTrainGraphs()
    for i in range(1000):
        g = gen_graph(rand_weight=True)
        api.InsertGraph(g, is_test=False)

def PrepareValidData(opt):
    print 'generating validation graphs'
    for i in range(n_valid):
        g = gen_graph(i)
        api.InsertGraph(g, is_test=True)

if __name__ == '__main__':
    api = MaxcutLib(sys.argv)
    
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
                frac += api.lib.TestNoStop(idx)
            print 'iter', iter, 'eps', eps, 'average cut size: ', frac / n_valid
            sys.stdout.flush()
            model_path = '%s/nrange_%d_%d_iter_%d.model' % (opt['save_dir'], int(opt['min_n']), int(opt['max_n']), iter)
            api.SaveModel(model_path)

        if iter % 1000 == 0:
            api.TakeSnapshot()

        api.lib.Fit()
