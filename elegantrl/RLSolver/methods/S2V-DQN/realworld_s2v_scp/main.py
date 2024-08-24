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

sys.path.append( '%s/../memetracker' % os.path.dirname(os.path.realpath(__file__)) )
from meme import *

nvalid = 100

def gen_new_graphs(opt):
    print 'generating new training graphs'
    sys.stdout.flush()
    api.ClearTrainGraphs()
    for i in tqdm(range(100)):
        g = get_scp_graph(g_undirected, float(opt['pq']))
        api.InsertGraph(g, is_test=False)

def PrepareValidData(opt):
    print 'generating validation graphs'
    sys.stdout.flush()
    f = open(opt['data_test'], 'rb')
    for i in tqdm(range(nvalid)):
        g = cp.load(f)
        api.InsertGraph(g, is_test=True)

if __name__ == '__main__':
    api = SetCoverLib(sys.argv)
    
    opt = {}
    for i in range(1, len(sys.argv), 2):
        opt[sys.argv[i][1:]] = sys.argv[i + 1]

    g_undirected, _ = build_full_graph('%s/InfoNet5000Q1000NEXP.txt' % opt['data_root'],'undirected')

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
            for idx in range(nvalid):
                frac += api.lib.Test(idx)
            print 'iter', iter, 'eps', eps, 'average size of cover: ', frac / nvalid
            sys.stdout.flush()
            model_path = '%s/pq-%s_iter_%d.model' % (opt['save_dir'], opt['pq'], iter)
            api.SaveModel(model_path)

        if iter % 500 == 0:
            api.TakeSnapshot()

        api.lib.Fit()
