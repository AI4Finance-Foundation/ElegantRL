import sys
sys.path.insert(0, '/nv/hcoc1/hdai8/data/local/lib/python2.7/site-packages')
import numpy as np
import networkx as nx
import cPickle as cp
import random
import ctypes
import os
import time
from tqdm import tqdm

sys.path.append( '%s/maxcut_lib' % os.path.dirname(os.path.realpath(__file__)) )
from maxcut_lib import MaxcutLib
    
def find_model_file(opt):
    max_n = int(opt['max_n'])
    min_n = int(opt['min_n'])
    log_file = '%s/log-%d-%d.txt' % (opt['save_dir'], min_n, max_n)

    best_r = -10000000
    best_it = -1
    with open(log_file, 'r') as f:
        for line in f:
            if 'average' in line:
                line = line.split(' ')
                it = int(line[1].strip())
                r = float(line[-1].strip())
                if r > best_r:
                    best_r = r
                    best_it = it
    assert best_it >= 0
    print 'using iter=', best_it, 'with r=', best_r
    return '%s/nrange_%d_%d_iter_%d.model' % (opt['save_dir'], min_n, max_n, best_it)

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

def gen_graph(idx):
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
            g.add_edge(x, y, weight=w)
    
    assert len(g) == num_nodes
    assert len(g.edges()) == num_edges
    return g

if __name__ == '__main__':
    api = MaxcutLib(sys.argv)
    
    opt = {}
    for i in range(1, len(sys.argv), 2):
        opt[sys.argv[i][1:]] = sys.argv[i + 1]

    model_file = find_model_file(opt)
    assert model_file is not None
    print 'loading', model_file
    sys.stdout.flush()
    api.LoadModel(model_file)

    test_name = 'optsicom'
    result_file = '%s/test-%s-gnn-%s-%s.csv' % (opt['save_dir'], test_name, opt['min_n'], opt['max_n'])

    frac = 0.0
    with open(result_file, 'w') as f_out:
        print 'testing'
        sys.stdout.flush()
        idx = 0
        for idx in tqdm(range(len(file_list))):
            g = gen_graph(idx)
            api.InsertGraph(g, is_test=True)
            t1 = time.time()
            val, sol = api.GetSol(idx, nx.number_of_nodes(g))
            t2 = time.time()
            f_out.write('%.8f,' % val)
            f_out.write('%d' % sol[0])
            for i in range(sol[0]):
                f_out.write(' %d' % sol[i + 1])
            f_out.write(',%.6f\n' % (t2 - t1))
            frac += val
            idx += 1

    print 'average cut size: ', frac / len(file_list)
