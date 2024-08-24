import numpy as np
import networkx as nx
import cPickle as cp
import random
import ctypes
import os
import sys
import time
from tqdm import tqdm

sys.path.append( '%s/tsp2d_lib' % os.path.dirname(os.path.realpath(__file__)) )
from tsp2d_lib import Tsp2dLib
    
def find_model_file(opt):
    max_n = int(opt['max_n'])
    min_n = int(opt['min_n'])
    log_file = '%s/log-%d-%d.txt' % (opt['save_dir'], min_n, max_n)

    best_r = 10000000
    best_it = -1
    with open(log_file, 'r') as f:
        for line in f:
            if 'average' in line:
                line = line.split(' ')
                it = int(line[1].strip())
                r = float(line[-1].strip())
                if r < best_r:
                    best_r = r
                    best_it = it
    assert best_it >= 0
    print 'using iter=', best_it, 'with r=', best_r
    return '%s/nrange_%d_%d_iter_%d.model' % (opt['save_dir'], min_n, max_n, best_it)

def TestSet():
    folder = '%s/test_tsp2d/tsp_min-n=%s_max-n=%s_num-graph=1000_type=%s' % (opt['data_root'], opt['test_min_n'], opt['test_max_n'], opt['g_type'])

    with open('%s/paths.txt' % folder, 'r') as f:
        for line in f:
            fname = '%s/%s' % (folder, line.split('/')[-1].strip())
            coors = {}
            in_sec = False
            n_nodes = -1
            with open(fname, 'r') as f_tsp:
                for l in f_tsp:
                    if 'DIMENSION' in l:
                        n_nodes = int(l.split(' ')[-1].strip())
                    if in_sec:
                        idx, x, y = [int(w.strip()) for w in l.split(' ')]
                        coors[idx - 1] = [float(x) / 1000000.0, float(y) / 1000000.0]
                        assert len(coors) == idx
                    elif 'NODE_COORD_SECTION' in l:
                        in_sec = True
            assert len(coors) == n_nodes
            g = nx.Graph()
            g.add_nodes_from(range(n_nodes))
            nx.set_node_attributes(g, 'pos', coors)
            yield g            

if __name__ == '__main__':
    api = Tsp2dLib(sys.argv)
    
    opt = {}
    for i in range(1, len(sys.argv), 2):
        opt[sys.argv[i][1:]] = sys.argv[i + 1]

    model_file = find_model_file(opt)
    assert model_file is not None
    print 'loading', model_file
    sys.stdout.flush()
    api.LoadModel(model_file)

    test_name = '-'.join([opt['g_type'], opt['test_min_n'], opt['test_max_n']])
    result_file = '%s/test-%s-gnn-%s-%s.csv' % (opt['save_dir'], test_name, opt['min_n'], opt['max_n'])

    n_test = 1000
    frac = 0.0
    with open(result_file, 'w') as f_out:
        print 'testing'
        sys.stdout.flush()
        idx = 0
        for g in tqdm(TestSet()):
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

    print 'average tour length: ', frac / n_test
