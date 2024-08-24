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

def get_num_primal_dual(g):
    n_primal = 0
    n_dual = 0
    for i in range(nx.number_of_nodes(g)):
        if g.node[i]['bipartite'] == 0:
            n_primal += 1
        else:
            n_dual += 1
    return n_primal, n_dual

def cplex_instance(g, var_type):
    n_primal, n_dual = get_num_primal_dual(g)

    variable_names = []
    for node in range(n_primal):
        variable_names.append("x" + str(node))

    rows = []
    for i in range(n_primal, n_primal + n_dual):
        vars = []
        for j in nx.neighbors(g, i):
            vars.append("x" + str(j))
        coeff = [1] * len(vars)
        rows.append([vars, coeff])

    objective_coeffs = [1] * n_primal
    variable_types = var_type * n_primal
    variable_ub = [1] * n_primal
    constraint_rhs = [1] * n_dual
    constraint_senses = "G" * n_dual

    prob = cplex.Cplex()
    prob.objective.set_sense(prob.objective.sense.minimize)
    prob.variables.add(obj=objective_coeffs, ub=variable_ub, types=variable_types, names=variable_names)
    prob.linear_constraints.add(lin_expr=rows, senses=constraint_senses, rhs=constraint_rhs)

    prob.set_log_stream(None)
    prob.set_error_stream(None)
    prob.set_warning_stream(None)
    prob.set_results_stream(None)
    prob.parameters.timelimit.set(3600)
    prob.parameters.workmem.set(2000)

    return prob

def solve_ip(g):
    ip = cplex_instance(g, 'B')
    ip.solve()
    solution = ip.solution.get_values()
    obj = ip.solution.get_objective_value()
    return obj

if __name__ == '__main__':
    opt = {}
    for i in range(1, len(sys.argv), 2):
        opt[sys.argv[i][1:]] = sys.argv[i + 1]

    g_undirected, _ = build_full_graph('%s/InfoNet5000Q1000NEXP.txt' % opt['data_root'],'undirected')

    # sol_list = []
    with open('%s/meme-ntest-%s-pq-%s.pkl' % (opt['out_root'], opt['num'], opt['pq']), 'wb') as f:
        for i in tqdm(range(int(opt['num']))):
            g = get_scp_graph(g_undirected, float(opt['pq']))
            print len(g)
            cp.dump(g, f, cp.HIGHEST_PROTOCOL)
            # sol_list.append(solve_ip(g))
    
    # with open('%s/meme-opt-ntest-%s-pq-%s.txt' % (opt['out_root'], opt['num'], opt['pq']), 'w') as f:
    #     for i in range(len(sol_list)):
    #         f.write('%.2f\n' % sol_list[i])
        
