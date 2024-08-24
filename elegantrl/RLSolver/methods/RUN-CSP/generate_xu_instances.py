import numpy as np
import itertools
import scipy.sparse as sp
import csp_utils
import random
import networkx as nx


def generate_instance(n, k, r, p):
    a = np.log(k) / np.log(n)
    v = k * n
    s = int(p * (n ** (2 * a)))
    iterations = int(r * n * np.log(n) - 1)

    parts = np.reshape(np.int64(range(v)), (n, k))
    nand_clauses = []

    for i in parts:
        nand_clauses += itertools.combinations(i, 2)

    edges = set()
    for _ in range(iterations):
        i, j = np.random.choice(n, 2, replace=False)
        all = set(itertools.product(parts[i, :], parts[j, :]))
        all -= edges
        edges |= set(random.sample(tuple(all), k=min(s, len(all))))

    nand_clauses += list(edges)
    clauses = {'NAND': nand_clauses}

    instance = csp_utils.CSP_Instance(language=csp_utils.is_language,
                                      n_variables=v,
                                      clauses=clauses)
    return instance


def get_random_instance():
    n = np.random.randint(10, 26)
    k = np.random.randint(5, 21)
    p = np.random.uniform(0.3, 1.0)
    a = np.log(k) / np.log(n)
    r = - a / np.log(1 - p)

    i = generate_instance(n, k, r, p)
    G = nx.Graph()
    G.add_edges_from(i.clauses['NAND'])

    return G