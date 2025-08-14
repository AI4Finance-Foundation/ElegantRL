# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
# Generates MILP instances of one of the following types: config['problems']    #
# Four directories are created: train, test, valid, and transfer.               #
# Usage: python 01_generate_instances.py <problem> -s <seed>                    #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #

import os
import json
import argparse
import util
from util import valid_seed
import numpy as np
import scipy as sp
import networkx as nx
from tqdm import trange
from itertools import combinations


class Graph:
    """
    Container for a graph.

    Parameters
    ----------
    n_nodes : int
        The number of nodes in the graph.
    edges : set of tuples (int, int)
        The edges of the graph, where the integers refer to the nodes.
    degrees : numpy array of integers
        The degrees of the nodes in the graph.
    neighbors : dictionary of type {int: set of ints}
        The neighbors of each node in the graph.
    """

    def __init__(self, n_nodes, edges, degrees, neighbors):
        self.n_nodes = n_nodes
        self.edges = edges
        self.degrees = degrees
        self.neighbors = neighbors

    def __len__(self):
        return self.n_nodes

    def __iter__(self):
        return iter(range(self.n_nodes))

    def greedy_clique_partition(self):
        """
        Partition the graph into cliques using a greedy algorithm.

        Returns
        -------
        list of sets
            The resulting clique partition.
        """
        cliques = []
        leftover_nodes = (-self.degrees).argsort().tolist()

        while leftover_nodes:
            clique_center, leftover_nodes = leftover_nodes[0], leftover_nodes[1:]
            clique = {clique_center}
            neighbors = self.neighbors[clique_center].intersection(leftover_nodes)
            densest_neighbors = sorted(neighbors, key=lambda x: -self.degrees[x])
            for neighbor in densest_neighbors:
                # Can you add it to the clique, and maintain clique-ness?
                if all([neighbor in self.neighbors[clique_node] for clique_node in clique]):
                    clique.add(neighbor)
            cliques.append(clique)
            leftover_nodes = [node for node in leftover_nodes if node not in clique]

        return cliques

    @staticmethod
    def erdos_renyi(n_nodes, edge_probability, random):
        """
        Generate an Erdös-Rényi random graph with a given edge probability.

        Parameters
        ----------
        n_nodes : int
            The number of nodes in the graph.
        edge_probability : float in [0,1]
            The probability of generating each edge.
        random : np.random.Generator
            A random number generator.

        Returns
        -------
        Graph
            The generated graph.
        """
        edges = set()
        degrees = np.zeros(n_nodes, dtype=int)
        neighbors = {node: set() for node in range(n_nodes)}
        for edge in combinations(np.arange(n_nodes), 2):
            if random.uniform() < edge_probability:
                edges.add(edge)
                degrees[edge[0]] += 1
                degrees[edge[1]] += 1
                neighbors[edge[0]].add(edge[1])
                neighbors[edge[1]].add(edge[0])

        return Graph(n_nodes, edges, degrees, neighbors)

    @staticmethod
    def barabasi_albert(n_nodes, affinity, random):
        """
        Generate a Barabási-Albert random graph with a given edge probability.

        Parameters
        ----------
        n_nodes : int
            The number of nodes in the graph.
        affinity : integer >= 1
            The number of nodes each new node will be attached to, in the sampling scheme.
        random : np.random.Generator
            A random number generator.

        Returns
        -------
        Graph
            The generated graph.
        """
        assert 1 <= affinity < n_nodes

        edges = set()
        degrees = np.zeros(n_nodes, dtype=int)
        neighbors = {node: set() for node in range(n_nodes)}
        for new_node in range(affinity, n_nodes):
            # first node is connected to all previous ones (star-shape)
            if new_node == affinity:
                neighborhood = np.arange(new_node)
            # remaining nodes are picked stochastically
            else:
                neighbor_prob = degrees[:new_node] / (2 * len(edges))
                neighborhood = random.choice(new_node, affinity, replace=False, p=neighbor_prob)
            for node in neighborhood:
                edges.add((node, new_node))
                degrees[node] += 1
                degrees[new_node] += 1
                neighbors[node].add(new_node)
                neighbors[new_node].add(node)

        return Graph(n_nodes, edges, degrees, neighbors)


def generate_indset(graph, filename):
    """
    Generate a Maximum Independent Set (also known as Maximum Stable Set) instance
    in CPLEX LP format from a previously generated graph.

    Parameters
    ----------
    graph : Graph
        The graph from which to build the independent set problem.
    filename : str
        Path to the file to save.
    """
    cliques = graph.greedy_clique_partition()
    inequalities = set(graph.edges)
    for clique in cliques:
        clique = tuple(sorted(clique))
        for edge in combinations(clique, 2):
            inequalities.remove(edge)
        if len(clique) > 1:
            inequalities.add(clique)

    # Put trivial inequalities for nodes that didn't appear
    # in the constraints, otherwise SCIP will complain
    used_nodes = set()
    for group in inequalities:
        used_nodes.update(group)
    for node in range(10):
        if node not in used_nodes:
            inequalities.add((node,))

    with open(filename, 'w') as lp_file:
        lp_file.write("maximize\nOBJ: ")
        lp_file.write(" + ".join([f"x{node}" for node in graph]) + "\n")
        lp_file.write("\nsubject to\n")
        for count, group in enumerate(inequalities):
            lp_file.write(f"C{count}: " + " + ".join([f"x{node}" for node in sorted(group)]) + " <= 1\n")
        lp_file.write("\nbinary\n" + " ".join([f"x{node}" for node in graph]) + "\n")

def generate_general_indset(graph, filename, alphaE2, random):
    """
    Generate a General Independent Set instance
    in CPLEX LP format from a previously generated graph.

    Parameters
    ----------
    graph : Graph
        The graph from which to build the independent set problem.
    filename : str
        Path to the file to save.
    alphaE2 : float
        The probability of an edge being removable
    random : np.random.Generator
        A random number generator.
    """
    # Generate the set of removable edges
    E2 = [edge for edge in graph.edges if random.random() < alphaE2]

    # Create IP, write it to file, and solve it with CPLEX
    with open(filename, 'w') as lp_file:
        lp_file.write("maximize\nOBJ: " + " + ".join([f"10x{node}" for node in graph])
                      + "".join([f" - y{node1}_{node2}" for node1, node2 in E2]))
        lp_file.write("\n\nsubject to\n")
        for count, (node1, node2) in enumerate(graph.edges):
            y = f" - y{node1}_{node2}" if (node1, node2) in E2 else ""
            lp_file.write(f"C{count + 1}: x{node1} + x{node2}" + y + " <= 1\n")
        lp_file.write("\nbinary\n" + " ".join([f"x{node}" for node in graph]))
        lp_file.write("".join([f" y{node1}_{node2}" for node1, node2 in E2]))

# -------------------- KNAPSACK GENERATORS ----------------------
def generate_weights_and_values(n_items, random, min_range=10, max_range=20, scheme="weakly correlated"):
    """
    Parameters
    ----------
    n_items : int
        The number of items.
    random : numpy.random.Generator
        A random number generator.
    min_range : int, optional
        The lower range from which to sample the item weights. Default 10.
    max_range : int, optional
        The upper range from which to sample the item weights. Default 20.
    scheme : str, optional
        One of 'uncorrelated', 'weakly correlated', 'strongly correlated', 'subset-sum'. Default 'weakly correlated'.
    """
    weights = random.integers(min_range, max_range, n_items)

    if scheme == "subset-sum":
        values = weights
    elif scheme == "uncorrelated":
        values = random.integers(min_range, max_range, n_items)
    elif scheme == "weakly correlated":
        arr = np.vstack([np.maximum(weights - (max_range - min_range), 1), weights + (max_range - min_range)])
        values = np.apply_along_axis(lambda x: random.integers(x[0], x[1]), axis=0, arr=arr)
    elif scheme == "strongly correlated":
        values = weights + (max_range - min_range) / 10
    else:
        raise NotImplementedError
    return weights, values

def generate_knapsack(n_items, filename, random):
    """
    Generation of (hard) knapsack as described by:
        Moura, L. F. D. S. (2013). An efficient dynamic programming algorithm for the unbounded knapsack problem.

    Saves it as a CPLEX LP file.

    Parameters
    ----------
    n_items : int
        The number of items.
    filename : str
        Path to the file to save.
    random : numpy.random.Generator
        A random number generator.
    """
    weights, values = generate_weights_and_values(n_items, random)
    capacity = 0.5 * weights.sum()
    with open(filename, 'w') as file:
        file.write("maximize\nOBJ:")
        file.write(" + ".join([f"{value}x_{i + 1}"for i, value in enumerate(values)]))
        file.write("\n\nsubject to\n")
        file.write(" + ".join([f"{weight}x_{i + 1}" for i, weight in enumerate(weights)]) + f" <= {capacity}")

        file.write("\n\ninteger\n")
        file.write(" ".join([f"x_{i + 1}" for i in range(len(values))]))

def generate_mknapsack(n_items, n_knapsacks, filename, random):
    """
    Generate a Multiple Knapsack problem following a scheme among those found in section 2.1. of
        Fukunaga, Alex S. (2011). A branch-and-bound algorithm for hard multiple knapsack problems.
        Annals of Operations Research (184) 97-119.

    Saves it as a CPLEX LP file.

    Parameters
    ----------
    n_items : int
        The number of items.
    n_knapsacks : int
        The number of knapsacks.
    filename : str
        Path to the file to save.
    random : numpy.random.Generator
        A random number generator.
    """
    weights, values = generate_weights_and_values(n_items, random)  # scheme='subset-sum'
    capacities = np.zeros(n_knapsacks, dtype=int)
    capacities[:-1] = random.integers(0.4 * weights.sum() // n_knapsacks,
                                      0.6 * weights.sum() // n_knapsacks,
                                      n_knapsacks - 1)
    # Expectation capacities[-1] = 0.5 * weights.sum() // n_knapsacks
    capacities[-1] = 0.5 * weights.sum() - capacities[:-1].sum()

    with open(filename, 'w') as file:
        file.write("maximize\nOBJ: ")
        file.write(" + ".join([f"{values[i]}x_{i + 1}_{k + 1}"
                               for i in range(n_items)
                               for k in range(n_knapsacks)]))

        file.write("\n\nsubject to\n")
        for k in range(n_knapsacks):
            file.write(f"capacity_{k + 1}: " +
                       " + ".join([f"{weights[i]}x_{i + 1}_{k + 1}"
                                   for i in range(n_items)]) +
                       f" <= {capacities[k]}\n")

        for i in range(n_items):
            file.write(f"C_{i + 1}: " + " + ".join([f"x_{i + 1}_{k + 1}" for k in range(n_knapsacks)]) + " <= 1\n")

        file.write("\nbinary\n" + " ".join([f"x_{i + 1}_{k + 1}" for i in range(n_items) for k in range(n_knapsacks)]))

def generate_capacitated_facility_location(n_customers, n_facilities, ratio, filename, random):
    """
    Generate a Capacitated Facility Location problem following
        Cornuejols G, Sridharan R, Thizy J-M (1991)
        A Comparison of Heuristics and Relaxations for the Capacitated Plant Location Problem.
        European Journal of Operations Research 50:280-297.

    Saves it as a CPLEX LP file.

    Parameters
    ----------
    n_customers: int
        The desired number of customers.
    n_facilities: int
        The desired number of facilities.
    ratio: float
        The desired capacity / demand ratio.
    filename : str
        Path to the file to save.
    random : numpy.random.Generator
        A random number generator.
    """
    demands = random.integers(5, 35, size=n_customers)
    capacities = random.integers(10, 60, size=n_facilities)  # original: 10, 160
    fixed_costs = (random.integers(100, 110, size=n_facilities) * np.sqrt(capacities) +
                   random.integers(90, size=n_facilities)).astype(int)

    total_demand = demands.sum()
    total_capacity = capacities.sum()

    # adjust capacities according to ratio
    capacities = capacities * ratio * total_demand / total_capacity
    capacities = capacities.astype(int)

    # transportation costs
    c_x = random.random(n_customers).reshape((-1, 1))
    c_y = random.random(n_customers).reshape((-1, 1))

    f_x = random.random((n_facilities,))
    f_y = random.random((n_facilities,))

    trans_costs = np.sqrt((c_x - f_x) ** 2 + (c_y - f_y) ** 2) * 10 * demands.reshape((-1, 1))
    trans_costs = trans_costs.astype(int)

    # write problem
    with open(filename, 'w') as file:
        file.write("minimize\nOBJ: ")
        file.write(" + ".join([f"{trans_costs[i, j]}x_{i + 1}_{j + 1}"
                               for i in range(n_customers)
                               for j in range(n_facilities)] +
                              [f"{fixed_costs[j]}y_{j + 1}"
                               for j in range(n_facilities)]))

        file.write("\n\nsubject to\n")
        for i in range(n_customers):
            file.write(f"demand_{i + 1}: " + " + ".join([f"x_{i + 1}_{j + 1}" for j in range(n_facilities)]) + f" => 1\n")
        for j in range(n_facilities):
            file.write(f"capacity_{j + 1}: " +
                       " + ".join([f"{demands[i]}x_{i + 1}_{j + 1}"
                                   for i in range(n_customers)]) +
                       f" - {capacities[j]}y_{j + 1} <= 0\n")

        # optional constraints for LP relaxation tightening
        file.write("total_capacity: " +
                   " + ".join([f" -{capacities[j]}y_{j + 1}"
                               for j in range(n_facilities)]) +
                   f" <= -{total_demand}\n")
        for i in range(n_customers):
            for j in range(n_facilities):
                file.write(f"affectation_{i + 1}_{j + 1}: x_{i + 1}_{j + 1} - y_{j + 1} <= 0\n")

        file.write("\nbinary\n")
        file.write(" ".join([f"x_{i + 1}_{j + 1}"
                             for i in range(n_customers)
                             for j in range(n_facilities)] +
                            [f"y_{j + 1}"
                             for j in range(n_facilities)]))

def generate_multicommodity_network_flow(graph, n_nodes, n_commodities, filename, random):
    """
    Generate a Loose Fixed-Charge Multi-Commodity Network Flow problem following

    Saves it as a CPLEX LP file.

    Parameters
    ----------
    graph : Graph
        The graph from which to build the fcmcnf problem.
    n_nodes int
        The desired number of nodes in the graph.
    n_commodities: int
        The desired number of commodities.
    filename : str
        Path to the file to save.
    random : np.random.Generator
        A random number generator.
    """
    demands = random.integers(10, 100, size=n_commodities)
    adj_mat = [[0 for _ in range(n_nodes)] for _ in range(n_nodes)]
    incomings = dict([(j, []) for j in range(n_nodes)])
    outgoings = dict([(i, []) for i in range(n_nodes)])

    c = random.integers(12, 50, size=graph.size())
    f = random.integers(100, 250, size=graph.size())

    for i, j in graph.edges:
        c_ij = int(random.uniform(12, 50))  # variable_cost
        f_ij = int(random.uniform(100, 250))  # fixed_cost
        u_ij = int(random.uniform(1, n_commodities + 1) * random.uniform(10, 100))  # capacity

        adj_mat[i][j] = (c_ij, f_ij, u_ij)
        outgoings[i].append(j)
        incomings[j].append(i)

    commodities = []
    for k in range(n_commodities):
        while True:
            o_k = random.integers(0, n_nodes)  # origin_k
            d_k = random.integers(0, n_nodes)  # destination_k

            if o_k != d_k and nx.has_path(graph, o_k, d_k):
                commodities.append((o_k, d_k)); break

    with open(filename, 'w') as file:
        file.write("minimize\nOBJ: ")
        # demand_k * variable_cost * fraction of demand over edge (i, j) for commodity k
        file.write(" + ".join([f"{demands[k] * adj_mat[i][j][0]}x_{i + 1}_{j + 1}_{k + 1}"
                               for i, j in graph.edges for k in range(n_commodities)] +
                              [f"{adj_mat[i][j][1]}y_{i + 1}_{j + 1}" for i, j in graph.edges]))

        file.write("\n\nsubject to\n")
        for i in range(n_nodes):
            for k in range(n_commodities):
                # 1 if source, -1 if sink, 0 if else
                file.write(f"flow_{i + 1}_{k + 1}: " +
                           " + ".join([f"x_{i + 1}_{j + 1}_{k + 1}" for j in outgoings[i]]) +
                           "".join([f" - x_{j + 1}_{i + 1}_{k + 1}" for j in incomings[i]]) +
                           f" = {int(commodities[k][0] == i) - int(commodities[k][1] == i)}\n")

        for i, j in graph.edges:
            file.write(f"arc_{i + 1}_{j + 1}: " +
                       " + ".join([f"{demands[k]}x_{i + 1}_{j + 1}_{k + 1}"
                                   for k in range(n_commodities)]) +
                       f" - {adj_mat[i][j][2]}y_{i + 1}_{j + 1} <= 0\n")

        file.write("\nbinary\n" + " ".join([f"y_{i + 1}_{j + 1}" for i, j in graph.edges]))

def generate_setcover(n_rows, n_cols, density, max_coef, filename, random):
    """
    Generates a setcover instance with specified characteristics,
    and writes it to a file in the LP format.

    Approach described in:
    E.Balas and A.Ho, Set covering algorithms using cutting planes, heuristics,
    and subgradient optimization: A computational study, Mathematical
    Programming, 12 (1980), 37-60.

    Parameters
    ----------
    n_rows : int
        Desired number of rows
    n_cols : int
        Desired number of columns
    density: float between 0 (excluded) and 1 (included)
        Desired density of the constraint matrix
    max_coef: int
        Maximum objective coefficient (>=1)
    filename: str
        File to which the LP will be written
    random: np.random.Generator
        Random number generator
    """
    nnzrs = int(n_rows * n_cols * density)

    assert nnzrs >= n_rows  # at least 1 col per row
    assert nnzrs >= 2 * n_cols  # at leats 2 rows per col

    # compute number of rows per column
    indices = random.choice(n_cols, size=nnzrs)  # random column indexes
    indices[:2 * n_cols] = np.repeat(np.arange(n_cols), 2)  # force at least 2 rows per col
    _, col_nrows = np.unique(indices, return_counts=True)

    # for each column, sample random rows
    indices[:n_rows] = random.permutation(n_rows)  # force at least 1 column per row
    i = 0
    indptr = [0]
    for n in col_nrows:

        # empty column, fill with random rows
        if i >= n_rows:
            indices[i:i + n] = random.choice(n_rows, size=n, replace=False)

        # partially filled column, complete with random rows among remaining ones
        elif i + n > n_rows:
            remaining_rows = np.setdiff1d(np.arange(n_rows), indices[i:n_rows], assume_unique=True)
            indices[n_rows:i + n] = random.choice(remaining_rows, size=i + n - n_rows, replace=False)

        i += n
        indptr.append(i)

    # objective coefficients
    c = random.integers(max_coef, size=n_cols) + 1

    # sparce CSC to sparse CSR matrix
    A = sp.sparse.csc_matrix(
        (np.ones(len(indices), dtype=int), indices, indptr),
        shape=(n_rows, n_cols)).tocsr()
    indices = A.indices
    indptr = A.indptr

    # write problem
    with open(filename, 'w') as file:
        file.write("minimize\nOBJ: ")
        file.write(" + ".join([f"{c[j]}x{j + 1}" for j in range(n_cols)]))

        file.write("\n\nsubject to\n")
        for i in range(n_rows):
            slice = indices[indptr[i]:indptr[i + 1]]
            file.write(f"C{i}: " + " + ".join([f"x{j + 1}" for j in slice]) + f" >= 1\n")

        file.write("\nbinary\n" + " ".join([f"x{j + 1}" for j in range(n_cols)]))

# ------------------ COMBINATORIAL AUCTIONS ------------------------
def generate_cauctions(n_items, n_bids, filename, random, min_value=1, max_value=100,
                       value_deviation=0.5, add_item_prob=0.7, max_n_sub_bids=5,
                       additivity=0.2, budget_factor=1.5, resale_factor=0.5, integers=False):
    """
    Generate a Combinatorial Auction problem following the 'arbitrary' scheme found in section 4.3. of
        Kevin Leyton-Brown, Mark Pearson, and Yoav Shoham. (2000).
        Towards a universal test suite for combinatorial auction algorithms.
        Proceedings of ACM Conference on Electronic Commerce (EC-00) 66-76.

    Saves it as a CPLEX LP file.

    Parameters
    ----------
    random : np.random.Generator
        A random number generator.
    filename : str
        Path to the file to save.
    n_items : int
        The number of items.
    n_bids : int
        The number of bids.
    min_value : int
        The minimum resale value for an item.
    max_value : int
        The maximum resale value for an item.
    value_deviation : int
        The deviation allowed for each bidder's private value of an item, relative from max_value.
    add_item_prob : float in [0, 1]
        The probability of adding a new item to an existing bundle.
    max_n_sub_bids : int
        The maximum number of substitutable bids per bidder (+1 gives the maximum number of bids per bidder).
    additivity : float
        Additivity parameter for bundle prices.
        Note that additivity < 0 gives sub-additive bids,while additivity > 0 gives super-additive bids.
    budget_factor : float
        The budget factor for each bidder, relative to their initial bid's price.
    resale_factor : float
        The resale factor for each bidder, relative to their initial bid's resale value.
    integers : logical
        Should bid's prices be integral ?
    warnings : logical
        Should warnings be printed ?
    """

    assert 0 <= min_value <= max_value
    assert 0 <= add_item_prob <= 1

    def choose_next_item(bundle_mask, interests, compats, random):
        n_items = len(interests)
        prob = (1 - bundle_mask) * interests * compats[bundle_mask, :].mean(axis=0)
        prob /= prob.sum()
        return random.choice(n_items, p=prob)

    # common item values (resale price)
    values = min_value + (max_value - min_value) * random.random(n_items)

    # item compatibilities
    compats = np.triu(random.random((n_items, n_items)), k=1)
    compats = compats + compats.transpose()
    compats = compats / compats.sum(1)

    bids = []
    n_dummy_items = 0

    # create bids, one bidder at a time
    while len(bids) < n_bids:

        # bidder item values (buy price) and interests
        private_interests = random.random(n_items)
        private_values = values + max_value * value_deviation * (2 * private_interests - 1)

        # substitutable bids of this bidder
        bidder_bids = {}

        # generate initial bundle, choose first item according to bidder interests
        prob = private_interests / private_interests.sum()
        item = random.choice(n_items, p=prob)
        bundle_mask = np.full(n_items, 0)
        bundle_mask[item] = 1

        # add additional items, according to bidder interests and item compatibilities
        while random.random() < add_item_prob:
            # stop when bundle full (no item left)
            if bundle_mask.sum() == n_items: break
            item = choose_next_item(bundle_mask, private_interests, compats, random)
            bundle_mask[item] = 1

        bundle = np.nonzero(bundle_mask)[0]

        # compute bundle price with value additivity
        price = private_values[bundle].sum() + np.power(len(bundle), 1 + additivity)
        if integers:
            price = int(price)

        # drop negatively priced bundles
        if price < 0:
            continue

        # bid on initial bundle
        bidder_bids[frozenset(bundle)] = price

        # generate candidates substitutable bundles
        sub_candidates = []
        for item in bundle:

            # at least one item must be shared with initial bundle
            bundle_mask = np.full(n_items, 0)
            bundle_mask[item] = 1

            # add additional items, according to bidder interests and item compatibilities
            while bundle_mask.sum() < len(bundle):
                item = choose_next_item(bundle_mask, private_interests, compats, random)
                bundle_mask[item] = 1

            sub_bundle = np.nonzero(bundle_mask)[0]

            # compute bundle price with value additivity
            sub_price = private_values[sub_bundle].sum() + np.power(len(sub_bundle), 1 + additivity)
            if integers:
                sub_price = int(sub_price)

            sub_candidates.append((sub_bundle, sub_price))

        # filter valid candidates, higher priced candidates first
        budget = budget_factor * price
        min_resale_value = resale_factor * values[bundle].sum()
        for bundle, price in [sub_candidates[i] for i in np.argsort([-price for bundle, price in sub_candidates])]:

            if len(bidder_bids) >= max_n_sub_bids + 1 or len(bids) + len(bidder_bids) >= n_bids:
                break

            if not 0 < price < budget:
                continue

            if values[bundle].sum() < min_resale_value:
                continue

            if frozenset(bundle) in bidder_bids:
                continue

            bidder_bids[frozenset(bundle)] = price

        # add XOR constraint if needed (dummy item)
        if len(bidder_bids) > 2:
            dummy_item = [n_items + n_dummy_items]
            n_dummy_items += 1
        else:
            dummy_item = []

        # place bids
        for bundle, price in bidder_bids.items():
            bids.append((list(bundle) + dummy_item, price))

    # generate the LP file
    with open(filename, 'w') as file:
        bids_per_item = [[] for item in range(n_items + n_dummy_items)]

        file.write("maximize\nOBJ: ")
        # file.write(" + ".join([f"{price}x{i + 1}" for i, (bundle, price) in enumerate(bids)]))
        for i, bid in enumerate(bids):
            bundle, price = bid
            file.write(f" + {price}x{i + 1}")
            for item in bundle:
                bids_per_item[item].append(i)

        file.write("\n\nsubject to\n")
        for item_bids in bids_per_item:
            if item_bids:
                file.write(" + ".join([f"x{i + 1}" for i in item_bids]) + f" <= 1\n")

        file.write("\nbinary\n" + " ".join([f"x{i + 1}" for i in range(len(bids))]))


if __name__ == '__main__':
    # read default config file
    with open('config.json') as f:
        config = json.load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help="Instance type to process.",
        choices=config['problems'],
    )
    parser.add_argument(
        '-s', '--seed',
        help="Random generator seed.",
        type=valid_seed,
        default=config['seed'],
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    # smaller size for debugging purposes
    config['num_instances'] = [("train", 10),
                               ("valid", 2),
                               ("test", 1),
                               ("transfer", 1)]

    instance_dir = f'data/{args.problem}/instances'
    if args.problem == "indset":
        affinity = 4
        for instance_type, num_instances in config['num_instances']:
            n_nodes = 1000 if instance_type == "transfer" else 500
            out_dir = instance_dir + f'/{instance_type}_{n_nodes}_{affinity}'
            os.makedirs(out_dir); print(f"{num_instances} instances in {out_dir}")
            for i in trange(num_instances):
                filename = os.path.join(out_dir, f'instance_{i + 1}.lp')
                graph = Graph.barabasi_albert(n_nodes, affinity, rng)
                generate_indset(graph, filename)

    elif args.problem == "gisp":
        edge_prob = 0.6
        drop_rate = 0.5
        for instance_type, num_instances in config['num_instances']:
            n_nodes = 80 if instance_type == "transfer" else 60
            out_dir = instance_dir + f'/{instance_type}_{n_nodes}_{drop_rate}'
            os.makedirs(out_dir); print(f"{num_instances} instances in {out_dir}")
            for i in trange(num_instances):
                filename = os.path.join(out_dir, f'instance_{i + 1}.lp')
                graph = Graph.erdos_renyi(n_nodes, edge_prob, rng)
                generate_general_indset(graph, filename, drop_rate, rng)

    elif args.problem == "mkp":
        n_items = 100
        for instance_type, num_instances in config['num_instances']:
            n_knapsacks = 12 if instance_type == "transfer" else 6
            out_dir = instance_dir + f'/{instance_type}_{n_items}_{n_knapsacks}'
            os.makedirs(out_dir); print(f"{num_instances} instances in {out_dir}")
            for i in trange(num_instances):
                filename = os.path.join(out_dir, f'instance_{i + 1}.lp')
                generate_mknapsack(n_items, n_knapsacks, filename, rng)

    elif args.problem == "cflp":
        n_facilities = 25  # original: 35
        ratio = 2  # original: 5
        for instance_type, num_instances in config['num_instances']:
            n_customers = 60 if instance_type == "transfer" else 25  # original: 35
            out_dir = instance_dir + f'/{instance_type}_{n_customers}_{n_facilities}_{ratio}'
            os.makedirs(out_dir); print(f"{num_instances} instances in {out_dir}")
            for i in trange(num_instances):
                filename = os.path.join(out_dir, f'instance_{i + 1}.lp')
                generate_capacitated_facility_location(n_customers, n_facilities, ratio, filename, rng)

    elif args.problem == "fcmcnf":
        edge_prob = 0.33
        for instance_type, num_instances in config['num_instances']:
            n_nodes = 20 if instance_type == "transfer" else 15
            n_commodities = 30 if instance_type == "transfer" else 22
            out_dir = instance_dir + f'/{instance_type}_{n_nodes}_{n_commodities}_{edge_prob}'
            os.makedirs(out_dir); print(f"{num_instances} instances in {out_dir}")
            for i in trange(num_instances):
                filename = os.path.join(out_dir, f'instance_{i + 1}.lp')
                # graph = Graph.erdos_renyi(n_nodes, edge_prob, rng)
                graph = nx.erdos_renyi_graph(n_nodes, edge_prob, seed=args.seed, directed=True)
                generate_multicommodity_network_flow(graph, n_nodes, n_commodities, filename, rng)

    elif args.problem == "setcover":
        density = 0.05
        max_coef = 100
        for instance_type, num_instances in config['num_instances']:
            n_rows = 500 if instance_type == "transfer" else 400
            n_cols = 1000 if instance_type == "transfer" else 750
            out_dir = instance_dir + f'/{instance_type}_{n_rows}_{n_cols}_{density}'
            os.makedirs(out_dir); print(f"{num_instances} instances in {out_dir}")
            for i in trange(num_instances):
                filename = os.path.join(out_dir, f'instance_{i + 1}.lp')
                generate_setcover(n_rows, n_cols, density, max_coef, filename, rng)

    elif args.problem == "cauctions":
        for instance_type, num_instances in config['num_instances']:
            n_items = 200 if instance_type == "transfer" else 100
            n_bids = 1000 if instance_type == "transfer" else 500
            out_dir = instance_dir + f'/{instance_type}_{n_items}_{n_bids}'
            os.makedirs(out_dir); print(f"{num_instances} instances in {out_dir}")
            for i in trange(num_instances):
                filename = os.path.join(out_dir, f'instance_{i + 1}.lp')
                generate_cauctions(n_items, n_bids, filename, rng)

    print("done.")
