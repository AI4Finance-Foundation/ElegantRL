from model import RUN_CSP
from csp_utils import CSP_Instance, Constraint_Language

import data_utils
import argparse
import os
import numpy as np
import glob
import networkx as nx
import csv

from tqdm import tqdm


def get_P_value(n, d, z):
    return ((z / n) - (d / 4.0)) / np.sqrt(d / 4.0)


def evaluate_boosted(network, instances, degree, t_max, attempts=64):
    best_conflict_ratios = []
    mean_conflict_ratios = []
    best_Ps = []
    mean_Ps = []

    for i, instance in enumerate(instances):
        output_dict = network.predict_boosted(instance, iterations=t_max, attempts=attempts)

        conflicts = np.int32([instance.count_conflicts(assignment) for assignment in output_dict['all_assignments'][:, :, t_max-1]])
        conflict_ratios = conflicts / instance.n_clauses

        least_conflicts = output_dict['conflicts']
        best_conflict_ratios.append(least_conflicts)
        mean_conflict_ratios.append(np.mean(conflict_ratios))

        cut_values = instance.n_clauses - conflicts
        P = [get_P_value(instance.n_variables, degree, z) for z in cut_values]
        best_P = get_P_value(instance.n_variables, degree, instance.n_clauses - least_conflicts) #np.max(P)
        best_Ps.append(best_P)
        mean_Ps.append(np.mean(P))

        print(f'Conflicts for instance {i}: {least_conflicts}, Best P: {best_P}')

    print(f'Best Conflicts: {np.mean(best_conflict_ratios)}, Mean Conflicts: {np.mean(mean_conflict_ratios)}')
    print(f'Best P: {np.mean(best_Ps)}, Mean P: {np.mean(mean_Ps)}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_dir', type=str, help='Path to the trained RUN-CSP instance')
    parser.add_argument('-t', '--t_max', type=int, default=100, help='Number of iterations t_max for which RUN-CSP runs on each instance')
    parser.add_argument('-a', '--attempts', type=int, default=64, help='Attempts for each graph')
    parser.add_argument('-d', '--data_path', default=None, help='Path to the evaluation data. Expects a directory with graphs in dimacs format.')
    parser.add_argument('-v', '--n_variables', type=int, default=100, help='Number of variables in each training instance. Only used when --data_path is not specified.')
    parser.add_argument('--degree', type=int, default=3, help='The uniform degree of the regular graphs.')
    parser.add_argument('-i', '--n_instances', type=int, default=1000, help='Number of instances for training. Only used when --data_path is not specified.')
    args = parser.parse_args()

    language = Constraint_Language.get_coloring_language(2)
    network = RUN_CSP.load(args.model_dir)

    if args.data_path is not None:
        print('loading graphs...')
        names, graphs = data_utils.load_graphs(args.data_path)
        instances = [CSP_Instance.graph_to_csp_instance(g, language, 'NEQ') for g in graphs]
    else:
        print(f'Generating {args.n_instances} training instances')
        graphs = [nx.random_regular_graph(args.degree, args.n_variables) for _ in range(args.n_instances)]
        instances = [CSP_Instance.graph_to_csp_instance(g, language, 'NEQ') for g in graphs]

    evaluate_boosted(network, instances, args.degree, args.t_max, args.attempts)


if __name__ == '__main__':
    main()
