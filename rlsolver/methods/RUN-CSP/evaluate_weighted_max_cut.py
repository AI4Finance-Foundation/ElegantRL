from model import RUN_CSP
from evaluate import evaluate_and_save
from csp_utils import CSP_Instance, mc_weighted_language

import data_utils
import argparse
import os
import numpy as np
import glob
import networkx as nx

from tqdm import tqdm


def compute_weighted_score(instance, assignment):
    matrices = mc_weighted_language.relation_matrices
    valid_pos = np.float32([matrices['NEQ'][assignment[u], assignment[v]] for [u, v] in instance.clauses['NEQ']])
    valid_neg = np.float32([matrices['NEQ'][assignment[u], assignment[v]] for [u, v] in instance.clauses['EQ']])
    score = np.sum(valid_pos) - np.sum(valid_neg)
    return score


def evaluate_boosted(network, eval_instances, t_max, attempts=64):
    """
    Evaluate RUN-CSP Network with boosted predictions
    :param network: A RUN_CSP network
    :param eval_instances: A list of CSP instances for evaluation
    :param t_max: Number of RUN_CSP iterations on each instance
    :param attempts: Number of parallel attempts for each instance
    """

    conflict_ratios = []
    for i, instance in enumerate(eval_instances):
        output_dict = network.predict_boosted(instance, iterations=t_max, attempts=attempts)

        all_assignment = output_dict['all_assignments']
        best_per_attempt = [np.max([compute_weighted_score(instance, all_assignment[a, :, t]) for t in range(t_max)]) for a in range(attempts)]

        best = np.max(best_per_attempt)
        mean = np.mean(best_per_attempt)
        std = np.std(best_per_attempt)

        print(f'Cut size for instance {i if instance.name is None else instance.name}: {best}, {mean} (+-{std})')

    mean_conflict_ratio = np.mean(conflict_ratios)
    print(f'mean conflict ratio for evaluation instances: {mean_conflict_ratio}')
    return conflict_ratios


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_dir', type=str, help='The model directory of a trained network')
    parser.add_argument('-t', '--t_max', type=int, default=100,
                        help='Number of iterations t_max for which RUN-CSP runs on each instance')
    parser.add_argument('-a', '--attempts', type=int, default=64, help='Attempts for each graph')
    parser.add_argument('-d', '--data_path', default=None,
                        help='A path to a training set of graphs in the NetworkX adj_list format. If left unspecified, random instances are used.')
    parser.add_argument('-v', '--n_variables', type=int, default=400,
                        help='Number of variables in each training instance. Only used when --data_path is not specified.')
    parser.add_argument('--degree', type=int, default=3,
                        help='Number of clauses in each training instance. Only used when --data_path is not specified.')
    parser.add_argument('-i', '--n_instances', type=int, default=100,
                        help='Number of instances for training. Only used when --data_path is not specified.')
    parser.add_argument('-s', '--save_path', type=str, help='Path to a csv file to store results')
    args = parser.parse_args()

    language = mc_weighted_language

    if args.data_path is not None:
        print('loading graphs...')
        graphs = [data_utils.load_col_graph(p) for p in tqdm(glob.glob(args.data_path))]
        names = [os.path.basename(p) for p in glob.glob(args.data_path)]
        instances = [CSP_Instance.graph_to_weighted_mc_instance(g, name=name) for g, name in zip(graphs, names)]
    else:
        print(f'Generating {args.n_instances} training instances')
        # instances = [CSP_Instance.generate_random(args.n_variables, args.n_clauses, language) for _ in tqdm(range(args.n_instances))]
        graphs = [nx.random_regular_graph(args.degree, args.n_variables) for _ in range(args.n_instances)]
        instances = [CSP_Instance.graph_to_csp_instance(g, language, 'NEQ') for g in tqdm(graphs)]

    net = RUN_CSP.load(args.model_dir)

    if args.save_path is None:
        conflicting_edges = evaluate_boosted(net, instances, args.t_max, attempts=args.attempts)
    else:
        conflicting_edges = evaluate_and_save(args.save_path, net, instances, args.t_max, attempts=args.attempts)


if __name__ == '__main__':
    main()
