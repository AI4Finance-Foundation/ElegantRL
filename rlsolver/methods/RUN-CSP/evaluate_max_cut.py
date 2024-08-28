from model import RUN_CSP
from evaluate import evaluate_boosted
from csp_utils import CSP_Instance, Constraint_Language

import data_utils
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_dir', type=str, help='Path to the trained RUN-CSP instance')
    parser.add_argument('-t', '--t_max', type=int, default=100, help='Number of iterations t_max for which RUN-CSP runs on each instance')
    parser.add_argument('-a', '--attempts', type=int, default=64, help='Attempts for each graph')
    parser.add_argument('-d', '--data_path', default=None, help='Path to the evaluation data. Expects a directory with graphs in dimacs format.')
    args = parser.parse_args()

    network = RUN_CSP.load(args.model_dir)
    language = Constraint_Language.get_coloring_language(2)

    print('loading graphs...')
    names, graphs = data_utils.load_graphs(args.data_path)
    instances = [CSP_Instance.graph_to_csp_instance(g, language, 'NEQ', name=n) for n, g in zip(names, graphs)]
    
    conflicting_edges = evaluate_boosted(network, instances, args.t_max, attempts=args.attempts)

if __name__ == '__main__':
    main()
