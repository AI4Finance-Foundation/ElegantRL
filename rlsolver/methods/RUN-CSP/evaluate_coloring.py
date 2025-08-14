from model import RUN_CSP
from evaluate import evaluate_boosted
from csp_utils import CSP_Instance

import data_utils
import argparse

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_dir', type=str, help='Path to the trained RUN-CSP instance')
    parser.add_argument('-t', '--t_max', type=int, default=100, help='Number of iterations t_max for which RUN-CSP runs on each instance')
    parser.add_argument('-a', '--attempts', type=int, default=64, help='Attempts for each graph')
    parser.add_argument('-d', '--data_path', default=None, help='Path to the evaluation data. Expects a directory with graphs in dimacs format.')
    parser.add_argument('-v', '--n_variables', type=int, default=400, help='Number of variables in each training instance. Only used when --data_path is not specified.')
    parser.add_argument('-c', '--n_clauses', type=int, default=1000, help='Number of clauses in each training instance. Only used when --data_path is not specified.')
    parser.add_argument('-i', '--n_instances', type=int, default=100, help='Number of instances for training. Only used when --data_path is not specified.')
    args = parser.parse_args()

    network = RUN_CSP.load(args.model_dir)
    language = network.language

    if args.data_path is not None:
        print('loading graphs...')
        names, graphs = data_utils.load_graphs(args.data_path)
        instances = [CSP_Instance.graph_to_csp_instance(g, language, 'NEQ', name=n) for n, g in zip(names, graphs)]
    else:
        print(f'Generating {args.n_instances} training instances')
        instances = [CSP_Instance.generate_random(args.n_variables, args.n_clauses, language) for _ in tqdm(range(args.n_instances))]
    
    conflicting_edges = evaluate_boosted(network, instances, args.t_max, attempts=args.attempts)

if __name__ == '__main__':
    main()
