from model import Max_2SAT_Network
from evaluate import evaluate_boosted, evaluate_and_save
from csp_utils import CSP_Instance, max_2sat_language

import data_utils
import argparse
import os
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_dir', type=str, help='Path to the trained RUN-CSP instance')
    parser.add_argument('-t', '--t_max', type=int, default=100, help='Number of iterations t_max for which RUN-CSP runs on each instance')
    parser.add_argument('-a', '--attempts', type=int, default=64, help='Attempts for each graph')
    parser.add_argument('-d', '--data_path', default=None, help='Path to the evaluation data. Expects a directory with graphs in dimacs format.')
    args = parser.parse_args()

    network = Max_2SAT_Network.load(args.model_dir)

    print('loading cnf formulas...')
    names, formulas = data_utils.load_formulas(args.data_path)
    print('Converting formulas to CSP instances')
    instances = [CSP_Instance.cnf_to_instance(f, name=n) for n, f in zip(names, formulas)]
    
    conflicting_edges = evaluate_boosted(network, instances, args.t_max, attempts=args.attempts)


if __name__ == '__main__':
    main()
