from model import Max_IS_Network
from csp_utils import CSP_Instance, is_language

import data_utils
import argparse
import os
import glob
import numpy as np
import csv
from tqdm import tqdm


def evaluate_boosted(network, eval_instances, t_max, attempts=64):
    """
    Evaluate Independent Set Network with boosted predictions
    :param network: A Max_IS_Network
    :param eval_instances: A list of CSP instances for evaluation
    :param t_max: Number of RUN_CSP iterations on each instance
    :param attempts: Number of parallel attempts for each instance
    """

    conflict_ratios = []
    is_sizes = []
    for i, instance in enumerate(eval_instances):

        # get boosted and corrected predictions
        output_dict = network.predict_boosted_and_corrected(instance, iterations=t_max, attempts=attempts)

        conflicts = output_dict['conflicts']
        conflict_ratio = output_dict['conflict_ratio']
        conflict_ratios.append(conflict_ratio)
        is_size = output_dict['is_size']
        is_sizes.append(is_size)
        print(f'Induced edges for instance {i}: {conflicts}, IS Size: {is_size}')

    mean_conflict_ratio = np.mean(conflict_ratios)
    mean_is_size = np.mean(is_sizes)
    print(f'Mean ratio of induced edges: {mean_conflict_ratio}, Mean Corrected Independent Set size: {mean_is_size}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_dir', type=str, help='Path to the trained RUN-CSP instance')
    parser.add_argument('-t', '--t_max', type=int, default=100, help='Number of iterations t_max for which RUN-CSP runs on each instance')
    parser.add_argument('-a', '--attempts', type=int, default=64, help='Attempts for each graph')
    parser.add_argument('-d', '--data_path', default=None, help='Path to the evaluation data. Expects a directory with graphs in dimacs format.')
    args = parser.parse_args()

    network = Max_IS_Network.load(args.model_dir)

    print('loading graphs...')
    names, graphs = data_utils.load_graphs(args.data_path)
    instances = [CSP_Instance.graph_to_csp_instance(g, is_language, 'NAND') for n, g in zip(names, graphs)]
    
    evaluate_boosted(network, instances, args.t_max, attempts=args.attempts)
    
if __name__ == '__main__':
    main()
