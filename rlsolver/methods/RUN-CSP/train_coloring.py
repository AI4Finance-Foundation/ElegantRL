from model import RUN_CSP
from csp_utils import CSP_Instance, Constraint_Language
from train import train

import data_utils

import argparse
import random
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('-t', '--t_max', type=int, default=25, help='Number of iterations t_max for which RUN-CSP runs on each instance')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('-m', '--model_dir', type=str, help='Model directory in which the trained model is stored')
    parser.add_argument('-d', '--data_path', help='A path to a training set of graphs in the dimacs graph format.')
    parser.add_argument('--n_colors', type=int, default=3, help='Number of colors')
    args = parser.parse_args()

    language = Constraint_Language.get_coloring_language(args.n_colors)

    print('loading graphs...')
    names, graphs = data_utils.load_graphs(args.data_path)
    random.shuffle(graphs)
    print('Converting graphs to CSP Instances')
    instances = [CSP_Instance.graph_to_csp_instance(g, language, 'NEQ') for g in tqdm(graphs)]
    
    # combine instances into batches
    train_batches = CSP_Instance.batch_instances(instances, args.batch_size)

    # construct and train new network
    network = RUN_CSP(args.model_dir, language)
    train(network, train_batches, epochs=args.epochs, t_max=args.t_max)


if __name__ == '__main__':
    main()
