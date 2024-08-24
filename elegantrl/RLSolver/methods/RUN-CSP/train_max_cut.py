from model import RUN_CSP
from train import train
from csp_utils import CSP_Instance, Constraint_Language

import data_utils
import argparse
import os
import numpy as np
import glob
import networkx as nx
import csv
import numpy as np

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--state_size', type=int, default=128, help='Size of the variable states in RUN-CSP')
    parser.add_argument('-b', '--batch_size', type=int, default=10, help='Batch size used during training')
    parser.add_argument('-e', '--epochs', type=int, default=25, help='Number of training epochs')
    parser.add_argument('-m', '--model_dir', default='model', type=str, help='The model directory of a trained network')
    # parser.add_argument('-m', '--model_dir', type=str, help='The model directory of a trained network')
    parser.add_argument('-t', '--t_max', type=int, default=30, help='Number of iterations t_max for which RUN-CSP runs on each instance')
    parser.add_argument('-d', '--data_path', default='data.G1.dimacs', help='A path to a training set of graphs in the dimacs graph format')
    # parser.add_argument('-d', '--data_path', help='A path to a training set of graphs in the dimacs graph format')
    args = parser.parse_args()

    language = Constraint_Language.get_coloring_language(2)

    print('loading graphs...')
    names, graphs = data_utils.load_graphs(args.data_path)
    instances = [CSP_Instance.graph_to_csp_instance(g, language, 'NEQ') for g in graphs]

    train_batches = CSP_Instance.batch_instances(instances, args.batch_size)
    network = RUN_CSP(args.model_dir, language=language, state_size=args.state_size)
    train(network, train_batches, t_max=args.t_max, epochs=args.epochs)


if __name__ == '__main__':
    main()
