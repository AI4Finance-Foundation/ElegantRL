from model import Max_2SAT_Network
from csp_utils import CSP_Instance
from train import train

import data_utils

import argparse
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--state_size', type=int, default=128, help='Size of the variable states in RUN-CSP')
    parser.add_argument('-e', '--epochs', type=int, default=25, help='Number of training epochs')
    parser.add_argument('-t', '--t_max', type=int, default=30, help='Number of iterations t_max for which RUN-CSP runs on each instance')
    parser.add_argument('-b', '--batch_size', type=int, default=10, help='Batch size for training')
    parser.add_argument('-m', '--model_dir', type=str, help='Model directory in which the trained model is stored')
    parser.add_argument('-d', '--data_path', help='A path to a training set of formulas in the DIMACS cnf format.')
    args = parser.parse_args()

    print('loading cnf formulas...')
    names, formulas = data_utils.load_formulas(args.data_path)
    random.shuffle(formulas)
    print('Converting formulas to CSP instances')
    instances = [CSP_Instance.cnf_to_instance(f) for f in formulas]
        
    # combine instances into batches
    train_batches = CSP_Instance.batch_instances(instances, args.batch_size)

    # construct and train new network
    network = Max_2SAT_Network(args.model_dir, state_size=args.state_size)
    train(network, train_batches, t_max=args.t_max, epochs=args.epochs)


if __name__ == '__main__':
    main()
