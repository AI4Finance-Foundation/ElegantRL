from model import Max_IS_Network
from csp_utils import CSP_Instance, is_language

import data_utils
import argparse
import os
import random
from tqdm import tqdm


def train(network, train_data, t_max, epochs):
    '''
    Trains an Independent Set Network on the given data
    :param network: The Max_IS_Network instance
    :param train_data: A list of CSP instances that are used for training
    :param t_max: Number of RUN_CSP iterations on each instance
    :param epochs: Number of training epochs
    '''

    best_ratio = 0.0
    for e in range(epochs):
        print('Epoch: {}'.format(e))

        # train one epoch
        output_dict = network.train(train_data, iterations=t_max)

        # Get average percentage of conflicting edges and relative size of independent set
        conflict_ratio = output_dict['conflict_ratio']
        is_ratio = output_dict['is_ratio']
        corrected_ratio = output_dict['corrected_ratio']
        print(f'Ratio of violated constraints: {conflict_ratio}, IS Ratio: {is_ratio}, Corrected: {corrected_ratio}')

        # if network improved, save new model
        if corrected_ratio > best_ratio:
            network.save_checkpoint('best')
            best_ratio = corrected_ratio
            

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--state_size', type=int, default=128, help='Size of the variable states in RUN-CSP')
    parser.add_argument('-k', '--kappa', type=float, default=1.0, help='The parameter kappa for the loss function')
    parser.add_argument('-e', '--epochs', type=int, default=25, help='Number of training epochs')
    parser.add_argument('-t', '--t_max', type=int, default=30, help='Number of iterations t_max for which RUN-CSP runs on each instance')
    parser.add_argument('-b', '--batch_size', type=int, default=10, help='Batch size for training')
    parser.add_argument('-m', '--model_dir', type=str, help='Model directory in which the trained model is stored')
    parser.add_argument('-d', '--data_path', help='A path to a training set of graphs in the dimacs format.')
    args = parser.parse_args()
 
    print('loading graphs...')
    names, graphs = data_utils.load_graphs(args.data_path)
    random.shuffle(graphs)
    print('Converting graphs to CSP Instances')
    instances = [CSP_Instance.graph_to_csp_instance(g, is_language, 'NAND') for g in graphs]
    
    # combine instances into batches
    train_batches = CSP_Instance.batch_instances(instances, args.batch_size)

    # construct new network
    network = Max_IS_Network(args.model_dir, state_size=args.state_size)
    train(network, train_batches, t_max=args.t_max, epochs=args.epochs)


if __name__ == '__main__':
    main()
