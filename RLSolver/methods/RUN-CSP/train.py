from model import RUN_CSP
from csp_utils import Constraint_Language, CSP_Instance

import argparse
import numpy as np
from tqdm import tqdm


def train(network, train_data, t_max, epochs):
    """
    Trains a RUN-CSP Network on the given data
    :param network: The RUN_CSP network
    :param train_data: A list of CSP instances that are used for training
    :param t_max: Number of RUN_CSP iterations on each instance
    :param epochs: Number of training epochs
    """

    best_conflict_ratio = 1.0
    for e in range(epochs):
        print('Epoch: {}'.format(e))

        # train one epoch
        output_dict = network.train(train_data, iterations=t_max)
        conflict_ratio = output_dict['conflict_ratio']
        print(f'Ratio of violated constraints: {conflict_ratio}')

        # if network improved, save new model
        if conflict_ratio < best_conflict_ratio:
            network.save_checkpoint('best')
            best_conflict_ratio = conflict_ratio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--language_config_path', type=str,  help='The path to a json file that specifies the constraint language')
    parser.add_argument('-m', '--model_dir', type=str, help='Path to the model directory where the trained RUN-CSP instance will be stored')
    parser.add_argument('-v', '--n_variables', type=int, default=100, help='Number of variables in each training instance.')
    parser.add_argument('--c_min', type=int, default=100, help='Minimum number of clauses in each training instance.')
    parser.add_argument('--c_max', type=int, default=600, help='Maximum number of clauses in each training instance.')
    parser.add_argument('-i', '--n_instances', type=int, default=4000, help='Number of instances for training.')
    parser.add_argument('-t', '--t_max', type=int, default=30, help='Number of iterations t_max for which RUN-CSP runs on each instance')
    parser.add_argument('-s', '--state_size', type=int, default=128, help='Size of the variable states in RUN-CSP')
    parser.add_argument('-b', '--batch_size', type=int, default=10, help='Batch size used during training')
    parser.add_argument('-e', '--epochs', type=int, default=25, help='Number of training epochs')
    args = parser.parse_args()

    print(f'Loading constraint language from {args.language_config_path}')
    language = Constraint_Language.load(args.language_config_path)
    # create RUN_CSP instance for given constraint language
    network = RUN_CSP(args.model_dir, language, args.state_size)

    print(f'Generating {args.n_instances} training instances')
    train_instances = [CSP_Instance.generate_random(args.n_variables, np.random.randint(args.c_min, args.c_max), language) for _ in tqdm(range(args.n_instances))]
    # combine instances into batches
    train_batches = CSP_Instance.batch_instances(train_instances, args.batch_size)

    # train and store the network
    train(network, train_batches, args.t_max, args.epochs)


if __name__ == '__main__':
    main()
