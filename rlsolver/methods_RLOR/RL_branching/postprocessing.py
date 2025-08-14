# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
# Generates training samples for imitation learning                             #
# Usage: python 03_generate_il_samples.py <problem> <type> -s <seed> -j <njobs> #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #

import os
import csv
import json
import glob
import argparse
import numpy as np

from scipy.stats import gmean, gstd


if __name__ == '__main__':
    # read default config file
    with open('config.json') as f:
        config = json.load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=config['problems'],
    )
    parser.add_argument(
        'running_dir',
        help='Directory containing csv results.',
    )
    args = parser.parse_args()

    experiment_dir = f'experiments/{args.problem}/05_evaluate'
    result_files = glob.glob(experiment_dir + f'/{args.running_dir}/*.csv')
    for result_file in sorted(result_files, key=len):
        with open(result_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)

            nnodes = []
            stimes = []
            mean_nnodes = []
            mean_stimes = []
            current_instance = None
            for row in reader:
                # if current_instance is None:
                #     current_instance = row['instance']
                if row['instance'] != current_instance:
                    current_instance = row['instance']
                    if nnodes and stimes:
                        mean_nnodes.append(np.mean(nnodes))
                        mean_stimes.append(np.mean(stimes))
                        nnodes = []; stimes = []
                nnodes.append(int(row['nnodes']))
                stimes.append(float(row['stime']))
            mean_nnodes.append(np.mean(nnodes))
            mean_stimes.append(np.mean(stimes))
            print(f"result_file: {os.path.basename(result_file)}"
                  f" | nnodes: {gmean(mean_nnodes):.0f}*/{gstd(mean_nnodes):.2f}"
                  f" | stimes: {gmean(mean_stimes):.2f}*/{gstd(mean_stimes):.2f}")
