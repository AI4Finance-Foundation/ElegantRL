# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #
# Generates training samples for imitation learning                             #
# Usage: python 03_generate_il_samples.py <problem> <type> -s <seed> -j <njobs> #
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * #

import os
import glob
import json
import queue
import argparse
import util
from util import init_scip_params
import multiprocessing as mp
import numpy as np
import pyscipopt as scip

from nodesels.nodesel_oracle import NodeselOracle, RandomSampler


def make_samples(in_queue, out_queue, tmp_dir, k_sols, sampling):
    """
    Worker loop: fetch an instance, run an episode and record samples.

    Parameters
    ----------
    in_queue: mp.Queue
        Instance files from which to collect samples.
    out_queue : mp.Queue
        Output queue in which to put solutions.
    tmp_dir : str
        Directory in which to write samples.
    k_sols : int
        Number of solutions to save. (Default: 10)
    """
    while True:
        # Fetch an instance...
        episode, instance, seed = in_queue.get()
        instance_id = f"[w {os.getpid()}] episode {episode}"
        print(f"{instance_id}: Processing instance '{instance}'...")

        # Retrieve available solution files
        solution_files = sorted(glob.glob(f'{instance[:-3]}-*.sol'), key=len)
        print(f"{instance_id}: Retrieved {len(solution_files)} solutions")
        if len(solution_files) == 0:
            print("ABORT: No solutions")
            continue

        # Initialize SCIP model
        m = scip.Model()
        m.hideOutput()
        m.readProblem(instance)

        # 1: CPU user seconds, 2: wall clock time
        m.setIntParam('timing/clocktype', 1)
        m.setRealParam('limits/time', 90)
        init_scip_params(m, seed)

        solutions = []
        for solution_file in solution_files[:k_sols]:
            solution = m.readSolFile(solution_file)
            solutions.append(solution)

        sampler = RandomSampler(episode, tmp_dir, out_queue, seed)
        oracle = NodeselOracle(sampler, sampling, solutions)

        m.includeNodesel(nodesel=oracle,
                         name="nodesel_oracle",
                         desc="BestEstimate node selector that saves samples based on a diving oracle",
                         stdpriority=999999,
                         memsavepriority=999999)

        out_queue.put({
            'type': "start",
            'episode': episode,
        })

        m.optimize()
        m.freeProb()

        count = max(sampler.sample_count, 1)
        print(f"{instance_id}: {[f'{x / count:.2f}' for x in sampler.action_count]}")
        print(f"{instance_id}: Process completed, {sampler.sample_count} samples")

        out_queue.put({
            'type': "done",
            'episode': episode,
        })


def send_orders(orders_queue, instances, random):
    """
    Dispatcher: Continuously send sampling orders to workers.
                Relies on limited queue capacity.

    Parameters
    ----------
    orders_queue : mp.Queue
        Limited-size queue to which to send orders.
    instances : list
        Instance file names from which to sample episodes.
    random: np.random.Generator
        Random number generator
    """
    episode = 0
    while True:
        instance = random.choice(instances)
        # blocks the process until a free slot in the queue is available
        orders_queue.put([episode, instance, random.integers(2**31)])
        episode += 1


def collect_samples(instances, sample_dir, n_jobs, k_sols, max_samples, sampling, random):
    """
    Runs branch-and-bound episodes on the given set of instances,
    and collects (state, action) pairs from the diving oracle.

    Parameters
    ----------
    instances : list
        Instance files from which to collect samples.
    sample_dir : str
        Directory in which to write samples.
    n_jobs : int
        Number of jobs for parallel sampling.
    k_sols : int
        Number of solutions to save per instance.
    max_samples : int
        Number of samples to collect.
    sampling : str
        Type of sampling to perform.
    random: np.random.Generator
        Random number generator
    """
    tmp_dir = sample_dir + '/tmp'
    os.makedirs(tmp_dir, exist_ok=True)

    # start workers
    in_queue = mp.Queue(2 * n_jobs)
    out_queue = mp.Queue()

    # temp solution for limited threads
    # removes the need for the dispatcher
    # in_queue = [(episode, instance, random.integers(2**31))
    #             for episode, instance in enumerate(instances)]
    # removes the need for the workers
    # make_samples(in_queue, out_queue, tmp_dir, k_sols, sampling)

    workers = []
    for i in range(n_jobs):
        p = mp.Process(
            target=make_samples,
            args=(in_queue, out_queue, tmp_dir, k_sols, sampling),
            daemon=True)
        workers.append(p)
        p.start()

    # start dispatcher
    dispatcher = mp.Process(
        target=send_orders,
        args=(in_queue, instances, random),
        daemon=True)
    dispatcher.start()
    print(f"[m {os.getpid()}] dispatcher started...")

    # record answers and write samples
    buffer = {}
    episode_i = 0
    n_samples = 0
    in_buffer = 0

    while n_samples < max_samples:
        try: sample = out_queue.get(timeout=100)
        # if no response is given in time_limit seconds,
        # the solver has crashed and the worker is dead:
        # start a new worker to pick up the pieces.
        except queue.Empty:
            p = mp.Process(
                target=make_samples,
                args=(in_queue, out_queue, tmp_dir, k_sols, sampling),
                daemon=True)
            workers.append(p)
            p.start()
            print(f"[m {os.getpid()}] new worker started...")
            episode_i += 1
            continue

        # add received sample to buffer
        if sample['type'] == 'start':
            # Create a new episode object
            buffer[sample['episode']] = []
        else:
            # Add samples to correct episode
            buffer[sample['episode']].append(sample)
            if sample['type'] == 'sample':
                in_buffer += 1

        # early stop dispatcher (hard)
        if in_buffer + n_samples >= max_samples and dispatcher.is_alive():
            dispatcher.terminate()
            print(f"[m {os.getpid()}] dispatcher stopped...")

        # if current_episode object is not empty...
        while episode_i in buffer and buffer[episode_i]:
            samples_to_write = buffer[episode_i]
            buffer[episode_i] = []

            # write samples from current episode
            for sample in samples_to_write:
                # if final sample is processed...
                if sample['type'] == "done":
                    # move to next episode
                    del buffer[episode_i]

                    print(f"[m {os.getpid()}] episode {sample['episode']}:"
                          f" {n_samples} / {max_samples} samples written.")

                    episode_i += 1
                    break

                # else write sample
                in_buffer -= 1
                n_samples += 1
                os.rename(sample['filename'], f'{sample_dir}/sample_{n_samples}.pkl')

                # stop the episode as soon as
                # enough samples are collected
                if n_samples == max_samples:
                    buffer = {}
                    break

    # stop all workers (hard)
    for p in workers:
        p.terminate()
    print(f"[m {os.getpid()}] sampling completed")


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
        'sampling_type',
        help='Type of sampling to apply',
        choices=["Children", "Nodes"],
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed.',
        type=utilities.valid_seed,
        default=config['seed'],
    )
    parser.add_argument(
        '-k', '--ksols',
        help='Number of solutions to save.',
        default=config['k'],
        type=int,
    )
    parser.add_argument(
        '-j', '--njobs',
        help='Number of parallel jobs.',
        type=int,
        default=1,
    )
    parser.add_argument(
        '-r', '--ratio',
        help='Samples per instance ratio',
        type=int,
        default=50,
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    difficulty = config['difficulty'][args.problem]
    sample_dir = f'data/{args.problem}/samples/k={args.ksols}_{args.sampling_type}_adjusted'

    for instance_type in ["train", "valid"]:
        out_dir = sample_dir + f'/{instance_type}_{difficulty}'
        # os.makedirs(out_dir, exist_ok=True)
        instances = glob.glob(f'data/{args.problem}/instances/{instance_type}_{difficulty}/*.lp')
        num_samples = args.ratio * len(instances)
        print(f"{len(instances)} {instance_type} instances for {num_samples} {args.sampling_type} samples")
        collect_samples(instances, out_dir, args.njobs, args.ksols, num_samples, args.sampling_type, rng)
