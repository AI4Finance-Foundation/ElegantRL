import time

import torch

from rlsolver.methods.eco_s2v.config import *
from rlsolver.methods.eco_s2v.config import INFERENCE_DEVICE
from rlsolver.methods.eco_s2v.src.envs.spinsystem import SpinSystemFactory
from rlsolver.methods.eco_s2v.src.envs.util_envs import (SingleGraphGenerator,
                                                         RewardSignal, ExtraAction,
                                                         OptimisationTarget, SpinBasis,
                                                         DEFAULT_OBSERVABLES, Observable)
from rlsolver.methods.eco_s2v.src.networks.mpnn import MPNN
from rlsolver.methods.eco_s2v.util import test_network, load_graph_set_from_txt
from rlsolver.methods.util import calc_txt_files_with_prefixes
from rlsolver.methods.util_result import write_graph_result


def run(save_loc="BA_40spin/eco",
        graph_save_loc="../../data/syn_BA",
        network_save_path=None,
        batched=True,
        max_batch_size=None,
        max_parallel_jobs=4,
        prefixes=INFERENCE_PREFIXES,
        if_greedy=False):
    print("\n----- Running {} -----\n".format(os.path.basename(__file__)))

    data_folder = os.path.join(save_loc)
    print("save location :", data_folder)
    print("network params :", network_save_path)

    ####################################################
    # NETWORK SETUP
    ####################################################

    network_fn = MPNN
    network_args = {
        'n_layers': 3,
        'n_features': 64,
        'n_hid_readout': [],
        'tied_weights': False
    }

    ####################################################
    # SET UP ENVIRONMENTAL AND VARIABLES
    ####################################################

    if ALG in [Alg.s2v]:
        env_args = {'observables': [Observable.SPIN_STATE],
                    'reward_signal': RewardSignal.DENSE,
                    'extra_action': ExtraAction.NONE,
                    'optimisation_target': OptimisationTarget.CUT,
                    'spin_basis': SpinBasis.BINARY,
                    'norm_rewards': True,
                    'memory_length': None,
                    'horizon_length': None,
                    'stag_punishment': None,
                    'basin_reward': None,
                    'reversible_spins': False,
                    'if_greedy': if_greedy}
    else:
        env_args = {'observables': DEFAULT_OBSERVABLES,
                    'reward_signal': RewardSignal.BLS,
                    'extra_action': ExtraAction.NONE,
                    'optimisation_target': OptimisationTarget.CUT,
                    'spin_basis': SpinBasis.BINARY,
                    'norm_rewards': True,
                    'memory_length': None,
                    'horizon_length': None,
                    'stag_punishment': None,
                    'basin_reward': 1. / NUM_TRAIN_NODES,
                    'reversible_spins': True,
                    'if_greedy': if_greedy}
    if prefixes:
        file_names = calc_txt_files_with_prefixes(graph_save_loc, prefixes)

    else:
        file_names = os.listdir(graph_save_loc)

    for file in file_names:
        graph_name = os.path.basename(file)
        start_time = time.time()

        step_factor = 2
        graph_dict = os.path.join(graph_save_loc, graph_name).replace("\\", "/")
        graphs_test = load_graph_set_from_txt(graph_dict)

        ####################################################
        # SETUP NETWORK TO TEST
        ####################################################

        test_env = SpinSystemFactory.get(SingleGraphGenerator(graphs_test[0]),
                                         graphs_test[0].shape[0] * step_factor,  # step_factor is 1 here
                                         **env_args)

        torch.device(INFERENCE_DEVICE)
        print("Set torch default device to {}.".format(INFERENCE_DEVICE))

        network = network_fn(n_obs_in=test_env.observation_space.shape[1],
                             **network_args).to(INFERENCE_DEVICE)

        network.load_state_dict(torch.load(network_save_path, map_location=INFERENCE_DEVICE))
        for param in network.parameters():
            param.requires_grad = False
        network.eval()

        print("Successfully created agent with pre-trained MPNN.\nMPNN architecture\n\n{}".format(repr(network)))

        ####################################################
        # TEST NETWORK ON VALIDATION GRAPHS
        ####################################################
        results, results_raw, history = test_network(network, env_args, graphs_test, INFERENCE_DEVICE, step_factor, n_attempts=50,
                                                     # step_factor is 1
                                                     return_raw=True, return_history=True,
                                                     batched=batched, max_batch_size=max_batch_size)
        run_duration = time.time() - start_time
        results_fname = f"results_{os.path.splitext(graph_name)[0]}.pkl"
        results_raw_fname = f"results_{os.path.splitext(graph_name)[0]}_raw.pkl"
        history_fname = f"results_{os.path.splitext(graph_name)[0]}_history.pkl"

        for res, fname, label in zip([results, results_raw, history],
                                     [results_fname, results_raw_fname, history_fname],
                                     ["results", "results_raw", "history"]):
            if label == "results":
                result = (res['sol'][0] + 1) / 2
                result = result.astype(int)
                obj = res['cut'][0]
                num_nodes = len(result)
                write_graph_result(obj, run_duration, num_nodes, ALG.value, result, graph_dict, plus1=True)

            save_path = os.path.join(data_folder, fname).replace("\\", "/")


if __name__ == "__main__":
    prefixes = INFERENCE_PREFIXES
    run(max_parallel_jobs=3, prefixes=prefixes)
