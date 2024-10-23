import os

import matplotlib.pyplot as plt
import torch

import rlsolver.methods.eco_and_s2v_dqn.src.envs.core as ising_env
from rlsolver.methods.eco_and_s2v_dqn.experiments.utils import test_network, load_graph_set, mk_dir
from rlsolver.methods.eco_and_s2v_dqn.src.envs.utils import (SingleGraphGenerator,
                            RewardSignal, ExtraAction,
                            OptimisationTarget, SpinBasis,
                            DEFAULT_OBSERVABLES)
from rlsolver.methods.eco_and_s2v_dqn.src.networks.mpnn import MPNN
from rlsolver.methods.eco_and_s2v_dqn.config.eco_config import *

try:
    import seaborn as sns
    plt.style.use('seaborn')
except ImportError:
    pass

def run(save_loc="pretrained_agent/eco",
        network_save_loc="experiments_new/pretrained_agent/networks/eco/network_best_ER_200spin.pth",
        graph_save_loc="../../../_graphs/validation/ER_200spin_p15_100graphs.pkl",
        batched=True,
        max_batch_size=None,
        step_factor=None,
        n_attemps=50):

    print("\n----- Running {} -----\n".format(os.path.basename(__file__)))

    ####################################################
    # FOLDER LOCATIONS
    ####################################################

    print("save location :", save_loc)
    print("network params :", network_save_loc)
    mk_dir(save_loc)

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

    if step_factor is None:
        step_factor = 2

    env_args = {'observables': DEFAULT_OBSERVABLES,
                'reward_signal': RewardSignal.BLS,
                'extra_action': ExtraAction.NONE,
                'optimisation_target': OptimisationTarget.CUT,
                'spin_basis': SpinBasis.BINARY,
                'norm_rewards': True,
                'memory_length': None,
                'horizon_length': None,
                'stag_punishment': None,
                'basin_reward': None,
                'reversible_spins': True}

    ####################################################
    # LOAD VALIDATION GRAPHS
    ####################################################

    graphs_test = load_graph_set(graph_save_loc)

    ####################################################
    # SETUP NETWORK TO TEST
    ####################################################

    test_env = ising_env.make("SpinSystem",
                              SingleGraphGenerator(graphs_test[0]),
                              graphs_test[0].shape[0] * step_factor,
                              **env_args)

    device = DEVICE
    torch.device(device)
    print("Set torch default device to {}.".format(device))

    network = network_fn(n_obs_in=test_env.observation_space.shape[1],
                         **network_args).to(device)

    network.load_state_dict(torch.load(network_save_loc, map_location=device))
    for param in network.parameters():
        param.requires_grad = False
    network.eval()

    print("Sucessfully created agent with pre-trained MPNN.\nMPNN architecture\n\n{}".format(repr(network)))

    ####################################################
    # TEST NETWORK ON VALIDATION GRAPHS
    ####################################################

    results, results_raw, history = test_network(network, env_args, graphs_test, device, step_factor,
                                                 return_raw=True, return_history=True, n_attempts=n_attemps,
                                                 batched=batched, max_batch_size=max_batch_size)

    results_fname = "results_" + os.path.splitext(os.path.split(graph_save_loc)[-1])[0] + ".pkl"
    results_raw_fname = "results_" + os.path.splitext(os.path.split(graph_save_loc)[-1])[0] + "_raw.pkl"
    history_fname = "results_" + os.path.splitext(os.path.split(graph_save_loc)[-1])[0] + "_history.pkl"

    for res, fname, label in zip([results, results_raw, history],
                                 [results_fname, results_raw_fname, history_fname],
                                 ["results", "results_raw", "history"]):
        save_path = os.path.join(save_loc, fname)
        res.to_pickle(save_path)
        print("{} saved to {}".format(label, save_path))

if __name__ == "__main__":
    run()