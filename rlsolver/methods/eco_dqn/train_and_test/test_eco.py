import os

import matplotlib.pyplot as plt
import torch

import rlsolver.methods.eco_dqn.src.envs.core as ising_env
from rlsolver.methods.eco_dqn.utils import test_network, load_graph_set
from rlsolver.methods.eco_dqn.src.envs.utils import (SingleGraphGenerator,
                            RewardSignal, ExtraAction,
                            OptimisationTarget, SpinBasis,
                            DEFAULT_OBSERVABLES)
from rlsolver.methods.eco_dqn.src.networks.mpnn import MPNN

try:
    import seaborn as sns
    plt.style.use('seaborn')
except ImportError:
    pass

def run(save_loc="BA_40spin/eco",
        graph_save_loc="../../../_graphs/validation/BA_40spin_m4_100graphs.pkl",
        batched=True,
        max_batch_size=None):

    print("\n----- Running {} -----\n".format(os.path.basename(__file__)))

    ####################################################
    # NETWORK LOCATION
    ####################################################

    data_folder = os.path.join(save_loc, 'data')
    network_folder = os.path.join(save_loc, 'network')

    print("data folder :", data_folder)
    print("network folder :", network_folder)

    test_save_path = os.path.join(network_folder, 'test_scores.pkl')
    network_save_path = os.path.join(network_folder, 'network_best.pth')

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

    gamma = 0.95
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
                'basin_reward': 1. / 40,
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
                              graphs_test[0].shape[0]*step_factor,
                              **env_args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.device(device)
    print("Set torch default device to {}.".format(device))

    network = network_fn(n_obs_in=test_env.observation_space.shape[1],
                         **network_args).to(device)

    network.load_state_dict(torch.load(network_save_path,map_location=device))
    for param in network.parameters():
        param.requires_grad = False
    network.eval()

    print("Sucessfully created agent with pre-trained MPNN.\nMPNN architecture\n\n{}".format(repr(network)))

    ####################################################
    # TEST NETWORK ON VALIDATION GRAPHS
    ####################################################

    results, results_raw, history = test_network(network, env_args, graphs_test, device, step_factor,
                                                 return_raw=True, return_history=True,
                                                 batched=batched, max_batch_size=max_batch_size)

    results_fname = "results_" + os.path.splitext(os.path.split(graph_save_loc)[-1])[0] + ".pkl"
    results_raw_fname = "results_" + os.path.splitext(os.path.split(graph_save_loc)[-1])[0] + "_raw.pkl"
    history_fname = "results_" + os.path.splitext(os.path.split(graph_save_loc)[-1])[0] + "_history.pkl"

    for res, fname, label in zip([results, results_raw, history],
                                 [results_fname, results_raw_fname, history_fname],
                                 ["results", "results_raw", "history"]):
        save_path = os.path.join(data_folder, fname)
        res.to_pickle(save_path)
        print("{} saved to {}".format(label, save_path))

if __name__ == "__main__":
    run()