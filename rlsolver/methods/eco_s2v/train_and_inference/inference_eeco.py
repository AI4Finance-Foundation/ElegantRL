import os
import sys

cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../../../rlsolver')
sys.path.append(os.path.dirname(rlsolver_path))

import time
import torch
import networkx as nx
from rlsolver.methods.eco_s2v.src.envs.inference_network_env import SpinSystemFactory
from rlsolver.methods.eco_s2v.util import eeco_test_network, load_graph_from_txt
from rlsolver.methods.eco_s2v.src.envs.util_envs_eeco import (SetGraphGenerator,
                                                              )
from rlsolver.methods.eco_s2v.src.envs.util_envs import (RewardSignal, ExtraAction,
                                                         OptimisationTarget, SpinBasis,
                                                         DEFAULT_OBSERVABLES)
from rlsolver.methods.eco_s2v.src.networks.mpnn import MPNN

from rlsolver.methods.util_result import write_graph_result
from rlsolver.methods.eco_s2v.config import *


def run(graph_folder="../../data/syn_BA",
        if_greedy=False,
        n_sims=1,
        mini_sims=10,
        network_save_path=NEURAL_NETWORK_SAVE_PATH):  # 设置 mini_sims 以减少显存占用
    print("\n----- Running {} -----\n".format(os.path.basename(__file__)))

    print("Testing network: ", network_save_path)

    network_fn = MPNN
    network_args = {
        'n_layers': 3,
        'n_features': 64,
        'n_hid_readout': [],
        'tied_weights': False
    }
    network = network_fn(n_obs_in=7, **network_args).to(INFERENCE_DEVICE)

    network.load_state_dict(torch.load(network_save_path, map_location=INFERENCE_DEVICE))
    for param in network.parameters():
        param.requires_grad = False
    network.eval()

    if ALG == Alg.eeco:
        env_args = {
            'observables': DEFAULT_OBSERVABLES,
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
            'if_greedy': if_greedy,
            'use_tensor_core': False
        }

    step_factor = 2
    files = os.listdir(graph_folder)

    for prefix in INFERENCE_PREFIXES:
        graphs = []
        file_list = []
        for file in files:
            if prefix in file:
                file = os.path.join(graph_folder, file).replace("\\", "/")
                file_list.append(file)
                g = load_graph_from_txt(file)
                g_array = nx.to_numpy_array(g)
                g_tensor = torch.tensor(g_array, dtype=torch.float, device=INFERENCE_DEVICE)
                graphs.append(g_tensor)

        if len(graphs) > 0:
            for i, graph_tensor in enumerate(graphs):
                test_graph_generator = SetGraphGenerator(graph_tensor, device=INFERENCE_DEVICE)

                best_obj = float('-inf')
                best_sol = None

                num_batches = (n_sims + mini_sims - 1) // mini_sims  # 计算分批次数
                for batch in range(num_batches):
                    current_mini_sims = min(mini_sims, n_sims - batch * mini_sims)  # 防止超出 n_sims

                    test_env = SpinSystemFactory.get(
                        test_graph_generator,
                        graph_tensor.shape[0] * step_factor,
                        **env_args,
                        device=INFERENCE_DEVICE,
                        n_sims=current_mini_sims,  # 只处理 mini_sims 个环境
                    )

                    start_time = time.time()
                    result, sol = eeco_test_network(network, test_env, LOCAL_SEARCH_FREQUENCY)

                    if result['obj'] > best_obj:  # 记录最佳结果
                        best_obj = result['obj']
                        best_sol = result['sol']
                run_duration = time.time() - start_time
                sol = (best_sol + 1) / 2
                write_graph_result(best_obj, run_duration, sol.shape[0], ALG.value, sol.to(torch.int), file_list[i], plus1=True)


if __name__ == "__main__":
    run(graph_folder='../../../rlsolver/data/syn_BA',
        if_greedy=False,
        n_sims=NUM_INFERENCE_SIMS,
        mini_sims=MINI_INFERENCE_SIMS)
