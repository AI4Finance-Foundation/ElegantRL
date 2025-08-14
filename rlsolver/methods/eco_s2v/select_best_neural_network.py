import os
import sys

cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../../rlsolver')
sys.path.append(os.path.dirname(rlsolver_path))

import torch
import shutil
import numpy as np

from rlsolver.methods.eco_s2v.src.envs.inference_network_env import SpinSystemFactory
from rlsolver.methods.eco_s2v.util import eeco_test_network
from rlsolver.methods.eco_s2v.src.envs.util_envs_eeco import (SetGraphGenerator)
from rlsolver.methods.eco_s2v.src.envs.util_envs import (EdgeType, Observable,
                                                         RewardSignal, ExtraAction,
                                                         OptimisationTarget, SpinBasis,
                                                         DEFAULT_OBSERVABLES)
from rlsolver.methods.eco_s2v.src.networks.mpnn import MPNN
from rlsolver.methods.eco_s2v.util import test_network

from rlsolver.methods.eco_s2v.config import *
import json

"""
逻辑是先读网络文件夹中的网络，再读图文件夹中的图，对一张图开n个环境,取最大值，结果文件要以网络文件命名，结果文件中的内容是图的名称.
在测试网络的过程中，为提高效率，图文件夹中只放一张图
"""


def run(neural_network_folder, n_sims, mini_sims, num_generated_instances, alg, num_nodes, graph_type):
    print("\n----- Running {} -----\n".format(os.path.basename(__file__)))
    network_result_save_path = neural_network_folder + "/" + neural_network_folder.split("/")[-1] + ".json"
    networks = os.listdir(neural_network_folder)
    total_result = []
    network_path = []
    obj_vs_time = {}
    data = {}
    if alg == Alg.eco or alg == Alg.eeco:
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
            'if_greedy': False,
        }
    elif alg == Alg.s2v:
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
                    'if_greedy': False}
    if alg == Alg.eeco:
        from rlsolver.methods.eco_s2v.src.envs.util_envs_eeco import ValidationGraphGenerator
        validation_graph_generator = ValidationGraphGenerator(n_spins=num_nodes, graph_type=graph_type,
                                                              edge_type=EdgeType.DISCRETE, device=INFERENCE_DEVICE,
                                                              n_sims=num_generated_instances, seed=VALIDATION_SEED)
    elif alg == Alg.s2v or alg == Alg.eco:
        from rlsolver.methods.eco_s2v.src.envs.util_envs import ValidationGraphGenerator
        validation_graph_generator = ValidationGraphGenerator(n_spins=num_nodes, graph_type=graph_type,
                                                              edge_type=EdgeType.DISCRETE, seed=VALIDATION_SEED,
                                                              n_sims=num_generated_instances)

    graphs = validation_graph_generator.get()

    for network_name in networks:
        if network_name.endswith(".json"):
            with open(os.path.join(neural_network_folder, network_name), "r") as f:
                data = json.load(f)
        if network_name.endswith(".pth"):
            network_time = network_name.split("_")[-1].split(".")[0]
            network_results = []
            network_save_path = os.path.join(neural_network_folder, network_name)
            network_path.append(network_name)
            network_fn = MPNN
            network_args = {
                'n_layers': 3,
                'n_features': 64,
                'n_hid_readout': [],
                'tied_weights': False
            }
            if alg == Alg.eco or alg == Alg.eeco:
                network = network_fn(n_obs_in=7, **network_args).to(INFERENCE_DEVICE)
            elif alg == Alg.s2v:
                network = network_fn(n_obs_in=1, **network_args).to(INFERENCE_DEVICE)
            network.load_state_dict(torch.load(network_save_path, map_location=INFERENCE_DEVICE))
            for param in network.parameters():
                param.requires_grad = False
            network.eval()
            step_factor = 2
            if alg == Alg.eco or alg == Alg.s2v:
                for i, graph_array in enumerate(graphs):
                    results, results_raw, history = test_network(network, env_args, [graph_array], INFERENCE_DEVICE, step_factor, n_attempts=50,
                                                                 # step_factor is 1
                                                                 return_raw=True, return_history=True,
                                                                 batched=True, max_batch_size=None)
                    best_obj = results['cut'][0]
                    network_results.append(best_obj)
            elif alg == Alg.eeco:
                for i, graph_tensor in enumerate(graphs):
                    best_obj = -1e10
                    test_graph_generator = SetGraphGenerator(graph_tensor, device=INFERENCE_DEVICE)
                    num_batches = (n_sims + mini_sims - 1) // mini_sims  # 计算分批次数
                    for batch in range(num_batches):
                        current_mini_sims = min(mini_sims, n_sims - batch * mini_sims)  # 防止超出 n_sims

                        test_env = SpinSystemFactory.get(
                            test_graph_generator,
                            graph_tensor.shape[0] * step_factor,
                            **env_args, use_tensor_core=USE_TENSOR_CORE_IN_INFERENCE,
                            device=INFERENCE_DEVICE,
                            n_sims=current_mini_sims,  # 只处理 mini_sims 个环境
                        )

                        result, sol = eeco_test_network(network, test_env, LOCAL_SEARCH_FREQUENCY)

                        if result['obj'] > best_obj:  # 记录最佳结果
                            best_obj = result['obj']
                        network_results.append(best_obj)
            network_performance = np.mean(np.array(network_results))
            total_result.append(network_performance)
            obj_vs_time[network_time] = network_performance.item()
            obj_vs_time = dict(sorted(obj_vs_time.items(), key=lambda item: item[1]))
    data['obj_vs_time'] = obj_vs_time
    best_network = network_path[np.argmax(np.array(total_result))]
    best_network_path = os.path.join(neural_network_folder, best_network.replace('.pth', '_best.pth'))
    shutil.copy(os.path.join(neural_network_folder, best_network), best_network_path)
    with open(network_result_save_path, 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    LOCAL_SEARCH_FREQUENCY = 10000000000
    run(neural_network_folder=NEURAL_NETWORK_FOLDER, n_sims=NUM_INFERENCE_SIMS,
        mini_sims=MINI_INFERENCE_SIMS, num_generated_instances=NUM_GENERATED_INSTANCES_IN_SELECT_BEST,
        alg=ALG, num_nodes=NUM_TRAINED_NODES_IN_INFERENCE,
        graph_type=GRAPH_TYPE)
