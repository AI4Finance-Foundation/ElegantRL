import os
import sys

cur_path = os.path.dirname(os.path.abspath(__file__))
rlsolver_path = os.path.join(cur_path, '../../../../../')
sys.path.append(os.path.dirname(rlsolver_path))
import time
import json
import torch as th
import tqdm

from rlsolver.methods.eco_s2v.config import *
from rlsolver.methods.eco_s2v.jumanji.train.config import Config
from rlsolver.methods.eco_s2v.jumanji.agents.AgentPPO import AgentA2C
from rlsolver.methods.eco_s2v.src.envs.spinsystem_eeco import SpinSystemFactory

from rlsolver.methods.eco_s2v.src.envs.util_envs import (EdgeType, RewardSignal, ExtraAction,
                                                         OptimisationTarget, SpinBasis,
                                                         DEFAULT_OBSERVABLES)

from rlsolver.methods.eco_s2v.src.envs.util_envs_eeco import (RandomBarabasiAlbertGraphGenerator,
                                                              RandomErdosRenyiGraphGenerator, ValidationGraphGenerator,
                                                              )
from rlsolver.methods.eco_s2v.util import cal_txt_name


def run(save_loc, graph_save_loc):
    print("\n----- Running {} -----\n".format(os.path.basename(__file__)))
    pre_fix = save_loc + "/" + NEURAL_NETWORK_PREFIX
    pre_fix = cal_txt_name(pre_fix)
    network_save_path = pre_fix + "/" + NEURAL_NETWORK_PREFIX + ".pth"
    logger_save_path = pre_fix + f"/logger.json"
    sampling_speed_save_path = pre_fix + "/sampling_speed.json"
    print('pre_fix:', pre_fix.split("/")[-1])
    if not os.path.exists(pre_fix):
        os.makedirs(pre_fix)
    start_time = time.time()
    total_time = 0
    step_fact = 2

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
                }
    n_spins_train = NUM_TRAIN_NODES

    if GRAPH_TYPE == GraphType.ER:
        train_graph_generator = RandomErdosRenyiGraphGenerator(n_spins=n_spins_train, p_connection=0.15,
                                                               edge_type=EdgeType.DISCRETE, n_sims=NUM_TRAIN_SIMS)
    if GRAPH_TYPE == GraphType.BA:
        train_graph_generator = RandomBarabasiAlbertGraphGenerator(n_spins=n_spins_train, m_insertion_edges=4,
                                                                   edge_type=EdgeType.DISCRETE, n_sims=NUM_TRAIN_SIMS, device=TRAIN_DEVICE)

    validation_graph_generator = ValidationGraphGenerator(n_spins=NUM_VALIDATION_NODES, graph_type=GRAPH_TYPE,
                                                          edge_type=EdgeType.DISCRETE, device=TRAIN_DEVICE,
                                                          n_sims=NUM_VALIDATION_SIMS, seed=VALIDATION_SEED)
    graphs_validation = validation_graph_generator.get()
    n_validations = graphs_validation.shape[0]

    train_envs = SpinSystemFactory.get(train_graph_generator,
                                       int(n_spins_train * step_fact),
                                       **env_args, device=TRAIN_DEVICE,
                                       n_sims=NUM_TRAIN_SIMS)
    n_spins_test = validation_graph_generator.get().shape[1]
    test_envs = SpinSystemFactory.get(validation_graph_generator,
                                      int(n_spins_test * step_fact),
                                      **env_args, device=TRAIN_DEVICE,
                                      n_sims=n_validations)
    env_args_ = {
        'env_name': 'maxcut',
        'num_envs': NUM_TRAIN_SIMS,
        'num_nodes': NUM_TRAIN_NODES,
        'state_dim': 2,
        'action_dim': 1,
        'if_discrete': True,
    }
    args = Config(AgentA2C, "maxcut", env_args_)
    args.gpu_id = TRAIN_GPU_ID
    agent = args.agent_class(args.net_dims, args.state_dim, args.action_dim, gpu_id=args.gpu_id, args=args)
    path_main, path_ext = os.path.splitext(network_save_path)
    if path_ext == '':
        path_ext += '.pth'
    last_record_time = time.time()
    last_record_step = 0
    sampling_speed_vs_timestep = {}
    obj_vs_time = {}
    th.set_grad_enabled(False)
    obj_vs_time['0'] = th.mean(agent.inference(env=test_envs, max_steps=NUM_VALIDATION_NODES * step_fact)).item()
    best_score = float('-inf')

    buffer = []
    for i in tqdm.tqdm(range(JUMANJI_NB_STEPS)):
        buffer_items = agent._explore_vec_env(env=train_envs, horizon_len=HERIZON_LENGTH)
        buffer[:] = buffer_items
        if not TEST_SAMPLING_SPEED:
            th.set_grad_enabled(True)
            logging_tuple = agent.update_net(buffer)
        th.set_grad_enabled(False)

        if (time.time() - last_record_time >= JUMANJI_TEST_OBJ_FREQUENCY) and not TEST_SAMPLING_SPEED:
            total_time += time.time() - start_time
            test_score = th.mean(agent.inference(env=test_envs, max_steps=NUM_VALIDATION_NODES * step_fact)).item()
            obj_vs_time[f'{total_time}'] = test_score
            last_record_time = time.time()
            if test_score > best_score:
                agent.save(path_main + "_best" + path_ext)
                start_time = time.time()

        # if TEST_SAMPLING_SPEED:
        #     sampling_speed_vs_timestep[f'{i}'] = (NUM_TRAIN_SIMS * HERIZON_LENGTH)/(time.time()-last_record_time)
        #     last_record_time =time.time()

        result_dict = {}
        result_dict['alg'] = "jumanji"
        result_dict['n_sims'] = NUM_TRAIN_SIMS
    if TEST_SAMPLING_SPEED:
        result_dict['sampling_speed'] = sampling_speed_vs_timestep
        with open(sampling_speed_save_path, 'w') as json_file:
            json.dump(result_dict, json_file, indent=4)
    else:
        result_dict['obj_vs_time'] = obj_vs_time
        with open(logger_save_path, 'w') as json_file:
            json.dump(result_dict, json_file, indent=4)
    print(f"result saved to{pre_fix}")
