# from numba.cuda.cudadrv.nvrtc import NVRTC

import torch

import rlsolver.methods.eco_s2v.src.envs.core as ising_env
from rlsolver.methods.eco_s2v.config import *
from rlsolver.methods.eco_s2v.plot import plot_scatter
from rlsolver.methods.eco_s2v.src.agents.dqn_torch import DQN
from rlsolver.methods.eco_s2v.src.agents.utils import TestMetric
from rlsolver.methods.eco_s2v.src.envs.util_envs_torch import (SetGraphGenerator,
                                                               RandomBarabasiAlbertGraphGenerator, RandomErdosRenyiGraphGenerator,
                                                               EdgeType, RewardSignal, ExtraAction,
                                                               OptimisationTarget, SpinBasis,
                                                               DEFAULT_OBSERVABLES)
from rlsolver.methods.eco_s2v.src.networks.mpnn import MPNN
from rlsolver.methods.eco_s2v.util import write_sampling_speed, load_graph_set_from_folder

try:
    import seaborn as sns

    sns.set_style("whitegrid")
    # plt.style.use('seaborn')
except ImportError:
    pass

import time


def run(save_loc):
    print("\n----- Running {} -----\n".format(os.path.basename(__file__)))

    ####################################################
    # SET UP ENVIRONMENTAL AND VARIABLES
    ####################################################

    gamma = 0.95
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
                # 'basin_reward':1./200,
                'basin_reward': 1. / NUM_TRAIN_NODES,
                'reversible_spins': True}

    ####################################################
    # SET UP TRAINING AND TEST GRAPHS
    ####################################################
    start_time = time.time()
    n_spins_train = NUM_TRAIN_NODES

    if GRAPH_TYPE == GraphType.ER:
        train_graph_generator = RandomErdosRenyiGraphGenerator(n_spins=n_spins_train, p_connection=0.15,
                                                               edge_type=EdgeType.DISCRETE)
    if GRAPH_TYPE == GraphType.BA:
        train_graph_generator = RandomBarabasiAlbertGraphGenerator(n_spins=n_spins_train, m_insertion_edges=4,
                                                                   edge_type=EdgeType.DISCRETE)

    ####
    # Pre-generated test graphs
    ####
    graphs_test = load_graph_set_from_folder(NEURAL_NETWORK_SAVE_PATH)
    graphs_test = [torch.tensor(m, dtype=torch.float, device=TRAIN_DEVICE) for m in graphs_test]
    n_tests = len(graphs_test)

    test_graph_generator = SetGraphGenerator(graphs_test, ordered=True)

    ####################################################
    # SET UP TRAINING AND TEST ENVIRONMENTS
    ####################################################

    train_envs = [ising_env.make("SpinSystem",
                                 train_graph_generator,
                                 int(n_spins_train * step_fact),
                                 **env_args)]

    n_spins_test = train_graph_generator.get().shape[0]
    test_envs = [ising_env.make("SpinSystem",
                                test_graph_generator,
                                int(n_spins_test * step_fact),
                                **env_args)]

    ####################################################
    # SET UP FOLDERS FOR SAVING DATA
    ####################################################

    # pre_fix = save_loc + "/" + ALG.value + "_" + GRAPH_TYPE.value + "_" + str(NUM_TRAIN_NODES) + "_"
    pre_fix = save_loc + "/" + NEURAL_NETWORK_PREFIX
    # network_save_path = pre_fix + "network.pth"
    network_save_path = pre_fix + "/" + NEURAL_NETWORK_PREFIX + ".pth"
    test_save_path = pre_fix + "test_scores.pkl"
    loss_save_path = pre_fix + "losses.pkl"
    logger_save_path = pre_fix + "logger.txt"
    sampling_speed_save_path = pre_fix + "sampling_speed.txt"

    ####################################################
    # SET UP AGENT
    ####################################################

    nb_steps = NB_STEPS

    network_fn = lambda: MPNN(n_obs_in=train_envs[0].observation_space.shape[1],
                              n_layers=3,
                              n_features=64,
                              n_hid_readout=[],
                              tied_weights=False)
    args = {
        'envs': train_envs,
        'network': network_fn,
        'init_network_params': None,
        'init_weight_std': 0.01,
        'double_dqn': True,
        'clip_Q_targets': False,
        'replay_start_size': int(round(REPLAY_START_SIZE / (NUM_TRAIN_SIMS))),
        'replay_buffer_size': REPLAY_BUFFER_SIZE,
        'gamma': gamma,
        'update_learning_rate': False,
        'initial_learning_rate': 1e-4,
        'peak_learning_rate': 1e-3,
        'peak_learning_rate_step': 5000,
        'final_learning_rate': 1e-4,
        'final_learning_rate_step': 200000,
        'minibatch_size': 64,
        'max_grad_norm': None,
        'weight_decay': 0,
        'update_exploration': True,
        'initial_exploration_rate': 1,
        'final_exploration_rate': 0.05,
        'final_exploration_step': FINAL_EXPLORATION_STEP,
        'adam_epsilon': 1e-8,
        'logging': True,
        'evaluate': True,
        'update_target_frequency': max(1, int(round(UPDATE_TARGET_FREQUENCY / (NUM_TRAIN_SIMS)))),
        'update_frequency': max(1, int(UPDATE_FREQUENCY / (NUM_TRAIN_SIMS))),
        'save_network_frequency': SAVE_NETWORK_FREQUENCY,
        'loss': "mse",
        'network_save_path': network_save_path,
        'test_envs': test_envs,
        'test_episodes': n_tests,
        'test_obj_frequency': TEST_OBJ_FREQUENCY,
        'test_save_path': test_save_path,
        'test_metric': TestMetric.MAX_CUT,
        'logger_save_path': logger_save_path,
        'seed': None,
        'test_sampling_speed': TEST_SAMPLING_SPEED
    }
    # if TEST_SAMPLING_SPEED:
    #         nb_steps = 2000
    #         args['test_obj_frequency']=args['update_target_frequency']=args['update_frequency']=args['save_network_frequency']=1e6
    #         args['replay_start_size'] = 0
    agent = DQN(**args)

    print("\n Created DQN agent with network:\n\n", agent.network)

    #############
    # TRAIN AGENT
    #############
    sampling_start_time = time.time()
    agent.learn(timesteps=nb_steps, start_time=start_time, verbose=True)
    print(time.time() - start_time)
    if TEST_SAMPLING_SPEED:
        sampling_speed = NUM_TRAIN_SIMS * nb_steps / (time.time() - sampling_start_time)
        write_sampling_speed(sampling_speed_save_path, sampling_speed)

    else:
        plot_scatter(logger_save_path)


if __name__ == "__main__":
    run(save_loc=NEURAL_NETWORK_DIR)
