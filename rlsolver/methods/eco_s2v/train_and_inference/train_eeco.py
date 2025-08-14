import rlsolver.methods.eco_s2v.src.envs.core as ising_env
from rlsolver.methods.eco_s2v.config import *
from rlsolver.methods.eco_s2v.src.agents.dqn_eeco import DQN
from rlsolver.methods.eco_s2v.src.agents.utils import TestMetric
from rlsolver.methods.eco_s2v.src.envs.util_envs_eeco import (RandomBarabasiAlbertGraphGenerator,
                                                              RandomErdosRenyiGraphGenerator, ValidationGraphGenerator,
                                                              )
from rlsolver.methods.eco_s2v.src.envs.util_envs import (EdgeType, RewardSignal, ExtraAction,
                                                         OptimisationTarget, SpinBasis, DEFAULT_OBSERVABLES)
from rlsolver.methods.eco_s2v.src.networks.mpnn import MPNN
from rlsolver.methods.eco_s2v.util import (cal_txt_name)

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
                'basin_reward': 1. / NUM_TRAIN_NODES,
                'reversible_spins': True,
                }

    ####################################################
    # SET UP TRAINING AND TEST GRAPHS
    ####################################################
    start_time = time.time()
    n_spins_train = NUM_TRAIN_NODES

    if GRAPH_TYPE == GraphType.ER:
        train_graph_generator = RandomErdosRenyiGraphGenerator(n_spins=n_spins_train, p_connection=0.15,
                                                               edge_type=EdgeType.DISCRETE, n_sims=NUM_TRAIN_SIMS, device=TRAIN_DEVICE)
    if GRAPH_TYPE == GraphType.BA:
        train_graph_generator = RandomBarabasiAlbertGraphGenerator(n_spins=n_spins_train, m_insertion_edges=4,
                                                                   edge_type=EdgeType.DISCRETE, n_sims=NUM_TRAIN_SIMS, device=TRAIN_DEVICE)

    validation_graph_generator = ValidationGraphGenerator(n_spins=NUM_VALIDATION_NODES, graph_type=GRAPH_TYPE,
                                                          edge_type=EdgeType.DISCRETE, device=TRAIN_DEVICE,
                                                          n_sims=NUM_VALIDATION_SIMS, seed=VALIDATION_SEED)

    ####
    # Pre-generated test graphs
    ####
    graphs_validation = validation_graph_generator.get()
    n_validations = graphs_validation.shape[0]
    # validation_graph_generator = SetGraphGenerator(graphs_validation)
    ####################################################
    # SET UP TRAINING AND TEST ENVIRONMENTS
    ####################################################

    device = SAMPLE_DEVICE_IN_ECO_S2V if USE_TWO_DEVICES_IN_ECO_S2V else TRAIN_DEVICE
    train_envs = ising_env.make("SpinSystem",
                                train_graph_generator,
                                int(n_spins_train * step_fact),
                                **env_args, device=device,
                                n_sims=NUM_TRAIN_SIMS)

    n_spins_test = validation_graph_generator.get().shape[1]
    test_envs = ising_env.make("SpinSystem",
                               validation_graph_generator,
                               int(n_spins_test * step_fact),
                               **env_args, device=device,
                               n_sims=n_validations)

    pre_fix = save_loc + "/" + NEURAL_NETWORK_PREFIX
    pre_fix = cal_txt_name(pre_fix)
    # network_save_path = pre_fix + "/" + NEURAL_NETWORK_PREFIX + ".pth"
    network_save_path = pre_fix + "/" + NEURAL_NETWORK_PREFIX + ".pth"
    test_save_path = pre_fix + "/test_scores.pkl"
    loss_save_path = pre_fix + "/losses.pkl"
    logger_save_path = pre_fix + f"/logger.json"
    sampling_speed_save_path = pre_fix + "/sampling_speed.json"
    print('pre_fix:', pre_fix.split("/")[-1])

    ####################################################
    # SET UP AGENT
    ####################################################

    nb_steps = NB_STEPS

    network_fn = lambda: MPNN(n_obs_in=train_envs.observation_space.shape[1],
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
        'replay_start_size': REPLAY_START_SIZE,
        'replay_buffer_size': REPLAY_BUFFER_SIZE,
        'gamma': gamma,
        'update_learning_rate': False,
        'initial_learning_rate': 1e-3,
        'peak_learning_rate': 1e-4,
        'peak_learning_rate_step': 15000,
        'final_learning_rate': 1e-4,
        'final_learning_rate_step': 200000,
        # 'minibatch_size': int(NUM_TRAIN_SIMS*2),
        'minibatch_size': 64,
        'max_grad_norm': None,
        'weight_decay': 0,
        'update_exploration': False,
        'initial_exploration_rate': 0.05,
        'final_exploration_rate': 0.05,
        'final_exploration_step': FINAL_EXPLORATION_STEP,
        'adam_epsilon': 1e-8,
        'logging': True,
        'evaluate': True,
        'update_target_frequency': UPDATE_TARGET_FREQUENCY,
        'update_frequency': UPDATE_FREQUENCY,
        'save_network_frequency': SAVE_NETWORK_FREQUENCY,
        'loss': "mse",
        'network_save_path': network_save_path,
        'test_envs': test_envs,
        'test_episodes': n_validations,
        'test_obj_frequency': TEST_OBJ_FREQUENCY,
        'test_save_path': test_save_path,
        'test_metric': TestMetric.MAX_CUT,
        'logger_save_path': logger_save_path,
        'sampling_speed_save_path': sampling_speed_save_path,
        'seed': None,
        'test_sampling_speed': TEST_SAMPLING_SPEED,
        'sampling_patten': "best_score",
        'sample_device': device, # added
        'buffer_device': BUFFER_DEVICE, # added
    }
    args['args'] = args
    # if TEST_SAMPLING_SPEED:
    #     nb_steps = int(2e4)
    #     args['test_obj_frequency'] = args['update_target_frequency'] = args['update_frequency'] = args[
    #         'save_network_frequency'] = 1e6
    #     args['replay_start_size'] = args['initial_exploration_rate'] =0
    #     args['replay_buffer_size'] = NUM_TRAIN_SIMS
    #     args['update_exploration'] = False
    agent = DQN(**args)
    #############
    # TRAIN AGENT
    #############
    agent.learn(timesteps=nb_steps, start_time=start_time, verbose=True)


if __name__ == "__main__":
    run(save_loc=NEURAL_NETWORK_DIR)
