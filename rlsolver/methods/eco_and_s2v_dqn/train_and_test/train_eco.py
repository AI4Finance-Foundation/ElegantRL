import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

import rlsolver.methods.eco_and_s2v_dqn.src.envs.core as ising_env
from rlsolver.methods.eco_and_s2v_dqn.utils import load_graph_set, mk_dir,load_graph_set_from_folder
from rlsolver.methods.eco_and_s2v_dqn.src.agents.dqn.dqn import DQN
from rlsolver.methods.eco_and_s2v_dqn.src.agents.dqn.utils import TestMetric
from rlsolver.methods.eco_and_s2v_dqn.src.envs.utils import (SetGraphGenerator,
                            RandomBarabasiAlbertGraphGenerator,
                            EdgeType, RewardSignal, ExtraAction,
                            OptimisationTarget, SpinBasis,
                            DEFAULT_OBSERVABLES)
from rlsolver.methods.eco_and_s2v_dqn.src.networks.mpnn import MPNN
from rlsolver.methods.eco_and_s2v_dqn.config.eco_config import *

try:
    import seaborn as sns
    plt.style.use('seaborn')
except ImportError:
    pass

import time

def run(save_loc="BA_200spin/eco", graph_save_loc="../../data/syn_BA"):

    print("\n----- Running {} -----\n".format(os.path.basename(__file__)))

    ####################################################
    # SET UP ENVIRONMENTAL AND VARIABLES
    ####################################################

    gamma=0.95
    step_fact = 2

    env_args = {'observables':DEFAULT_OBSERVABLES,
                'reward_signal':RewardSignal.BLS,
                'extra_action':ExtraAction.NONE,
                'optimisation_target':OptimisationTarget.CUT,
                'spin_basis':SpinBasis.BINARY,
                'norm_rewards':True,
                'memory_length':None,
                'horizon_length':None,
                'stag_punishment':None,
                # 'basin_reward':1./200,
                'basin_reward': 1./NODES,
                'reversible_spins':True}

    ####################################################
    # SET UP TRAINING AND TEST GRAPHS
    ####################################################

    n_spins_train = N_SPINS_TRAIN
#m_insertion_edges、EdgeType.DISCRETE的作用有疑问
    train_graph_generator = RandomBarabasiAlbertGraphGenerator(n_spins=n_spins_train,m_insertion_edges=4,edge_type=EdgeType.DISCRETE)

    ####
    # Pre-generated test graphs
    ####

    # graph_save_loc = "../../data/syn_BA/BA_100_ID0.txt"
    graphs_test = load_graph_set_from_folder(graph_save_loc)
    n_tests = len(graphs_test)

    test_graph_generator = SetGraphGenerator(graphs_test, ordered=True)

    ####################################################
    # SET UP TRAINING AND TEST ENVIRONMENTS
    ####################################################

    train_envs = [ising_env.make("SpinSystem",
                                 train_graph_generator,
                                 int(n_spins_train*step_fact),
                                 **env_args)]


    n_spins_test = train_graph_generator.get().shape[0]
    test_envs = [ising_env.make("SpinSystem",
                                test_graph_generator,
                                int(n_spins_test*step_fact),
                                **env_args)]

    ####################################################
    # SET UP FOLDERS FOR SAVING DATA
    ####################################################

    data_folder = os.path.join(save_loc,'data')
    network_folder = os.path.join(save_loc, 'network')

    mk_dir(data_folder)
    mk_dir(network_folder)
    # print(data_folder)
    network_save_path = os.path.join(network_folder,'network.pth')
    test_save_path = os.path.join(network_folder,'test_scores.pkl')
    loss_save_path = os.path.join(network_folder, 'losses.pkl')

    ####################################################
    # SET UP AGENT
    ####################################################

    nb_steps = NB_STEPS

    network_fn = lambda: MPNN(n_obs_in=train_envs[0].observation_space.shape[1],
                              n_layers=3,
                              n_features=64,
                              n_hid_readout=[],
                              tied_weights=False)

    agent = DQN(train_envs,

                network_fn,

                init_network_params=None,
                init_weight_std=0.01,

                double_dqn=True,
                clip_Q_targets=False,

                replay_start_size=REPLAY_START_SIZE,
                replay_buffer_size=REPLAY_BUFFER_SIZE,  # 20000
                gamma=gamma,  # 1
                update_target_frequency=FINAL_EXPLORATION_STEP,  # 500

                update_learning_rate=False,
                initial_learning_rate=1e-4,
                peak_learning_rate=1e-4,
                peak_learning_rate_step=20000,
                final_learning_rate=1e-4,
                final_learning_rate_step=200000,

                update_frequency=32,  # 1
                minibatch_size=64,  # 128
                max_grad_norm=None,
                weight_decay=0,

                update_exploration=True,
                initial_exploration_rate=1,
                final_exploration_rate=0.05,  # 0.05
                final_exploration_step=FINAL_EXPLORATION_STEP,  # 40000

                adam_epsilon=1e-8,
                logging=False,
                loss="mse",

                # save_network_frequency=400000,
                save_network_frequency=SAVE_NETWORK_FREQUENCY,
                network_save_path=network_save_path,

                evaluate=True,
                test_envs=test_envs,
                test_episodes=n_tests,
                # test_frequency=50000,  # 10000
                test_frequency=10000,  # 10000
                test_save_path=test_save_path,
                test_metric=TestMetric.MAX_CUT,

                seed=None
                )

    print("\n Created DQN agent with network:\n\n", agent.network)

    #############
    # TRAIN AGENT
    #############
    start = time.time()
    agent.learn(timesteps=nb_steps, verbose=True)
    #训完之后会输出时间
    print(time.time() - start)

    agent.save()


    ############
    # PLOT - learning curve
    ############
    data = pickle.load(open(test_save_path,'rb'))
    data = np.array(data)

    fig_fname = os.path.join(network_folder,"training_curve")

    plt.plot(data[:,0],data[:,1])
    plt.xlabel("Timestep")
    plt.ylabel("Mean reward")
    if agent.test_metric==TestMetric.ENERGY_ERROR:
      plt.ylabel("Energy Error")
    elif agent.test_metric==TestMetric.BEST_ENERGY:
      plt.ylabel("Best Energy")
    elif agent.test_metric==TestMetric.CUMULATIVE_REWARD:
      plt.ylabel("Cumulative Reward")
    elif agent.test_metric==TestMetric.MAX_CUT:
      plt.ylabel("Max Cut")
    elif agent.test_metric==TestMetric.FINAL_CUT:
      plt.ylabel("Final Cut")

    plt.savefig(fig_fname + ".png", bbox_inches='tight')
    plt.savefig(fig_fname + ".pdf", bbox_inches='tight')

    plt.clf()

    ############
    # PLOT - losses
    ############
    data = pickle.load(open(loss_save_path,'rb'))
    data = np.array(data)

    fig_fname = os.path.join(network_folder,"loss")

    N=50
    data_x = np.convolve(data[:,0], np.ones((N,))/N, mode='valid')
    data_y = np.convolve(data[:,1], np.ones((N,))/N, mode='valid')

    plt.plot(data_x,data_y)
    plt.xlabel("Timestep")
    plt.ylabel("Loss")

    plt.yscale("log")
    plt.grid(True)

    plt.savefig(fig_fname + ".png", bbox_inches='tight')
    plt.savefig(fig_fname + ".pdf", bbox_inches='tight')

if __name__ == "__main__":
    run()