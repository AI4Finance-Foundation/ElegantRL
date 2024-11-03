import os
import pickle
import networkx as nx
import time
import numpy as np
import scipy as sp
import pandas as pd
import torch

from collections import namedtuple
from copy import deepcopy

import rlsolver.methods.eco_s2v.src.envs.core as ising_env
from rlsolver.methods.eco_s2v.src.envs.util import (SingleGraphGenerator, SpinBasis)
from rlsolver.methods.eco_s2v.src.agents.solver import Network, Greedy
from rlsolver.methods.eco_s2v.config.config import *


####################################################
# TESTING ON GRAPHS
####################################################

def test_network(network, env_args, graphs_test, device=None, step_factor=1, batched=True,
                 n_attempts=50, return_raw=False, return_history=False, max_batch_size=None):
    if batched:
        return __test_network_batched(network, env_args, graphs_test, device, step_factor,
                                      n_attempts, return_raw, return_history, max_batch_size)
    else:
        if max_batch_size is not None:
            print("Warning: max_batch_size argument will be ignored for when batched=False.")
        return __test_network_sequential(network, env_args, graphs_test, step_factor,
                                         n_attempts, return_raw, return_history)


def __test_network_batched(network, env_args, graphs_test, device=None, step_factor=1,
                           n_attempts=50, return_raw=False, return_history=False, max_batch_size=None):
    if device is None:
        device = "cuda:7" if torch.cuda.is_available() else "cpu"
    torch.device(device)

    # HELPER FUNCTION FOR NETWORK TESTING

    acting_in_reversible_spin_env = env_args['reversible_spins']

    if env_args['reversible_spins']:
        # If MDP is reversible, both actions are allowed.
        if env_args['spin_basis'] == SpinBasis.BINARY:
            allowed_action_state = (0, 1)
        elif env_args['spin_basis'] == SpinBasis.SIGNED:
            allowed_action_state = (1, -1)
    else:
        # If MDP is irreversible, only return the state of spins that haven't been flipped.
        if env_args['spin_basis'] == SpinBasis.BINARY:
            allowed_action_state = 0
        if env_args['spin_basis'] == SpinBasis.SIGNED:
            allowed_action_state = 1

    def predict(states):
        qs = network(states)
        if acting_in_reversible_spin_env:
            if qs.dim() == 1:
                actions = [qs.argmax().item()]
            else:
                actions = qs.argmax(1, True).squeeze(1).cpu().numpy()
            return actions
        else:
            if qs.dim() == 1:
                x = (states.squeeze()[:, 0] == allowed_action_state).nonzero()
                actions = [x[qs[x].argmax().item()].item()]
            else:
                disallowed_actions_mask = (states[:, :, 0] != allowed_action_state)
                qs_allowed = qs.masked_fill(disallowed_actions_mask, -1000)
                actions = qs_allowed.argmax(1, True).squeeze(1).cpu().numpy()
            return actions

    # NETWORK TESTING

    results = []
    results_raw = []
    if return_history:
        history = []

    n_attempts = n_attempts if env_args["reversible_spins"] else 1

    for j, test_graph in enumerate(graphs_test):
        i_comp = 0
        i_batch = 0
        t_total = 0
        n_spins = test_graph.shape[0]
        n_steps = int(n_spins * step_factor)
        test_env = ising_env.make("SpinSystem",
                                  SingleGraphGenerator(test_graph),
                                  n_steps,
                                  **env_args)
        print("Running greedy solver with +1 initialisation of spins...", end="...")
        # Calculate the greedy cut with all spins initialised to +1
        greedy_env = deepcopy(test_env)
        greedy_env.reset(spins=np.array([1] * test_graph.shape[0]))

        greedy_agent = Greedy(greedy_env)
        greedy_agent.solve()

        greedy_single_cut = greedy_env.get_best_cut()
        greedy_single_spins = greedy_env.best_spins

        print("done.")

        if return_history:
            actions_history = []
            rewards_history = []
            scores_history = []

        best_cuts = []
        init_spins = []
        best_spins = []

        greedy_cuts = []
        greedy_spins = []

        while i_comp < n_attempts:
            if max_batch_size is None:
                batch_size = n_attempts
            else:
                batch_size = min(n_attempts - i_comp, max_batch_size)
            i_comp_batch = 0
            if return_history:
                actions_history_batch = [[None] * batch_size]
                rewards_history_batch = [[None] * batch_size]
                scores_history_batch = []
            test_envs = [None] * batch_size
            best_cuts_batch = [-1e3] * batch_size
            init_spins_batch = [[] for _ in range(batch_size)]
            best_spins_batch = [[] for _ in range(batch_size)]

            greedy_envs = [None] * batch_size
            greedy_cuts_batch = []
            greedy_spins_batch = []

            obs_batch = [None] * batch_size
            print("Preparing batch of {} environments for graph {}.".format(batch_size, j), end="...")
            for i in range(batch_size):
                env = deepcopy(test_env)
                obs_batch[i] = env.reset()
                test_envs[i] = env
                greedy_envs[i] = deepcopy(env)
                init_spins_batch[i] = env.best_spins
            if return_history:
                scores_history_batch.append([env.calculate_score() for env in test_envs])
            print("done.")

            # Calculate the max cut acting w.r.t. the network
            t_start = time.time()
            # pool = mp.Pool(processes=16)
            k = 0
            while i_comp_batch < batch_size:
                t1 = time.time()
                # Note: Do not convert list of np.arrays to FloatTensor, it is very slow!
                # see: https://github.com/pytorch/pytorch/issues/13918
                # Hence, here we convert a list of np arrays to a np array.
                obs_batch = torch.FloatTensor(np.array(obs_batch)).to(device)
                actions = predict(obs_batch)
                obs_batch = []

                if return_history:
                    scores = []
                    rewards = []

                i = 0
                for env, action in zip(test_envs, actions):
                    if env is not None:
                        obs, rew, done, info = env.step(action)
                        if return_history:
                            scores.append(env.calculate_score())
                            rewards.append(rew)
                        if not done:
                            obs_batch.append(obs)
                        else:
                            best_cuts_batch[i] = env.get_best_cut()
                            best_spins_batch[i] = env.best_spins
                            i_comp_batch += 1
                            i_comp += 1
                            test_envs[i] = None
                    i += 1
                    k += 1
                if return_history:
                    actions_history_batch.append(actions)
                    scores_history_batch.append(scores)
                    rewards_history_batch.append(rewards)
                # print("\t",
                #       "Par. steps :", k,
                #       "Env steps : {}/{}".format(k/batch_size,n_steps),
                #       'Time: {0:.3g}s'.format(time.time()-t1))

            t_total += (time.time() - t_start)
            i_batch += 1
            print("Finished agent testing batch {}.".format(i_batch))
            if env_args["reversible_spins"]:
                print("Running greedy solver with {} random initialisations of spins for batch {}...".format(batch_size,                                                                                      i_batch), end="...")
                for env in greedy_envs:
                    Greedy(env).solve()
                    cut = env.get_best_cut()
                    greedy_cuts_batch.append(cut)
                    greedy_spins_batch.append(env.best_spins)
                print("done.")
            if return_history:
                actions_history += actions_history_batch
                rewards_history += rewards_history_batch
                scores_history += scores_history_batch
            best_cuts += best_cuts_batch
            init_spins += init_spins_batch
            best_spins += best_spins_batch
            if env_args["reversible_spins"]:
                greedy_cuts += greedy_cuts_batch
                greedy_spins += greedy_spins_batch
            # print("\tGraph {}, par. steps: {}, comp: {}/{}".format(j, k, i_comp, batch_size),
            #       end="\r" if n_spins<100 else "")

        i_best = np.argmax(best_cuts)
        best_cut = best_cuts[i_best]
        sol = best_spins[i_best]

        mean_cut = np.mean(best_cuts)
        if env_args["reversible_spins"]:
            idx_best_greedy = np.argmax(greedy_cuts)
            greedy_random_cut = greedy_cuts[idx_best_greedy]
            greedy_random_spins = greedy_spins[idx_best_greedy]
            greedy_random_mean_cut = np.mean(greedy_cuts)
        else:
            greedy_random_cut = greedy_single_cut
            greedy_random_spins = greedy_single_spins
            greedy_random_mean_cut = greedy_single_cut
        print(
            'Graph {}, best(mean) cut: {}({}), greedy cut (rand init / +1 init) : {} / {}.  ({} attempts in {}s)\t\t\t'.format(
                j, best_cut, mean_cut, greedy_random_cut, greedy_single_cut, n_attempts, np.round(t_total, 2)))

        results.append([best_cut, sol,
                        mean_cut,
                        greedy_single_cut, greedy_single_spins,
                        greedy_random_cut, greedy_random_spins,
                        greedy_random_mean_cut,
                        t_total / (n_attempts)])

        results_raw.append([init_spins,
                            best_cuts, best_spins,
                            greedy_cuts, greedy_spins])

        if return_history:
            history.append([np.array(actions_history).T.tolist(),
                            np.array(scores_history).T.tolist(),
                            np.array(rewards_history).T.tolist()])

    results = pd.DataFrame(data=results, columns=["cut", "sol",
                                                  "mean cut",
                                                  "greedy (+1 init) cut", "greedy (+1 init) sol",
                                                  "greedy (rand init) cut", "greedy (rand init) sol",
                                                  "greedy (rand init) mean cut",
                                                  "time"])

    results_raw = pd.DataFrame(data=results_raw, columns=["init spins",
                                                          "cuts", "sols",
                                                          "greedy cuts", "greedy sols"])

    if return_history:
        history = pd.DataFrame(data=history, columns=["actions", "scores", "rewards"])

    if return_raw == False and return_history == False:
        return results
    else:
        ret = [results]
        if return_raw:
            ret.append(results_raw)
        if return_history:
            ret.append(history)
        return ret


def __test_network_sequential(network, env_args, graphs_test, step_factor=1,
                              n_attempts=50, return_raw=False, return_history=False):
    if return_raw or return_history:
        raise NotImplementedError("I've not got to this yet!  Used the batched test script (it's faster anyway).")

    results = []

    n_attempts = n_attempts if env_args["reversible_spins"] else 1

    for i, test_graph in enumerate(graphs_test):

        n_steps = int(test_graph.shape[0] * step_factor)

        best_cut = -1e3
        best_spins = []

        greedy_random_cut = -1e3
        greedy_random_spins = []

        greedy_single_cut = -1e3
        greedy_single_spins = []

        times = []

        test_env = ising_env.make("SpinSystem",
                                  SingleGraphGenerator(test_graph),
                                  n_steps,
                                  **env_args)
        net_agent = Network(network, test_env,
                            record_cut=False, record_rewards=False, record_qs=False)

        greedy_env = deepcopy(test_env)
        greedy_env.reset(spins=np.array([1] * test_graph.shape[0]))
        greedy_agent = Greedy(greedy_env)

        greedy_agent.solve()

        greedy_single_cut = greedy_env.get_best_cut()
        greedy_single_spins = greedy_env.best_spins

        for k in range(n_attempts):

            net_agent.reset(clear_history=True)
            greedy_env = deepcopy(test_env)
            greedy_agent = Greedy(greedy_env)

            tstart = time.time()
            net_agent.solve()
            times.append(time.time() - tstart)

            cut = test_env.get_best_cut()
            if cut > best_cut:
                best_cut = cut
                best_spins = test_env.best_spins

            greedy_agent.solve()

            greedy_cut = greedy_env.get_best_cut()
            if greedy_cut > greedy_random_cut:
                greedy_random_cut = greedy_cut
                greedy_random_spins = greedy_env.best_spins

            # print('\nGraph {}, attempt : {}/{}, best cut : {}, greedy cut (rand init / +1 init) : {} / {}\t\t\t'.format(
            #     i + 1, k, n_attemps, best_cut, greedy_random_cut, greedy_single_cut),
            #     end="\r")
            print('\nGraph {}, attempt : {}/{}, best cut : {}, greedy cut (rand init / +1 init) : {} / {}\t\t\t'.format(
                i + 1, k, n_attempts, best_cut, greedy_random_cut, greedy_single_cut),
                end=".")

        results.append([best_cut, best_spins,
                        greedy_single_cut, greedy_single_spins,
                        greedy_random_cut, greedy_random_spins,
                        np.mean(times)])

    return pd.DataFrame(data=results, columns=["cut", "sol",
                                               "greedy (+1 init) cut", "greedy (+1 init) sol",
                                               "greedy (rand init) cut", "greedy (rand init) sol",
                                               "time"])


####################################################
# LOADING GRAPHS
####################################################

Graph = namedtuple('Graph', 'name n_vertices n_edges matrix bk_val bk_sol')


def load_graph(graph_dir, graph_name):
    inst_loc = os.path.join(graph_dir, 'instances', graph_name + '.mc')
    val_loc = os.path.join(graph_dir, 'bkvl', graph_name + '.bkvl')
    sol_loc = os.path.join(graph_dir, 'bksol', graph_name + '.bksol')

    vertices, edges, matrix = 0, 0, None
    bk_val, bk_sol = None, None

    with open(inst_loc) as f:
        for line in f:
            arr = list(map(int, line.strip().split(' ')))
            if len(arr) == 2:  # contains the number of vertices and edges
                n_vertices, n_edges = arr
                matrix = np.zeros((n_vertices, n_vertices))
            else:
                assert type(matrix) == np.ndarray, 'First line in file should define graph dimensions.'
                i, j, w = arr[0] - 1, arr[1] - 1, arr[2]
                matrix[[i, j], [j, i]] = w

    with open(val_loc) as f:
        bk_val = float(f.readline())

    with open(sol_loc) as f:
        bk_sol_str = f.readline().strip()
        bk_sol = np.array([int(x) for x in list(bk_sol_str)] + [np.random.choice([0, 1])])  # final spin is 'no-action'

    return Graph(graph_name, n_vertices, n_edges, matrix, bk_val, bk_sol)


def load_graph_set(graph_save_loc):
    graphs_test = pickle.load(open(graph_save_loc, 'rb'))

    def graph_to_array(g):
        if type(g) == nx.Graph:
            g = nx.to_numpy_array(g)
        elif type(g) == sp.sparse.csr_matrix:
            g = g.toarray()
        return g

    graphs_test = [graph_to_array(g) for g in graphs_test]
    print('{} target graphs loaded from {}'.format(len(graphs_test), graph_save_loc))
    return graphs_test


def load_graph_from_txt(file_path):
    """
    从txt文件中加载图，并返回图对象
    """
    with open(file_path, 'r', encoding='UTF-8') as f:
        lines = f.readlines()

    # 第一行是节点数量和边数量
    node_count, edge_count = map(int, lines[0].strip().split())

    # 创建一个空的无向图
    g = nx.Graph()

    # 添加边到图中
    for line in lines[1:]:
        x, y, _ = map(int, line.strip().split())
        g.add_edge(x, y)

    return g


def load_graph_set_from_txt(graph_save_folder):
    """
    从指定文件夹中读取所有txt文件并转换为图对象集合
    """
    graphs_test = []

    # 遍历文件夹中的所有txt文件

    g = load_graph_from_txt(graph_save_folder)
    graphs_test.append(g)

    # 将图转换为数组形式
    def graph_to_array(g):
        if isinstance(g, nx.Graph):
            g = nx.to_numpy_array(g)
        elif isinstance(g, sp.sparse.csr_matrix):
            g = g.toarray()
        return g

    graphs_test = [graph_to_array(g) for g in graphs_test]
    print('{} target graphs loaded from {}'.format(len(graphs_test), graph_save_folder))

    return graphs_test


def load_graph_set_from_folder(graph_save_folder):
    """
    从指定文件夹中读取所有txt文件并转换为图对象集合
    """
    graphs_test = []

    # 遍历文件夹中的所有txt文件
    for file_name in os.listdir(graph_save_folder):
        if file_name.endswith('.txt') and str(NUM_TRAIN_NODES) in file_name.split("_"):
            file_path = os.path.join(graph_save_folder, file_name)
            g = load_graph_from_txt(file_path)
            graphs_test.append(g)

    # 将图转换为数组形式
    def graph_to_array(g):
        if isinstance(g, nx.Graph):
            g = nx.to_numpy_array(g)
        elif isinstance(g, sp.sparse.csr_matrix):
            g = g.toarray()
        return g

    graphs_test = [graph_to_array(g) for g in graphs_test]
    print('{} target graphs loaded from {}'.format(len(graphs_test), graph_save_folder))

    return graphs_test


####################################################
# FILE UTILS
####################################################

def mk_dir(export_dir, quite=False):
    if not os.path.exists(export_dir):
        try:
            os.makedirs(export_dir)
            print('created dir: ', export_dir)
        except OSError as exc:  # Guard against race condition
            if exc.errno != exc.errno.EEXIST:
                raise
        except Exception:
            pass
    else:
        print('dir already exists: ', export_dir)
