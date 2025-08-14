RL Methods
==========

This section summarizes the reinforcement-learning–driven algorithms in RLSolver and how they approach combinatorial optimization.

ECO-DQN
--------
ECO-DQN (Exploratory Combinatorial Optimization with Deep Q-Networks) is a reinforcement learning framework designed for solving NP-hard combinatorial optimization problems over graphs. It builds upon the S2V-DQN model by integrating deep graph embeddings (such as Structure2Vec) with Q-learning to learn greedy construction policies. Unlike sequence-based models such as Pointer Networks, ECO-DQN incrementally builds a solution by selecting one node at a time based on the current graph state and partial solution. The action-value function is parameterized by the graph embedding network, and training is performed using n-step Q-learning with fitted Q-iteration for improved sample efficiency. ECO-DQN supports generalization to graphs of varying sizes and has been shown to perform competitively or even outperform traditional approximation algorithms on tasks like Minimum Vertex Cover, Max-Cut, and TSP.

In implementation, ECO-DQN is supported within the `ElegantRL quickstart guide <https://github.com/AI4Finance-Foundation/ElegantRL>`_, where the ``train_and_evaluate(args)`` function trains a DQN agent in the ECO environment.

S2V-DQN
--------
S2V-DQN (Structure2Vec Deep Q-Network) is the approach introduced by Dai et al. in `Learning Combinatorial Optimization Algorithms over Graphs <https://arxiv.org/abs/1704.01665>`_.
. It uses a graph-neural message-passing network (Structure2Vec) to embed each graph into a fixed-length vector capturing both local and global structure. A DQN agent then selects actions (e.g., which edge to cut) based on this embedding.

To train S2V-DQN on a distribution of graphs:

.. code-block:: bash

   python methods/eco_s2v/main.py  # train S2V-DQN on distribution-wise maxcut instances

Once trained, inference can be run by setting `TRAIN_INFERENCE = 1` in `config.py` and re-running the same script.

MCPG
-----
MCPG (Monte Carlo Policy Gradient) estimates the gradient of expected reward by running full Monte Carlo rollouts under the current policy. Each complete-solution sample is scored, and the policy network is updated by back-propagating the reward signal through the sampled trajectories, effectively performing on-policy policy gradient updates.

iSCO
----
iSCO (improved Sampling algorithm for Combinatorial Optimization) instantiates the generic sampling framework with efficiency and parallelism enhancements.  
It leverages discrete‐space Markov Chain Monte Carlo (MCMC) moves together with just‐in‐time compilation for accelerators, achieving faster convergence on high‐quality solutions across diverse combinatorial instances.

Jumanji
--------
Jumanji is a modular RL environment framework tailored for combinatorial problems. It provides standard Gym-style interfaces and helpers, letting you plug in any policy architecture or training algorithm with minimal boilerplate. Jumanji focuses on reproducibility and extensibility, making it easy to benchmark new methods on a wide range of combinatorial tasks.

