.. ElegantRL documentation master file, created by
   sphinx-quickstart on Mon Mar  1 09:26:26 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ElegantRL's documentation!
=====================================

.. image:: ./images/logo.png
   :width: 100%
   :align: center
   :target: https://github.com/AI4Finance-LLC/ElegantRL

`ElegantRL <https://github.com/AI4Finance-LLC/ElegantRL>`_ is developped for researchers and practitioners with the following advantages:.

   - **Lightweight**: The core codes  <1,000 lines (check elegantrl/tutorial), using PyTorch (train), OpenAI Gym (env), NumPy, Matplotlib (plot).

   - **Efficient**: more efficient than `Ray RLlib <https://github.com/ray-project/ray>`_ in many testing cases.

   - **Stable**: much more stable than `Stable Baseline 3 <https://github.com/DLR-RM/stable-baselines3>`_.

ElegantRL implements the following model-free deep reinforcement learning (DRL) algorithms:

   - **DDPG, TD3, SAC, A2C, PPO(GAE) for continuous actions**
   
   - **DQN, DoubleDQN, D3QN for discrete actions**

For the details of DRL algorithms, please check out the educational webpage `OpenAI Spinning Up <https://spinningup.openai.com/en/latest/>`_. 

.. toctree::
    :maxdepth: 1
    :hidden:

    Home <self>

.. toctree::
   :maxdepth: 1
   :caption: Overview

   about/overview
   about/installation
   about/quickstart

.. toctree::
   :maxdepth: 1
   :caption: API

   api/net
   api/agent
   api/envs
   api/run

.. toctree::
   :maxdepth: 1
   :caption: Algorithms
   
   algorithms/ddpg
   algorithms/td3
   algorithms/sac
   algorithms/intersac
   algorithms/modsac
   algorithms/a2c
   algorithms/ppo
   algorithms/interppo
   algorithms/gaeppo
   algorithms/dqn
   algorithms/double_dqn
   algorithms/dueling_dqn
   algorithms/d3qn
   
.. toctree::
   :maxdepth: 1
   :caption: Examples
   
   examples/BipedalWalker-v3
   examples/LunarLanderContinuous-v2


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
