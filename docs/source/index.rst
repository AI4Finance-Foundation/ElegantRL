.. ElegantRL documentation master file, created by
   sphinx-quickstart on Mon Mar  1 09:26:26 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ElegantRL!
=====================================

.. image:: ./images/logo.png
   :width: 50%
   :align: center
   :target: https://github.com/AI4Finance-Foundation/ElegantRL

`ElegantRL <https://github.com/AI4Finance-Foundation/ElegantRL>`_ is developped for researchers and practitioners with the following features:

   - **Lightweight**: The core codes  <1,000 lines (check elegantrl/tutorial), using PyTorch (train), OpenAI Gym (env), NumPy, Matplotlib (plot).

   - **Efficient**: in many testing cases, we find it more efficient than `Ray RLlib <https://github.com/ray-project/ray>`_ in many testing cases.

   - **Stable**: much more stable than `Stable Baseline 3 <https://github.com/DLR-RM/stable-baselines3>`_.

ElegantRL implements the following model-free deep reinforcement learning (DRL) algorithms:

   - **DDPG, TD3, SAC, A2C, PPO (GAE) for continuous actions**
   
   - **DQN, DoubleDQN, D3QN for discrete actions**

For DRL algorithms, please check out the OpenAI's educational webpage `Spinning Up <https://spinningup.openai.com/en/latest/>`_. 

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
   :caption: TUTORIAL

   tutorial/intro
   tutorial/net
   tutorial/agent
   tutorial/env
   tutorial/run

.. toctree::
   :maxdepth: 1
   :caption: Algorithms
   
   algorithms/dqn
   algorithms/double_dqn
   algorithms/ddpg
   algorithms/td3
   algorithms/sac
   algorithms/intersac
   algorithms/a2c
   algorithms/ppo
   algorithms/interppo
   algorithms/maddpg
   algorithms/qmix
   algorithms/redq
   
   
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
