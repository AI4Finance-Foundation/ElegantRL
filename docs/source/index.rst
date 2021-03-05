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

`ElegantRL <https://github.com/AI4Finance-LLC/ElegantRL>`_ is featured with lightweight, efficient and stable, for researchers and practitioners.

   - **Lightweight**: The core codes <1,000 lines, using PyTorch, OpenAI Gym, and NumPy.

   - **Efficient**: performance is comparable with Ray RLlib.

   - **Stable**: as stable as Stable Baseline 3.

Model-free deep reinforcement learning (DRL) algorithms:

   - **DDPG, TD3, SAC, A2C, PPO(GAE) for continuous actions**
   
   - **DQN, DoubleDQN, D3QN for discrete actions**

For algorithm details, please check out OpenAI Spinning Up.

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
