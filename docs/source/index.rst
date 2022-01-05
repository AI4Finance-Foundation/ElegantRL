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
   
   

`ElegantRL <https://github.com/AI4Finance-Foundation/ElegantRL>`_ is an open-source massively parallel framework for DRL algorithms implemented in PyTorch. We aim to provide a *next-generation* framework that embraces recent breakthroughs in massively parallel simulation, ensemble methods, and population-based training. 


ElegantRL features strong **scalability**, **elasticity** and **lightweightness**, and allows users to conduct **efficient** training on either one GPU or hundreds of GPUs: 

   - **Scalability**: ElegantRL fully exploits the parallelism of DRL algorithms at multiple levels, making it easily scale out to hundreds or thousands of computing nodes on a cloud platform, say, thousands of GPUs.
   
   - **Elasticity**: ElegantRL can elastically allocate computing resources, which helps adapt to available resources and prevents over-provisioning and under-provisioning on the cloud.
   
   - **Lightweightness**: The core codes <1,000 lines (check `elegantrl_helloworld <https://github.com/AI4Finance-Foundation/ElegantRL/tree/master/elegantrl_helloworld>`_).
   
   - **Efficient**: in many testing cases, we find it more efficient than `Ray RLlib <https://github.com/ray-project/ray>`_.

ElegantRL implements the following deep reinforcement learning (DRL) algorithms:

   - **DDPG, TD3, SAC, A2C, PPO, REDQ for continuous actions**
   
   - **DQN, DoubleDQN, D3QN, PPO-Discrete for discrete actions**
   
   - **QMIX, VDN; MADDPG, MAPPO, MATD3 for multi-agent RL**


For beginners, we maintain `*ElegantRL-HelloWorld* <https://github.com/AI4Finance-Foundation/ElegantRL/tree/master/elegantrl_helloworld>`_ as a tutorial. It is a lightweight DRL implementation with <1,000 lines of core codes. More details are available `here <https://elegantrl.readthedocs.io/en/latest/tutorial/intro.html>`_.

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
   :caption: Hello-World

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
   algorithms/a2c
   algorithms/ppo
   algorithms/redq
   algorithms/maddpg
   algorithms/qmix
   algorithms/mappo
   algorithms/vdn
   
   
.. toctree::
   :maxdepth: 1
   :caption: Tutorials
   
   examples/Creating_VecEnv
   examples/BipedalWalker-v3
   examples/LunarLanderContinuous-v2
   
   
.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/config
   api/run
   api/worker
   api/learner
   api/replay
   api/evaluator
   api/utils
   
 
.. toctree::
   :maxdepth: 1
   :caption: Other

   other/faq


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
