.. RLSolver documentation master file, created by

Welcome to RLSolver!
=====================================


   :width: 50%
   :align: center
   :target: https://github.com/zhumingpassional/RLSolver
   
   

`RLSolver <https://github.com/zhumingpassional/RLSolver>`_ : GPU-based Massively Parallel Environments for Combinatorial Optimization (CO) Problems Using Reinforcement Learning

We aim to showcase the effectiveness of massively parallel environments for combinatorial optimization (CO) problems using reinforcement learning (RL). RL with the help of GPU based parallel environments can significantly improve the sampling speed and can obtain high-quality solutions within short time.

Overview

RLSolver has three layers:

-Environments: providing massively parallel environments using GPUs.
-RL agents: providing RL algorithms, e.g., REINFORCE.
-Problems: typical CO problems, e.g., graph maxcut and TNCO.

Key Technologies
-**GPU-based Massively parallel environments** of Markov chain Monte Carlo (MCMC) simulations on GPU using thousands of CUDA cores and tensor cores.
-**Distribution-wise** is **much much faster** than the instance-wise methods, such as MCPG and iSCO, since we can obtain the results directly by inference.

Why Use GPU-based Massively Parallel Environments?

The bottleneck of using RL for solving CO problems is the sampling speed since existing solver engines (a.k.a, gym-style environments) are implemented on CPUs. Training the policy network is essentially estimating the gradients via a Markov chain Monte Carlo (MCMC) simulation, which requires a large number of samples from environments.

Existing CPU-based environments have two significant disadvantages: 1) The number of CPU cores is typically small, generally ranging from 16 to 256, resulting in a small number of parallel environments. 2) The communication link between CPUs and GPUs has limited bandwidth. The massively parallel environments can overcome these disadvantages, since we can build thounsands of environments and the communication bottleneck between CPUs and GPUs is bypassed, therefore the sampling speed is significantly improved.

Installation
---------------------------------------

RLSolver generally requires:

- Python>=3.6

- PyTorch>=1.0.2

- gym, matplotlib, numpy, torch

You can simply install ElegantRL from PyPI with the following command:

.. code-block:: bash
   :linenos:

   pip3 install rlsolver --upgrade

Or install with the newest version through GitHub:

.. code-block:: bash
   :linenos:

   git clone https://github.com/zhumingpassional/RLSolver
   cd RLSolver
   pip3 install .


.. toctree::
    :maxdepth: 1
    :hidden:

    Home <self>
    
.. toctree::
   :maxdepth: 1
   :caption: HelloWorld

   helloworld/intro

.. toctree::
   :maxdepth: 1
   :caption: Overview

   about/overview
   about/cloud
   about/parallel
   
   
.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorial/maxcut


.. toctree::
   :maxdepth: 1
   :caption: Algorithms
   
   algorithms/REINFORCE


.. toctree::
   :maxdepth: 1
   :caption: RLSolver
   
   RLSolver/overview
   RLSolver/helloworld
   RLSolver/datasets
   RLSolver/environments
   RLSolver/benchmarks
   
   
.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/config
   api/util
   
 
.. toctree::
   :maxdepth: 1
   :caption: Other

   other/faq


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
