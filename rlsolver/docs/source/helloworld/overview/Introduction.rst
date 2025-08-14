Introduction
============

**RLSolver: GPU-based Massively Parallel Environments for Large-Scale  
Combinatorial Optimization (CO) Problems Using Reinforcement Learning**


We aim to showcase the effectiveness of GPU-based massively parallel environments for large-scale combinatorial optimization (CO) problems using reinforcement learning (RL).  
RL with GPU-based parallel environments can significantly improve the sampling speed and obtain high-quality solutions in a short time.

Overview
--------

RLSolver has three layers:

- **Environments**: providing massively parallel environments using GPUs.  
- **RL agents**: implementing reinforcement-learning algorithms, e.g., REINFORCE and DQN.  
- **Problems**: typical CO problems, e.g., graph Max-Cut, Knapsack, and TSP.


.. figure:: /_static/RLSolver_framework.png
   :alt: RLSolver Architecture
   :width: 75%
   :align: center

   **Figure 1: RLSolver overall architecture.**

Key Technologies
----------------

- **GPU-based massively parallel environments** for Markov chain Monte Carlo (MCMC) simulations on thousands of CUDA cores and tensor cores.  
- **Distribution-wise training** is orders of magnitude faster than instance-wise methods (e.g., MCPG, iSCO), since inference can be done directly on batched instances.


.. raw:: html

   <h2 style="margin-top: 1em;">Why Use GPU-based Massively Parallel Environments?</h2>


The bottleneck of using RL for solving large-scale CO problems — especially in distribution-wise scenarios — is the low sampling speed, since existing solver engines (a.k.a. “gym-style” environments) are implemented on CPUs.  Training the policy network is essentially estimating the gradients via a Markov chain Monte Carlo (MCMC) simulation, which requires a large number of samples from the environments.

Existing CPU-based environments have two significant disadvantages:

1.  The number of CPU cores is typically small, generally ranging from 16 to 256, resulting in a small number of parallel environments.  
2.  The communication link between CPUs and GPUs has limited bandwidth.  

GPU-based massively parallel environments overcome these disadvantages: we can build thousands of environments on the GPU, and bypass the CPU–GPU communication bottleneck; therefore the sampling speed is significantly improved.

.. raw:: html

   <div style="font-size:1.5em; font-weight:bold; margin:1em 0;">
     Improving the Sampling Speed
   </div>

.. figure:: /_static/sampling_efficiency_maxcut.png
   :alt: Sampling speed comparison between CPU and GPU environments
   :width: 90%
   :align: center

From the above figures, we used CPU and GPU based environments.  
We see that the sampling speed is improved by at least 2 orders by using GPU-based massively parallel environments compared with conventional CPUs.


.. raw:: html

   <div style="font-size:1.5em; font-weight:bold; margin:1em 0;">
     Improving the Convergence Speed
   </div>


.. figure:: /_static/obj_time.png
   :alt: Objective value versus wall-clock time
   :width: 90%
   :align: center

To achieve the same objective value, if we use more parallel environments, the less running time.

.. raw:: html

   <div style="font-size:1.5em; font-weight:bold; margin:1em 0;">
     Improving the Quality of Solutions
   </div>

.. figure:: /_static/objectives_epochs.png
   :alt: Convergence curves on GSET G22 with varying parallel environments
   :width: 80%
   :align: center

GPU-based parallel environments can significantly improve the quality of solutions during training, since RL methods require many high-quality samples from the environments for training. Take graph maxcut as an example. We select G22 in the Gset dataset. The above figure shows the objective values vs. number of epochs with different number of GPU-based parallel environments. We see that, generally, the more parallel environments, the higher objective values, and the faster convergence.