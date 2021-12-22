How to creating a VecEnv on GPUs
===============================

Deep reinforcement learning (DRL) employs a trial-and-error manner to collect data from agent-environment interactions, along with the learning procedure. ElegantRL speed-ups the data collection through **worker parallelism** (``VecEnv``), by making use of hardware accelerators, e.g., GPUs. 

A ``VecEnv`` runs thousands of independent sub-environments in parallel. In practice, it takes a batch of actions and returns a batch of transitions for each step.  

Advantages of GPU-accelerated ``VecEnv``:
--------------

- Easily scaling out to thousands of parallel simulations.
- Significantly speeding up the simulation
- Maximizing GPU utilization.

Here, we 
