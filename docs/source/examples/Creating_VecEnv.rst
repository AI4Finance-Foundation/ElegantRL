How to create a VecEnv on GPUs
===============================

Deep reinforcement learning (DRL) employs a trial-and-error manner to collect data from agent-environment interactions, along with the learning procedure. ElegantRL speeds up the data collection through **worker parallelism (VecEnv)**, by making use of hardwares, e.g., GPUs. 

A VecEnv runs thousands of independent sub-environments in parallel. In practice, it takes a batch of actions and returns a batch of transitions for each step.

Why creating a VecEnv on GPUs?
--------------

- Running thousands of parallel simulations, since the manycore GPU architecture is natually suited for highly parallel simulations.
- Speeding up the matrix computations of each simulation using GPU tensor cores.
- Reducing the communication overhead by bypassing the bottleneck between CPUs and GPUs.
- Maximizing GPU utilization through pipeline parallelism.

ElegantRL supports external GPU-accelerated simulators, namely Isaac Gym, and user-customized VecEnv. Here we explain in detail how to use Isaac Gym and how to define your own VecEnv. 

Running an environmnet from Isaac Gym
------------------------------------------






Here, we 
