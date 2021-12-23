How to create a VecEnv on GPUs
===============================

Deep reinforcement learning (DRL) employs a trial-and-error manner to collect data from agent-environment interactions, along with the learning procedure. ElegantRL speeds up the data collection through **worker parallelism (VecEnv)**, by making use of hardwares, e.g., GPUs. 

A VecEnv runs thousands of independent sub-environments in parallel. In practice, it takes a batch of actions and returns a batch of transitions for each step.

Why creating a VecEnv on GPUs?

- Running thousands of parallel simulations, since the manycore GPU architecture is natually suited for highly parallel simulations.
- Speeding up the matrix computations of each simulation using GPU tensor cores.
- Reducing the communication overhead by bypassing the bottleneck between CPUs and GPUs.
- Maximizing GPU utilization through pipeline parallelism.

ElegantRL supports external GPU-accelerated simulators, namely NVIDIA Isaac Gym, and user-customized VecEnv. Here we explain in detail how to use Isaac Gym and how to define your own VecEnv. 

Running an environmnet from NVIDIA Isaac Gym
------------------------------------------

Isaac Gym is NVIDIA’s prototype physics simulation environment for reinforcement learning research. Isaac Gym includes a straightforward RL task system that can be used with it, e.g., Cartpole, Ant, Humanoid, Shadow Hand Object Manipulation, and supports users for customization. To download NVIDIA Isaac Gym, please follow the installation instructions from https://developer.nvidia.com/isaac-gym. 

ElegantRL provides a wrapper to process a defined Isaac Gym environment ``PreprocessIsaacVecEnv``:

.. code-block:: python

    from elegantrl.envs.IsaacGym import PreprocessIsaacVecEnv

    env_name = 'Ant'
    
    env = PreprocessIsaacVecEnv(env_name, if_print=False, env_num=4096, device_id=0)


Building an environmnet from scatch
------------------------------------------

We show you an example of how to construct a VecEnv from scatch. We create a simple chasing environment, with discrete actions, continuous state space, and mildly stochastic dynamics. The objective is to move the agent from any point of the room towards the goal point. The agent takes a penalty at every step equal to the distance to the objective. When the agent reaches the goal the episode ends. The agent can move in the room by using one of the 4 discrete actions, North, South, West, East.
