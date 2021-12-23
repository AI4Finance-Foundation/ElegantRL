How to create a VecEnv on GPUs
===============================

Deep reinforcement learning (DRL) employs a trial-and-error manner to collect data from agent-environment interactions, along with the learning procedure. ElegantRL speeds up the data collection through **worker parallelism (VecEnv)**, by making use of hardwares, e.g., GPUs. 

A VecEnv runs thousands of independent sub-environments in parallel. In practice, it takes a batch of actions and returns a batch of transitions for each step.

Why creating a VecEnv on GPUs?

- Running thousands of parallel simulations, since the manycore GPU architecture is natually suited for highly parallel simulations.
- Speeding up the matrix computations of each simulation using GPU tensor cores.
- Reducing the communication overhead by bypassing the bottleneck between CPUs and GPUs.
- Maximizing GPU utilization through pipeline parallelism.

For GPU-accelerated VecEnv, ElegantRL supports external GPU-accelerated VecEnv, namely NVIDIA Isaac Gym, and user-customized VecEnv. Here we explain in detail how to use Isaac Gym and how to define your own VecEnv in ElegantRL. 

Running an environmnet from NVIDIA Isaac Gym
------------------------------------------

Isaac Gym is NVIDIA’s prototype physics simulation environment for reinforcement learning research. Isaac Gym includes a straightforward RL task system that can be used with it, e.g., Cartpole, Ant, Humanoid, Shadow Hand Object Manipulation, and supports users for customization. To download NVIDIA Isaac Gym, please follow the installation instructions from https://developer.nvidia.com/isaac-gym. 

ElegantRL provides a wrapper to process a defined Isaac Gym environment ``PreprocessIsaacVecEnv``:

.. code-block:: python

    from elegantrl.envs.IsaacGym import PreprocessIsaacVecEnv

    env_name = 'Ant'
    
    env = PreprocessIsaacVecEnv(env_name, if_print=False, env_num=4096, device_id=0)


Building an environmnet from scratch
------------------------------------------

We show you an example of how to construct a VecEnv from scratch. We create a simple chasing environment, a deterministic environment with continuous actions and continuous state space. The objective is to move the agent to chase a randomly moving robot. The reward depends on the distance between the agent and the robot. The environment terminates when the agent catches the robot or the max step is reached.

To keep the example simple, we only import two packages, PyTorch and Numpy.

.. code-block:: python

    import torch
    import numpy as np
    
Now, we start to create the environment. For initialization, we define the number of environments ``env_num``, the GPU id ``device_id``, and the dimension of the chasing space ``dim``. In the chasing environment, we needs to keep track of the positions and velocities of the agent and the robot.

.. code-block:: python

    class ChasingVecEnv:
    def __init__(self, dim=2, env_num=4096, device_id=0):
        self.dim = dim
        self.init_distance = 8.0

        # reset
        self.p0s = None  # position
        self.v0s = None  # velocity
        self.p1s = None
        self.v1s = None

        self.distances = None
        self.steps = None

        '''env info'''
        self.env_name = 'ChasingVecEnv'
        self.state_dim = self.dim * 4
        self.action_dim = self.dim
        self.max_step = 2 ** 10
        self.if_discrete = False
        self.target_return = 6.3

        self.env_num = env_num
        self.device = torch.device(f"cuda:{device_id}")
 


