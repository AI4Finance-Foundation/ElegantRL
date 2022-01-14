How to create a VecEnv on GPUs
===============================

ElegantRL supports massively parallel simulation through GPU-accelerated VecEnv.

Here, we talk about how to create a VecEnv on GPUs from scratch and go through a simple chasing example, a deterministic environment with continuous actions and continuous state space. The goal is to move an agent to chase a randomly moving robot. The reward depends on the distance between agent and robot. The environment terminates when the agent catches the robot or the max step is reached.

To keep the example simple, we only use two packages, PyTorch and Numpy.

.. code-block:: python

    import torch
    import numpy as np
    
Now, we start to create the environment, which usually includes initialization function, reset function, and step function. 

For **initialization function**, we specify the number of environments ``env_num``, the GPU id ``device_id``, and the dimension of the chasing space ``dim``. In the chasing environment, we keep track of positions and velocities of the agent and the robot.

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
            self.current_steps = None

            '''env info'''
            self.env_name = 'ChasingVecEnv'
            self.state_dim = self.dim * 4
            self.action_dim = self.dim
            self.max_step = 2 ** 10
            self.if_discrete = False
            self.target_return = 6.3

            self.env_num = env_num
            self.device = torch.device(f"cuda:{device_id}")
          
The second step is to implement a **reset function**. The reset function is called at the beginning of each episode and sets initial state to current state. To utilize GPUs, we use data structures for multi-dimensional tensors provided by the torch package.

.. code-block:: python

    def reset(self):
        self.p0s = torch.zeros((self.env_num, self.dim), dtype=torch.float32, device=self.device)
        self.v0s = torch.zeros((self.env_num, self.dim), dtype=torch.float32, device=self.device)
        self.p1s = torch.zeros((self.env_num, self.dim), dtype=torch.float32, device=self.device)
        self.v1s = torch.zeros((self.env_num, self.dim), dtype=torch.float32, device=self.device)

        self.current_steps = np.zeros(self.env_num, dtype=np.int)

        for env_i in range(self.env_num):
            self.reset_env_i(env_i)

        self.distances = ((self.p0s - self.p1s) ** 2).sum(dim=1) ** 0.5

        return self.get_state()
        
The last function is the **step function**, that includes a transition function and a reward function, and signals the terminal state. To compute the transition function, we utilize mathematical operations from the torch package over the data (tensors). These operations allow us to compute transitions and rewards of thousands of environments in parallel.

.. note::
    Unlike computing the transition function and reward function in parallel, we check the terminal state in a sequential way. Since sub-environments may terminate at different time steps, when a sub-environment is at terminal state, we have to reset it manually.
    
.. code-block:: python

    def step(self, action1s):
        '''transition function'''
        action0s = torch.rand(size=(self.env_num, self.dim), dtype=torch.float32, device=self.device)
        action0s_l2 = (action0s ** 2).sum(dim=1, keepdim=True) ** 0.5
        action0s = action0s / action0s_l2.clamp_min(1.0)

        self.v0s *= 0.50
        self.v0s += action0s
        self.p0s += self.v0s * 0.01

        action1s_l2 = (action1s ** 2).sum(dim=1, keepdim=True) ** 0.5
        action1s = action1s / action1s_l2.clamp_min(1.0)

        self.v1s *= 0.75
        self.v1s += action1s
        self.p1s += self.v1s * 0.01

        '''reward function'''
        distances = ((self.p0s - self.p1s) ** 2).sum(dim=1) ** 0.5
        rewards = self.distances - distances - action1s_l2.squeeze(1) * 0.02
        self.distances = distances

        '''check terminal state'''
        self.steps += 1  # array
        dones = torch.zeros(self.env_num, dtype=torch.float32, device=self.device)
        for env_i in range(self.env_num):
            done = 0
            if distances[env_i] < 1:
                done = 1
                rewards[env_i] += self.init_distance
            elif self.steps[env_i] == self.max_step:
                done = 1

            if done:
                self.reset_env_i(env_i)
            dones[env_i] = done

        '''next_state'''
        next_states = self.get_state()
        return next_states, rewards, dones, None
        
For more information about the chasing environment, we provide a `Colab version <https://github.com/AI4Finance-Foundation/ElegantRL/blob/master/tutorial_Creating_ChasingVecEnv.ipynb>`_ to play with, and its code can be found `here <https://github.com/AI4Finance-Foundation/ElegantRL/blob/master/elegantrl/envs/ChasingEnv.py>`_.


