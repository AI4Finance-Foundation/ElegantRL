Worker: *worker.py*
=================================

Deep reinforcement learning (DRL) employs a trial-and-error manner to collect training data (transitions) from agent-environment interactions, along with the learning procedure. ElegantRL utilizes ``Worker`` to generate transitions and achieves worker parallelism, thus greatly speeding up the data collection.

Based on the type of the environment, we support two different worker parallelism to generate transitions in parallel:
  - A vectorized environment (VecEnv) runs thousands of independent sub-environments in parallel. In each step, it takes a batch of actions and returns a batch of transitions. When the environment is a VecEnv, if we want the parallelism to be 64, we can simply set #sub-environments to 64 and #workers to 1 in ``Arguments`` in *Config.py*.
  - When the environment is not a VecEnv, e.g., environments from OpenAI Gym or MuJoCo, if we want the parallelism to be 64, we can directly set #workers to 64.
  
.. warning::
  For VecEnv, if users want to increase the degree of parallelism, we recommend to increase #sub-environments and make #workers unchaged. In pratice, there is no need to set #workers > 1 for GPU-accelerated VecEnv. 
  
We highly recommend users to use GPU-accelerated VecEnv to achieve massively parallel simulations. A GPU-accelerated VecEnv can:
  - Running thousands of parallel simulations, since the manycore GPU architecture is natually suited for parallel simulations.
  - Speeding up the matrix computations of each simulation using GPU tensor cores.
  - Reducing the communication overhead by bypassing the bottleneck between CPUs and GPUs.
  - Maximizing GPU utilization through pipeline parallelism.
  
A tutorial on how to create a GPU-accelerated VecEnv is available `here <https://elegantrl.readthedocs.io/en/latest/examples/Creating_VecEnv.html>`_.

Implementations
---------------------

.. autoclass:: elegantrl.train.worker.PipeWorker
   :members:
   
Utils
---------------------

.. autoclass:: elegantrl.train.worker.act_dict_to_device

.. autoclass:: elegantrl.train.worker.trajectory_to_device
