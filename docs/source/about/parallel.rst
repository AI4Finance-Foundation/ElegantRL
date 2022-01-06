Muti-level Parallelism
==============================================

ElegantRL is a massively parallel framework for DRL algorithms. We fully exploit the parallelism of DRL algorithms at multiple levels, including agent parallelism of population-based training and worker-learner parallelism of a single agent.

Here, we follow a *bottom-up* approach to describe the parallelism of DRL algorithms at multiple levels.


Worker parallelism
-----------------------------------------------------------

At the bottom, an worker generates transitions (collect training data) from interactions between policy network and environment. Based on the type of the environment, we support two different worker parallelism to generate transitions in parallel:

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


Learner parallelism
-----------------------------------------------------------

In the middle, a learner fetches a batch of transitions to train neural networks, e.g., value net and policy net. We support multiple-critics and multiple actors running in parallel for ensemble DRL methods. Due to the stochastic nature of the training process (e.g., random seeds), an ensemble DRL algorithm increases the diversity of the data collection, improves the stability of the learning process, and reduces the overestimation bias.


Agent parallelism
-----------------------------------------------------------

On the top, an agent is self-contained and encapsulated, including the components, *worker, learner, and evaluator*. We adopt the population-based training to train hundreds of agents in parallel, which offers a flexibility for ensemble methods on the cloud.
