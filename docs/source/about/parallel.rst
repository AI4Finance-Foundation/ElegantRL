Muti-level Parallelism
==============================================

ElegantRL is a massively parallel framework for DRL algorithms. We fully exploit the parallelism of DRL algorithms at multiple levels, including agent parallelism of population-based training and worker-learner parallelism of a single agent.

Here, we follow a *bottom-up* approach to describe the parallelism of DRL algorithms at multiple levels.


Worker parallelism
-----------------------------------------------------------

At the bottom, an worker generates transitions (collect training data) from interactions between policy network and environment. ElegantRL enables massively parallel simulation via a vectorized environment (VecEnv) class, which supports thousands of sub-environments on one GPU. ElegantRL also support the worker parallelism for non-vectorized environment, e.g., OpenAI Gym and MuJoCo.


Learner parallelism
-----------------------------------------------------------

In the middle, a learner fetches a batch of transitions to train neural networks, e.g., value net and policy net. We support multiple-critics and multiple actors running in parallel for ensemble DRL methods. Due to the stochastic nature of the training process (e.g., random seeds), an ensemble DRL algorithm increases the diversity of the data collection, improves the stability of the learning process, and reduces the overestimation bias.


Agent parallelism
-----------------------------------------------------------

On the top, an agent is self-contained and encapsulated, including the components, *worker, learner, and evaluator*. We adopt the population-based training to train hundreds of agents in parallel, which offers a flexibility for ensemble methods.
