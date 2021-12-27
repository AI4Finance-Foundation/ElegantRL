Worker: *worker.py*
=================================

Deep reinforcement learning (DRL) employs a trial-and-error manner to collect training data (transitions) from agent-environment interactions, along with the learning procedure. ElegantRL utilizes ``Worker`` to greatly speeds up the data collection through worker parallelism (VecEnv), by making full use of hardwares, GPUs in particular.
A ``Worker`` generates transitions from interactions between the actor (policy) network and the environment. ElegantRL achieves worker parallelism to 
