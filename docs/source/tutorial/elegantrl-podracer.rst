Cloud Example 2: Tournament-based Ensemble Training
======================================================================

In this section, we provide a tutorial of *tournament-based ensemble training*, to show ElegantRL's scalability on hundreds of computing nodes on a cloud platform, say, hundreds of GPUs.

For detailed description, please check our recent paper, presented at NeurIPS 2021: Deep RL Workshop: `ElegantRL-Podracer: Scalable and Elastic Library for Cloud-Native Deep Reinforcement Learning <https://arxiv.org/abs/2112.05923>`_.  

What is a tournament-based ensemble training?
------------------------------------------------------------

The key of the tournament-based ensemble training scheme is the interaction between a *training pool* and a *leaderboard*. The training pool contains hundreds of agents that 1) are trained in an asynchronous manner, and 2) can be initialized with different DRL algorithms/hyper-parameter setup for an ensemble purpose. The leaderboard records the agents with high performance and continually updates as more agents (pods) are trained.

.. image:: ../images/framework2.png
   :width: 100%
   :align: center


As shown in the figure above, the tournament-based ensemble training proceeds as follows:

  1. An *orchestrator* instantiates a new agent and put it into a training pool.
  
  2. A *generator* initializes an agent with networks and optimizers selected from a leaderboard. The generator is a class of subordinate functions associated with the leaderboard, which has different variations to support different evolution strategies
  
  3. An *updater* determines whether and where to insert an agent into the leaderboard according to its performance, after a pod has been trained for a certain number of steps or certain amount of time.





Comparison with generational evolution
---------------------------------------------------------------

In generational evolution, the entire population of agents is simultaneously updated for each generation.  However, this paradigm scales poorly on the cloud since it requires to finish training of every member of a large population before any further evolution can occur, imposing a significant computational burden.


Our tournament-based ensemble training updates agents asynchronously, which decouples population evolution and singleagent learning. Such an asynchronously distributed training reduce waiting time among parallel agents and reduce the agent-to-agent communication overhead. 




Example: Isaac Gym
-------------------------------------------------------

We select two canonical robotic control tasks, Ant and Humanoid, where each task has both MuJoCo version and Isaac Gym verison. We compare our tournament-based ensemble training with RLlib on these four tasks. 


We employ two different metrics to evaluate the agent’s performance:

  - **Episodic reward vs. training time (wall-clock time)**: we measure the episodic reward at different training time, which can be affected by the convergence speed, communication overhead, scheduling efficiency, etc.

  - **Episodic reward vs. training step**: from the same testings, we also measure the episodic reward at different training steps. This result can be used to investigate the massive parallel simulation capability of GPUs, and also check the algorithm’s performance.


.. image:: ../images/test1.png
   :width: 90%
   :align: center
   

.. image:: ../images/test2.png
   :width: 90%
   :align: center   


Run tournament-based ensemble training in ElegantRL
--------------------------------------------------------------
