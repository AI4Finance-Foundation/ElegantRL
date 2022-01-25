Run: *run.py*
=================================

In *run.py*, we provide functions to wrap the training (and evaluation) process. 

In ElegantRL, users follow a **two-step procedure** to train an agent in a lightweight and automatic way. 

1. Initializing the agent and environment, and setting hyper-parameters up in ``Arguments``.
2. Passing the ``Arguments`` to functions for the training process, e.g., ``train_and_evaluate`` for single-process training and ``train_and_evaluate_mp`` for multi-process training.

Let's look at a demo for the simple two-step procedure.

.. code-block:: python
   
   from elegantrl.train.config import Arguments
   from elegantrl.train.run import train_and_evaluate, train_and_evaluate_mp
   from elegantrl.envs.Chasing import ChasingEnv
   from elegantrl.agents.AgentPPO import AgentPPO
   
   # Step 1
   args = Arguments(agent=AgentPPO(), env_func=ChasingEnv)
   
   # Step 2
   train_and_evaluate_mp(args)

Single-process
---------------------

.. autofunction:: elegantrl.train.run.train_and_evaluate
   
Multi-process
---------------------

.. autofunction:: elegantrl.train.run.train_and_evaluate_mp
   
Utils
---------------------

.. autoclass:: elegantrl.train.run.safely_terminate_process

.. autoclass:: elegantrl.train.run.check_subprocess
