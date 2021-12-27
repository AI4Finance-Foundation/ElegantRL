Run: *run.py*
=================================

In *run.py*, we provide functions to wrap the training (and evaluation) process. 

To train an agent in ElegantRL, users mainly follow a two-step procedure. 
1. Initializing the agent, environment and setting hyper-parameters up in ``Arguments``.
2. Passing the ``Arguments`` to functions for the training process, e.g., ``train_and_evaluate`` for single-process training and ``train_and_evaluate_mp`` for multi-process training.

.. code-block:: python
   
   from elegantrl.train.config import Arguments
   from elegantrl.train.run import train_and_evaluate, train_and_evaluate_mp
   from elegantrl.envs.Chasing import ChasingEnv
   from elegantrl.agents.AgentPPO import AgentPPO
   
   # Step 1
   args = Arguments(agent=AgentPPO(), env_func=ChasingEnv)
   
   # Step 2
   train_and_evaluate_mp(args)

Single Process
---------------------

.. autofunction:: elegantrl.train.run.train_and_evaluate
   
Multi Process
---------------------

.. autofunction:: elegantrl.train.run.train_and_evaluate_mp
   
Utils
---------------------

.. autoclass:: elegantrl.train.run.process_safely_terminate
