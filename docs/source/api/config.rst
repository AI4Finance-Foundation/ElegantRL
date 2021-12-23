Configuration (config.py)
==================

To keep ElegantRL simple to use, we allow users to control the training process through an ``Arguments`` class. This class contains all adjustable parameters of the training process, including environment setup, model training, model evaluation, and resource allocation. 

The ``Arguments`` class provides users an unified interface to customize the training process and save the training profile. The class should be initialized at the start of the training process.

.. code-block:: python
   from elegantrl.train.config import Arguments
   from elegantrl.agents.AgentPPO import AgentPPO
   from elegantrl.envs.Gym import build_env
   import gym
   
   args = Arguments(build_env('Pendulum-v1'), AgentPPO())

The full list of parameters in ``Arguments``:

.. autoclass:: elegantrl.train.config.Arguments
   :members:

