Configuration: *config.py*
==========================


``Arguments``
---------------------

The ``Arguments`` class contains all parameters of the training process, including environment setup, model training, model evaluation, and resource allocation. It provides users an unified interface to customize the training process. 

The class should be initialized at the start of the training process. For example,

.. code-block:: python

   from elegantrl.train.config import Arguments
   from elegantrl.agents.AgentPPO import AgentPPO
   from elegantrl.envs.Gym import build_env
   import gym
   
   args = Arguments(build_env('Pendulum-v1'), AgentPPO())

The full list of parameters in ``Arguments``:

.. autoclass:: elegantrl.train.config.Arguments
   :members:


Environment registration
---------------------

.. autofunction:: elegantrl.train.config.build_env

.. autofunction:: elegantrl.train.config.check_env


Utils
---------------------

.. autofunction:: elegantrl.train.config.kwargs_filter
