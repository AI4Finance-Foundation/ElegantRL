.. automodule:: elegantrl.agent

DQN
==========

`Deep Q-Network (DQN) <https://arxiv.org/abs/1312.5602>`_ approximates a state-value function in a Q-Learning framework with a neural network. This implementation provides vanilla Deep Q-Learning and supports the following extensions:

-  Experience replay: ✔️
-  Target network: ✔️
-  Gradient clipping: ✔️
-  Reward clipping: ❌
-  Prioritized Experience Replay (PER): ✔️
-  Dueling DQN: ✔️

.. note::
    This implementation has no support for reward clipping because we introduce the hyper-paramter ``reward_scale`` as an alternative for reward scaling. We believe that the clipping function may omit information since it cannot map the clipped reward back to the original reward, however, the reward scaling function is able to map the reward back and forth.

Example
------------

.. code-block:: python
   :linenos:
        fsddfasdf



Parameters
---------------------

.. autoclass:: elegantrl.agent.AgentDQN
   :members:
   

Networks
-------------

.. autoclass:: elegantrl.net.QNet
   :members:

.. autoclass:: elegantrl.net.QNetDuel
   :members:
