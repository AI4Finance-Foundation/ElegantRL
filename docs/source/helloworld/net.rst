Networks: *net.py*
==================

In ElegantRL, there are three basic network classes: Q-net, Actor, and Critic. Here, we list several examples, which are the networks used by DQN, SAC, and PPO algorithms. 

The full list of networks are available `here <https://github.com/AI4Finance-Foundation/ElegantRL/blob/master/elegantrl/agents/net.py>`_

Q Net
-----

.. autoclass:: elegantrl_helloworld.net.QNet
   :members:

Actor Network
-------------

.. autoclass:: elegantrl_helloworld.net.ActorSAC
   :members:

.. autoclass:: elegantrl_helloworld.net.ActorPPO
   :members:

.. autoclass:: elegantrl_helloworld.net.ActorDiscretePPO
   :members:

Critic Network
--------------

.. autoclass:: elegantrl_helloworld.net.CriticTwin
   :members:

.. autoclass:: elegantrl_helloworld.net.CriticPPO
   :members:
