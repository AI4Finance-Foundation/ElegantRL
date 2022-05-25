.. _mappo:


MAPPO
==========

Multi-Agent Proximal Policy Optimization (MAPPO), a variant of PPO, is specialized for multi-agent settings. Using a 1-GPU desktop, we show that MAPPO achieves surprisingly strong performance in two popular multi-agent testbeds: the particle-world environments, and the Starcraft multi-agent challenge.

-  Shared network parameter for all agents ✔️
-  This class is under test, we temporarily add all utils in AgentMAPPO ✔️

MAPPO achieves strong results while exhibiting comparable sample efficiency. 


Parameters
---------------------

.. autoclass:: elegantrl.agents.AgentRODE.AgentREDQ
   :members:

.. _redq_networks:
   
Networks
-------------

.. autoclass:: elegantrl.agents.net.ActorSAC
   :members:
   
.. autoclass:: elegantrl.agents.net.Critic
   :members:
   
              
