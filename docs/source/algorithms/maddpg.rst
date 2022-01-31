.. _maddpg:


MADDPG
==========

`Multi-Agent Deep Deterministic Policy Gradient (MADDPG) <https://arxiv.org/abs/1706.02275>`_ is a multi-agent reinforcement learning algorithm for continuous action space:

-  Implementation is based on DDPG ✔️
-  Initialize n DDPG agents in MADDPG ✔️

Code Snippet
------------

.. code-block:: python

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        buffer.update_now_len()
        self.batch_size = batch_size
        self.update_tau = soft_update_tau
        rewards, dones, actions, observations, next_obs = buffer.sample_batch(self.batch_size)
        for index in range(self.n_agents):
            self.update_agent(rewards, dones, actions, observations, next_obs, index)

        for agent in self.agents:
            self.soft_update(agent.cri_target, agent.cri, self.update_tau)
            self.soft_update(agent.act_target, agent.act, self.update_tau)
    
        return           

Parameters
---------------------

.. autoclass:: elegantrl.agents.AgentMADDPG.AgentMADDPG
   :members:
.. _maddpg_networks:
   
Networks
-------------

.. autoclass:: elegantrl.agents.net.Actor
   :members:
   
.. autoclass:: elegantrl.agents.net.Critic
   :members:

