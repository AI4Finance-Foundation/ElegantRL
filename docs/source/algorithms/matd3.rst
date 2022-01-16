.. _matd3:


MATD3
==========

`Multi-Agent TD3(MATD3) <https://arxiv.org/abs/1910.01465>`_ uses double centralized critics to reduce overestimation bias in multi-agent domains.


Code Snippet
------------
.. code-block:: python

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        """
        Update the neural networks by sampling batch data from ``ReplayBuffer``.
        
        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :param repeat_times: the re-using times of each trajectory.
        :param soft_update_tau: the soft update parameter.
        :return Nonetype
        """
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

.. autoclass:: elegantrl.agents.AgentMATD3.AgentMATD3
   :members:
.. _matd3_networks:
   
Networks
-------------

.. autoclass:: elegantrl.agents.net.Actor
   :members:
   
.. autoclass:: elegantrl.agents.net.CriticTwin
   :members:

