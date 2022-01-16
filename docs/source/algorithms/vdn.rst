.. _vdn:


VDN
==========

`Value Decomposition Networks (VDN) <https://arxiv.org/abs/1706.05296>`_  trains individual agents with a novel value decomposition network architecture, which learns to decompose the team value function into agent-wise value functions.

Code Snippet
------------
.. code-block:: python

    def train(self, batch, t_env: int, episode_num: int):
    
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)



Parameters
---------------------

.. autoclass:: elegantrl.agents.AgentVDN.AgentVDN
   :members:
   
.. _vdn_networks:
Networks
-------------
   
.. autoclass:: elegantrl.agents.net.VDN
    :members:
