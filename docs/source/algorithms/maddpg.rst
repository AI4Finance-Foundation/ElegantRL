.. _maddpg:


MADDPG
==========

`Multi-Agent Deep Deterministic Policy Gradient (MADDPG) <https://arxiv.org/abs/1706.02275>`_  This implementation is based on DDPG and supports the following extensions:

-  Implement is based on DDPG ✔️
-  Init n DDPG Agent in MADDPG: ✔️

Code Snippet
------------
DDPG Agents is store in the list agents
.. code-block:: python

    def init(self,net_dim, state_dim, action_dim, learning_rate=1e-4,marl=True, n_agents = 1,   if_use_per=False, env_num=1, agent_id=0):
        self.agents = [AgentDDPG() for i in range(n_agents)]
        self.explore_env = self.explore_one_env
        self.if_off_policy = True
        self.n_agents = n_agents
        for i in range(self.n_agents):
            self.agents[i].init(net_dim, state_dim, action_dim, learning_rate=1e-4,marl=True, n_agents = self.n_agents,   if_use_per=False, env_num=1, agent_id=0)
        self.n_states = state_dim
        self.n_actions = action_dim
        
        self.batch_size = net_dim
        self.gamma = 0.95
        self.update_tau = 0
        self.device = torch.device(f"cuda:{agent_id}" if (torch.cuda.is_available() and (agent_id >= 0)) else "cpu")
              
              
              
