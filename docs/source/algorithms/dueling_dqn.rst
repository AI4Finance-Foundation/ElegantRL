Dueling DQN
=======================

Example
-------------------

.. code-block:: python
   :linenos:

    class AgentDuelingDQN(AgentDQN):
        def __init__(self):
            super().__init__()
            self.explore_rate = 0.25  # the probability of choosing action randomly in epsilon-greedy

        def init(self, net_dim, state_dim, action_dim):
            self.action_dim = action_dim
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.cri = QNetDuel(net_dim, state_dim, action_dim).to(self.device)
            self.cri_target = deepcopy(self.cri)
            self.act = self.cri

            self.criterion = torch.nn.MSELoss()
            self.optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate)
            """Contribution of Dueling DQN
            1. Advantage function (of A2C) --> Dueling Q value = val_q + adv_q - adv_q.mean()
            """
       
Value-based Methods
-----------------------

.. autoclass:: elegantrl.agent.AgentDuelingDQN
   :members:
   :undoc-members:
