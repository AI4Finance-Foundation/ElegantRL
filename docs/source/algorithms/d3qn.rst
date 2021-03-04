D3QN
==========

Example
-----------------

.. code-block:: python
   :linenos:

      class AgentD3QN(AgentDoubleDQN):  # D3QN: Dueling Double DQN
          def __init__(self):
              super().__init__()

          def init(self, net_dim, state_dim, action_dim):
              self.action_dim = action_dim
              self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

              self.cri = QNetTwinDuel(net_dim, state_dim, action_dim).to(self.device)
              self.cri_target = deepcopy(self.cri)
              self.act = self.cri

              self.criterion = torch.nn.SmoothL1Loss()
              self.optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate)
              """Contribution of D3QN (Dueling Double DQN)
              There are not contribution of D3QN.  
              Obviously, DoubleDQN is compatible with DuelingDQN.
              Any beginner can come up with this idea (D3QN) independently.
              """


Value-based Methods
------------------------
.. autoclass:: elegantrl.agent.AgentD3QN
   :members:
   :undoc-members:
