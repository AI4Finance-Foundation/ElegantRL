Double DQN
===========

Example
----------------

.. code-block:: python
   :linenos:

      class AgentDoubleDQN(AgentDQN):
          def __init__(self):
              super().__init__()
              self.explore_rate = 0.25  # the probability of choosing action randomly in epsilon-greedy
              self.softmax = torch.nn.Softmax(dim=1)

          def init(self, net_dim, state_dim, action_dim):
              self.action_dim = action_dim
              self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

              self.cri = QNetTwin(net_dim, state_dim, action_dim).to(self.device)
              self.cri_target = deepcopy(self.cri)
              self.act = self.cri

              self.criterion = torch.nn.SmoothL1Loss()
              self.optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate)


Value-based Methods
---------------------------

.. autoclass:: elegantrl.agent.AgentDoubleDQN
   :members:
   :undoc-members:
