TD3
=======

Example
---------------

.. code-block:: python
   :linenos:

      class AgentTD3(AgentBase):
          def __init__(self):
              super().__init__()
              self.explore_noise = 0.1  # standard deviation of explore noise
              self.policy_noise = 0.2  # standard deviation of policy noise
              self.update_freq = 2  # delay update frequency, for soft target update

          def init(self, net_dim, state_dim, action_dim):
              self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

              self.cri = CriticTwin(net_dim, state_dim, action_dim).to(self.device)
              self.cri_target = deepcopy(self.cri)

              self.criterion = torch.nn.MSELoss()
              self.optimizer = torch.optim.Adam([{'params': self.act.parameters(), 'lr': self.learning_rate},
                                                 {'params': self.cri.parameters(), 'lr': self.learning_rate}])



Actor-Critic
----------------

.. autoclass:: elegantrl.agent.AgentTD3
   :members:
   :undoc-members:
