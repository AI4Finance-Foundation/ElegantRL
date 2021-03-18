PPO
==========

Example
-----------------

.. code-block:: python
   :linenos:

      class AgentPPO(AgentBase):
          def __init__(self):
              super().__init__()
              self.clip = 0.3  # ratio.clamp(1 - clip, 1 + clip)
              self.lambda_entropy = 0.01  # larger lambda_entropy means more exploration
              self.noise = None

          def init(self, net_dim, state_dim, action_dim):
              self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

              self.act = ActorPPO(net_dim, state_dim, action_dim).to(self.device)
              self.cri = CriticAdv(state_dim, net_dim).to(self.device)

              self.criterion = torch.nn.SmoothL1Loss()
              self.optimizer = torch.optim.Adam([{'params': self.act.parameters(), 'lr': self.learning_rate},
                                                 {'params': self.cri.parameters(), 'lr': self.learning_rate}])



Actor-Critic
---------------

.. autoclass:: elegantrl.agent.AgentPPO
   :members:
   :undoc-members:
