DDPG
=======

Example
-----------------

.. code-block:: python
   :linenos:

      class AgentDDPG(AgentBase):
          def __init__(self):
              super().__init__()
              self.ou_explore_noise = 0.3  # explore noise of action
              self.ou_noise = None

          def init(self, net_dim, state_dim, action_dim):
              self.ou_noise = OrnsteinUhlenbeckNoise(size=action_dim, sigma=self.ou_explore_noise)
              # I don't recommend OU-Noise
              self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

              self.act = Actor(net_dim, state_dim, action_dim).to(self.device)
              self.act_target = deepcopy(self.act)
              self.cri = Critic(net_dim, state_dim, action_dim).to(self.device)
              self.cri_target = deepcopy(self.cri)

              self.criterion = torch.nn.MSELoss()
              self.optimizer = torch.optim.Adam([{'params': self.act.parameters(), 'lr': self.learning_rate},
                                                 {'params': self.cri.parameters(), 'lr': self.learning_rate}])


Actor-Critic
----------------

.. autoclass:: elegantrl.agent.AgentDDPG
   :members:
   :undoc-members:

