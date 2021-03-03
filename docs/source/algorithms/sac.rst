SAC
=======

Example
------------------------

.. code-block:: python
   :linenos:
   
      class AgentSAC(AgentBase):
          def __init__(self):
              super().__init__()
              self.target_entropy = None
              self.alpha_log = None

          def init(self, net_dim, state_dim, action_dim):
              self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
              self.target_entropy = np.log(action_dim)
              self.alpha_log = torch.tensor((-np.log(action_dim) * np.e,), dtype=torch.float32,
                                            requires_grad=True, device=self.device)  # trainable parameter

              self.act = ActorSAC(net_dim, state_dim, action_dim).to(self.device)
              self.act_target = deepcopy(self.act)
              self.cri = CriticTwin(int(net_dim * 1.25), state_dim, action_dim).to(self.device)
              self.cri_target = deepcopy(self.cri)

              self.criterion = torch.nn.SmoothL1Loss()
              self.optimizer = torch.optim.Adam([{'params': self.act.parameters(), 'lr': self.learning_rate},
                                                 {'params': self.cri.parameters(), 'lr': self.learning_rate},
                                                 {'params': (self.alpha_log,), 'lr': self.learning_rate}])


Actor-Critic
-----------------

.. autoclass:: elegantrl.agent.AgentSAC
   :members:
   :undoc-members:
