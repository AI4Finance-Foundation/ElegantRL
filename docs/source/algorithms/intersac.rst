Inter SAC
==============

Example
-------------

.. code-block:: python
   :linenos:

      class AgentInterSAC(AgentSAC):  # Integrated Soft Actor-Critic
          def __init__(self):
              super().__init__()
              self.obj_c = (-np.log(0.5)) ** 0.5  # for reliable_lambda

          def init(self, net_dim, state_dim, action_dim):
              self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
              self.target_entropy = np.log(action_dim)
              self.alpha_log = torch.tensor((-np.log(action_dim) * np.e,), dtype=torch.float32,
                                            requires_grad=True, device=self.device)  # trainable parameter

              self.act = InterSPG(net_dim, state_dim, action_dim).to(self.device)
              self.act_target = deepcopy(self.act)

              self.criterion = torch.nn.SmoothL1Loss()
              self.optimizer = torch.optim.Adam(
                  [{'params': self.act.enc_s.parameters(), 'lr': self.learning_rate * 0.9},  # more stable
                   {'params': self.act.enc_a.parameters(), },
                   {'params': self.act.net.parameters(), 'lr': self.learning_rate * 0.9},
                   {'params': self.act.dec_a.parameters(), },
                   {'params': self.act.dec_d.parameters(), },
                   {'params': self.act.dec_q1.parameters(), },
                   {'params': self.act.dec_q2.parameters(), },
                   {'params': (self.alpha_log,)}], lr=self.learning_rate)



Actor-Critic
--------------------
.. autoclass:: elegantrl.agent.AgentInterSAC
   :members:
   :undoc-members:
