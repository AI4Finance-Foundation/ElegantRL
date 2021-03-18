Inter PPO
===============

Example
------------

.. code-block:: python
   :linenos:
   
      class AgentInterPPO(AgentPPO):
          def __init__(self):
              super().__init__()
              self.clip = 0.25  # ratio.clamp(1 - clip, 1 + clip)
              self.lambda_entropy = 0.01  # could be 0.02
              self.lambda_gae_adv = 0.98  # could be 0.95~0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)
              self.obj_c = (-np.log(0.5)) ** 0.5  # for reliable_lambda

          def init(self, net_dim, state_dim, action_dim):
              self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

              self.act = InterPPO(state_dim, action_dim, net_dim).to(self.device)

              self.criterion = torch.nn.SmoothL1Loss()
              self.optimizer = torch.optim.Adam([
                  {'params': self.act.enc_s.parameters(), 'lr': self.learning_rate * 0.9},
                  {'params': self.act.dec_a.parameters(), },
                  {'params': self.act.a_std_log, },
                  {'params': self.act.dec_q1.parameters(), },
                  {'params': self.act.dec_q2.parameters(), },
              ], lr=self.learning_rate)


Actor-Critic
---------------------
.. autoclass:: elegantrl.agent.AgentInterPPO
   :members:
   :undoc-members:
