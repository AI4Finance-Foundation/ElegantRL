Mod SAC
==============

Example
-------------

.. code-block:: python
   :linenos:

      class AgentModSAC(AgentSAC):  # Modified SAC using reliable_lambda and TTUR (Two Time-scale Update Rule)
          def __init__(self):
              super().__init__()
              self.if_use_dn = False
              self.obj_c = (-np.log(0.5)) ** 0.5  # for reliable_lambda

          def init(self, net_dim, state_dim, action_dim):
              self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
              self.target_entropy = np.log(action_dim)
              self.alpha_log = torch.tensor((-np.log(action_dim) * np.e,), dtype=torch.float32,
                                            requires_grad=True, device=self.device)  # trainable parameter

              self.act = ActorSAC(net_dim, state_dim, action_dim, self.if_use_dn).to(self.device)
              self.act_target = deepcopy(self.act)
              self.cri = CriticTwin(int(net_dim * 1.25), state_dim, action_dim, self.if_use_dn).to(self.device)
              self.cri_target = deepcopy(self.cri)

              self.criterion = torch.nn.SmoothL1Loss()
              self.optimizer = torch.optim.Adam([{'params': self.act.parameters(), 'lr': self.learning_rate},
                                                 {'params': self.cri.parameters(), 'lr': self.learning_rate},
                                                 {'params': (self.alpha_log,), 'lr': self.learning_rate}])




Actor-Critic
---------------

.. autoclass:: elegantrl.agent.AgentModSAC
   :members:
   :undoc-members:
