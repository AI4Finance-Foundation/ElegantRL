A2C
=======


Example
---------------

.. code-block:: python
   :linenos:

    class AgentInterAC(AgentBase):  # use InterSAC instead of InterAC .Warning: sth. wrong with this code, need to check
        def __init__(self):
            super().__init__()
            self.explore_noise = 0.2  # standard deviation of explore noise
            self.policy_noise = 0.4  # standard deviation of policy noise
            self.update_freq = 2 ** 7  # delay update frequency, for hard target update
            self.avg_loss_c = (-np.log(0.5)) ** 0.5  # old version reliable_lambda

        def init(self, net_dim, state_dim, action_dim):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.act = InterDPG(state_dim, action_dim, net_dim).to(self.device)
            self.act_target = deepcopy(self.act)

            self.criterion = torch.nn.MSELoss()
            self.optimizer = torch.optim.Adam(self.act.parameters(), lr=self.learning_rate)



Actor-Critic
----------------

.. autoclass:: elegantrl.agent.AgentInterAC		
   :members:
   :undoc-members:
