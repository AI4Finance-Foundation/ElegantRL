.. _dqn:
.. automodule:: elegantrl.agent
DQN
==========
`Deep Q-Network (DQN) <https://arxiv.org/abs/1312.5602>`_ approximates a state-value function in a Q-Learning framework with a neural network. This implementation provides vanilla Deep Q-Learning and supports the following extensions:
-  Experience replay: ✔️
-  Target network: ✔️
-  Gradient clipping: ✔️
-  Reward clipping: ❌
-  Prioritized Experience Replay (PER): ✔️

.. note::
    This implementation has no support for reward clipping because we introduce the hyper-paramter ``reward_scale`` as an alternative for reward scaling. We believe that the clipping function may omit information since it cannot map the clipped reward back to the original reward, however, the reward scaling function is able to manipulate the reward back and forth.
    This implementation has no support for reward clipping because we introduce the hyper-paramter ``reward_scale`` as an alternative for reward scaling. We believe that the clipping function may omit information since it cannot map the clipped reward back to the original reward, however, the reward scaling function is able to map the reward back and forth.

Example
------------
.. code-block:: python
   :linenos:
      class AgentDQN(AgentBase):
          def __init__(self):
              super().__init__()
              self.explore_rate = 0.1  # the probability of choosing action randomly in epsilon-greedy
              self.action_dim = None  # chose discrete action randomly in epsilon-greedy
              self.state = None  # set for self.update_buffer(), initialize before training
              self.learning_rate = 1e-4
              self.act = None
              self.cri = self.cri_target = None
              self.criterion = None
              self.optimizer = None
          def init(self, net_dim, state_dim, action_dim):
              self.action_dim = action_dim
              self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
              self.cri = QNet(net_dim, state_dim, action_dim).to(self.device)
              self.cri_target = deepcopy(self.cri)
              self.act = self.cri  # to keep the same from Actor-Critic framework
              self.criterion = torch.torch.nn.MSELoss()
              self.optimizer = torch.optim.Adam(self.cri.parameters(), lr=self.learning_rate)
Parameters
---------------------
.. autoclass:: elegantrl.agent.AgentDQN
   :members:
   
.. _dqn_networks:
   
Networks
-------------
.. autoclass:: elegantrl.net.QNet
   :members:
