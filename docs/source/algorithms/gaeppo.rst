Gae PPO
=============

Example
---------------

.. code-block:: python
   :linenos:

      class AgentGaePPO(AgentPPO):
          def __init__(self):
              super().__init__()
              self.clip = 0.25  # ratio.clamp(1 - clip, 1 + clip)
              self.lambda_entropy = 0.01  # could be 0.02
              self.lambda_gae_adv = 0.98  # could be 0.95~0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)


Actor-Critic
-----------------
.. autoclass:: elegantrl.agent.AgentGaePPO
   :members:
   :undoc-members:
