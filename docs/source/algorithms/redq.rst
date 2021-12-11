.. _redq:


REDQ
==========

`REDQ <https://arxiv.org/abs/2101.05982>`_ REDQ has three carefully integrated ingredients which allow it to achieve its high performance: (i) a UTD ratio >> 1; (ii) an ensemble of Q functions; (iii) in-target minimization across a random subset of Q functions from the ensemble. This implementation is based on SAC and supports the following extensions:

- Implement G, M, N Parameter
- Based On SAC class
- Works well in Mujoco


Code Snippet
------------
You can change G,M,N when call AgentREDQ.init 
.. code-block:: python

    AgentREDQ.init(self, net_dim=256, state_dim=8, action_dim=2, reward_scale=1.0, gamma=0.99,
            learning_rate=3e-4, if_per_or_gae=False, env_num=1, gpu_id=0, G=20, M=2, N=10):
              
              
              
