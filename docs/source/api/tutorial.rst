Tutorial
========

Networks: *net.py*
------------------

class QNet(*nn.Module*)
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
   :linenos:

    class QNet(nn.Module):  # nn.Module is a standard PyTorch Network
        def __init__(self, mid_dim, state_dim, action_dim):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, action_dim))

        def forward(self, state):
            return self.net(state)  # q value

- __init__(*self, mid_dim, state_dim, action_dim*)

The network has four layers with ReLU activation functions, where the input size is ``state_dim`` and the output size is ``action_dim``, with ReLU activation functions.

- forward(*self, state*)

Take ``state`` as the input and output Q values.

class QNetTwin(*nn.Module*) 
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
   :linenos:

    class QNetTwin(nn.Module):  # Double DQN
        def __init__(self, mid_dim, state_dim, action_dim):
            super().__init__()
            self.net_state = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU(),
                                        nn.Linear(mid_dim, mid_dim), nn.ReLU())
            self.net_q1 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                        nn.Linear(mid_dim, action_dim))  # q1 value
            self.net_q2 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                        nn.Linear(mid_dim, action_dim))  # q2 value

        def forward(self, state):
            tmp = self.net_state(state)
            return self.net_q1(tmp)  # one Q value

        def get_q1_q2(self, state):
            tmp = self.net_state(state)
            return self.net_q1(tmp), self.net_q2(tmp)  # two Q values

- __init__(*self, mid_dim, state_dim, action_dim*)

There are three networks:

The **net_state** network has two layers,  where the input size is ``state_dim`` and the output size is ``mid_dim``.

The **net_q1** and **net_q2** network has two layers,  where the input size is mid_dim and the output size is action_dim.

The **net_state** network is connected to both the **net_q1** network and **net_q2** network, with ReLU activation functions.

- forward(*self, state*)

Take ``state`` as the input and output one Q value.

- get_q1_q2(*self, state*)

Take ``state`` as the input and output two Q values.

Agents: *agent.py*
------------------

class AgentBase
^^^^^^^^^^^^^^^



Environment: *env.py*
---------------------

Main: *run.py*
--------------