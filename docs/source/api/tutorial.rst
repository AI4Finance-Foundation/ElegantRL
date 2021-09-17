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

The network has four layers with ReLU activation functions, where the input size is ``state_dim`` and the output size is ``action_dim``.

nn.ReLU() is used as the activation function.

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

Create three three-layer neural networks:

The first has ``state_dim`` amount of nodes in input layer, ``mid_dim`` amount of nodes in hidden layer and output layer.

The second and third both have ``mid_dim`` amount of nodes in input layer, ``state_dim`` amount of nodes in hidden layer and output layer.

nn.ReLU() is used as the activation function.

- forward(*self, state*)

Take ``state`` as the input, and connect the first neural network with the second in series. Return the ouputs of the second network, which is one Q value.

- get_q1_q2(*self, state*)

Take ``state`` as the input. Then separately connect the first neural network with the second and third in series. Return the ouputs of the second and third networks, which are two Q values.

Agents: *agent.py*
------------------

class AgentBase
^^^^^^^^^^^^^^^



Environment: *env.py*
---------------------

Main: *run.py*
--------------