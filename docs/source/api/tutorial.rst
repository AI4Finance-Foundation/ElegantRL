Tutorial
========

Networks-net.py
---------------

class QNet(nn.Module)
^^^^^^^^^^^^^^^^^^^^^

.. .. code-block:: python
..    :linenos:

..     __init__(self, mid_dim, state_dim, action_dim)

**__init__(self, mid_dim, state_dim, action_dim)**

Create a four-layer neural network with *mid_dim* nodes in input layer, *state_dim* nodes in hidden layers, and *action_dim* nodes in output layer. nn.ReLU() is used as activation function.

.. .. code-block:: python
..    :linenos:

..     forward(self, state)

**forward(self, state)**

Take *state* as the input of the neural network and return the Q values.


Agents-agent.py
---------------

Environment-env.py
------------------

Main-run.py
-----------