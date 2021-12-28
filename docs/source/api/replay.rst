Replay Buffer: *replay_buffer.py*
=================================

ElegantRL provides ``ReplayBuffer`` to store sampled transitions. 

In ElegantRL, we utilize ``Worker`` for exploration (data sampling) and ``Learner`` for exploitation (model learning), and we view such a relationship as a "producer-consumer" model, where a worker produces transitions and a learner consumes, and a learner updates the actor net at worker to produce new transitions. In this case, the ``ReplayBuffer`` is the storage buffer that connects the worker and learner.

Each transition is in a format (state, (reward, done, action)).

.. note::
  We allocate the ``ReplayBuffer`` on continuous RAM for high performance training. Since the collected transitions are packed in sequence, the addressing speed increases dramatically when a learner randomly samples a batch of transitions.
  
Implementations
---------------------

.. autoclass:: elegantrl.train.replay_buffer.ReplayBuffer
   :members:
   
Multiprocessing
---------------------

.. autoclass:: elegantrl.train.replay_buffer.ReplayBufferMP
   :members:
   
Initialization
---------------------

.. autofunction:: elegantrl.train.replay_buffer.init_replay_buffer

Utils
---------------------

.. autoclass:: elegantrl.train.replay_buffer.BinarySearchTree
