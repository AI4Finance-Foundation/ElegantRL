Worker: *worker.py*
=================================

Deep reinforcement learning (DRL) employs a trial-and-error manner to collect training data (transitions) from agent-environment interactions, along with the learning procedure. ElegantRL utilizes ``Worker`` to generate transitions and achieves worker parallelism, thus greatly speeding up the data collection.

Implementations
---------------------

.. autoclass:: elegantrl.train.worker.PipeWorker
   :members:
   
