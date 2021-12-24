Evaluator (evaluator.py)
===============================

ElegantRL provide an ``evaluator`` for agent's performance evaluation and model saving. In the course of training, the evaluator periodically evaluates agents and outputs corresponding scores. Commonly used performance metrics are mean and variance of episodic rewards.


The score from the evaluator has following two functions:

  1. **Metrics for leaderboard**: ElegantRL supports a tournament-based ensemble training scheme to empower the population-based training (PBT). We maintain a leaderboard to keep track of agents with high scores and then we perform a tournament-based evolution among these agents. In this case, the score from the evaluator serves as a metric for leaderboard.
  
  2. **Early stopping mechanism**: ElegantRL provides an early stopping mechanism to mitigate the performance collapse and the waste of computing power brought by overfitting. When the score from the evaluator continuously drops, we would terminate the training process early. Therefore, the score serves as a criterion to determine overfitting of models. 
  
We implement the ``evaluator`` as a microservice, which can be ran as an independent process. When an evaluator is running, it can automatically monitors parallel agents, and provide evaluation when any agent needs, and communicate agent information with the leaderboard.

.. autoclass:: elegantrl.train.evaluator.Evaluator
   :members:
   
