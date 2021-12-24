Evaluator (evaluator.py)
===============================

ElegantRL provide an ``evaluator`` for agent's performance evaluation and model saving. In the course of training, the evaluator periodically evaluates agents and outputs corresponding scores. Commonly used performance metrics are mean and variance of episodic rewards. 


The score from the evaluator has following two functions:

- **Metrics for leaderboard**: ElegantRL supports a tournament-based ensemble training scheme to empower the population-based training (PBT). We maintain a leaderboard to keep track of agents with high scores and then perform a tournament-based evolution among these agents. In this case, the score from the evaluator serves as a metric for leaderboard.
  
- **Metrics for termination**: The evaluator determines whether or not to terminate the training of an agent based on the score.
  
    - Case 1: when the score reaches the target score, the evaluator would terminate the training process since it means that the goal of the task is reached. 
    
    - Case 2: when the score continuously drops, the evaluator would terminate the training process early to mitigate the performance collapse and the waste of computing power brought by overfitting. In this case, the score serves as a criterion to determine overfitting of models. 
  
We implement the ``evaluator`` as a microservice, which can be ran as an independent process. When an evaluator is running, it can automatically monitors parallel agents, and provide evaluation when any agent needs, and communicate agent information with the leaderboard.

.. autoclass:: elegantrl.train.evaluator.Evaluator
   :members:
   
