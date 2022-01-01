Evaluator: *evaluator.py*
===============================

In the course of training, ElegantRL provide an ``evaluator`` to periodically evaluate agent's performance and save models.

For agent evaluation, the evaluator runs agent's actor (policy) network on the testing environment and outputs corresponding scores. Commonly used performance metrics are mean and variance of episodic rewards. The score is useful in following two cases:
  - Case 1: the score serves as a goal signal. When the score reaches the target score, it means that the goal of the task is achieved. 
  - Case 2: the score serves as a criterion to determine overfitting of models. When the score continuously drops, we can terminate the training process early to mitigate the performance collapse and the waste of computing power brought by overfitting.

.. note::
  ElegantRL supports a tournament-based ensemble training scheme to empower the population-based training (PBT). We maintain a leaderboard to keep track of agents with high scores and then perform a tournament-based evolution among these agents. In this case, the score from the evaluator serves as a metric for leaderboard.

For model saving, the evaluator saves following three types of files:
  - actor.pth: actor (policy) network of the agent.
  - plot_learning_curve.jpg: learning curve of the agent. 
  - recorder.npy: log file, including total training steps, reward average, reward standard deviation, reward exp, actor loss, and critic loss.  
  
We implement the ``evaluator`` as a microservice, which can be ran as an independent process. When an evaluator is running, it can automatically monitors parallel agents, and provide evaluation when any agent needs, and communicate agent information with the leaderboard.

Implementations
---------------------

.. autoclass:: elegantrl.train.evaluator.Evaluator
   :members:
   

Utils
---------------------

.. autofunction:: elegantrl.train.evaluator.get_episode_return_and_step

.. autofunction:: elegantrl.train.evaluator.save_learning_curve
   
