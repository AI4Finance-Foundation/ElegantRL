Quickstart
=============

As a quickstart, we select the Pendulum task from the demo.py to show how to train a DRL agent in ElegantRL. 

Step 1: Import packages
-------------------------------

.. code-block:: python
   
   from elegantrl_helloworld.demo import *

   gym.logger.set_level(40) # Block warning
   
Step 2: Specify Agent and Environment
--------------------------------------

.. code-block:: python

   env = PendulumEnv('Pendulum-v0', target_return=-500)
   args = Arguments(AgentSAC, env)
   
Part 3: Specify hyper-parameters
--------------------------------------

.. code-block:: python

   args.reward_scale = 2 ** -1  # RewardRange: -1800 < -200 < -50 < 0
   args.gamma = 0.97
   args.target_step = args.max_step * 2
   args.eval_times = 2 ** 3
   
Step 4: Train and Evaluate the Agent
--------------------------------------

.. code-block:: python

   train_and_evaluate(args)
   
Try by yourself through this `Colab <https://github.com/AI4Finance-Foundation/ElegantRL/blob/master/quickstart_Pendulum_v1.ipynb>`_!

.. tip::
    - By default, it will train a stable-SAC agent in the Pendulum-v0 environment for 400 seconds.

    - It will choose to utilize the CPU or GPU automatically. Don't worry, we never use ``.cuda()``.

    - It will save the log and model parameters file in ``'./{Environment}_{Agent}_{GPU_ID}'``.

    - It will print the total reward while training. (Maybe we should use TensorBoardX?)

    - The code is heavily commented. We believe these comments can answer some of your questions.
