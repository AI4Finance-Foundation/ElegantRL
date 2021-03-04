Quickstart
=============

Open  ``./elegantrl/run.py``:

.. code-block:: python
   :linenos:
   
    import os
    import time
    from copy import deepcopy

    import torch
    import numpy as np
    import numpy.random as rd

    from elegantrl.agent import ReplayBuffer, ReplayBufferMP
    from elegantrl.env import PreprocessEnv
    import gym

    gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'

    '''DEMO'''


Run the example:

.. code-block:: python

  python run.py
  
You can see ``run__demo(gpu_id=0, cwd='AC_BasicAC')`` in Main.py.

.. tip::
    - In default, it will train a stable-DDPG in LunarLanderContinuous-v2 for 2000 second.
    
    - It would choose CPU or GPU automatically. Don't worry, We never use .cuda().
    
    - It would save the log and model parameters file in Current Working Directory cwd='AC_BasicAC'.
    
    - It would print the total reward while training. Maybe We should use TensorBoardX?
    
    - There are many comment in the code. We believe these comments can answer some of your questions.
