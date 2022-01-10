Quickstart
=============

Inside ``./elegantrl_helloworld/run.py``, you will find some demo code that looks like this:

.. code-block:: python
   :linenos:

    import time
    from agent import *
    from env import *
    from typing import Tuple

    gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'

    '''MORE DEMO CODE FOLLOWS'''


If you run the file in your terminal, you'll see an agent training live:

.. code-block:: python

  python run.py

You can see ``demo_continuous_action_on_policy()`` called at the bottom of the file.

.. code-block:: python
   :linenos:

    if __name__ == '__main__':
      # demo_continuous_action_off_policy()
      demo_continuous_action_on_policy()
      # demo_discrete_action_off_policy()
      # demo_discrete_action_on_policy()

.. tip::
    - By default, it will train a stable-PPO agent in the Pendulum-v1 environment for 400 seconds.

    - It will choose to utilize the CPU or GPU automatically. Don't worry, we never use ``.cuda()``.

    - It will save the log and model parameters file in ``'./{Agent}_{Environment}_0'``.

    - It will print the total reward while training. (Maybe we should use TensorBoardX?)

    - The code is heavily commented. We believe these comments can answer some of your questions.
