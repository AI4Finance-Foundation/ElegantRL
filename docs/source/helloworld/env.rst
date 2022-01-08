Environment: *env.py*
=====================

The environment part is straightforward.  env.py is constructed by an environment wrapper class and a helper function that is used inside the wrapper class.

The environment wrapper class *PreprocessEnv* inherits *gym.wrapper*. It has an *init()* that takes an gym environment (OpenAI style) and creates things we need, a re-write of *reset()* and *step()* that make the return as we want.


Continuous action tasks
-----------------------

1. `Pendulum <https://gym.openai.com/envs/Pendulum-v0/>`_

2. `Lunar Lander Continuous <https://gym.openai.com/envs/LunarLanderContinuous-v2/>`_

3. `Bipedal Walker <https://gym.openai.com/envs/BipedalWalker-v2/>`_

Discrete action tasks
---------------------

1. `Cart Pole <https://gym.openai.com/envs/CartPole-v0/>`_

2. `Lunar Lander <https://gym.openai.com/envs/LunarLander-v2/>`_
