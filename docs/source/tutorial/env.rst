Environment: *env.py*
=====================

Our environment part is quite straightforward. env.py is constructed by a environment wrapper class and a helper function that is used inside the wrapper class.

The environment wrapper class *PreprocessEnv* inherits *gym.wrapper*. It has an *init()* that takes an OpenAI gym environment and create things we need, a re-write of *reset()* and *step()* that make the return as we want.

Listed below, we include 5 different tasks (3 continuous, 2 discrete) in the helloworld setup. Feel free to modify the code and try more tasks using different environments.

Continuous action tasks
-----------------------

1. `Pendulum <https://gym.openai.com/envs/Pendulum-v0/>`_

2. `Lunar Lander Continuous <https://gym.openai.com/envs/LunarLanderContinuous-v2/>`_

3. `Bipedal Walker <https://gym.openai.com/envs/BipedalWalker-v2/>`_

Discrete action tasks
---------------------

1. `Cart Pole <https://gym.openai.com/envs/CartPole-v0/>`_

2. `Lunar Lander <https://gym.openai.com/envs/LunarLander-v2/>`_
