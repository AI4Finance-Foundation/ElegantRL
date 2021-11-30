====================
ElegantRL HelloWorld
====================

Welcome to ElegantRL Helloworld!   In this page, we will help you understand and use ElegantRL by introducing the main structure, code functionalities, and how to run.

.. contents:: Table of Contents
    :depth: 3

Structure
=========

.. figure:: ../images/File_structure.png
    :align: center

An agent (*agent.py*) with Actor-Critic networks (*net.py*) is trained (*run.py*) by interacting with an environment (*env.py*).

net.py
------

Our `net.py <https://github.com/AI4Finance-Foundation/ElegantRL/blob/master/elegantrl_helloworld/net.py>`_ contains classes of Q-Net, Actor network, Critic network, and their variations of according to different DRL algorithms.

Networks are the core of DRL, which will be updated in each step (might not be the case for some specific algs) during training time.

For detail explanation, please refer to the page of `Networks <https://elegantrl.readthedocs.io/en/latest/tutorial/net.html>`_.

agent.py
--------

`agent.py <https://github.com/AI4Finance-Foundation/ElegantRL/blob/master/elegantrl_helloworld/agent.py>`_ contains classes of different DRL agents which implement different DRL algorithms and their variations.

In this helloWorld, we will focus on DQN, PPO, SAC, and a discrete version of PPO, which are the most popular and commonly used DRL algorithms.

For detail explanation, please refer to the page of `Networks <https://elegantrl.readthedocs.io/en/latest/tutorial/agent.html>`_.

env.py
------

`env.py <https://github.com/AI4Finance-Foundation/ElegantRL/blob/master/elegantrl_helloworld/env.py>`_ contains a class that preprocess env from OpenAI Gym, and another helper function to get the objects we need. We still have the reset() and step() functions, which are just calling the reset() and step() functions in OpenAI Gym envs, in the class PreprocessEnv. Refer to `OpenAI's explanation <https://github.com/openai/gym/blob/master/gym/core.py>`_ to better understand what role does envs play in DRL.

run.py
------

`run.py <https://github.com/AI4Finance-Foundation/ElegantRL/blob/master/elegantrl_helloworld/run.py>`_ contains classes and functions for training and evaluating, and four functions available to run. Those four functions are the four big categories of DRL: 1. continuous action using off polocy algorithm 2. continuous action using off polocy algorithm 3. discrete action using off polocy algorithm 4. discrete action using on polocy algorithm.

Tasks
=====

As explained in environment section, our env class is a wrapper of `OpenAI Gym <https://gym.openai.com/>`_ env. So in this tutorial, we are using few classic tasks in OpenAI Gym:

Continuous action tasks
-----------------------

1. `Pendulum <https://gym.openai.com/envs/Pendulum-v0/>`_

2. `Lunar Lander Continuous <https://gym.openai.com/envs/LunarLanderContinuous-v2/>`_

3. `Bipedal Walker <https://gym.openai.com/envs/BipedalWalker-v2/>`_

Discrete action tasks
---------------------

1. `Cart Pole <https://gym.openai.com/envs/CartPole-v0/>`_

2. `Lunar Lander <https://gym.openai.com/envs/LunarLander-v2/>`_

Run The Code
============

In `run.py <https://github.com/AI4Finance-Foundation/ElegantRL/blob/master/elegantrl_helloworld/run.py>`_, there are four functions that are available to run in the main function.

- demo_continuous_action_off_policy()
- demo_continuous_action_on_policy()
- demo_discrete_action_off_policy()
- demo_discrete_action_on_policy()

Chose the task you want by setting the boolean to 1 (others to 0) in the function, then uncomment one of the four functions and run it. 

If everything works fine, then congratulation! You have successfully run a DRL trial using ElegantRL!
