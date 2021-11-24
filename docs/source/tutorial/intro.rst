====================
ElegantRL HelloWorld
====================

Welcome to ElegantRL Helloworld! In this section, we will help you understand and use ElegantRL by introducing the main structure, functionality of codes, and how to run.

.. contents:: Table of Contents
    :depth: 2

Structure
=========

.. figure:: ../images/File_structure.png
    :align: center

An agent (*agent.py*) with Actor-Critic networks (*net.py*) is trained (*run.py*) by interacting with an environment (*env.py*).

net.py
------
Our `net.py <https://github.com/AI4Finance-Foundation/ElegantRL/blob/master/elegantrl_helloworld/net.py>`_ contains classes of Q-Net, Actor network, Critic network, and their variations according to different DRL algs.

Network are the core of DRL, which will be updated in each step (might not be the case for some specific algs) during training time.

For detail explanation, please refer to the page of `Networks <https://elegantrl.readthedocs.io/en/latest/tutorial/net.html>`_.

agent.py
--------

env.py
------

run.py
------
