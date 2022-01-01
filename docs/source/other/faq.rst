FAQ
=============================

:Version: 1.0
:Date: 12-31-2021
:Contributors: Steven Li, Xiao-Yang Liu



Description
----------------

This document contains the most frequently asked questions related to the ElegantRL Library, based on questions posted on the slack channels and Github_ issues.

.. _Github: https://github.com/AI4Finance-Foundation/ElegantRL


Outline
----------------

    - :ref:`Section-1`

    - :ref:`Section-2`

    - :ref:`Section-3`

    - :ref:`Section-4`

		- :ref:`Section-4-1`

		- :ref:`Section-4-2`
		
		- :ref:`Section-4-3`

    - :ref:`Section-5`


.. _Section-1:

Section 1  Where to start?
--------------------------------

    - Get started with ElegantRL-helloworld, a lightweight and stable subset of ElegantRL. 
    
        - Read the introductary `post <https://towardsdatascience.com/elegantrl-a-lightweight-and-stable-deep-reinforcement-learning-library-95cef5f3460b>`_ of ElegantRL-helloworld.

        - Read the `post <https://towardsdatascience.com/elegantrl-mastering-the-ppo-algorithm-part-i-9f36bc47b791>`_ to learn how an algorithm is implemented.
        
        - Read the posts (`Part I <https://medium.com/mlearning-ai/elegantrl-demo-stock-trading-using-ddpg-part-i-e77d7dc9d208>`_, `Part II <https://medium.com/mlearning-ai/elegantrl-demo-stock-trading-using-ddpg-part-ii-d3d97e01999f>`_) to learn a demo of ElegantRL-helloworld on a stock trading task.
    
    - Read the `post <https://towardsdatascience.com/elegantrl-podracer-scalable-and-elastic-library-for-cloud-native-deep-reinforcement-learning-bafda6f7fbe0>`_ and the `paper <https://arxiv.org/abs/2112.05923>`_ that describe our cloud solution, ElegantRL-Podracer.

    - Run the Colab-based notebooks on simple Gym environments.
    
    - Install the library following the instructions at the official Github `repo <https://github.com/AI4Finance-Foundation/ElegantRL>`_.
    
    - Run the demos from MuJoCo to Isaac Gym provided in the library `folder <https://github.com/AI4Finance-Foundation/ElegantRL/tree/master/elegantrl>`_.

    - Enter on the AI4Finance `slack <https://join.slack.com/t/ai4financeworkspace/shared_invite/zt-kq0c9het-FCSU6Y986OnSw6Wb5EkEYw>`_.


.. _Section-2:

Section 2 What to do when you experience problems?
----------------------------------------------------------------

    - If any questions arise, please follow this sequence of activities:

        - Check if it is not already answered on this FAQ

        - Check if it is not posted on the Github repo `issues <https://github.com/AI4Finance-Foundation/ElegantRL/issues>`_.  

        - If you cannot find your question, please report it as a new issue or ask it on the AI4Finance slack (Our members will get to you ASAP).


.. _Section-3:

Section 3 Most frequently asked questions related to the ElegantRL Library
---------------------------------------------------------------------------

	.. role:: raw-html(raw)
	   :format: html

    - :raw-html:`<font color="#A52A2A">What kinds of environment can I use? </font>`

	ElegantRL supports any gym-style environment and provides wrappers for MuJoCo and Isaac Gym.

    - :raw-html:`<font color="#A52A2A">How can I use a VecEnv? </font>`

	You can use `VecEnv <https://elegantrl.readthedocs.io/en/latest/examples/Creating_VecEnv.html>`_ imported from Isaac Gym or write your own VecEnv by yourself. There is no VecEnv wrapper to process a non-VecEnv to VecEnv.   

    - :raw-html:`<font color="#A52A2A">What is ElegantRL-helloworld? </font>`

	It is a tutorial-level implementation for users (e.g., beginners) who do not have a demand for parallel computing. 
	
    - :raw-html:`<font color="#A52A2A">What DRL algorithms can I use with ElegantRL? </font>`

	In the `folder <https://github.com/AI4Finance-Foundation/ElegantRL/tree/master/elegantrl/agents>`_, we currently have DQN, DDQN, DDPG, TD3, SAC, A2C, REDQ, and PPO. 
	
    - :raw-html:`<font color="#A52A2A">What kinds of parallelism does ElegantRL support? </font>`

	ElegantRL support parallelism of DRL algorithms at multiple levels, including agent parallelism of population-based training and worker-learner parallelism of a single agent.
	
    - :raw-html:`<font color="#A52A2A">What is agent parallelism?  </font>`

	Agent parallelism is to train hundreds of agents in parallel through population-based training (PBT), which offers a flexibility for ensemble methods.
	
    - :raw-html:`<font color="#A52A2A">What is worker parallelism? </font>`

	Worker parallelism is to generate transitions in parallel, thus accelerating the data collection. We currently support two different parallelism to adapt different types of environments.
	
		- use a `VecEnv <https://elegantrl.readthedocs.io/en/latest/examples/Creating_VecEnv.html>`_ to generate transitions in batch.
		
		- if the environment is not a VecEnv, use multiple workers to generate transitions in parallel.

    - :raw-html:`<font color="#A52A2A">What is learner parallelism? </font>`

	Learner parallelism is to train multiple-critics and multiple actors running in parallel for ensemble DRL methods. Due to the stochastic nature of the training process (e.g., random seeds), an ensemble DRL algorithm increases the diversity of the data collection, improves the stability of the learning process, and reduces the overestimation bias.

    - :raw-html:`<font color="#A52A2A">What kinds of ensemble methods can I use?  </font>`

	We currently support three ensemble methods, which are weighted average, model fusion, and tournament-based ensemble training scheme.

    - :raw-html:`<font color="#A52A2A">What is tournament-based ensemble training scheme?  </font>`

	Tournament-based ensemble training scheme is our cloud orchestration mechanism, scheduling the interactions between a leaderboard and a training pool with hundreds of agents (pods). More details are available in the `post <https://towardsdatascience.com/elegantrl-podracer-scalable-and-elastic-library-for-cloud-native-deep-reinforcement-learning-bafda6f7fbe0>`_ and the `paper <https://arxiv.org/abs/2112.05923>`_.

    - :raw-html:`<font color="#A52A2A">Can I use a pre-trained model? </font>`

	Yes, you can load a model to continue the training. A tutorial is coming soon.

    - :raw-html:`<font color="#A52A2A">Can I use Tensorboard for logging?  </font>`

	No, we cannot support Tensorboard.

    - :raw-html:`<font color="#A52A2A">Does ElegantRL supports multi-agent reinforcement learning (MARL)? </font>`

	Yes, we are implementing MARL algorithms and adapting them to the massively parallel framework. Currently, we provide several MARL algorithms, such as QMix, MADDPG, MAPPO, and VDN. The tutorials are coming soon.

    - :raw-html:`<font color="#A52A2A">Does ElegantRL supports GPU training?   </font>`

	ElegantRL supports flexible resource allocation from zero to hundreds of GPUs.

    - :raw-html:`<font color="#A52A2A">Can I use ElegantRL without GPUs?  </font>`

	Of course! You can use ElegantRL-helloworld for non-GPU training or use ElegantRL by setting GPU_ids to None (you cannot use GPU-accelerated VecEnv in this case).

    - :raw-html:`<font color="#A52A2A">How can I contribute to the development?  </font>`

	You can participate on the slack channels, check the current issues and the roadmap, and help any way you can (sharing the library with others, testing the library of different applications, contributing with code development, etc).


.. _Section-4:

Section 4 References for diving deep into Deep Reinforcement Learning (DRL)
------------------------------------------------------------------------------

.. _Section-4-1:

Subsection 4.1 Open-source softwares and materials
-----------------------------------------------------------------

	.. role:: raw-html(raw)
	   :format: html

    - OpenAI Gym
        https://gym.openai.com/	
    
    - MuJoCo
        https://mujoco.org/
      
    - Isaac Gym
        https://developer.nvidia.com/isaac-gym	
	
    - OpenAI Spinning Up
        https://spinningup.openai.com/en/latest/

    - Stable Baselines3
        https://github.com/DLR-RM/stable-baselines3

    - Ray RLlib
        https://docs.ray.io/en/master/rllib.html
	
    - Tianshou
        https://github.com/thu-ml/tianshou
	
    - ChainerRL
        https://github.com/chainer/chainerrl
	
    - MushroomRL
        https://github.com/MushroomRL/mushroom-rl/tree/master
	
    - ACME
        https://github.com/deepmind/acme
	
    - PFRL
        https://github.com/pfnet/pfrl
	
    - SURREAL
        https://github.com/SurrealAI/surreal
	
    - rlpyt
        https://github.com/astooke/rlpyt
	
    - MAlib
        https://github.com/sjtu-marl/malib	

    - Policy gradient algorithms
        https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html

    

.. _Section-4-2:

Subsection 4.2 DRL algorithms
-----------------------------------------------------------------

	.. role:: raw-html(raw)
	   :format: html

    - David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, et al. *Mastering the game of Go without human knowledge*. Nature, 550(7676):354–359, 2017.
    
    - V. Mnih, K. Kavukcuoglu, D. Silver, A. Graves, Ioannis Antonoglou, Daan Wierstra, and Martin A. Riedmiller. *Playing atari with deep reinforcement learning*. ArXiv, abs/1312.5602, 2013.
    
    - H. V. Hasselt, Arthur Guez, and David Silver. *Deep reinforcement learning with double q-learning*. ArXiv, abs/1509.06461, 2016.
    
    - Timothy P Lillicrap, Jonathan J Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. *Continuous control with deep reinforcement learning*. In ICLR, 2016.
    
    - J. Schulman, F. Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. *Proximal policy optimization algorithms*. ArXiv, abs/1707.06347, 2017.
    
    - Matteo Hessel, Joseph Modayil, H. V. Hasselt, Tom Schaul, Georg Ostrovski, Will Dabney, Dan Horgan,Bilal Piot, Mohammad Gheshlaghi Azar, and David Silver. *Rainbow:  Combining improvements in deepreinforcement learning*. In AAAI, 2018.
    
    - Scott Fujimoto, Herke Hoof, and David Meger. *Addressing function approximation error in actor-critic methods*. In International Conference on Machine Learning, pages 1587–1596. PMLR, 2018.
    
    - Tuomas Haarnoja, Aurick Zhou, P. Abbeel, and Sergey Levine. *Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor*. In ICML, 2018.
    
    - Xinyue Chen, Che Wang, Zijian Zhou, and Keith W. Ross. *Randomized ensembled double q-learning: Learning fast without a model*. In International Conference on Learning Representations, 2021.
  
    
    
.. _Section-4-3:

Subsection 4.2 Other resources
-----------------------------------------------------------------

	.. role:: raw-html(raw)
	   :format: html

    - Richard S. Sutton and Andrew G. Barto. *Reinforcement learning: An introduction*. IEEE Transactions on Neural Networks, 16:285–286, 2005.
    
    - Arun Nair, Praveen Srinivasan, Sam Blackwell, Cagdas Alcicek, Rory Fearon, Alessandro De Maria, Vedavyas Panneershelvam, Mustafa Suleyman, Charlie Beattie, Stig Petersen, Shane Legg, Volodymyr Mnih, Koray Kavukcuoglu, and David Silver. *Massively parallel methods for deep reinforcement learning*. ArXiv, abs/1507.04296, 2015.
    
    - Philipp Moritz, Robert Nishihara, Stephanie Wang, Alexey Tumanov, Richard Liaw, Eric Liang, Melih Elibol, Zongheng Yang, William Paul, Michael I Jordan, et al. *Ray: A distributed framework for emerging ai applications*. In 13th USENIX Symposium on Operating Systems Design and Implementation (OSDI), pages 561–577, 2018.
    
    - Lasse Espeholt, Rapha¨el Marinier, Piotr Stanczyk, Ke Wang, and Marcin Michalski. *Seed rl: Scalable and efficient deep-rl with accelerated central inference*. In International Conference on Machine Learning. PMLR, 2019.
    
    - Agrim Gupta, Silvio Savarese, Surya Ganguli, and Fei-Fei Li. *Embodied intelligence via learning and evolution*. Nature Communications, 2021.
    
    - Matteo Hessel, Manuel Kroiss, Aidan Clark, Iurii Kemaev, John Quan, Thomas Keck, Fabio Viola, and Hado van Hasselt. *Podracer architectures for scalable reinforcement learning*. arXiv preprint arXiv:2104.06272, 2021.
    
    - Zechu Li, Xiao-Yang Liu, Jiahao Zheng, Zhaoran Wang, Anwar Walid, and Jian Guo. *FinRL-podracer: High performance and scalable deep reinforcement learning for quantitative finance*. ACM International Conference on AI in Finance (ICAIF), 2021.
    
    - Nikita Rudin, David Hoeller, Philipp Reist, and Marco Hutter. *Learning to walk in minutes using massively parallel deep reinforcement learning*. In Conference on Robot Learning, 2021.
    
    - Brijen Thananjeyan, Kirthevasan Kandasamy, Ion Stoica, Michael I. Jordan, Ken Goldberg, and Joseph Gonzalez. *Resource allocation in multi-armed bandit exploration: Overcoming nonlinear scaling with adaptive parallelism*. In ICML, 2021.


.. _Section-5:
    
Section 5  Common issues/bugs
--------------------------------
- When running Isaac Gym, found error *ImportError: libpython3.7m.so.1.0: cannot open shared object file: No such file or directory*:\
    Run the following code in bash to add the path of Isaac Gym conda environment.
    
    	**export LD_LIBRARY_PATH=$PATH$**
    
    For example, the name of Isaac Gym conda environment is rlgpu:
    
    	**export LD_LIBRARY_PATH=/xfs/home/podracer_steven/anaconda3/envs/rlgpu/lib**
