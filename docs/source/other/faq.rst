:github_url: https://github.com/AI4Finance-LLC/FinRL-Library


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

	*ElegantRL supports any gym-style environment and provides wrappers for MuJoCo and Isaac Gym. *

    - :raw-html:`<font color="#A52A2A">How can I use a VecEnv? </font>`

	*You can use `VecEnv <https://elegantrl.readthedocs.io/en/latest/examples/Creating_VecEnv.html>`_ imported from Isaac Gym or write your own VecEnv by yourself. There is no VecEnv wrapper to process a non-VecEnv to VecEnv.*   

    - :raw-html:`<font color="#A52A2A">What is ElegantRL-helloworld? </font>`

	*It is a tutorial-level implementation for users (e.g., beginners) who do not have a demand for parallel computing. *
	
    - :raw-html:`<font color="#A52A2A">What DRL algorithms can I use with ElegantRL? </font>`

	*In the `folder <https://github.com/AI4Finance-Foundation/ElegantRL/tree/master/elegantrl/agents>`_, we currently have DQN, DDQN, DDPG, TD3, SAC, A2C, REDQ, and PPO. *
	
    - :raw-html:`<font color="#A52A2A">What kinds of parallelism does ElegantRL support? </font>`

	*ElegantRL support parallelism of DRL algorithms at multiple levels, including agent parallelism of population-based training and worker-learner parallelism of a single agent.*
	
    - :raw-html:`<font color="#A52A2A">What is agent parallelism?  </font>`

	*Agent parallelism is to train hundreds of agents in parallel through population-based training (PBT), which offers a flexibility for ensemble methods.*
	
    - :raw-html:`<font color="#A52A2A">What is worker parallelism? </font>`

	*Worker parallelism is to generate transitions in parallel, thus accelerating the data collection. We currently support two different parallelism to adapt different types of environments.*
	
		- *use a `VecEnv <https://elegantrl.readthedocs.io/en/latest/examples/Creating_VecEnv.html>`_ to generate transitions in batch.*
		
		- *if the environment is not a VecEnv, use multiple workers to generate transitions in parallel.*

    - :raw-html:`<font color="#A52A2A">What is learner parallelism? </font>`

	*Learner parallelism is to train multiple-critics and multiple actors running in parallel for ensemble DRL methods. Due to the stochastic nature of the training process (e.g., random seeds), an ensemble DRL algorithm increases the diversity of the data collection, improves the stability of the learning process, and reduces the overestimation bias.*

    - :raw-html:`<font color="#A52A2A">What kinds of ensemble methods can I use?  </font>`

	*We currently support three ensemble methods, which are weighted average, model fusion, and tournament-based ensemble training scheme.*

    - :raw-html:`<font color="#A52A2A">What is tournament-based ensemble training scheme?  </font>`

	*Tournament-based ensemble training scheme is our cloud orchestration mechanism, scheduling the interactions between a leaderboard and a training pool with hundreds of agents (pods). More details are available in the `post <https://towardsdatascience.com/elegantrl-podracer-scalable-and-elastic-library-for-cloud-native-deep-reinforcement-learning-bafda6f7fbe0>`_ and the `paper <https://arxiv.org/abs/2112.05923>`_.*

    - :raw-html:`<font color="#A52A2A">Can I use a pre-trained model? </font>`

	*Yes, you can load a model to continue the training. A tutorial is coming soon.*

    - :raw-html:`<font color="#A52A2A">Can I use Tensorboard for logging?  </font>`

	*No, we cannot support Tensorboard.*

    - :raw-html:`<font color="#A52A2A">Does ElegantRL supports multi-agent reinforcement learning (MARL)? </font>`

	*Yes, we are implementing MARL algorithms and adapting them to the massively parallel framework. Currently, we provide several MARL algorithms, such as QMix, MADDPG, MAPPO, and VDN. The tutorials are coming soon.*

    - :raw-html:`<font color="#A52A2A">Does ElegantRL supports GPU training?   </font>`

	*ElegantRL supports flexible resource allocation from zero to hundreds of GPUs.*

    - :raw-html:`<font color="#A52A2A">Can I use ElegantRL without GPUs?  </font>`

	*Of course! You can use ElegantRL-helloworld for non-GPU training or use ElegantRL by setting GPU_ids to None (you cannot use GPU-accelerated VecEnv in this case).*

    - :raw-html:`<font color="#A52A2A">How can I contribute to the development?  </font>`

	*participate on the slack channels, check the current issues and the roadmap, and help any way you can (sharing the library with others, testing the library of different applications, contributing with code development, etc)*


.. _Section-4:

Section 4 References for diving deep into Deep Reinforcement Learning (DRL)
------------------------------------------------------------------------------

.. _Section-4-1:

Subsection 4.1 General resources
-----------------------------------------------------------------

	.. role:: raw-html(raw)
	   :format: html

    - OpenAI Spinning UP DRL, educational resource
        https://spinningup.openai.com/en/latest/

    - Awesome-ai-in-finance
        https://github.com/georgezouq/awesome-ai-in-finance

    - penAI Gym
        https://github.com/openai/gym

    - Stable Baselines 3
        contains the implementations of all models used by FinRL
        https://github.com/DLR-RM/stable-baselines3

    - Ray RLlib
        https://docs.ray.io/en/master/rllib.html

    - Policy gradient algorithms
        https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html

    - Fischer, T.G., 2018. Reinforcement learning in financial markets-a survey (No. 12/2018). FAU Discussion Papers in Economics. (:raw-html:`<font color="#A52A2A">a survey on the use of RL for finance </font>`)

    - Li, Y., 2018. Deep reinforcement learning. arXiv preprint arXiv:1810.06339. (:raw-html:`<font color="#A52A2A">an in-depth review of DRL and its main models and components</font>`)

    - Charpentier, A., Elie, R. and Remlinger, C., 2020. Reinforcement learning in economics and finance. arXiv preprint arXiv:2003.10014. (:raw-html:`<font color="#A52A2A">an in-depth review of uses of RL and DRL in finance</font>`)

    - Kolm, P.N. and Ritter, G., 2020. Modern perspectives on reinforcement learning in finance. Modern Perspectives on Reinforcement Learning in Finance (September 6, 2019). The Journal of Machine Learning in Finance, 1(1) (:raw-html:`<font color="#A52A2A">an in-depth review of uses of RL and DRL in finance</font>`)

    - Practical Deep Reinforcement Learning Approach for Stock Trading, paper and codes, Workshop on Challenges and Opportunities for AI in Financial Services, NeurIPS 2018.


.. _Section-4-2:

Subsection 4.2 Papers related to the implemented DRL models
-----------------------------------------------------------------

	.. role:: raw-html(raw)
	   :format: html

    - Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D. and Riedmiller, M., 2013. Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602 	(:raw-html:`<font color="#A52A2A">the first paper that proposed (with success) the use of DL in RL</font>`)

    - Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A.A., Veness, J., Bellemare, M.G., Graves, A., Riedmiller, M., Fidjeland, A.K., Ostrovski, G. and Petersen, S., 2015. Human-level control through deep reinforcement learning. Nature, 518(7540), pp.529-533 (:raw-html:`<font color="#A52A2A">an excellent review paper of important concepts on DRL</font>`)

    - Lillicrap, T.P., Hunt, J.J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., Silver, D. and Wierstra, D., 2015. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971 (:raw-html:`<font color="#A52A2A">paper that proposed the DDPG model</font>`)

    - Fujimoto, S., Hoof, H. and Meger, D., 2018, July. Addressing function approximation error in actor-critic methods. In International Conference on Machine Learning (pp. 1587-1596). PMLR (:raw-html:`<font color="#A52A2A">paper that proposed the TD3 model</font>`)

    - Schulman, J., Wolski, F., Dhariwal, P., Radford, A. and Klimov, O., 2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347 (:raw-html:`<font color="#A52A2A">paper that proposed the PPO model</font>`)

    - Mnih, V., Badia, A.P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., Silver, D. and Kavukcuoglu, K., 2016, June. Asynchronous methods for deep reinforcement learning. In International conference on machine learning (pp. 1928-1937). PMLR (:raw-html:`<font color="#A52A2A">paper that proposed the A3C model</font>`)

    - https://openai.com/blog/baselines-acktr-a2c/ (:raw-html:`<font color="#A52A2A">description of the implementation of the A2C model</font>`)

    - Schulman, J., Levine, S., Abbeel, P., Jordan, M. and Moritz, P., 2015, June. Trust region policy optimization. In International conference on machine learning (pp. 1889-1897). PMLR (:raw-html:`<font color="#A52A2A">description of the implementation of the TRPO model</font>`)

.. _Section-5:
    
Section 5  Common issues/bugs
--------------------------------
- Package trading_calendars reports errors in Windows system:\
    Trading_calendars is not maintained now. It may report erros in Windows system (python>=3.7). These are two possible solutions: 1.Use python=3.6 environment 2.Replace trading_calendars with exchange_caldenars.
