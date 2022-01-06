BipedalWalker-v3
===============================

BipedalWalker-v3 is a classic task in robotics that performs a fundamental skill: moving. The goal is to get a 2D biped walker to walk through rough terrain. BipedalWalker is considered to be a difficult task in the continuous action space, and there are only a few RL implementations that can reach the target reward.

Check out a **video** on bilibili: `Crack the BipedalWalkerHardcore-v2 with total reward 310 using IntelAC <https://www.bilibili.com/video/BV1wi4y187tC>`_.

Step 1: Install ElegantRL
------------------------------

.. code-block:: python
   :linenos:
   
     pip install git+https://github.com/AI4Finance-LLC/ElegantRL.git
  
Step 2: Import Packages
-------------------------------

   - ElegantRL
   
   - OpenAI Gym: a toolkit for developing and comparing reinforcement learning algorithms.
   
   - PyBullet Gym: an open-source implementation of the OpenAI Gym MuJoCo environments.

.. code-block:: python
   :linenos:
   
      from elegantrl.run import *
      from elegantrl.agent import AgentGaePPO
      from elegantrl.env import PreprocessEnv
      import gym
      gym.logger.set_level(40) # Block warning


Step 3: Specify Agent and Environment
---------------------------------------------

   - args.agent: firstly chooses a DRL algorithm, and the user is able to choose one from a set of agents in agent.py
   
   - args.env: creates and preprocesses an environment, and the user can either customize own environment or preprocess environments from OpenAI Gym and PyBullet Gym in env.py.

.. code-block:: python
   :linenos:
   
      args = Arguments(if_off_policy=True)
      args.agent = AgentGaePPO() # AgentSAC(), AgentTD3(), AgentDDPG()
      args.env = PreprocessEnv(env=gym.make(‘BipedalWalker-v3’))
      args.reward_scale = 2 ** -1 # RewardRange: -200 < -150 < 300 < 334
      args.gamma = 0.95
      args.rollout_num = 2 # the number of rollout workers (larger is not always faster)

Step 4: Train and Evaluate the Agent
----------------------------------------

The training and evaluating processes are inside function **train_and_evaluate__multiprocessing(args)**, and the parameter is args. It includes two fundamental objects in DRL:

   - agent
   
   - environment (env)

And the parameters for training:

   - batch_size
   
   - target_step
   
   - reward_scale
   
   - gamma, etc

Also the parameters for evaluation:

   - break_step
   
   - random_seed, etc

.. code-block:: python
   :linenos:
   
      train_and_evaluate__multiprocessing(args) # the training process will terminate once it reaches the target reward.

Step 5: Testing Results
----------------------------------------

After reaching the target reward, we generate the frame for each state and compose frames as a video result. From the video, the walker is able to move forward constantly.

.. code-block:: python
   :linenos:

      for i in range(1024):
          frame = gym_env.render('rgb_array')
          cv2.imwrite(f'{save_dir}/{i:06}.png', frame)

          states = torch.as_tensor((state,), dtype=torch.float32, device=device)
          actions = agent.act(states)
          action = actions.detach().cpu().numpy()[0]
          next_state, reward, done, _ = env.step(action)
          if done:
              state = env.reset()
          else:
              state = next_state

.. image:: ../images/BipedalWalker-v3_1.gif
.. image:: ../images/BipedalWalker-v3_2.gif
