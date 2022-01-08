Example 2: BipedalWalker-v3
===============================

BipedalWalker-v3 is a classic task in robotics that performs a fundamental skill: moving forward as fast as possible. The goal is to get a 2D biped walker to walk through rough terrain. BipedalWalker is considered to be a difficult task in the continuous action space, and there are only a few RL implementations that can reach the target reward.

When a biped walker takes random action:

.. image:: ../images/BipedalWalker-v3_1.gif
   :width: 80%
   :align: center


Step 1: Install ElegantRL
------------------------------

.. code-block:: python
   
     pip install git+https://github.com/AI4Finance-LLC/ElegantRL.git
  
Step 2: Import packages
-------------------------------

   - ElegantRL
   
   - OpenAI Gym: a toolkit for developing and comparing reinforcement learning algorithms (collections of environments).
   
.. code-block:: python
   
   import gym

   from elegantrl.agents.AgentPPO import AgentPPO
   from elegantrl.envs.Gym import get_gym_env_args
   from elegantrl.train.config import Arguments
   from elegantrl.train.run import train_and_evaluate, train_and_evaluate_mp

Step 3: Get environment information
--------------------------------------------------

.. code-block:: python
   
   get_gym_env_args(gym.make('BipedalWalker-v3'), if_print=True)
   

Output: 

.. code-block:: python

   env_args = {
       'env_num': 1,
       'env_name': 'BipedalWalker-v3',
       'max_step': 1600,
       'state_dim': 24,
       'action_dim': 4,
       'if_discrete': False,
       'target_return': 300,
   }


Step 4: Initialize agent and environment
---------------------------------------------

   - agent: chooses a agent (DRL algorithm) from a set of agents in the `directory <https://github.com/AI4Finance-Foundation/ElegantRL/tree/master/elegantrl/agents>`_.
   
   - env_func: the function to create an environment, in this case, we use ``gym.make`` to create BipedalWalker-v3.
   
   - env_args: the environment information.

.. code-block:: python
   
   env_func = gym.make
   env_args = {
       'env_num': 1,
       'env_name': 'BipedalWalker-v3',
       'max_step': 1600,
       'state_dim': 24,
       'action_dim': 4,
       'if_discrete': False,
       'target_return': 300,
       'id': 'BipedalWalker-v3',
   }

   args = Arguments(agent=AgentPPO, env_func=env_func, env_args=env_args)

Step 5: Specify hyper-parameters
----------------------------------------

A list of hyper-parameters is available `here <https://elegantrl.readthedocs.io/en/latest/api/config.html>`_.

.. code-block:: python

   args.net_dim = 2 ** 8
   args.batch_size = args.net_dim * 2
   args.target_step = args.max_step * 2
   args.worker_num = 4

   args.save_gap = 2 ** 9
   args.eval_gap = 2 ** 8
   args.eval_times1 = 2 ** 4
   args.eval_times2 = 2 ** 5
   

Step 6: Train your agent
----------------------------------------

In this tutorial, we provide four different modes to train an agent:

   - **Single-process**: utilize one GPU for a single-process training. No parallelism.
   
   - **Multi-process**: utilize one GPU for a multi-process training. Support worker and learner parallelism.

   - **Multi-GPU**: utilize multi-GPUs to train an agent through model fusion. Specify the GPU ids you want to use. 
   
   - **Tournament-based ensemble training**: utilize multi-GPUs to run tournament-based ensemble training.
   
   
.. code-block:: python

   flag = 'SingleProcess'

   if flag == 'SingleProcess':
       args.learner_gpus = 0
       train_and_evaluate(args)
       
   elif flag == 'MultiProcess':
       args.learner_gpus = 0
       train_and_evaluate_mp(args)
       
   elif flag == 'MultiGPU':
       args.learner_gpus = [0, 1, 2, 3]
       train_and_evaluate_mp(args)
       
   elif flag == 'Tournament-based':
       args.learner_gpus = [[i, ] for i in range(4)]  # [[0,], [1, ], [2, ]] or [[0, 1], [2, 3]]
       python_path = '.../bin/python3'
       train_and_evaluate_mp(args, python_path)
       
   else:
       raise ValueError(f"Unknown flag: {flag}")
   
   
Step 7: Testing Results
----------------------------------------

After reaching the target reward, we generate the frame for each state and compose frames as a video result. From the video, the walker is able to move forward constantly.

.. code-block:: python

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


Performance of a trained agent:

.. image:: ../images/BipedalWalker-v3_2.gif
   :width: 80%
   :align: center
   
   
Check out our **video** on bilibili: `Crack the BipedalWalkerHardcore-v2 with total reward 310 using IntelAC <https://www.bilibili.com/video/BV1wi4y187tC>`_.
