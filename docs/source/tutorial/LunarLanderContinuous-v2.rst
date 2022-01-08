Example 1: LunarLanderContinuous-v2
========================================

LunarLanderContinuous-v2 is a robotic control task. The goal is to get a Lander to rest on the landing pad. If lander moves away from landing pad it loses reward back. Episode finishes if the lander crashes or comes to rest, receiving additional -100 or +100 points. Detailed description of the task can be found at `OpenAI Gym <https://gym.openai.com/envs/LunarLanderContinuous-v2/>`_.

When a Lander takes random actions:

.. image:: ../images/LunarLander.gif
   :width: 80%
   :align: center

Our Python code is available `here <https://github.com/AI4Finance-Foundation/ElegantRL/blob/master/examples/tutorial_LunarLanderContinous-v2.py>`_.

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

   from elegantrl.agents.AgentPPO import AgentSAC
   from elegantrl.envs.Gym import get_gym_env_args
   from elegantrl.train.config import Arguments
   from elegantrl.train.run import train_and_evaluate

Step 3: Get environment information
--------------------------------------------------

.. code-block:: python
   
   get_gym_env_args(gym.make('LunarLanderContinuous-v2'), if_print=True)
   

Output: 

.. code-block:: python

   env_args = {
       'env_num': 1,
       'env_name': 'LunarLanderContinuous-v2',
       'max_step': 1000,
       'state_dim': 8,
       'action_dim': 4,
       'if_discrete': True,
       'target_return': 200,
       'id': 'LunarLanderContinuous-v2'
   }


Step 4: Initialize agent and environment
---------------------------------------------

   - agent: chooses a agent (DRL algorithm) from a set of agents in the `directory <https://github.com/AI4Finance-Foundation/ElegantRL/tree/master/elegantrl/agents>`_.
   
   - env_func: the function to create an environment, in this case, we use ``gym.make`` to create LunarLanderContinuous-v2.
   
   - env_args: the environment information.

.. code-block:: python
   
   env_func = gym.make
   env_args = {
       'env_num': 1,
       'env_name': 'LunarLanderContinuous-v2',
       'max_step': 1000,
       'state_dim': 8,
       'action_dim': 4,
       'if_discrete': True,
       'target_return': 200,
       'id': 'LunarLanderContinuous-v2'
   }

   args = Arguments(agent=AgentSAC(), env_func=env_func, env_args=env_args)

Step 5: Specify hyper-parameters
----------------------------------------

A list of hyper-parameters is available `here <https://elegantrl.readthedocs.io/en/latest/api/config.html>`_.

.. code-block:: python

   args.net_dim = 2 ** 9
   args.max_memo = 2 ** 22
   args.repeat_times = 2 ** 1
   args.reward_scale = 2 ** -2
   args.batch_size = args.net_dim * 2
   args.target_step = 2 * env_args['max_step']

   args.eval_gap = 2 ** 8
   args.eval_times1 = 2 ** 1
   args.eval_times2 = 2 ** 4
   args.break_step = int(8e7)
   args.if_allow_break = False
   args.worker_num = 1
   

Step 6: Train your agent
----------------------------------------

In this tutorial, we provide a single-process demo to train an agent **without any GPU usage and parallelism**:
   
.. code-block:: python

   args.learner_gpus = -1  # no GPU usage
   train_and_evaluate(args)
   
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

.. image:: ../images/LunarLanderTwinDelay3.gif
   :width: 80%
   :align: center
