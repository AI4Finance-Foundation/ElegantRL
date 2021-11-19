.. _a2c:


A2C
==========

`Advantage Actor-Critic (A2C) <https://arxiv.org/abs/1602.01783>`_ is a synchronous and deterministic version of Asynchronous Advantage Actor-Critic (A3C). This implementation of the A2C algorithm is built on PPO algorithm for simplicity, and it supports the following extensions:

-  Target network: ✔️
-  Gradient clipping: ✔️
-  Reward clipping: ❌
-  Generalized Advantage Estimation (GAE): ✔️
-  Discrete version: ✔️

.. warning::
    The implementation of A2C serves as a pedagogical goal. For practitioners, we recommend using the PPO algorithm for training agents. Without the trust-region and clipped ratio, hyper-parameters in A2C, e.g., ``repeat_times``, need to be fine-tuned to avoid performance collapse.
    

Code Snippet
------------

.. code-block:: python

    import torch
    from elegantrl.run import train_and_evaluate
    from elegantrl.config import Arguments
    from elegantrl.envs.gym import build_env
    from elegantrl.agents.AgentA2C import AgentA2C
    
    # train and save
    args = Arguments(env=build_env('Pendulum-v0'), agent=AgentA2C())
    args.cwd = 'demo_Pendulum_A2C'
    args.env.target_return = -200
    args.reward_scale = 2 ** -2
    train_and_evaluate(args) 
    
    # test
    agent = AgentA2C()
    agent.init(args.net_dim, args.state_dim, args.action_dim)
    agent.save_or_load_agent(cwd=args.cwd, if_save=False)
    
    env = build_env('Pendulum-v0')
    state = env.reset()
    episode_reward = 0
    for i in range(2 ** 10):
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        
        episode_reward += reward
        if done:
            print(f'Step {i:>6}, Episode return {episode_reward:8.3f}')
            break
        else:
            state = next_state
        env.render()
              
              
              
Parameters
---------------------

.. autoclass:: elegantrl.agents.AgentA2C.AgentA2C
   :members:
   
.. autoclass:: elegantrl.agents.AgentA2C.AgentDiscreteA2C
   :members:
   
.. _a2c_networks:
   
Networks
-------------

.. autoclass:: elegantrl.agents.net.ActorPPO
   :members:
   
.. autoclass:: elegantrl.agents.net.ActorDiscretePPO
   :members:
   
.. autoclass:: elegantrl.agents.net.CriticPPO
   :members:
