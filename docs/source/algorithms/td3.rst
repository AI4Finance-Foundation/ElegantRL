.. _td3:


TD3
==========

`Twin Delayed DDPG (TD3) <https://arxiv.org/abs/1802.09477>`_ is a successor of DDPG algorithm with the usage of three additional tricks. In TD3, the usage of **Clipped Double-Q Learning**,  **Delayed Policy Updates**, and **Target Policy Smoothing** overcomes the overestimation of Q-values and smooths out Q-values along with changes in action, which shows improved performance over baseline DDPG. This implementation provides TD3 and supports the following extensions:

-  Experience replay: ✔️
-  Target network: ✔️
-  Gradient clipping: ✔️
-  Reward clipping: ❌
-  Prioritized Experience Replay (PER): ✔️

.. note::
    For the clipped Double-Q learning, we implement two Q-networks with shared parameters under a single Class ``CriticTwin``. Such an implementation allows a lower computational and training time cost.

.. warning::
    In the TD3 implementation, it contains a number of highly sensitive hyper-parameters, which requires the user to carefully tune these hyper-parameters to obtain a satisfied result.

Code Snippet
------------

.. code-block:: python

    import torch
    from elegantrl.run import train_and_evaluate
    from elegantrl.config import Arguments
    from elegantrl.envs.gym import build_env
    from elegantrl.agents.AgentTD3 import AgentTD3
    
    # train and save
    args = Arguments(env=build_env('Pendulum-v0'), agent=AgentTD3())
    args.cwd = 'demo_Pendulum_TD3'
    args.env.target_return = -200
    args.reward_scale = 2 ** -2
    train_and_evaluate(args) 
    
    # test
    agent = AgentTD3()
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

.. autoclass:: elegantrl.agents.AgentTD3.AgentTD3
   :members:
   
.. _td3_networks:
   
Networks
-------------

.. autoclass:: elegantrl.agents.net.Actor
   :members:
   
.. autoclass:: elegantrl.agents.net.CriticTwin
   :members:
