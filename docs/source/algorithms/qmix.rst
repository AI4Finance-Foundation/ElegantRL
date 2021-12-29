.. _ddpg:


QMix
==========

`QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning <https://arxiv.org/abs/1803.11485>`_ is a value-based method that can train decentralised policies in a centralised end-to-end fashion.QMIX employs a network that estimates joint action-values as a complex non-linear combination of per-agent values that condition only on local observations. 

-  Experience replay: ✔️
-  Target network: ✔️
-  Gradient clipping: ❌
-  Reward clipping: ❌
-  Prioritized Experience Replay (PER): ❌
-  Ornstein–Uhlenbeck noise: ❌



Code Snippet
------------

.. code-block:: python

    import torch
    from elegantrl.run import train_and_evaluate
    from elegantrl.config import Arguments
    from elegantrl.envs.gym import build_env
    from elegantrl.agents.AgentDDPG import AgentDDPG
    
    # train and save
    args = Arguments(env=build_env('Pendulum-v0'), agent=AgentDDPG())
    args.cwd = 'demo_Pendulum_DDPG'
    args.env.target_return = -200
    args.reward_scale = 2 ** -2
    train_and_evaluate(args) 
    
    # test
    agent = AgentDDPG()
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

.. autoclass:: elegantrl.agents.AgentQMix.AgentQMix
   :members:
   
.. _qmix_networks:
   
Networks
-------------

.. autoclass:: elegantrl.agents.net.QMix
   :members:
   
.. autoclass:: elegantrl.agents.net.Critic
   :members:
