.. _ppo:


PPO
==========

`Proximal Policy Optimization (PPO) <https://arxiv.org/abs/1707.06347>`_ is an on-policy Actor-Critic algorithm for both discrete and continuous action spaces. It has two primary variants: **PPO-Penalty** and **PPO-Clip**, where both utilize surrogate objectives to avoid the new policy changing too far from the old policy. This implementation provides PPO-Clip and supports the following extensions:

-  Target network: ✔️
-  Gradient clipping: ✔️
-  Reward clipping: ❌
-  Generalized Advantage Estimation (GAE): ✔️
-  Discrete version: ✔️

.. note::
    The surrogate objective is the key feature of PPO since it both regularizes the policy update and enables the reuse of training data.
    
A clear explanation of PPO algorithm and implementation in ElegantRL is available `here <https://towardsdatascience.com/elegantrl-mastering-the-ppo-algorithm-part-i-9f36bc47b791>`_.

Code Snippet
------------

.. code-block:: python

    import torch
    from elegantrl.run import train_and_evaluate
    from elegantrl.config import Arguments
    from elegantrl.envs.gym import build_env
    from elegantrl.agents.AgentPPO import AgentPPO
    
    # train and save
    args = Arguments(env=build_env('BipedalWalker-v3'), agent=AgentPPO())
    args.cwd = 'demo_BipedalWalker_PPO'
    args.env.target_return = 300
    args.reward_scale = 2 ** -2
    train_and_evaluate(args) 
    
    # test
    agent = AgentPPO()
    agent.init(args.net_dim, args.state_dim, args.action_dim)
    agent.save_or_load_agent(cwd=args.cwd, if_save=False)
    
    env = build_env('BipedalWalker-v3')
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

.. autoclass:: elegantrl.agents.AgentPPO.AgentPPO
   :members:
   
.. autoclass:: elegantrl.agents.AgentPPO.AgentDiscretePPO
   :members:
   
.. _ppo_networks:
   
Networks
-------------

.. autoclass:: elegantrl.agents.net.ActorPPO
   :members:
   
.. autoclass:: elegantrl.agents.net.ActorDiscretePPO
   :members:
   
.. autoclass:: elegantrl.agents.net.CriticPPO
   :members:
