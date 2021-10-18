.. _td3:


TD3
==========

`Twin Delayed DDPG (TD3) <https://arxiv.org/abs/1802.09477>`_ is an off-policy policy-gradient algorithm for continuous action space. Since computing the maximum over actions in the target is a challenge in continuous action spaces, DDPG deals with this by using a policy network to compute an action. This implementation provides DDPG and supports the following extensions:

-  Experience replay: ✔️
-  Target network: ✔️
-  Gradient clipping: ✔️
-  Reward clipping: ❌
-  Prioritized Experience Replay (PER): ✔️
-  Ornstein–Uhlenbeck noise: ✔️


.. note::
    In the DDPG paper, the authors use time-correlated Ornstein-Uhlenbeck Process to add noise to the action output. However, as shown in the later works, the Ornstein-Uhlenbeck Process is an overcomplication that does not have a noticeable effect on performance when compared to uncorrelated Gaussian noise.

Code Snippet
------------

.. code-block:: python

    import torch
    from elegantrl.run import Arguments, train_and_evaluate
    from elegantrl.env import build_env
    from elegantrl.agent import AgentTD3
    
    # train and save
    args = Arguments(env=build_env('Pendulum-v0'), agent=AgentTD3())
    args.cwd = 'demo_Pendulum_TD3'
    args.env.target_return = -200
    args.reward_scale = 2 ** -2
    train_and_evaluate(args) 
    
    # test
    agent = AgentTD3()
    agent.init(args.net_dim, args.state_dim, args.action_dim)
    agent.save_or_load_agent(cwd=agrs.cwd, if_save=False)
    
    env = build_env('Pendulum-v0')
    state = env.reset()
    episode_reward = 0
    for i in range(2 ** 10):
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        
        episode_reward += reward
        if done:
            print(f'Step {i:>6}, Episode return {episode_return:8.3f}')
            break
        else:
            state = next_state
        env.render()
              
              
              
Parameters
---------------------

.. autoclass:: elegantrl.agent.AgentTD3
   :members:
   
.. _ddpg_networks:
   
Networks
-------------

.. autoclass:: elegantrl.net.Actor
   :members:
   
.. autoclass:: elegantrl.net.Critic
   :members:
