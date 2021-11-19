.. _ddpg:


DDPG
==========

`Deep Deterministic Policy Gradient (DDPG) <https://arxiv.org/abs/1509.02971>`_ is an off-policy Actor-Critic algorithm for continuous action space. Since computing the maximum over actions in the target is a challenge in continuous action spaces, DDPG deals with this by using a policy network to compute an action. This implementation provides DDPG and supports the following extensions:

-  Experience replay: ✔️
-  Target network: ✔️
-  Gradient clipping: ✔️
-  Reward clipping: ❌
-  Prioritized Experience Replay (PER): ✔️
-  Ornstein–Uhlenbeck noise: ✔️


.. warning::
    In the DDPG paper, the authors use time-correlated Ornstein-Uhlenbeck Process to add noise to the action output. However, as shown in the later works, the Ornstein-Uhlenbeck Process is an overcomplication that does not have a noticeable effect on performance when compared to uncorrelated Gaussian noise.

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

.. autoclass:: elegantrl.agents.AgentDDPG.AgentDDPG
   :members:
   
.. _ddpg_networks:
   
Networks
-------------

.. autoclass:: elegantrl.agents.net.Actor
   :members:
   
.. autoclass:: elegantrl.agents.net.Critic
   :members:
