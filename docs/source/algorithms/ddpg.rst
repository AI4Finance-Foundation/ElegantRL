.. _ddpg:


DDPG
==========

`Deep Deterministic Policy Gradient (DDPG) <https://arxiv.org/abs/1509.02971>`_ is an off-policy policy-gradient algorithm for continuous action space. Since computing the maximum over actions in the target is a challenge in continuous action spaces, DDPG deals with this by using a policy network to compute an action. This implementation provides DDPG and supports the following extensions:

-  Experience replay: ✔️
-  Target network: ✔️
-  Gradient clipping: ✔️
-  Reward clipping: ❌
-  Prioritized Experience Replay (PER): ✔️
-  Ornstein–Uhlenbeck noise: ✔️

.. note::
    This implementation has no support for reward clipping because we introduce the hyper-paramter ``reward_scale`` as an alternative for reward scaling. We believe that the clipping function may omit information since it cannot map the clipped reward back to the original reward, however, the reward scaling function is able to manipulate the reward back and forth.


.. warning::
    PER leads to a faster learning speed and is also critical for environmnets with sparse reward. However, a replay buffer with small size may hurt the performance of PER. 


Code Snippet
------------

.. code-block:: python

    import torch
    from elegantrl.run import Arguments, train_and_evaluate
    from elegantrl.env import build_env
    from elegantrl.agent import AgentDDPG
    
    # train and save
    args = Arguments(env=build_env('CartPole-v0'), agent=AgentDDPG())
    args.cwd = 'demo_CartPole_DQN'
    args.target_return = 195
    train_and_evaluate(args) 
    
    # test
    agent = AgentDDPG()
    agent.init(args.net_dim, args.state_dim, args.action_dim)
    agent.save_or_load_agent(cwd=agrs.cwd, if_save=False)
    
    env = build_env('CartPole-v0')
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

.. autoclass:: elegantrl.agent.AgentDDPG
   :members:
   
.. _ddpg_networks:
   
Networks
-------------

.. autoclass:: elegantrl.net.Actor
   :members:
   
.. autoclass:: elegantrl.net.Critic
   :members:
