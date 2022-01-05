.. _dqn:


DQN
==========

`Deep Q-Network (DQN) <https://arxiv.org/abs/1312.5602>`_ is an off-policy value-based algorithm for discrete action space. It uses a deep neural network to approximate a Q function defined on a state-action pair. This implementation starts from a vanilla Deep Q-Learning and supports the following extensions:

-  Experience replay: ✔️
-  Target network (soft update): ✔️
-  Gradient clipping: ✔️
-  Reward clipping: ❌
-  Prioritized Experience Replay (PER): ✔️
-  Dueling network architecture: ✔️

.. note::
    This implementation has no support for reward clipping because we introduce the hyper-paramter ``reward_scale`` for reward scaling as an alternative. We believe that the clipping function may omit information since it cannot map the clipped reward back to the original reward, however, the reward scaling function is able to manipulate the reward back and forth.


.. warning::
    PER leads to a faster learning speed and is also critical for environments with sparse rewards. However, a replay buffer with small size may hurt the performance of PER. 


Code Snippet
------------

.. code-block:: python

    import torch
    from elegantrl.run import train_and_evaluate
    from elegantrl.config import Arguments
    from elegantrl.envs.gym import build_env
    from elegantrl.agents.AgentDQN import AgentDQN
    
    # train and save
    args = Arguments(env=build_env('CartPole-v0'), agent=AgentDQN())
    args.cwd = 'demo_CartPole_DQN'
    args.target_return = 195
    args.agent.if_use_dueling = True
    train_and_evaluate(args) 
    
    # test
    agent = AgentDQN()
    agent.init(args.net_dim, args.state_dim, args.action_dim)
    agent.save_or_load_agent(cwd=args.cwd, if_save=False)
    
    env = build_env('CartPole-v0')
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

.. autoclass:: elegantrl.agents.AgentDQN.AgentDQN
   :members:
   
.. _dqn_networks:
   
Networks
-------------

.. autoclass:: elegantrl.agents.net.QNet
   :members:
   
.. autoclass:: elegantrl.agents.net.QNetDuel
   :members:
