.. _dqn:
.. automodule:: elegantrl.agent

DQN
==========

`Deep Q-Network (DQN) <https://arxiv.org/abs/1312.5602>`_ approximates a state-value function in a Q-Learning framework with a neural network. This implementation provides vanilla Deep Q-Learning and supports the following extensions:

-  Experience replay: ✔️
-  Target network: ✔️
-  Gradient clipping: ✔️
-  Reward clipping: ❌
-  Prioritized Experience Replay (PER): ✔️
-  Dueling DQN: ✔️

.. note::
    This implementation has no support for reward clipping because we introduce the hyper-paramter ``reward_scale`` as an alternative for reward scaling. We believe that the clipping function may omit information since it cannot map the clipped reward back to the original reward, however, the reward scaling function is able to manipulate the reward back and forth.

Example
------------

.. code-block:: python
    import torch
    from elegantrl.run import Arguments, train_and_evaluate
    from elegantrl.env import build_env
    from elegantrl.agent import AgentDQN
    
    # train and save
    args = Arguments(env=build_env('CartPole-v0'), agent=AgentDQN())
    args.cwd = 'demo_CartPole_DQN'
    args.target_return = 195
    train_and_evaluate(args) 
    # test
    env = build_env('CartPole-v0')
    agent = AgentDQN()
    agent.init(args.net_dim, args.state_dim, args.action_dim)
    agent.save_or_load_agent(cwd=agrs.cwd, if_save=False)
    state = env.reset()
    episode_reward = 0
    for i in range(2 ** 10):
        s = torch.as_tensor((state,), dtype=torch.float32, device=agent.device)
        action = agent.act(s).detach().cpu().numpy()[0]
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

.. autoclass:: elegantrl.agent.AgentDQN
   :members:
   
.. _dqn_networks:
   
Networks
-------------

.. autoclass:: elegantrl.net.QNet
   :members:
