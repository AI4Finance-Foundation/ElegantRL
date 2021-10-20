.. _ppo:


PPO
==========

`Proximal Policy Optimization (PPO) <https://arxiv.org/abs/1707.06347>`_ is a successor of DDPG algorithm with the usage of three additional tricks. In TD3, the usage of **Clipped Double-Q Learning**,  **Delayed Policy Updates**, and **Target Policy Smoothing** overcomes the overestimation of Q-values and smooths out Q-values along with changes in action, which shows improved performance over baseline DDPG. This implementation provides TD3 and supports the following extensions:

-  Target network: ✔️
-  Gradient clipping: ✔️
-  Reward clipping: ❌
-  Generalized Advantage Estimation (GAE): ✔️
-  Entropy

.. note::
    For the clipped Double-Q learning, we implement two Q-networks with shared parameters under a single Class ``CriticTwin``. Such an implementation allows a lower computational and training time cost.

.. warning::
    In the TD3 implementation, it contains a number of highly sensitive hyper-parameters, which requires the user to carefully tune these hyper-parameters to obtain a satisfied result.

Code Snippet
------------

.. code-block:: python

    import torch
    from elegantrl.run import Arguments, train_and_evaluate
    from elegantrl.env import build_env
    from elegantrl.agent import AgentPPO
    
    # train and save
    args = Arguments(env=build_env('Pendulum-v0'), agent=AgentPPO())
    args.cwd = 'demo_Pendulum_PPO'
    args.env.target_return = -200
    args.reward_scale = 2 ** -2
    train_and_evaluate(args) 
    
    # test
    agent = AgentPPO()
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

.. autoclass:: elegantrl.agent.AgentPPO
   :members:
   
.. _ppo_networks:
   
Networks
-------------

.. autoclass:: elegantrl.net.ActorPPO
   :members:
   
.. autoclass:: elegantrl.net.CriticPPO
   :members:
