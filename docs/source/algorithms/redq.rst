.. _redq:


REDQ
==========

`Randomized Ensembled Double Q-Learning: Learning Fast Without a Model (REDQ) <https://arxiv.org/abs/2101.05982>`_ has 
three carefully integrated ingredients which allow it to achieve its high performance:  

-  update-to-data (UTD) ratio >> 1.
-  an ensemble of Q functions.
-  in-target minimization across a random subset of Q functions.

This implementation is based on SAC.


Code Snippet
------------

.. code-block:: python

    import torch
    from elegantrl.run import train_and_evaluate
    from elegantrl.config import Arguments
    from elegantrl.envs.gym import build_env
    from elegantrl.agents.AgentREDQ import AgentREDQ
    
    # train and save
    args = Arguments(env=build_env('Hopper-v2'), agent=AgentREDQ())
    args.cwd = 'demo_Hopper_REDQ'
    train_and_evaluate(args) 
    
    # test
    agent = AgentREDQ()
    agent.init(args.net_dim, args.state_dim, args.action_dim)
    agent.save_or_load_agent(cwd=args.cwd, if_save=False)
    
    env = build_env('Pendulum-v0')
    state = env.reset()
    episode_reward = 0
    for i in range(125000):
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

.. autoclass:: elegantrl.agents.AgentREDQ.AgentREDQ
   :members:
   
.. _redq_networks:
   
Networks
-------------

.. autoclass:: elegantrl.agents.net.ActorSAC
   :members:
   
.. autoclass:: elegantrl.agents.net.Critic
   :members:
   
