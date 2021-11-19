.. _sac:


SAC
==========

`Soft Actor-Critic (SAC) <https://arxiv.org/abs/1801.01290>`_ is an off-policy Actor-Critic algorithm for continuous action space. In SAC, it introduces an entropy regularization to the loss function, which has a close connection with the trade-off of the exploration and exploitation. In our implementation, we employ a **learnable entropy regularization coefficienct** to dynamic control the scale of the entropy, which makes it consistent with a pre-defined target entropy. SAC also utilizes **Clipped Double-Q Learning** (mentioned in TD3) to overcome the overestimation of Q-values. This implementation provides SAC and supports the following extensions:

-  Experience replay: ✔️
-  Target network: ✔️
-  Gradient clipping: ✔️
-  Reward clipping: ❌
-  Prioritized Experience Replay (PER): ✔️
-  Leanable entropy regularization coefficient: ✔️

.. note::
    Inspired by the delayed policy update from TD3, we implement a modified version of SAC ``AgentModSAC`` with a dynamic adjustment of the frequency of the policy update. The adjustment is based on the loss of critic networks: a small loss leads to a high update frequency and vise versa. 

Code Snippet
------------

.. code-block:: python

    import torch
    from elegantrl.run import train_and_evaluate
    from elegantrl.config import Arguments
    from elegantrl.envs.gym import build_env
    from elegantrl.agents.AgentSAC import AgentSAC
    
    # train and save
    args = Arguments(env=build_env('Pendulum-v0'), agent=AgentSAC())
    args.cwd = 'demo_Pendulum_SAC'
    args.env.target_return = -200
    args.reward_scale = 2 ** -2
    train_and_evaluate(args) 
    
    # test
    agent = AgentSAC()
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

.. autoclass:: elegantrl.agents.AgentSAC.AgentSAC
   :members:
   
.. autoclass:: elegantrl.agents.AgentSAC.AgentModSAC
   :members:
   
.. _sac_networks:
   
Networks
-------------

.. autoclass:: elegantrl.agents.net.ActorSAC
   :members:
   
.. autoclass:: elegantrl.agents.net.CriticTwin
   :members:
