Single File
=============

As a quickstart, we introduce how to train a DRL agent efficiently by using massive parallel simulation in ElegantRL. We take PPO and multi-stock task as example.

Step 1: Main Modules
-------------------------------

.. code-block:: python
   
   # Environment Class, wrapped with VMAP for massive parallel simulation
   class StockTradingVmapEnv:
      # gym-style functions:
      self.reset()
      self.step()
      
      # vectorized functions:
      self.vmap_get_total_asset = vmap(func=_get_total_asset, in_dims=(None, 0, 0), out_dims=0)
      self.vmap_get_state = vmap(func=_get_state, in_dims=(0, 0, None, None), out_dims=0)
      self.vmap_inplace_amount_shares_when_buy = vmap(func=_inplace_amount_shares_when_buy, in_dims=(0, 0, 0, None, None), out_dims=0)
      self.vmap_inplace_amount_shares_when_sell = vmap(func=_inplace_amount_shares_when_sell, in_dims=(0, 0, 0, None, None), out_dims=0)
      
   # Agent Class, algorithm control module
   class AgentBase
      # main functions:
      self.init() # Init agent hyperparameters.
      self.optimizer_update() # Perform backward propagation for optimizer
      self.soft_update() # Soft update target network
   
   class AgentPPO
      # main functions:
      self.explore_env() # Perform a rollout and save it into the Replaybuffer
      self.update_net() # Update the agent: computation the loss function and call the backward propagation.
      self.get_advantages() # Compute the advantage value.
   
   # Neural Network Class, pytorch network module
   class ActorPPO 
      # main functions:
      self.get_action(state) # input state and output (action, log_probability).
   class CriticPPO
      # main functions:
      self.forward(state) # input state and output the corresponding advantage value.   
   # Evaluator Class, evaluation and test module
   class Evaluator
      # main functions:
      self.evaluate_and_save() # Perform evaluation and save the model.
   
   # Hyperparameters Class, setting hyperparameters
   class Config
   
   
Step 2: Preperation of Agent, Environment and Hyperparameters
--------------------------------------

.. code-block:: python

    agent_class = AgentPPO  # DRL algorithm name
    env_class = StockTradingVmapEnv # run a finance env with massive parallel simulation.
    env_args = {
        'env_name': 'StockTradingVmapEnv',  # Store the environment class in the hyperparmeters. 
        'state_dim':   # number of shares + price + technique factors + amount dimension
        'action_dim':  # number of shares
        'if_discrete': False  # continuous action space
    }
    get_gym_env_args(env=StockTradingVmapEnv(), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `config.py Arguments()` for hyperparameter explanation
    args.break_step = int(2e5)  # break training if 'total_step > break_step'
    args.net_dims = (64, 32)  # the middle layer dimension of MultiLayer Perceptron
    args.gamma = 0.97  # discount factor of future rewards
    args.repeat_times = 16  # repeatedly update network using ReplayBuffer to keep critic's loss small.

    train_agent(args) # Pass the hyperparameters and start the training flow.
   
Part 3: Train and Evaluate the Agent
--------------------------------------

.. code-block:: python

   def train_agent(args: Config):
       
       args.init_before_training() # Set the path for saving a trained model
       env = build_env(args.env_class, args.env_args) # Initialize an environment instance for sampling.
       agent = args.agent_class(args.net_dims, args.state_dim, args.action_dim, gpu_id=args.gpu_id, args=args) # Initialize an agent.
       
       # Initialize the evluator with an indepedent environment for evluation. 
       evaluator = Evaluator(eval_env=build_env(args.env_class, args.env_args),
                             eval_per_step=args.eval_per_step,
                             eval_times=args.eval_times,
                             cwd=args.cwd)
       
       torch.set_grad_enabled(False)
       agent.last_state = env.reset() # Store the last state so that agent continues a rollout without a restart.
       while True:  # start training
           buffer_items = agent.explore_env(env, args.horizon_len) # Perform a rollout with a length of horizon_len.

           torch.set_grad_enabled(True)
           logging_tuple = agent.update_net(buffer_items) # Update the agent by using data from replay buffer.
           torch.set_grad_enabled(False)

           evaluator.evaluate_and_save(agent.act, args.horizon_len, logging_tuple) # Evaluate the performance of the agent.
           if (evaluator.total_step > args.break_step) or os.path.exists(f"{args.cwd}/stop"): # Stop the training process by make a new directory "stop".
               break  # stop training when reach `break_step` or `mkdir cwd/stop`
 
