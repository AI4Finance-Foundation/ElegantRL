# API List (brief)


---
## `class AgentBase()` 

### level 1 `init(self, net_dim, state_dim, action_dim)` 
explict call init() to build networks
- `int net_dim`
- `int state_dim`
- `int action_dim` the dimension of action (or the number of discrete action)

### level 1 `store_transition(self, env, buffer, target_step, reward_scale, gamma)`
store transition (state, action, reward, ...) to ReplayBuffer
- `env` DRL training environment, it has `.reset()` and `.step()`
- `buffer` experience replay buffer.
- `int target_step` target number of exploration steps in the environment
- `float reward_scale` scale the reward size
- `float gamma` discount factor

plan to move `reward_scale` and `gamma` to `init()`

### level 1 `update_net(self, buffer, target_step, batch_size, repeat_times)`
update networks using sampling data from ReplayBuffer
- `buffer` experience replay buffer. ReplayBuffer has `.sample_batch()`
- `int target_step` number of target steps that add to ReplayBuffer
- `int batch_size` number of samples for stochastic gradient decent

### level 2 `select_action(self, state)` 
select action for exploration
- `array state`
- return `array action`

### level 2`soft_update(self, target_net, current_net)` 
soft target update using self.tau (set as $2^{-8}$ in default)
- `target_net` update via soft target update
- `current_net` update via gradient decent



---
## `class ReplayBuffer`

### level 1 `__init__(self, max_len, state_dim, action_dim, if_on_policy, if_gpu)`
creat a continuous memory space to store data
- `int max_len` maximum capacity, First In First Out.
- `int state_dim`
- `int action_dim`
- `bool if_on_policy` on-policy or off-policy
- `bool if_gpu` creat memory space on CPU RAM or GPU RAM

### level 2 `append_buffer(self, state, other)`
append to ReplayBuffer, same as `list.append()`
- `array state`
- `array other` including `action, reward, ...`

### level 2 `extend_buffer(self, state, other)`
extend to ReplayBuffer, same as `list.extend()`
- `array state`
- `array other` including `action, reward, ...`

### level 2 `sample_batch(self, batch_size)`
sample a batch of data from ReplayBuffer randomly for stochastic gradient decent
- `int batch_size` number of data in a batch
- return `float reward` a batch of reward
- return `float mask` a batch of mask (`mask = 0 if done else gamma`)
- return `array action` a batch of action
- return `array state` a batch of state
- return `array state` a batch of next state 

### level 2 `sample_for_ppo(self)`
sample all the data from ReplayBuffer
- return `float reward`
- return `float mask`
- return `array action`
- return `array noise`
- return `array state`



---
## `class Evaluator`
evaluate and save the best policy. Then plot the learning curve for fine-tuning.

### level 1 `evaluate_save(self, act, steps, obj_a, obj_c)`
evaluate the save the best policy. Save some training information for plotting.
- `act` actor network. (or $\arg\max(Q(s))$ in DQN variants)
- `int steps` increased number of training steps
- `float obj_a` mean of actor objective
- `float obj_c` mean of critic objective
- return `bool if_reach_goal` average episode return `r_avg > target_reward` means reach the goal.

### level 2 `save_npy_draw_plot(self)`
save the training information into `cwd/recorder.npy`. Then draw a plot.



---
## `class Preprocess(gym.Wrapper)`
preprocess a OpenAI standard gym environment for training

### level 1 `__init__(self, env)` 
get environemnt information

### level 2 `reset(self)`
reset the environment and start next episode
- return `array state`

### level 2 `step(self, action)`
actor explore in environment
- `array action`
- return `array state` next state
- return `float reward`
- return `bool done` end of this episode
- return `dict info` OpenAI standard operation. return a `None` is ok.



---
## `Net(nn.Moudule)`
`nn.Moudule` is a PyTorch standard base class of neural network. We build net
including QNet, Actor, Critic and their variants.

### level 2 `__init__(self, mid_dim, state_dim, action_dim)`
randomly initialize the neural network parameters
- `int net_dim`
- `int state_dim`
- `int action_dim` the dimension of action (or the number of discrete action)

### level 2 `forward(self, state)`
the policy of agent. map state to action. (if discrete action space, map state to Q value)
- `tensor state`
- return `tensor action`




# API List (full)

---
## level 0 `class Arguement()` 
set hyperparameters and prepare training environment

### level 1 `init_before_training(self, if_main)` 
prepare training environment.
- input `bool if_main` build current working directory

---
## level 0 `train_and_evaluate(args)` 
single processing, `args=Arguement()`

## level 0 `train_and_evaluate__multiprocessing(args)` 
multiprocessing, `args=Arguement()`

### level 0 `mp__update_params(args, pipe1_eva, pipe1_exp_list)`
update network parameters using ReplayBuffer data

- input `Argument args` hyperparameters
- input `pipe1_eva` pipe connect to process `mp_evaluate_agent`. Send actor network, receive `if_reach_goal` signal.
- input `pipe1_exp_list` pipes connect to multiple process `mp_explore_in_env`. Send actor network, receive ReplayBuffer data.

### level 0 `mp_explore_in_env(args, pipe2_exp, worker_id)`
explore the environment and save data into ReplayBuffer

- input `Argument args` hyperparameters
- input `pipe2_exp` pipe connect to process `mp__update_params`. Receive actor network, send ReplayBuffer data.

### level 0 `mp_evaluate_agent(args, pipe2_eva)`
evaluate the agent (actor network) and collect training information for fine-tuning.

- input `Argument args` hyperparameters
- input `pipe2_eva` pipe connect to process `mp__update_params`. Receive actor network, send `if_reach_goal` signal.




---
## `class AgentBase()` 

### level 0 `__init__()` 
default initialize


### level 1 `init(self, net_dim, state_dim, action_dim)` 
explict call init() to build networks
- `int net_dim` the dimension of networks 
- `int state_dim` the dimension of state 
- `int action_dim` the dimension of action (or the number of discrete action)


### level 1 `store_transition(self, env, buffer, target_step, reward_scale, gamma)`
store transition (state, action, reward, ...) to ReplayBuffer
- `env` DRL training environment, it has `.reset()` and `.step()`
- `buffer` experience replay buffer. ReplayBuffer has `.append_buffer()` 
- `int target_step` target number of exploration steps in the environment
- `float reward_scale` scale the reward size
- `float gamma` discount factor

plan to move `reward_scale` and `gamma` to `init()`


### level 1 `update_net(self, buffer, target_step, batch_size, repeat_times)`
update networks using sampling data from ReplayBuffer
- `buffer` experience replay buffer. ReplayBuffer has `.sample_batch()`
- `int target_step` number of target steps that add to ReplayBuffer
- `int batch_size` number of samples for stochastic gradient decent

### level 2 `select_action(self, state)` 
select action for exploration
- `array state` the shape of state is `(state_dim, )`
- return `array action` the shape of action is `(action_dim, )`


### level 2 `save_load_model(self, cwd, if_save)`
save neural network to cwd
- `str cwd` current working directory
- `bool if_save` save or load model


### level 2`soft_update(self, target_net, current_net)` 
soft target update using self.tau (set as $2^{-8}$ in default)
- `target_net` update via soft target update of `current_net`
- `current_net` update via gradient decent





---
## `class ReplayBuffer`
single processing

## `class ReplayBufferMP`
multiple processing. Same as single processing

### level 1 `__init__(self, max_len, state_dim, action_dim, if_on_policy, if_gpu)`
creat a continuous memory space to store data
- `int max_len` maximum capacity, First In First Out.
- `int state_dim` the dimension of state 
- `int action_dim` the dimension of action (`action_dim=1` for discrete action)
- `bool if_on_policy` on-policy or off-policy
- `bool if_gpu` creat memory space on CPU RAM or GPU RAM


### level 2 `append_buffer(self, state, other)`
append to ReplayBuffer, same as `list.append()`
- `array state` the shape is `(state_dim, )`
- `array other` the shape is `(other_dim, )`, including `action, reward, ...`


### level 2 `extend_buffer(self, state, other)`
extend to ReplayBuffer, same as `list.extend()`
- `array state` the shape is `(-1, state_dim)`
- `array other` the shape is `(-1, other_dim)`, including `action, reward, ...`


### level 2 `sample_batch(self, batch_size)`
sample a batch of data from ReplayBuffer randomly for stochastic gradient decent
- `int batch_size` number of data in a batch
- return `float reward` a batch of reward
- return `float mask` a batch of mask (`mask = 0 if done else gamma`)
- return `array action` a batch of action
- return `array state` a batch of state
- return `array state` a batch of next state 


### level 2 `sample_for_ppo(self)`
sample all the data from ReplayBuffer. `now_len` is current length of ReplayBuffer.
- return `float reward` shape is `(now_len, 1)`
- return `float mask` shape is `(now_len, 1)`
- return `array action` shape is `(now_len, action_dim)`
- return `array noise` shape is `(now_len, action_dim)`
- return `array state` shape is `(now_len, state_dim)`


### level 2 `update__now_len__before_sample(self)`
update `now_len` (pointer) before sample data form ReplayBuffer


### level 2 `empty_memories__before_explore(self)`
empty the memories of ReplayBuffer before exploring for on-policy


### level 2`print_state_norm(self)`
print the `avg` and `std` for state normalization. compute using the state in ReplayBuffer after finishing the training pipeline




---
## `class Evaluator`
evaluate and save the best policy. Then plot the learning curve for fine-tuning.

### level 1 `__init__(self, ...)`

### level 1 `evaluate_save(self, act, steps, obj_a, obj_c)`
evaluate the save the best policy. Save some training information for plotting.
- `act` actor network. (or $\arg\max(Q(s))$ in DQN variants)
- `int steps` increased number of training steps
- `float obj_a` mean of actor objective
- `float obj_c` mean of critic objective
- return `if_reach_goal` average episode return `r_avg > target_reward` means reach the goal.

### level 2 `save_npy_draw_plot(self)`
save the training information into `cwd/recorder.npy`. Then draw a plot.



---
## `class Preprocess(gym.Wrapper)`
preprocess a OpenAI standard gym environment for training

### level 1 `__init__(self, env)` 
get environemnt information
- `str env_name` for example LunarLander-v2
- `int net_dim` the dimension of networks 
- `int state_dim` the dimension of state 
- `int action_dim` the dimension of action (or the number of discrete action)
- `int action_max` the max of continuous action. set as 1 when it is discrete action
- `int max_step` the max step of an episode 
- `bool if_discrete` discrete or continuous action space
- `float target_reward` the gold score of this environment
- `array neg_state_avg` for state normalization
- `array div_state_std` for state normalization

### level 2 `reset(self)`
reset the environment and start next episode. `.reset_norm()` mean do normalization on state.
- return `array state`

### level 2 `step(self, action)`
actor inact in environment. `.step_norm()` mean do normalization on state.
- `array action`
- return `array state` next state
- return `float reward`
- return `bool done` end of this episode
- return `dict info` OpenAI standard operation. return a `None` is ok.



---
## `class nn.Moudule`
PyTorch standard base class of neural network

## `Net(nn.Moudule)`
including QNet, Actor, Critic and their variants.

### level 2 `__init__(self, mid_dim, state_dim, action_dim)`
randomly initialize the neural network parameters
- `int net_dim` the dimension of networks 
- `int state_dim` the dimension of state 
- `int action_dim` the dimension of action (or the number of discrete action)


### level 2 `forward(self, state)`
the policy of agent. map state to action. (if discrete action space, map state to Q value)
- `tensor state` the shape is `(-1, state_dim)`



---
## Utils
some useful tools

### level 2 `get_episode_return(env, act, device)`
evaluate the policy (actor network)
- `env` environment has `.reset()` ,`.step()`, `.max_step` and `if_discrete` for evaluating.
- `act` actor network has `.forward()`
- `device` choose GPU `torch.device("cuda")` or CPU `torch.device("cpu")`
- output `float score` episode return


###  level 2 `save_learning_curve(recorder, cwd, save_title)`
plot the learning curve.
- `array recorder` its shape is `(-1, 4)`. Its items are `total_step, r_avg, r_std, obj_a, obj_c`
- `str cwd` current working directory
- `str save_title` title of plot


###  level 2 `explore_before_training(env, buffer, target_step, reward_scale, gamma)`
explore env using random action before training an off-policy algorithms. Then save these trajectories into ReplayBuffer.
- `env` Env
- `buffer` ReplayBuffer
- `int target_step` target number of exploration steps in the environment
- `float reward_scale` scale the reward size
- `float gamma` discount factor


###  level 2 `get_gym_env_info(env, if_print)`
get the environment inforation for training.
- `env` environment
- `bool if_print` print the information
- return `str env_name` for example LunarLander-v2
- return `int state_dim` the dimension of state 
- return `int action_dim` the dimension of action (or the number of discrete action)
- return `int action_max` the max range of continuous action. action_max=1 when it is discrete action
- return `int max_step` the max step of an episode 
- return `bool if_discrete` discrete or continuous action space
- return `float target_reward` the gold score of this environment
- return `array neg_state_avg` for state normalization
- return `array div_state_std` for state normalization



## Name list of ElegantRL (plan to)

enviroment
- `str env_name` the environment name. Such as `'CartPole-v0', 'LunarLander-v2'`
- `int state_dim` the dimension of state (the number of state vector)
- `int action_dim` the dimension of action (the number of discrete action)
- `int max_step` the max step of an episode. The actor will break this episode of environment exploration when `done=Ture` or `steps > max_step`. 
- `bool if_discrete` if swith to di

training 
- `int net_dim` the dimension of networks (the width of neural networks)

other
- `bool if_xxx` a Boolean value. Call it a `flag` in English?
- `bool if_on_policy` it shows that it is an on-policy algorithm.