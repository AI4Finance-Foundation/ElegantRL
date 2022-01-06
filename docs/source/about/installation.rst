Algorithm Implementation
=================================

ElegantRL implements DRL algorithms as different agents, where each agent follows a hierarchy from the base class ``AgentBase``. Such an hierarchical implementation effectively reduces code duplication, making our implementations lightweight. 

Here, we describe DQN-series algorithms. 


DQN-series
--------------------------------------------

.. image:: ../images/overview_2.png
   :alt: Fig. 2

As shown in the figure, the inheritance hierarchy of the DQN-series algorithms is as follows: 
  
  - **AgentBase**: the base class.
  
  - **AgentDQN**: a standard DQN agent.
  
  - **AgentDoubleDQN**: a Double-DQN agent with two Q-Nets for reducing overestimation, inheriting from AgentDQN.
  
.. code-block:: python
   
    class AgentBase:
      def init(net_dim, state_dim, action_dim, reward_scale, gamma,
             learning_rate, if_per_or_gae, env_num, gpu_id);
      def select_action(state);
      def select_actions(state);
      def explore_one_env(env, target_step);
      def explore_vec_env(env, target_step);
      def update_net(buffer, batch_size, repeat_times, soft_update_tau);
      def optim_update(optimizer, objective);
      def soft_update(target_net, current_net, tau);
      def save_or_load_agent(cwd, if_save);
      def convert_trajectory(traj_list);
   
    class AgentDQN(AgentBase):
      def init(net_dim, state_dim, action_dim, reward_scale, gamma,
             learning_rate, if_per_or_gae, env_num, gpu_id);
      def select_actions(states);
      def explore_one_env(env, target_step);
      def explore_vec_env(env, target_step);
      def update_net(buffer, batch_size, repeat_times, soft_update_tau);
      def get_obj_critic_raw(buffer, batch_size);
      def get_obj_critic_per(buffer, batch_size);

    class AgentDoubleDQN(AgentDQN):
      def init(net_dim, state_dim, action_dim, reward_scale, gamma,
             learning_rate, if_per_or_gae, env_num, gpu_id);
      def select_actions(states);
      def get_obj_critic_raw(buffer, batch_size);
      def get_obj_critic_per(buffer, batch_size);
   
