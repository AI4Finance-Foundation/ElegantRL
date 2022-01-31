.. _mappo:


MAPPO
==========

`Multi-Agent Proximal Policy Optimization (MAPPO) <https://arxiv.org/abs/2103.01955>`_ is a variant of PPO which is specialized for multi-agent settings. MAPPO achieves surprisingly strong performance in two popular multi-agent testbeds: the particle-world environments, the Starcraft multi-agent challenge.

-  Shared network parameter for all agents ✔️


MAPPO achieves strong results while exhibiting comparable sample efficiency. 

Code Snippet
------------

.. code-block:: python

   def ppo_update(self, sample, update_actor=True):

      share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
      value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
      adv_targ, available_actions_batch = sample

      old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
      adv_targ = check(adv_targ).to(**self.tpdv)
      value_preds_batch = check(value_preds_batch).to(**self.tpdv)
      return_batch = check(return_batch).to(**self.tpdv)
      active_masks_batch = check(active_masks_batch).to(**self.tpdv)

      # Reshape to do in a single forward pass for all steps
      values, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch,
                                                                           obs_batch, 
                                                                           rnn_states_batch, 
                                                                           rnn_states_critic_batch, 
                                                                           actions_batch, 
                                                                           masks_batch, 
                                                                           available_actions_batch,
                                                                           active_masks_batch)
      # actor update
      imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

      surr1 = imp_weights * adv_targ
      surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ


Parameters
---------------------

.. autoclass:: elegantrl.agents.AgentMAPPO.AgentMAPPO
   :members:   
.. _mappo_networks:
   
Networks
-------------

.. autoclass:: elegantrl.agents.net.ActorMAPPO
   :members:
   
.. autoclass:: elegantrl.agents.net.CriticMAPPO
   :members:

              
              
