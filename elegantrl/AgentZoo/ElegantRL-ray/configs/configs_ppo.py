from ray_elegantrl.agent import *

# Default config for on policy | ppo
max_step = 1024
rollout_num = 4
config = {
    'gpu_id': 0,
    'cwd': None,
    'if_cwd_time': True,
    'random_seed': 0,
    'env': {
        'id': 'HopperPyBulletEnv-v0',
        'state_dim': 15,
        'action_dim': 2,
        'if_discrete_action': False,
        'reward_dim': 1,
        'target_reward': 0,
        'max_step': max_step,
    },
    'agent': {
        'class_name': AgentPPO,
        'net_dim': 2 ** 8,
        'ratio_clip': 0.3,
        'lambda_entropy': 0.04,
        'lambda_gae_adv': 0.97,
        'if_use_gae': True,
        'if_use_dn': False,
        'learning_rate': 1e-4,
        'soft_update_tau': 2 ** -8,
    },
    'trainer': {
        'batch_size': 2 ** 8,
        'policy_reuse': 2 ** 4,
        'sample_step': max_step * rollout_num,
    },
    'interactor': {
        'horizon_step': max_step * rollout_num,
        'reward_scale': 2 ** 0,
        'gamma': 0.99,
        'rollout_num': rollout_num,
    },
    'buffer': {
        'max_buf': max_step * rollout_num,
        'if_on_policy': True,
        'if_per': False,
    },
    'evaluator': {
        'pre_eval_times': 2,  # for every rollout_worker 0 means cencle pre_eval
        'eval_times': 4,  # for every rollout_worker
        'if_save_model': True,
        'break_step': 2e6,
        'satisfy_reward_stop': False,
    }

}
