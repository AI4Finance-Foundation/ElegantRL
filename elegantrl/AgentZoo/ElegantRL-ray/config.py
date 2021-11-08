from ray_elegantrl.agent import *

# from gym_carla_feature.env_a.mp_config import params

# Default config for offpolicy
max_step = 1024
rollout_num = 4
default_config = {
    'gpu_id': 0,
    'cwd': None,
    'if_cwd_time': True,
    'random_seed': 0,
    'env': {
        'id': '',
        'state_dim': 0,
        'action_dim': 0,
        'if_discrete_action': False,
        'reward_dim': 1,
        'target_reward': 0,
        'max_step': max_step,
        # 'params_name': {'params': params}
    },
    'agent': {
        'class_name': AgentModSAC,
        'net_dim': 2 ** 8,
        'if_use_gae': True,  # for on policy ppo
    },
    'trianer': {
        'batch_size': 2 ** 8,
        'policy_reuse': 2 ** 0,
        'sample_step': max_step * rollout_num,
    },
    'interactor': {
        'horizon_step': max_step * rollout_num,
        'reward_scale': 2 ** 0,
        'gamma': 0.99,
        'rollout_num': rollout_num,
    },
    'buffer': {
        'max_buf': 2 ** 17,
        'if_off_policy': True,
        'if_per': False,  # for off policy
    },
    'evaluator': {
        'pre_eval_times': 1,  # for every rollout_worker 0 means cencle pre_eval
        'eval_times': 6,  # for every rollout_worker
        'if_save_model': True,
        'break_step': 2e6,
        'satisfy_reward_stop': False,
    }

}
