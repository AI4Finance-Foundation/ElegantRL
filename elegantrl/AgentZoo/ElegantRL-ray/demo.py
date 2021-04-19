from ray_elegantrl.interaction import beginer
import ray


def demo1_sac():
    from ray_elegantrl.configs.configs_modsac import config
    env = {
        'id': 'ReacherBulletEnv-v0',
        'state_dim': 9,
        'action_dim': 2,
        'if_discrete_action': False,
        'reward_dim': 1,
        'target_reward': 20,
        'max_step': 150,
    }
    config['interactor']['rollout_num'] = 4
    config['trainer']['sample_step'] = 1024  # env['max_step'] * config['interactor']['rollout_num']
    config['interactor']['horizon_step'] = config['trainer']['sample_step']
    config['interactor']['gamma'] = 0.99
    config['env'] = env
    config['gpu_id'] = 2
    config['if_cwd_time'] = True
    config['random_seed'] = 10087
    beginer(config)


def demo_carla_sac():
    from ray_elegantrl.configs.configs_modsac import config
    from gym_carla_feature.env_a.mp_config import params
    env = {
        'id': 'carla-v0',
        'state_dim': 42,
        'action_dim': 2,
        'if_discrete_action': False,
        'reward_dim': 1,
        'target_reward': 600,
        'max_step': 512,
        'params_name': {'params': params}
    }
    config['interactor']['rollout_num'] = 4
    config['trainer']['sample_step'] = env['max_step'] * config['interactor']['rollout_num']
    config['interactor']['horizon_step'] = config['trainer']['sample_step']
    config['interactor']['gamma'] = 0.99
    config['env'] = env
    config['gpu_id'] = 2
    config['if_cwd_time'] = True
    config['random_seed'] = 0
    beginer(config)


def demo1_ppo():
    from ray_elegantrl.configs.configs_ppo import config
    env = {
        'id': 'ReacherBulletEnv-v0',
        'state_dim': 9,
        'action_dim': 2,
        'if_discrete_action': False,
        'reward_dim': 1,
        'target_reward': 20,
        'max_step': 150,
    }
    config['agent']['lambda_entropy'] = 0.01
    config['agent']['lambda_gae_adv'] = 0.98
    config['interactor']['rollout_num'] = 4
    config['trainer']['sample_step'] = 4096  # env['max_step'] * config['interactor']['rollout_num']
    config['interactor']['horizon_step'] = config['trainer']['sample_step']
    config['trainer']['policy_reuse'] = 2 ** 4
    config['interactor']['gamma'] = 0.99
    config['buffer']['max_buf'] = config['interactor']['horizon_step']
    config['env'] = env
    config['gpu_id'] = 2
    config['if_cwd_time'] = True
    config['random_seed'] = 10087
    beginer(config)


def demo2_ppo():
    from ray_elegantrl.configs.configs_ppo import config

    env = {
        'id': 'AntBulletEnv-v0',
        'state_dim': 28,
        'action_dim': 8,
        'if_discrete_action': False,
        'reward_dim': 1,
        'target_reward': 2500,
        'max_step': 1000,
    }
    config['agent']['lambda_entropy'] = 0.05
    config['agent']['lambda_gae_adv'] = 0.97
    config['interactor']['rollout_num'] = 4
    config['trainer']['sample_step'] = 2 ** 11  # env['max_step'] * config['interactor']['rollout_num']
    config['trainer']['batch_size'] = 2 ** 11
    config['interactor']['horizon_step'] = config['trainer']['sample_step']
    config['trainer']['policy_reuse'] = 2 ** 3
    config['interactor']['gamma'] = 0.99
    config['evaluator']['break_step'] = int(8e6 * 8)
    config['buffer']['max_buf'] = config['interactor']['horizon_step']
    config['env'] = env
    config['gpu_id'] = 2
    config['if_cwd_time'] = True
    config['random_seed'] = 0
    beginer(config)


def demo_carla_ppo():
    from ray_elegantrl.configs.configs_ppo import config
    from gym_carla_feature.env_a.mp_config import params
    env = {
        'id': 'carla-v0',
        'state_dim': 42,
        'action_dim': 2,
        'if_discrete_action': False,
        'reward_dim': 1,
        'target_reward': 600,
        'max_step': 512,
        'params_name': {'params': params}
    }
    config['agent']['lambda_entropy'] = 0.05
    config['agent']['lambda_gae_adv'] = 0.97
    env['params_name']['params']['port'] = 2016
    config['interactor']['rollout_num'] = 6
    config['agent']['net_dim'] = 2 ** 9
    config['trainer']['batch_size'] = 2 ** 9
    config['trainer']['policy_reuse'] = 2 ** 3
    config['interactor']['gamma'] = 0.99
    config['trainer']['sample_step'] = 2 * env['max_step'] * config['interactor']['rollout_num']
    config['interactor']['horizon_step'] = config['trainer']['sample_step']
    config['env'] = env
    config['gpu_id'] = 2
    config['if_cwd_time'] = True
    config['random_seed'] = 0
    beginer(config)


if __name__ == '__main__':
    ray.init()
    # ray.init(local_mode=True)
    # ray.init(num_cpus=12,num_gpus=2)
    # ray.init(num_cpus=12, num_gpus=0)

    # demo1_sac()
    demo1_ppo()
    # demo2_ppo()
    # demo_carla_sac()
    # demo_carla_ppo()
