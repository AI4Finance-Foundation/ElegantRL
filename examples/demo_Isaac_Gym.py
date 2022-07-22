import isaacgym
import torch
import sys
import wandb

from elegantrl.train.run import train_and_evaluate
from elegantrl.train.config import Arguments, build_env
from elegantrl.agents.AgentPPO import AgentPPO
from elegantrl.envs.IsaacGym import IsaacVecEnv, IsaacOneEnv


def demo(seed, config):
    agent_class = AgentPPO
    env_func = IsaacVecEnv
    gpu_id = 0

    env_args = {
        'env_num': config['env_num'],
        'env_name': config['env_name'],
        'max_step': config['max_step'],
        'state_dim': config['state_dim'],
        'action_dim': config['action_dim'],
        'if_discrete': False,
        'target_return': 10000.,
        'sim_device_id': gpu_id,
        'rl_device_id': gpu_id,
    }
    env = build_env(env_func=env_func, env_args=env_args)
    args = Arguments(agent_class, env=env)
    args.if_Isaac = True
    args.if_use_old_traj = True
    args.if_use_gae = True
    args.obs_norm = True
    args.value_norm = False

    args.reward_scale = config['reward_scale']
    args.horizon_len = config['horizon_len']
    args.batch_size = config['batch_size']
    args.repeat_times = 5
    args.gamma = 0.99
    args.lambda_gae_adv = 0.95
    args.learning_rate = 5e-4
    args.lambda_entropy = 0.0

    args.eval_gap = 1e6
    args.learner_gpus = gpu_id
    args.random_seed = seed
    args.cwd = f'./result/{args.env_name}_{args.agent_class.__name__[5:]}_{args.env_num}envs/{args.random_seed}'

    train_and_evaluate(args)


if __name__ == '__main__':
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    config = {
        'env_name': 'Ant',
        'env_num': 2048,
        'state_dim': 60,
        'action_dim': 8,
        'max_step': 1000,
        'reward_scale': 0.01,
        'horizon_len': 32,
        'batch_size': 16384,
    }
    # config = {
    #     'env_name': 'Humanoid',
    #     'env_num': 2048,
    #     'state_dim': 108,
    #     'action_dim': 21,
    #     'max_step': 1000,
    #     'reward_scale': 0.01,
    #     'horizon_len': 32,
    #     'batch_size': 16384,
    # }
    # config = {
    #     'env_name': 'ShadowHand',
    #     'env_num': 16384,
    #     'state_dim': 211,
    #     'action_dim': 20,
    #     'max_step': 600,
    #     'reward_scale': 0.01,
    #     'horizon_len': 8,
    #     'batch_size': 32768,
    # }
    # config = {
    #     'env_name': 'Anymal',
    #     'env_num': 4096,
    #     'state_dim': 48,
    #     'action_dim': 12,
    #     'max_step': 2500,
    #     'reward_scale': 1,
    #     'horizon_len': 32,
    #     'batch_size': 16384,
    # }
    # config = {
    #     'env_name': 'Ingenuity',
    #     'env_num': 4096,
    #     'state_dim': 13,
    #     'action_dim': 6,
    #     'max_step': 2000,
    #     'reward_scale': 1,
    #     'horizon_len': 16,
    #     'batch_size': 16384,
    # }
    cwd = config['env_name'] + '_PPO_' + str(seed)
    wandb.init(
        project=config['env_name'] + '_PPO_' + str(config['env_num']),
        entity=None,
        sync_tensorboard=True,
        config=config,
        name=cwd,
        monitor_gym=True,
        save_code=True,
    )
    config = wandb.config
    demo(seed, config)
