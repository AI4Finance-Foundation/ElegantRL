import sys
import gym

from elegantrl.train.run import train_and_evaluate, train_and_evaluate_mp
from elegantrl.train.config import Arguments
from elegantrl.agents.AgentA2C import AgentA2C
from elegantrl.agents.AgentPPO import AgentPPO, AgentPPOHterm
from elegantrl.envs.CustomGymEnv import GymNormaEnv


def demo_a2c_ppo(gpu_id, drl_id, env_id):
    env_name = ['Pendulum-v0',
                'Pendulum-v1',
                'LunarLanderContinuous-v2',
                'BipedalWalker-v3',
                'Hopper-v2',
                'HalfCheetah-v3',
                'Humanoid-v3', ][env_id]
    agent_class = [AgentA2C, AgentPPO][drl_id]

    if env_name in {'Pendulum-v0', 'Pendulum-v1'}:
        from elegantrl.envs.CustomGymEnv import PendulumEnv
        env = PendulumEnv(env_name, target_return=-500)
        "TotalStep: 1e5, TargetReward: -200, UsedTime: 600s"
        args = Arguments(agent_class, env=env)
        args.reward_scale = 2 ** -1  # RewardRange: -1800 < -200 < -50 < 0
        args.gamma = 0.97
        args.target_step = args.max_step * 8
        # args.eval_times = 2 ** 3

        args.gamma = 0.95
        args.target_step = args.max_step * 16
        args.net_dim = 2 ** 7
        args.eval_times = 2 ** 6
        args.break_step = int(6e5)
        args.if_allow_break = False
    elif env_name == 'LunarLanderContinuous-v2':
        env_func = gym.make
        env_args = {'env_num': 1,
                    'env_name': 'LunarLanderContinuous-v2',
                    'max_step': 1000,
                    'state_dim': 8,
                    'action_dim': 2,
                    'if_discrete': False,
                    'target_return': 200,

                    'id': 'LunarLanderContinuous-v2'}
        args = Arguments(agent_class, env_func=env_func, env_args=env_args)

        args.target_step = args.max_step * 2
        args.reward_scale = 2 ** -1
        args.gamma = 0.99

        args.net_dim = 2 ** 7
        args.num_layer = 3
        args.batch_size = int(args.net_dim * 2)
        args.repeat_times = 2 ** 4

        args.eval_times = 2 ** 5

        args.lambda_h_term = 2 ** -5
    elif env_name == 'BipedalWalker-v3':
        env_func = gym.make
        env_args = {'env_num': 1,
                    'env_name': 'BipedalWalker-v3',
                    'max_step': 1600,
                    'state_dim': 24,
                    'action_dim': 4,
                    'if_discrete': False,
                    'target_return': 300, }
        args = Arguments(agent_class, env_func=env_func, env_args=env_args)

        args.gamma = 0.98
        args.eval_times = 2 ** 4
        args.reward_scale = 2 ** -1

        args.target_step = args.max_step * 4
        args.worker_num = 2
        args.net_dim = 2 ** 7
        args.num_layer = 3
        args.batch_size = int(args.net_dim * 2)
        args.repeat_times = 2 ** 4
        args.ratio_clip = 0.25
        args.lambda_gae_adv = 0.96
        args.lambda_entropy = 0.02
        args.if_use_gae = True

        args.lambda_h_term = 2 ** -5
    elif env_name == 'Hopper-v2':
        env_func = gym.make
        env_args = {
            'env_num': 1,
            'env_name': 'Hopper-v2',
            'max_step': 1000,
            'state_dim': 11,
            'action_dim': 3,
            'if_discrete': False,
            'target_return': 3800.,
        }
        args = Arguments(agent_class, env_func=env_func, env_args=env_args)
        args.eval_times = 2 ** 2
        args.reward_scale = 2 ** -4

        args.target_step = args.max_step * 4  # 6
        args.worker_num = 2

        args.net_dim = 2 ** 7
        args.num_layer = 3
        args.batch_size = int(args.net_dim * 2)
        args.repeat_times = 2 ** 4
        args.ratio_clip = 0.1

        args.gamma = 0.98
        args.lambda_gae_adv = 0.92
        args.if_use_gae = True
        args.lambda_entropy = 2 ** -8
        args.lambda_h_term = 2 ** -5

        args.if_allow_break = False
        args.break_step = int(4e6)
    elif env_name == 'HalfCheetah-v3':
        env_func = gym.make
        env_args = {
            'env_num': 1,
            'env_name': 'HalfCheetah-v3',
            'max_step': 1000,
            'state_dim': 17,
            'action_dim': 6,
            'if_discrete': False,
            'target_return': 4800.0,
        }

        args = Arguments(agent_class, env_func=env_func, env_args=env_args)
        args.eval_times = 2 ** 2
        args.reward_scale = 2 ** -3

        args.target_step = args.max_step * 4
        args.worker_num = 2

        args.net_dim = 2 ** 8
        args.num_layer = 3
        args.batch_size = int(args.net_dim * 2)
        args.repeat_times = 2 ** 4
        args.ratio_clip = 0.3
        args.gamma = 0.99
        args.lambda_gae_adv = 0.96
        args.if_use_gae = True
        args.clip_grad_norm = 0.8

        args.lambda_entropy = 2 ** -6
        # args.lambda_h_term = 2 ** -5

        args.if_allow_break = False
        args.break_step = int(8e6)

    elif env_name == 'Humanoid-v3':
        from elegantrl.envs.CustomGymEnv import HumanoidEnv
        env_func = HumanoidEnv
        env_args = {
            'env_num': 1,
            'env_name': 'Humanoid-v3',
            'max_step': 1000,
            'state_dim': 376,
            'action_dim': 17,
            'if_discrete': False,
            'target_return': 8000.,
        }
        args = Arguments(agent_class, env_func=env_func, env_args=env_args)
        args.reward_scale = 2 ** -4

        args.if_cri_target = False

        args.target_step = args.max_step * 8
        args.worker_num = 4
        args.net_dim = 2 ** 8
        args.batch_size = args.net_dim * 2
        args.repeat_times = 2 ** 5
        args.gamma = 0.985  # important
        args.lambda_gae_adv = 0.93
        args.if_use_gae = True
        args.learning_rate = 2 ** -15

        args.net_dim = 2 ** 9
        args.batch_size = args.net_dim * 2
        args.target_step = args.max_step * 4  # todo
        args.repeat_times = 2 ** 4

        args.lambda_entropy = 0.01

        args.eval_gap = 2 ** 9
        args.eval_times = 2 ** 2
        args.break_step = int(4e7)
        from elegantrl.envs.CustomGymEnv import HumanoidEnv
        env_func = HumanoidEnv
        env_args = {
            'env_num': 1,
            'env_name': 'Humanoid-v3',
            'max_step': 1000,
            'state_dim': 376,
            'action_dim': 17,
            'if_discrete': False,
            'target_return': 5000.,
        }
        args = Arguments(agent_class, env_func=env_func, env_args=env_args)
        args.reward_scale = 2 ** -5

        args.learning_rate = 2 ** -15
        args.num_layer = 3
        args.net_dim = 2 ** 9  # todo
        args.target_step = args.max_step * 8
        args.worker_num = 4
        args.batch_size = args.net_dim * 2
        args.repeat_times = 2 ** 5
        args.gamma = 0.995  # important
        args.if_use_gae = True
        args.lambda_gae_adv = 0.98
        args.lambda_entropy = 0.01

        args.eval_times = 2 ** 2
        args.break_step = int(5e7)
    else:
        raise ValueError('env_name:', env_name)

    args.learner_gpus = gpu_id
    args.random_seed += gpu_id

    if_check = 0
    if if_check:
        train_and_evaluate(args)
    else:
        train_and_evaluate_mp(args)


def demo_ppo_h_term(gpu_id, drl_id, env_id):
    env_name = ['Hopper-v3',
                'Swimmer-v3',
                'Ant-v3',
                'Humanoid-v3',
                'HalfCheetah-v3',
                'Walker2d-v3',
                ][env_id]
    agent_class = [AgentA2C, AgentPPO, AgentPPOHterm][drl_id]
    # from elegantrl.train.config import get_gym_env_args
    # get_gym_env_args(gym.make(env_name), if_print=True)
    # exit()

    if env_name == 'Hopper-v3':
        env_func = GymNormaEnv  # gym.make
        env_args = {
            'env_num': 1,
            'env_name': 'Hopper-v3',
            'max_step': 1000,
            'state_dim': 11,
            'action_dim': 3,
            'if_discrete': False,
            'target_return': 3500.,
        }
        args = Arguments(agent_class, env_func=env_func, env_args=env_args)

        args.num_layer = 3
        args.net_dim = 2 ** 8
        args.batch_size = int(args.net_dim * 2)

        args.worker_num = 4
        args.target_step = args.max_step * 1
        args.repeat_times = 2 ** 4

        args.learning_rate = 2 ** -15
        args.clip_grad_norm = 1.0
        args.reward_scale = 2 ** -4
        args.gamma = 0.993

        args.lambda_gae_adv = 0.95
        args.lambda_entropy = 0.05
        args.if_use_gae = True
        args.ratio_clip = 0.20

        '''H-term'''
        args.h_term_drop_rate = 2 ** -2
        args.h_term_lambda = 2 ** -3
        args.act_update_gap = 1
        args.h_term_k_step = 64  # 16

        args.eval_times = 2 ** 1
        args.eval_gap = 2 ** 8
        args.if_allow_break = False
        args.break_step = int(2e6)
    elif env_name == 'Swimmer-v3':

        # env_func = GymNormaEnv  # gym.make
        env_func = gym.make
        env_args = {
            'action_dim': 2,
            'env_name': 'Swimmer-v3',
            'env_num': 1,
            'if_discrete': False,
            'max_step': 1000,
            'state_dim': 8,
            'target_return': 360.0
        }
        args = Arguments(agent_class, env_func=env_func, env_args=env_args)

        args.num_layer = 3
        args.net_dim = 2 ** 8
        args.batch_size = int(args.net_dim * 2)

        args.worker_num = 4
        args.target_step = args.max_step * 1
        args.repeat_times = 2 ** 4

        args.learning_rate = 2 ** -14
        args.clip_grad_norm = 3.0
        args.reward_scale = 2 ** -1
        args.gamma = 0.993

        args.lambda_gae_adv = 0.95
        args.lambda_entropy = 0.05
        args.if_use_gae = True
        args.ratio_clip = 0.20

        if gpu_id == 1:
            args.repeat_times = 2 ** 2
        if gpu_id == 2:
            args.lambda_entropy = 2 ** 0
        if gpu_id == 3:
            args.repeat_times = 2 ** 2  # HK

        '''H-term'''
        args.h_term_drop_rate = 2 ** -2
        args.h_term_lambda = 2 ** -3
        args.act_update_gap = 1
        args.h_term_k_step = 64  # 16

        args.eval_times = 2 ** 1
        args.eval_gap = 2 ** 8
        args.if_allow_break = False
    elif env_name == 'Ant-v3':
        env_func = GymNormaEnv  # gym.make
        # env_func = gym.make
        env_args = {
            'env_num': 1,
            'env_name': 'Ant-v3',
            'max_step': 1000,
            'state_dim': 111,
            'action_dim': 8,
            'if_discrete': False,
            'target_return': 6000.0,
        }

        args = Arguments(agent_class, env_func=env_func, env_args=env_args)
        args.reward_scale = 2 ** -4

        args.num_layer = 3
        args.net_dim = 2 ** 8
        args.batch_size = int(args.net_dim * 2)

        args.worker_num = 4
        args.target_step = args.max_step * 1
        args.repeat_times = 2 ** 5
        args.reward_scale = 2 ** -4

        args.learning_rate = 2 ** -15
        args.clip_grad_norm = 0.6
        args.gamma = 0.99

        if gpu_id == 5:
            args.clip_grad_norm = 2.0
        if gpu_id == 2:
            args.clip_grad_norm = 2.0  # HK
        if gpu_id == 3:
            args.clip_grad_norm = 0.6  # HK

        args.lambda_gae_adv = 0.94
        args.lambda_entropy = 2 ** -6
        args.if_use_gae = True
        args.ratio_clip = 0.3

        '''H-term'''
        args.h_term_drop_rate = 2 ** -3
        args.h_term_lambda = 2 ** -3
        args.h_term_k_step = 12

        args.save_gap = 2 ** 4
        args.eval_gap = 2 ** 9
        args.eval_times = 2 ** 1
        args.break_step = int(2e7)
        args.if_allow_break = False
    elif env_name == 'Humanoid-v3':
        from elegantrl.envs.CustomGymEnv import HumanoidEnv
        env_func = HumanoidEnv
        env_args = {
            'env_num': 1,
            'env_name': 'Humanoid-v3',
            'max_step': 1000,
            'state_dim': 376,
            'action_dim': 17,
            'if_discrete': False,
            'target_return': 8000.,
        }
        args = Arguments(agent_class, env_func=env_func, env_args=env_args)

        args.num_layer = 3
        args.net_dim = 2 ** 9  #
        args.batch_size = int(args.net_dim * 2)

        args.worker_num = 8
        args.target_step = args.max_step * 3
        args.repeat_times = 2 ** 4  #

        args.learning_rate = 2 ** -15
        args.clip_grad_norm = 1.0
        args.reward_scale = 2 ** -3
        args.gamma = 0.985

        args.lambda_gae_adv = 0.94
        args.lambda_entropy = 2 ** -2  # important
        args.if_use_gae = True
        args.ratio_clip = 0.30

        '''H-term'''
        args.h_term_drop_rate = 2 ** -2
        args.h_term_lambda = 2 ** -3
        args.h_term_k_step = 12  # 4, 16

        args.save_gap = 2 ** 3
        args.eval_gap = 2 ** 9
        args.eval_times = 2 ** 2
        args.if_allow_break = False
        args.break_step = int(2e8)
    elif env_name == 'HalfCheetah-v3':
        env_func = GymNormaEnv
        # env_func = gym.make
        env_args = {
            'env_num': 1,
            'env_name': 'HalfCheetah-v3',
            'max_step': 1000,
            'state_dim': 17,
            'action_dim': 6,
            'if_discrete': False,
            'target_return': 4800.0,
        }
        args = Arguments(agent_class, env_func=env_func, env_args=env_args)

        args.num_layer = 3
        args.net_dim = 2 ** 7
        args.batch_size = int(args.net_dim * 2)

        args.worker_num = 4
        args.target_step = args.max_step * 1
        args.repeat_times = 2 ** 5

        args.learning_rate = 2 ** -15
        args.clip_grad_norm = 0.8
        args.reward_scale = 2 ** -5
        args.gamma = 0.99

        args.lambda_gae_adv = 0.94
        args.lambda_entropy = 2 ** -9
        args.if_use_gae = True
        args.ratio_clip = 0.3

        '''H-term'''
        args.h_term_drop_rate = 2 ** -2
        args.h_term_lambda = 2 ** -3
        args.h_term_k_step = 4

        args.eval_gap = 2 ** 8
        args.eval_times = 2 ** 1
        args.if_allow_break = False
        args.break_step = int(2e7)

    elif env_name == 'Walker2d-v3':
        env_func = GymNormaEnv
        # env_func = gym.make
        env_args = {
            'env_num': 1,
            'env_name': 'Walker2d-v3',
            'if_discrete': False,
            'max_step': 1000,
            'state_dim': 17,
            'action_dim': 6,
            'target_return': 65536
        }
        args = Arguments(agent_class, env_func=env_func, env_args=env_args)

        args.num_layer = 3
        args.net_dim = 2 ** 8
        args.batch_size = int(args.net_dim * 2)

        args.worker_num = 4
        args.target_step = args.max_step * 2
        args.repeat_times = 2 ** 5
        args.reward_scale = 2 ** -3

        args.learning_rate = 2 ** -15
        args.clip_grad_norm = 2.0
        args.gamma = 0.985

        args.lambda_gae_adv = 0.94
        args.if_use_gae = True
        args.ratio_clip = 0.4
        args.lambda_entropy = 2 ** -4
        if gpu_id == 2:
            args.ratio_clip = 0.6
        if gpu_id == 3:
            args.ratio_clip = 0.4

        '''H-term'''
        args.h_term_sample_rate = 2 ** -2
        args.h_term_drop_rate = 2 ** -3
        args.h_term_lambda = 2 ** -3
        args.h_term_k_step = 12

        args.save_gap = 2 ** 3
        args.eval_gap = 2 ** 8
        args.eval_times = 2 ** 1
        args.if_allow_break = False
        args.break_step = int(2e7)

    else:
        raise ValueError('env_name:', env_name)

    args.learner_gpus = gpu_id
    args.random_seed += gpu_id + 19431

    if_check = 0
    if if_check:
        train_and_evaluate(args)
    else:
        train_and_evaluate_mp(args)


if __name__ == '__main__':
    GPU_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # >=0 means GPU ID, -1 means CPU
    DRL_ID = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    ENV_ID = int(sys.argv[3]) if len(sys.argv) > 3 else 2

    # demo_ppo_h_term(GPU_ID, DRL_ID, ENV_ID)
    demo_a2c_ppo(GPU_ID, DRL_ID, ENV_ID)

    from elegantrl.train.evaluator import demo_load_pendulum_and_render

    demo_load_pendulum_and_render()
