import sys
import gym

from elegantrl.train.run import train_and_evaluate, train_and_evaluate_mp
from elegantrl.train.config import Arguments
from elegantrl.agents.AgentPPO import AgentPPO, AgentPPOHterm
from elegantrl.envs.CustomGymEnv import GymNormaEnv



def demo_ppo_h_term(gpu_id, drl_id, env_id):
    env_name = ['Hopper-v3',
                'Swimmer-v3',
                'Ant-v3',
                'Humanoid-v3',
                'HalfCheetah-v3',
                'Walker2d-v3',
                ][env_id]
    agent_class = [AgentPPO, AgentPPOHterm][drl_id]


    if env_name == 'Hopper-v3':
        env_func = GymNormaEnv 
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
        args.repeat_times = 2 ** 2

        '''H-term'''
        args.h_term_drop_rate = 2 ** -2
        args.h_term_lambda = 2 ** -3
        args.act_update_gap = 1
        args.h_term_k_step = 64  # 16

        args.eval_times = 2 ** 1
        args.eval_gap = 2 ** 8
        args.if_allow_break = False
    elif env_name == 'Ant-v3':
        env_func = GymNormaEnv
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
        args.clip_grad_norm = 2.0
        
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
        from rl.envs.CustomGymEnv import HumanoidEnv
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
        args.lambda_entropy = 2 ** -2  
        args.if_use_gae = True
        args.ratio_clip = 0.30

        '''H-term'''
        args.h_term_drop_rate = 2 ** -2
        args.h_term_lambda = 2 ** -3
        args.h_term_k_step = 16 

        args.save_gap = 2 ** 3
        args.eval_gap = 2 ** 9
        args.eval_times = 2 ** 2
        args.if_allow_break = False
        args.break_step = int(2e8)
    elif env_name == 'HalfCheetah-v3':
        env_func = GymNormaEnv
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
    AGENT_ID = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    ENV_ID = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    demo_ppo_h_term(GPU_ID, AGENT_ID, ENV_ID)

