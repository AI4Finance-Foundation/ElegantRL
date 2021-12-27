from elegantrl.IsaacGym import *

from elegantrl.train.config import Arguments
from elegantrl.train.run_parallel import train_and_evaluate_mp
from elegantrl.train.run_tutorial import train_and_evaluate
from elegantrl.agents.AgentPPO import AgentPPO, AgentModPPO


def demo_isaac_gym_on_policy1():
    agent = [AgentPPO(), AgentModPPO()][0]

    '''set env'''
    env_num = 1024
    target_return = 14000

    env = None
    env_class = PreprocessIsaacVecEnv
    env_args = {
        'env_name': 'Ant',
        'target_return': target_return,
        'if_print': False,
        'env_num': env_num,
        'device_id': 0,
        'rl_device_id': -1,  # plan to set it >=0
    }
    env_info = {'env_name': 'IsaacGymAnt',
                'env_num': env_num,
                'max_step': 1000,
                'state_dim': 60,
                'action_dim': 8,
                'if_discrete': False,
                'target_return': target_return, }

    '''hyper-parameters'''
    args = Arguments(agent=agent, env=env, env_func=env_class, env_args=env_args, env_info=env_info)

    args.if_per_or_gae = True
    args.net_dim = 2 ** 9
    args.batch_size = int(args.net_dim * 2)
    args.repeat_times = 2 ** 6
    args.reward_scale = 2 ** -1

    args.target_step = args.max_step * 2  # 4  IP 111 GPU 0

    args.worker_num = 1
    args.learner_gpus = (1,)
    args.eval_env = env
    args.eval_env_class = PreprocessIsaacOneEnv
    args.eval_env_args = env_args.copy()
    args.eval_env_args['env_num'] = 1
    args.eval_env_args['device_id'] = args.learner_gpus[0]

    if_use_mp = 1
    if if_use_mp:
        train_and_evaluate_mp(args)
    else:
        train_and_evaluate(args)


def demo_isaac_gym_on_policy():
    agent = [AgentPPO(), AgentModPPO()][0]

    '''set env'''
    env_num = 1024
    target_return = 14000

    env = None
    env_class = PreprocessIsaacVecEnv
    env_args = {
        'env_name': 'Ant',
        'target_return': target_return,
        'if_print': False,
        'env_num': env_num,
        'device_id': 0,
        'rl_device_id': -1,  # plan to set it >=0
    }
    env_info = {'env_name': 'IsaacGymAnt',
                'env_num': env_num,
                'max_step': 1000,
                'state_dim': 60,
                'action_dim': 8,
                'if_discrete': False,
                'target_return': target_return, }

    '''hyper-parameters'''
    args = Arguments(agent=agent, env=env, env_func=env_class, env_args=env_args, env_info=env_info)

    args.if_per_or_gae = True
    args.net_dim = 2 ** 9
    args.batch_size = int(args.net_dim * 2)
    args.repeat_times = 2 ** 4
    args.reward_scale = 2 ** -2

    args.target_step = args.max_step * 2  # 4  IP 111 GPU 0

    args.worker_num = 1
    args.learner_gpus = (1,)
    args.eval_gpu_id = args.learner_gpus[0]
    args.eval_env = env
    args.eval_env_class = PreprocessIsaacOneEnv
    args.eval_env_args = env_args.copy()
    args.eval_env_args['env_num'] = 1
    args.eval_env_args['device_id'] = args.eval_gpu_id

    train_and_evaluate_mp(args)


def check_isaac_gym():
    check_isaac_gym_vec_env_multiple_process()


if __name__ == '__main__':
    # check_isaac_gym()
    # exit()
    demo_isaac_gym_on_policy()
