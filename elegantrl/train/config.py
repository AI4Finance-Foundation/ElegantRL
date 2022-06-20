import os
import torch
import numpy as np
from copy import deepcopy
from pprint import pprint

'''config for agent'''


class Arguments:
    def __init__(self, agent, env=None, env_func=None, env_args=None):
        self.env = env  # the environment for training
        self.env_func = env_func  # env = env_func(*env_args)
        self.env_args = env_args  # env = env_func(*env_args)
        self.if_Isaac = True

        self.env_num = self.update_attr('env_num')  # env_num = 1. In vector env, env_num > 1.
        self.max_step = self.update_attr('max_step')  # the max step of an episode
        self.env_name = self.update_attr('env_name')  # the env name. Be used to set 'cwd'.
        self.state_dim = self.update_attr('state_dim')  # vector dimension (feature number) of state
        self.action_dim = self.update_attr('action_dim')  # vector dimension (feature number) of action
        self.if_discrete = self.update_attr('if_discrete')  # discrete or continuous action space
        self.target_return = self.update_attr('target_return')  # target average episode return
        self.target_step = 1e8

        self.agent = agent  # DRL algorithm
        self.net_dim = 2 ** 7  # the network width
        self.layer_num = 2  # layer number of MLP (Multi-layer perception, `assert layer_num>=2`)
        self.if_off_policy = self.get_if_off_policy()  # agent is on-policy or off-policy
        self.if_use_old_traj = True  # save old data to splice and get a complete trajectory (for vector env)
        self.if_act_target = False
        self.if_cri_target = True
        if self.if_off_policy:  # off-policy
            self.replay_buffer_size = 1e6  # capacity of replay buffer
            self.horizon_len = 1  # repeatedly update network to keep critic's loss small
            self.batch_size = self.net_dim  # num of transitions sampled from replay buffer.
            self.repeat_times = 2 ** 0  # collect horizon_len, then update network
            self.if_use_per = False  # use PER (Prioritized Experience Replay) for sparse reward
            self.num_seed_steps = 2
            self.num_steps_per_episode = 128 
        else:  # on-policy
            self.replay_buffer_size = 2 ** 12  # capacity of replay buffer
            self.horizon_len = self.replay_buffer_size  # repeatedly update network to keep critic's loss small
            self.batch_size = self.net_dim * 2  # num of transitions sampled from replay buffer.
            self.repeat_times = 2 ** 4  # collect horizon_len, then update network
            self.if_use_gae = True  # use PER: GAE (Generalized Advantage Estimation) for sparse reward

        '''Arguments for training'''
        self.gamma = 0.99  # discount factor of future rewards
        self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256
        self.learning_rate = 2 ** -12  # 2 ** -15 ~= 3e-5
        self.soft_update_tau = 2 ** -8  # 2 ** -8 ~= 5e-3

        '''Arguments for device'''
        self.worker_num = 1  # rollout workers number pre GPU (adjust it to get high GPU usage)
        self.thread_num = 8  # cpu_num for pytorch, `torch.set_num_threads(self.num_threads)`
        self.random_seed = 0  # initialize random seed in self.init_before_training()
        self.learner_gpus = 0  # `int` means the ID of single GPU, -1 means CPU

        '''Arguments for evaluate'''
        self.cwd = None  # current working directory to save model. None means set automatically
        self.if_remove = True  # remove the cwd folder? (True, False, None:ask me)
        self.break_step = +np.inf  # break training if 'total_step > break_step'
        self.if_over_write = False  # over write the best policy network (actor.pth)
        self.if_allow_break = True  # allow break training when reach goal (early termination)

        '''Arguments for evaluate'''
        self.tracker_len = 100 # todo
        self.eval_gap = 1e6  # evaluate the agent per eval_gap seconds
        self.reevaluate = False
        self.eval_times = 2 ** 4  # number of times that get episode return
        
    def init_before_training(self):
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.set_num_threads(self.thread_num)
        torch.set_default_dtype(torch.float32)

        '''auto set'''
        if self.cwd is None:
            self.cwd = f'./result/{self.env_name}_{self.agent.__name__[5:]}_{self.random_seed}'

        '''remove history'''
        if self.if_remove is None:
            self.if_remove = bool(input(f"| Arguments PRESS 'y' to REMOVE: {self.cwd}? ") == 'y')
        elif self.if_remove:
            import shutil
            shutil.rmtree(self.cwd, ignore_errors=True)
            print(f"| Arguments Remove cwd: {self.cwd}")
        else:
            print(f"| Arguments Keep cwd: {self.cwd}")
        os.makedirs(self.cwd, exist_ok=True)

    def update_attr(self, attr: str):
        return getattr(self.env, attr) if self.env_args is None else self.env_args[attr]

    def get_if_off_policy(self):
        name = self.agent.__name__
        return all((name.find('PPO') == -1, name.find('A2C') == -1))  # if_off_policy

    def print(self):
        # prints out args in a neat, readable format
        pprint(vars(self))


'''config for env(simulator)'''


def get_gym_env_args(env, if_print) -> dict:  # [ElegantRL.2021.12.12]
    """get a dict `env_args` about a standard OpenAI gym env information.

    env_args = {
        'env_num': 1,
        'env_name': env_name,            # [str] the environment name, such as XxxXxx-v0
        'max_step': max_step,            # [int] the steps in an episode. (from env.reset to done).
        'state_dim': state_dim,          # [int] the dimension of state
        'action_dim': action_dim,        # [int] the dimension of action
        'if_discrete': if_discrete,      # [bool] action space is discrete or continuous
        'target_return': target_return,  # [float] We train agent to reach this target episode return.
    }

    :param env: a standard OpenAI gym env
    :param if_print: [bool] print the dict about env inforamtion.
    :return: env_args [dict]
    """
    import gym

    env_num = getattr(env, 'env_num') if hasattr(env, 'env_num') else 1
    target_return = getattr(env, 'target_return') if hasattr(env, 'target_return') else +np.inf

    if {'unwrapped', 'observation_space', 'action_space', 'spec'}.issubset(dir(env)):  # isinstance(env, gym.Env):
        env_name = getattr(env, 'env_name', None)
        env_name = env.unwrapped.spec.id if env_name is None else env_name

        state_shape = env.observation_space.shape
        state_dim = state_shape[0] if len(state_shape) == 1 else state_shape  # sometimes state_dim is a list

        target_return = getattr(env, 'target_return', None)
        target_return_default = getattr(env.spec, 'reward_threshold', None)
        if target_return is None:
            target_return = target_return_default
        if target_return is None:
            target_return = 2 ** 16

        max_step = getattr(env, 'max_step', None)
        max_step_default = getattr(env, '_max_episode_steps', None)
        if max_step is None:
            max_step = max_step_default
        if max_step is None:
            max_step = 2 ** 10

        if_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        if if_discrete:  # make sure it is discrete action space
            action_dim = env.action_space.n
        elif isinstance(env.action_space, gym.spaces.Box):  # make sure it is continuous action space
            action_dim = env.action_space.shape[0]
            if not any(env.action_space.high - 1):
                print('WARNING: env.action_space.high', env.action_space.high)
            if not any(env.action_space.low - 1):
                print('WARNING: env.action_space.low', env.action_space.low)
        else:
            raise RuntimeError('\n| Error in get_gym_env_info()'
                               '\n  Please set these value manually: if_discrete=bool, action_dim=int.'
                               '\n  And keep action_space in (-1, 1).')
    else:
        env_name = getattr(env, 'env_num', env_num)
        max_step = env.max_step
        state_dim = env.state_dim
        action_dim = env.action_dim
        if_discrete = env.if_discrete
        target_return = getattr(env, 'target_return', target_return)

    env_args = {'env_num': env_num,
                'env_name': env_name,
                'max_step': max_step,
                'state_dim': state_dim,
                'action_dim': action_dim,
                'if_discrete': if_discrete,
                'target_return': target_return, }
    if if_print:
        env_args_repr = repr(env_args)
        env_args_repr = env_args_repr.replace(',', f",\n   ")
        env_args_repr = env_args_repr.replace('{', "{\n    ")
        env_args_repr = env_args_repr.replace('}', ",\n}")
        print(f"env_args = {env_args_repr}")
    return env_args


def kwargs_filter(func, kwargs: dict):  # [ElegantRL.2021.12.12]
    import inspect

    sign = inspect.signature(func).parameters.values()
    sign = set([val.name for val in sign])

    common_args = sign.intersection(kwargs.keys())
    filtered_kwargs = {key: kwargs[key] for key in common_args}
    return filtered_kwargs


def build_env(env=None, env_func=None, env_args=None):  # [ElegantRL.2021.12.12]
    if env is not None:
        env = deepcopy(env)
    elif env_func.__module__ == 'gym.envs.registration':
        import gym
        gym.logger.set_level(40)  # Block warning
        env = env_func(id=env_args['env_name'])
    else:
        env = env_func(**kwargs_filter(env_func.__init__, env_args.copy()))

    for attr_str in ('state_dim', 'action_dim', 'max_step', 'if_discrete', 'target_return'):
        if (not hasattr(env, attr_str)) and (attr_str in env_args):
            setattr(env, attr_str, env_args[attr_str])
    # env.max_step = env.max_step if hasattr(env, 'max_step') else env_args['max_step']
    # env.if_discrete = env.if_discrete if hasattr(env, 'if_discrete') else env_args['if_discrete']
    return env
