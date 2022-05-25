import gym
import numpy as np


class PendulumEnv(gym.Wrapper):  # a demo of custom gym env
    def __init__(self, gym_env_name=None):
        gym.logger.set_level(40)  # Block warning
        if gym_env_name is None:
            gym_env_name = "Pendulum-v0"  if gym.__version__ < '0.18.0' else "Pendulum-v1"
        super().__init__(env=gym.make(gym_env_name) )

        self.env_name = gym_env_name  # the name of this env.
        self.max_step = getattr(self.env, '_max_episode_steps', None)  # the max step of an episode
        self.state_dim = self.observation_space.shape[0]  # feature number of state
        self.action_dim = self.action_space.shape[0]  # feature number of action
        self.if_discrete = False  # discrete action or continuous action

    def reset(self):
        return self.env.reset().astype(np.float32)

    def step(self, action: np.ndarray):
        # OpenAI Pendulum env set its action space as (-2, +2). It is bad.
        # We suggest that adjust action space to (-1, +1) when designing your own env.
        state, reward, done, info_dict = self.env.step(action * 2)
        return state.astype(np.float32), reward, done, info_dict


def get_gym_env_args(env, if_print) -> dict:  # [ElegantRL.2021.12.12]
    """Get a dict ``env_args`` about a standard OpenAI gym env information.

    :param env: a standard OpenAI gym env
    :param if_print: [bool] print the dict about env information.
    :return: env_args [dict]

    env_args = {
        'env_num': 1,               # [int] the environment number, 'env_num>1' in vectorized env
        'env_name': env_name,       # [str] the environment name, such as XxxXxx-v0
        'max_step': max_step,       # [int] the steps in an episode. (from env.reset to done).
        'state_dim': state_dim,     # [int] the dimension of state
        'action_dim': action_dim,   # [int] the dimension of action or the number of discrete action
        'if_discrete': if_discrete, # [bool] action space is discrete or continuous
    }
    """

    env_num = getattr(env, 'env_num') if hasattr(env, 'env_num') else 1

    if {'unwrapped', 'observation_space', 'action_space', 'spec'}.issubset(dir(env)):  # isinstance(env, gym.Env):
        env_name = getattr(env, 'env_name', None)
        env_name = env.unwrapped.spec.id if env_name is None else env_name

        state_shape = env.observation_space.shape
        state_dim = state_shape[0] if len(state_shape) == 1 else state_shape  # sometimes state_dim is a list

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
            if any(env.action_space.high - 1):
                print('WARNING: env.action_space.high', env.action_space.high)
            if any(env.action_space.low + 1):
                print('WARNING: env.action_space.low', env.action_space.low)
        else:
            raise RuntimeError('\n| Error in get_gym_env_info()'
                               '\n  Please set these value manually: if_discrete=bool, action_dim=int.'
                               '\n  And keep action_space in (-1, 1).')
    else:
        env_name = env.env_name
        max_step = env.max_step
        state_dim = env.state_dim
        action_dim = env.action_dim
        if_discrete = env.if_discrete

    env_args = {'env_num': env_num,
                'env_name': env_name,
                'max_step': max_step,
                'state_dim': state_dim,
                'action_dim': action_dim,
                'if_discrete': if_discrete, }
    if if_print:
        env_args_repr = repr(env_args)
        env_args_repr = env_args_repr.replace(',', f",\n{' ':11}").replace('{', "{")
        print(f"env_args = {env_args_repr}")
    return env_args


def build_env(env_func=None, env_args=None):
    if env_func.__module__ == 'gym.envs.registration':  # special rule
        import gym
        gym.logger.set_level(40)  # Block warning
        env = env_func(id=env_args['env_name'])
    else:
        def kwargs_filter(func, kwargs: dict):
            import inspect

            sign = inspect.signature(func).parameters.values()
            sign = {val.name for val in sign}

            common_args = sign.intersection(kwargs.keys())
            return {key: kwargs[key] for key in common_args}  # filtered kwargs

        env = env_func(**kwargs_filter(env_func.__init__, env_args.copy()))
    for attr_str in ('state_dim', 'action_dim', 'max_step', 'if_discrete'):
        setattr(env, attr_str, env_args[attr_str])
    return env
