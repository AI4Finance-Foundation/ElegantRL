from copy import deepcopy

import gym

gym.logger.set_level(40)  # Block warning


def get_gym_env_args(env, if_print) -> dict:
    """
    Get a dict ``env_args`` about a standard OpenAI gym env information.

    'env_num': 1,

    'env_name': env_name,            # [str] the environment name, such as XxxXxx-v0

    'max_step': max_step,            # [int] the steps in an episode. (from env.reset to done).

    'state_dim': state_dim,          # [int] the dimension of state

    'action_dim': action_dim,        # [int] the dimension of action

    'if_discrete': if_discrete,      # [bool] action space is discrete or continuous

    'target_return': target_return,  # [float] We train agent to reach this target episode return.

    :param env: a standard OpenAI gym env.
    :param if_print: print the dict about env information.
    :return: a dict of env_args.
    """
    env_num = getattr(env, "env_num") if hasattr(env, "env_num") else 1

    if isinstance(env, gym.Env):
        env_name = getattr(env, "env_name", None)
        env_name = env.unwrapped.spec.id if env_name is None else env_name

        state_shape = env.observation_space.shape
        state_dim = (
            state_shape[0] if len(state_shape) == 1 else state_shape
        )  # sometimes state_dim is a list

        target_return = getattr(env, "target_return", None)
        target_return_default = getattr(env.spec, "reward_threshold", None)
        if target_return is None:
            target_return = target_return_default
        if target_return is None:
            target_return = 2**16

        max_step = getattr(env, "max_step", None)
        max_step_default = getattr(env, "_max_episode_steps", None)
        if max_step is None:
            max_step = max_step_default
        if max_step is None:
            max_step = 2**10

        if_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        if if_discrete:  # make sure it is discrete action space
            action_dim = env.action_space.n
        elif isinstance(
            env.action_space, gym.spaces.Box
        ):  # make sure it is continuous action space
            action_dim = env.action_space.shape[0]
            assert not any(env.action_space.high - 1)
            assert not any(env.action_space.low + 1)
        else:
            raise RuntimeError(
                "\n| Error in get_gym_env_info()"
                "\n  Please set these value manually: if_discrete=bool, action_dim=int."
                "\n  And keep action_space in (-1, 1)."
            )
    else:
        env_name = env.env_name
        max_step = env.max_step
        state_dim = env.state_dim
        action_dim = env.action_dim
        if_discrete = env.if_discrete
        target_return = env.target_return

    env_args = {
        "env_num": env_num,
        "env_name": env_name,
        "max_step": max_step,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "if_discrete": if_discrete,
        "target_return": target_return,
    }
    if if_print:
        env_args_repr = repr(env_args)
        env_args_repr = env_args_repr.replace(",", ",\n   ")
        env_args_repr = env_args_repr.replace("{", "{\n    ")
        env_args_repr = env_args_repr.replace("}", ",\n}")
        print(f"env_args = {env_args_repr}")
    return env_args


def kwargs_filter(func, kwargs: dict):
    """
    Filter the variable in env func.

    :param func: the function for creating an env.
    :param kwargs: args for the env.
    :return: filtered args.
    """
    import inspect

    sign = inspect.signature(func).parameters.values()
    sign = {val.name for val in sign}

    common_args = sign.intersection(kwargs.keys())
    return {key: kwargs[key] for key in common_args}  # filtered kwargs


def build_env(env=None, env_func=None, env_args=None):
    """
    Create an environment. If pass an existed env, copy a new one.

    :param env: an existed environment. (please pass None for now)
    :param env_func: the function for creating an env.
    :param env_args: the args for the env. Please take look at the demo.
    :return: an environment.
    """
    if env is not None:
        env = deepcopy(env)
    elif env_func.__module__ == "gym.envs.registration":
        import gym

        gym.logger.set_level(40)  # Block warning
        env = env_func(id=env_args["env_name"])
    else:
        env = env_func(**kwargs_filter(env_func.__init__, env_args.copy()))

    env.max_step = env.max_step if hasattr(env, "max_step") else env_args["max_step"]
    env.if_discrete = (
        env.if_discrete if hasattr(env, "if_discrete") else env_args["if_discrete"]
    )
    return env
